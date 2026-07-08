"""Orchestration STT → MT → TTS.

PoC séquentiel : un segment audio complet entre, ressort traduit en audio.
Pas de pipelining inter-étapes (priorité simplicité pour valider qualité/latence).
"""

from collections.abc import Iterator

import numpy as np

from .config import SAMPLE_RATE, MAX_SEGMENT_SECONDS, TTS_ENABLED
from .stt import VoxtralSTT
from .mt import MadladMT
from .tts import VoxtralTTS


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """PCM16 little-endian → float32 [-1, 1]."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] → PCM16 little-endian bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class Session:
    """Un flux de traduction (un couple speaker→listener)."""

    def __init__(
        self,
        stt: VoxtralSTT,
        mt: MadladMT,
        tts: VoxtralTTS | None,
        src: str,
        tgt: str,
        voice,
        text_only: bool = False,
    ):
        self._stt, self._mt, self._tts = stt, mt, tts
        self.src, self.tgt, self.voice = src, tgt, voice
        self.text_only = text_only
        self._buf = bytearray()
        self._max_bytes = int(MAX_SEGMENT_SECONDS * SAMPLE_RATE * 2)  # 2 octets/sample

    def feed(self, pcm: bytes) -> None:
        self._buf.extend(pcm)
        # Garde-fou : évite qu'un segment sans commit explose la RAM.
        if len(self._buf) > self._max_bytes:
            del self._buf[: len(self._buf) - self._max_bytes]

    def flush(self):
        """Traite le buffer accumulé. Yield: ("text", stage, str) ou ("audio", bytes)."""
        if not self._buf:
            return
        audio = pcm16_to_float32(bytes(self._buf))
        self._buf.clear()

        # 1. STT
        text_src = self._stt.transcribe(audio, lang=self.src)
        yield ("text", "stt", text_src)
        if not text_src:
            return

        # 2. MT
        text_tgt = self._mt.translate(text_src, self.src, self.tgt)
        yield ("text", "mt", text_tgt)
        if not text_tgt:
            return

        # Mode sous-titres : le texte traduit suffit — skip TTS (étape la plus
        # coûteuse) → 'done' arrive 2-3× plus vite, zéro GPU synthèse.
        # Idem si le serveur tourne sans TTS (TTS_ENABLED=false) : une session
        # audio reçoit quand même le texte (dégradation gracieuse, pas de crash).
        if self.text_only or self._tts is None:
            return

        # 3. TTS (streaming)
        for chunk in self._tts.synthesize(text_tgt, lang=self.tgt, voice=self.voice):
            yield ("audio", float32_to_pcm16(chunk))

    def close(self) -> None:
        self._buf.clear()


class TranslationPipeline:
    """Charge les 3 modèles une fois, fabrique des sessions à la demande."""

    def __init__(self):
        self.stt = VoxtralSTT()
        self.mt = MadladMT()
        # TTS_ENABLED=false → serveur sous-titres only : pas de client TTS,
        # pas besoin du serveur vLLM Omni (~21 Go VRAM économisés).
        self.tts = VoxtralTTS() if TTS_ENABLED else None
        if not TTS_ENABLED:
            print("[pipeline] TTS désactivé (TTS_ENABLED=false) — sortie texte uniquement")
        self._warmup()
        print("[pipeline] ready")

    def _warmup(self) -> None:
        """Force la compilation des kernels Metal (MLX) sur une inférence bidon.

        Sans ça, la 1ère vraie phrase paie ~30s de compilation lazy en plein
        meeting. Ici on l'absorbe au démarrage du serveur.
        """
        print("[pipeline] warmup (compilation kernels Metal)...")
        try:
            # ~0.5s d'audio silencieux → STT
            silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
            self.stt.transcribe(silence, lang="en")
            # MT
            self.mt.translate("warmup", "en", "fr")
            # TTS (consomme le générateur pour déclencher la génération)
            if self.tts is not None:
                for _ in self.tts.synthesize("warmup", lang="fr"):
                    pass
        except Exception as e:  # warmup best-effort : ne bloque pas le boot
            print(f"[pipeline] warmup partiel: {e}")

    def new_session(self, src: str, tgt: str, voice=None, text_only: bool = False) -> Session:
        return Session(self.stt, self.mt, self.tts, src, tgt, voice, text_only=text_only)
