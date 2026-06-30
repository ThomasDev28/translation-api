"""TTS — Voxtral 4B TTS via vLLM Omni (endpoint OpenAI-compatible `/v1/audio/speech`).

Mistral recommande vLLM Omni pour servir Voxtral TTS. Le serveur vLLM est lancé à
part (deploy/start.sh) et écoute sur TTS_VLLM_URL. Ici on est un simple client HTTP :
texte (langue cible) + voix → WAV → float32 @ SAMPLE_RATE.

Non-streaming : vLLM renvoie le WAV complet du segment (les segments sont déjà courts,
~3s, découpés par le commit côté proxy).

⚠️ Non testé hors GPU. Point d'intégration isolé : payload/route ajustables ici.
"""

import io
from collections.abc import Iterator

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import TTS_MODEL, TTS_VLLM_URL, SAMPLE_RATE, DEFAULT_VOICE


class VoxtralTTS:
    def __init__(self, base_url: str = TTS_VLLM_URL, model_id: str = TTS_MODEL):
        self._url = f"{base_url.rstrip('/')}/audio/speech"
        self._model = model_id
        self._client = httpx.Client(timeout=60.0)
        print(f"[tts] vLLM Omni client → {self._url}")

    def synthesize(
        self, text: str, lang: str | None = None, voice: str | None = None
    ) -> Iterator[np.ndarray]:
        """Yield le chunk audio float32 mono @ SAMPLE_RATE (segment complet)."""
        text = text.strip()
        if not text:
            return

        payload = {
            "model": self._model,
            "input": text,
            "voice": voice or DEFAULT_VOICE,
            "response_format": "wav",
        }
        resp = self._client.post(self._url, json=payload)
        resp.raise_for_status()

        audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)
        if audio.size:
            yield audio
