"""STT — Voxtral Mini (transformers, CUDA, full precision bf16).

API transformers (>= 4.54) : `VoxtralForConditionalGeneration` + `AutoProcessor`.
Transcription batch d'un segment audio accumulé (le proxy découpe déjà en ~3s via
commit). Voxtral attend du 16 kHz mono → resample depuis le 24 kHz du contrat client.

⚠️ Non testé hors GPU. Point d'intégration isolé : si l'API processor diffère selon
la version transformers, ajuster ici uniquement.
"""

import numpy as np
import torch
from scipy.signal import resample_poly

from .config import STT_MODEL, SAMPLE_RATE, STT_RATE, TORCH_DEVICE


class VoxtralSTT:
    def __init__(self, model_id: str = STT_MODEL):
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        print(f"[stt] loading {model_id} (bf16, {TORCH_DEVICE}) ...")
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = VoxtralForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=TORCH_DEVICE
        )
        print("[stt] ready")

    def transcribe(self, audio: np.ndarray, lang: str | None = None) -> str:
        """audio: float32 mono @ SAMPLE_RATE (24k), valeurs dans [-1, 1]."""
        if audio.size == 0:
            return ""

        if SAMPLE_RATE != STT_RATE:
            audio = resample_poly(audio, STT_RATE, SAMPLE_RATE).astype(np.float32)

        # Requête de transcription Voxtral (langue auto si lang=None).
        inputs = self._processor.apply_transcription_request(
            audio=audio,
            sampling_rate=STT_RATE,
            language=lang,
            model_id=STT_MODEL,
            format=["WAV"],
        ).to(self._model.device, dtype=self._model.dtype)

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=256)

        # On décode uniquement les tokens générés (après le prompt).
        decoded = self._processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return decoded[0].strip() if decoded else ""
