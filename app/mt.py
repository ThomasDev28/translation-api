"""MT — MADLAD-400-7B via CTranslate2 (CUDA, float16).

MADLAD = T5 multilingue 400+ langues. Poids normaux convertis CTranslate2 float16
au déploiement (scripts/download_models.py) → tourne sur GPU, ~14 GB VRAM.

MADLAD cible la langue via un préfixe `<2xx>` (ex: `<2en>` pour l'anglais).
"""

import ctranslate2
from transformers import AutoTokenizer

from .config import MT_MODEL, MT_CT2_DIR, MT_DEVICE, MT_COMPUTE_TYPE


class MadladMT:
    def __init__(self, ct2_dir: str = MT_CT2_DIR, hf_model: str = MT_MODEL):
        print(f"[mt] loading CTranslate2 {ct2_dir} ({MT_DEVICE}/{MT_COMPUTE_TYPE}) ...")
        self._translator = ctranslate2.Translator(
            ct2_dir, device=MT_DEVICE, compute_type=MT_COMPUTE_TYPE
        )
        # Tokenizer MADLAD (spiece) depuis le modèle HF d'origine.
        self._tok = AutoTokenizer.from_pretrained(hf_model)
        print("[mt] ready")

    def translate(self, text: str, src: str, tgt: str) -> str:
        text = text.strip()
        if not text:
            return ""

        # Préfixe langue cible MADLAD. (src non utilisé : MADLAD détecte la source.)
        source = self._tok.convert_ids_to_tokens(self._tok.encode(f"<2{tgt}> {text}"))

        results = self._translator.translate_batch(
            [source],
            max_decoding_length=256,
        )
        target_tokens = results[0].hypotheses[0]
        return self._tok.decode(
            self._tok.convert_tokens_to_ids(target_tokens),
            skip_special_tokens=True,
        ).strip()
