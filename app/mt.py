"""MT — MADLAD-400 via transformers (T5, CUDA bfloat16).

CTranslate2 abandonné : son converter (comme transformers>=5) mis-charge MADLAD.
MADLAD est un T5 `tie_word_embeddings=false` dont le checkpoint alias
shared / encoder.embed_tokens sur `lm_head` (std ~0.16) au lieu des VRAIS
embeddings vocab, qui vivent dans `decoder.embed_tokens` (std ~13). L'encodeur
embed alors du bruit → charabia ("们 们 们…"). Tout round-trip save/reload
(HF ou CT2) re-déclenche l'aliasing → on répare EN MÉMOIRE au chargement, une
fois, sans jamais resauvegarder : shared = encoder = decoder = vrais embeddings.

MADLAD cible la langue via un préfixe `<2xx>` (ex: `<2en>` pour l'anglais).
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .config import MT_MODEL, MT_DEVICE, MT_COMPUTE_TYPE, MT_BEAM_SIZE

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class MadladMT:
    def __init__(self, hf_model: str = MT_MODEL):
        dtype = _DTYPES.get(MT_COMPUTE_TYPE, torch.bfloat16)
        self._device = "cuda" if MT_DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        print(f"[mt] loading MADLAD {hf_model} (transformers {self._device}/{dtype}) ...")
        self._tok = AutoTokenizer.from_pretrained(hf_model)
        m = T5ForConditionalGeneration.from_pretrained(hf_model, dtype=dtype)
        self._fix_embeddings(m)
        self._model = m.to(self._device).eval()
        print("[mt] ready")

    @staticmethod
    def _fix_embeddings(m) -> None:
        """Remet les vrais embeddings vocab (decoder.embed_tokens, std ~13) dans
        shared / encoder / decoder. lm_head garde ses poids propres (std ~0.16)."""
        real_emb = m.decoder.embed_tokens.weight.data.clone()
        shared = nn.Parameter(real_emb)
        m.shared.weight = shared
        m.encoder.embed_tokens.weight = shared
        m.decoder.embed_tokens.weight = shared

    @torch.inference_mode()
    def translate(self, text: str, src: str, tgt: str) -> str:
        text = text.strip()
        if not text:
            return ""
        # Préfixe langue cible MADLAD. (src non utilisé : MADLAD détecte la source.)
        ids = self._tok(f"<2{tgt}> {text}", return_tensors="pt").input_ids.to(self._device)
        out = self._model.generate(ids, num_beams=MT_BEAM_SIZE, max_new_tokens=256)
        return self._tok.decode(out[0], skip_special_tokens=True).strip()
