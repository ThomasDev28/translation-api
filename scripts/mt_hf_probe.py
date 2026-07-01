"""Probe MADLAD via transformers direct (pas CTranslate2).

Diagnostic : vérifie si MADLAD-3B traduit proprement en HF pur. Si oui, le charabia
vient du converter CTranslate2 (mauvais mapping embeddings), pas du modèle.

Usage (sur le pod GPU) :
    /venv/main/bin/python scripts/mt_hf_probe.py
"""

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL = "google/madlad400-3b-mt"

TESTS = [
    "<2en> Bonjour a toutes et a tous, et merci d'etre presents pour cette reunion.",
    "<2fr> Hello everyone, and thank you for attending this meeting.",
    "<2es> Hello everyone, and thank you for attending this meeting.",
    "<2de> Hello everyone, and thank you for attending this meeting.",
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = T5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    for txt in TESTS:
        ids = tok(txt, return_tensors="pt").input_ids.cuda()
        out = m.generate(ids, max_new_tokens=80, num_beams=4)
        print(txt[:28], "->", tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
