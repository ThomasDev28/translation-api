"""Probe MADLAD-3B via transformers : inspecte les embeddings + teste le fix en mémoire.

Le bug : transformers>=5 charge mal MADLAD (T5 tie_word_embeddings=False). Il alias
shared/encoder.embed sur lm_head → l'encoder embed du bruit → charabia. Les VRAIS
embeddings vocab sont dans decoder.embed_tokens (std ~13) ; lm_head garde std ~0.16.

On répare EN MÉMOIRE (pas de save/reload qui re-casse) puis on génère.

Usage (pod GPU) :
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


def gen(tok, m, tag):
    print(f"\n===== {tag} =====")
    for txt in TESTS:
        ids = tok(txt, return_tensors="pt").input_ids.cuda()
        out = m.generate(ids, max_new_tokens=80, num_beams=4)
        print(txt[:28], "->", tok.decode(out[0], skip_special_tokens=True))


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = T5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

    # Inspecte : quel tensor porte les vrais embeddings ?
    print("--- stds (bf16) ---")
    for name, w in [
        ("shared", m.shared.weight),
        ("encoder.embed_tokens", m.encoder.embed_tokens.weight),
        ("decoder.embed_tokens", m.decoder.embed_tokens.weight),
        ("lm_head", m.lm_head.weight),
    ]:
        print(f"  {name:24s} std={w.float().std().item():.4f} "
              f"shape={tuple(w.shape)} id={id(w.data_ptr()) if False else w.data_ptr()}")

    gen(tok, m, "AVANT fix (brut)")

    # Fix en mémoire : shared = encoder = decoder = vrais embeddings (decoder.embed).
    import torch.nn as nn
    real_emb = m.decoder.embed_tokens.weight.data.clone()
    shared = nn.Parameter(real_emb)
    m.shared.weight = shared
    m.encoder.embed_tokens.weight = shared
    m.decoder.embed_tokens.weight = shared
    # lm_head garde ses poids propres (déjà chargés).

    gen(tok, m, "APRES fix (shared=decoder.embed)")


if __name__ == "__main__":
    main()
