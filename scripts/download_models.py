"""Prépare les 3 modèles sur le pod GPU. À lancer une fois après l'install.

- STT  : snapshot mistralai/Voxtral-Mini-3B-2507 (transformers le charge).
- TTS  : snapshot mistralai/Voxtral-4B-TTS-2603 (servi par vLLM).
- MADLAD: download MT_MODEL (défaut google/madlad400-7b-mt) + conversion CTranslate2 float16 (GPU).
"""

import os
import subprocess
import sys
from pathlib import Path

# Le backend Xet de HF crashe sur Vast.ai ("Internal Writer Error: Background
# writer channel closed") et ralentit le download. On force le HTTP classique
# AVANT d'importer huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import STT_MODEL, TTS_MODEL, MT_MODEL, MT_CT2_DIR  # noqa: E402


def fetch(repo_id: str, retries: int = 4) -> None:
    print(f"\n=== snapshot {repo_id} ===")
    for attempt in range(1, retries + 1):
        try:
            # max_workers bas = moins de connexions concurrentes, plus stable
            # sur les hosts Vast.ai. Reprend les fichiers déjà téléchargés.
            snapshot_download(repo_id=repo_id, max_workers=4)
            return
        except (HfHubHTTPError, RuntimeError, OSError) as e:
            if attempt == retries:
                raise
            print(f"  ⚠️  échec {attempt}/{retries} ({e}); retry…")


def _save_fixed_madlad(dst: Path) -> None:
    """Charge MADLAD, répare l'aliasing d'embeddings, sauvegarde un HF corrigé.

    MADLAD est un T5 `tie_word_embeddings=false` dont le checkpoint stocke
    `decoder.embed_tokens.weight` (les vrais embeddings) + `lm_head.weight`,
    sans `shared.weight`. transformers >= 5 charge mal ce layout : il alias
    `shared` / `encoder.embed_tokens` sur le `lm_head` au lieu des embeddings
    → l'encodeur embed du bruit → traduction en charabia ("ne pro ne pro…").
    On rétablit : shared = encoder.embed = decoder.embed = vrais embeddings ;
    lm_head garde ses poids propres. Le HF corrigé (avec un vrai `shared.weight`)
    se recharge ensuite correctement, y compris par ct2-transformers-converter.
    """
    import torch.nn as nn
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    print(f"=== réparation embeddings MADLAD → {dst} ===")
    m = T5ForConditionalGeneration.from_pretrained(MT_MODEL, torch_dtype="float32")
    real_emb = m.decoder.embed_tokens.weight.data.clone()   # std ~13 : vrais embeddings
    real_lmhead = m.lm_head.weight.data.clone()             # std ~0.16 : tête de sortie
    shared = nn.Parameter(real_emb)
    m.shared.weight = shared
    m.encoder.embed_tokens.weight = shared
    m.decoder.embed_tokens.weight = shared
    m.lm_head.weight = nn.Parameter(real_lmhead)
    m.save_pretrained(dst)
    AutoTokenizer.from_pretrained(MT_MODEL).save_pretrained(dst)


def convert_madlad() -> None:
    out = Path(MT_CT2_DIR)
    if out.exists():
        print(f"=== MADLAD CT2 déjà présent: {out} ===")
        return
    model_slug = MT_MODEL.rsplit("/", 1)[-1]
    fixed = out.parent / f"{model_slug}-hf-fixed"
    if not fixed.exists():
        _save_fixed_madlad(fixed)
    print(f"\n=== conversion MADLAD → CTranslate2 float16 ({out}) ===")
    subprocess.run(
        [
            "ct2-transformers-converter",
            "--model", str(fixed),
            "--output_dir", str(out),
            "--quantization", "float16",
            "--force",
        ],
        check=True,
    )


def main() -> None:
    fetch(STT_MODEL)
    fetch(TTS_MODEL)
    convert_madlad()
    print("\n✅ Modèles prêts.")


if __name__ == "__main__":
    main()
