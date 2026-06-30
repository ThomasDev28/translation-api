"""Prépare les 3 modèles sur le pod GPU. À lancer une fois après l'install.

- STT  : snapshot mistralai/Voxtral-Mini-3B-2507 (transformers le charge).
- TTS  : snapshot mistralai/Voxtral-4B-TTS-2603 (servi par vLLM).
- MADLAD: download google/madlad400-3b-mt + conversion CTranslate2 float16 (GPU).
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


def convert_madlad() -> None:
    out = Path(MT_CT2_DIR)
    if out.exists():
        print(f"=== MADLAD CT2 déjà présent: {out} ===")
        return
    print(f"\n=== conversion MADLAD → CTranslate2 float16 ({out}) ===")
    subprocess.run(
        [
            "ct2-transformers-converter",
            "--model", MT_MODEL,
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
