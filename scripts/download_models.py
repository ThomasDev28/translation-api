"""Prépare les 3 modèles sur le pod GPU. À lancer une fois après l'install.

- STT  : snapshot mistralai/Voxtral-Mini-3B-2507 (transformers le charge).
- TTS  : snapshot mistralai/Voxtral-4B-TTS-2603 (servi par vLLM).
- MADLAD: download google/madlad400-3b-mt + conversion CTranslate2 float16 (GPU).
"""

import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import STT_MODEL, TTS_MODEL, MT_MODEL, MT_CT2_DIR  # noqa: E402


def fetch(repo_id: str) -> None:
    print(f"\n=== snapshot {repo_id} ===")
    snapshot_download(repo_id=repo_id)


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
