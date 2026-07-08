"""Pré-télécharge les 3 modèles sur le pod GPU. À lancer une fois après l'install.

- STT   : mistralai/Voxtral-Mini-3B-2507 (transformers).
- TTS   : mistralai/Voxtral-4B-TTS-2603  (servi par vLLM).
- MADLAD : MT_MODEL (défaut google/madlad400-7b-mt-bt) — chargé DIRECT par transformers
  au runtime (app/mt.py), avec réparation embeddings EN MÉMOIRE. Pas de conversion
  CTranslate2 : le converter re-casse l'aliasing d'embeddings MADLAD (cf app/mt.py).
"""

import os

# Le backend Xet de HF crashe sur certains hosts et ralentit le download. On force
# le HTTP classique AVANT d'importer huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import STT_MODEL, TTS_MODEL, MT_MODEL  # noqa: E402


def fetch(repo_id: str, retries: int = 4) -> None:
    print(f"\n=== snapshot {repo_id} ===")
    for attempt in range(1, retries + 1):
        try:
            # max_workers bas = moins de connexions concurrentes, plus stable.
            # Reprend les fichiers déjà téléchargés.
            snapshot_download(repo_id=repo_id, max_workers=4)
            return
        except (HfHubHTTPError, RuntimeError, OSError) as e:
            if attempt == retries:
                raise
            print(f"  ⚠️  échec {attempt}/{retries} ({e}); retry…")


def main() -> None:
    fetch(STT_MODEL)
    fetch(TTS_MODEL)
    fetch(MT_MODEL)
    print("\n✅ Modèles prêts.")


if __name__ == "__main__":
    main()
