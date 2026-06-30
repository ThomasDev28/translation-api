#!/usr/bin/env bash
# Install + lancement direct sur un pod RunPod (template PyTorch/CUDA déjà fourni).
# Usage sur le pod :
#   git clone <repo> && cd translation-local && bash deploy/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "── GPU check ──"
nvidia-smi || { echo "Pas de GPU NVIDIA visible — abort."; exit 1; }

echo "── install deps ──"
pip install --upgrade pip
# torch compilé pour CUDA 12.4 (compatible drivers 12.4+, dont 12.8). Évite que
# pip tire un torch trop récent (cu12.9) incompatible avec le driver du pod.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install "vllm>=0.18"

echo "── lancement (vLLM TTS + download + API) ──"
bash deploy/start.sh
