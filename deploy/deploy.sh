#!/usr/bin/env bash
# Install + lancement direct sur un pod RunPod (template PyTorch/CUDA déjà fourni).
# Usage sur le pod :
#   git clone <repo> && cd translation-local && bash deploy/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "── GPU check ──"
nvidia-smi || { echo "Pas de GPU NVIDIA visible — abort."; exit 1; }

echo "── install deps ──"
# ⚠️ EXIGE un pod avec driver CUDA récent (≥ 13.x / driver ≥ 575). vLLM + torch 2.11
# sont compilés pour CUDA 13 → un host CUDA 12.x échoue (libcudart.so.13 manquant).
#   nvidia-smi  → vérifier "CUDA Version: 13.x" avant de lancer.
nvidia-smi | grep -i "cuda version" || true

pip install --upgrade pip
# On NE touche PAS à torch : on garde celui du template (cohérent avec vLLM/torchvision).
pip install -e .
pip install "vllm>=0.18"

echo "── lancement (vLLM TTS + download + API) ──"
bash deploy/start.sh
