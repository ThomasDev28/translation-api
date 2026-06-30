#!/usr/bin/env bash
# Démarrage complet sur le pod GPU (RunPod RTX 4090).
#   1. serveur vLLM Omni pour Voxtral TTS  (port 8001)
#   2. download STT + conversion MADLAD CT2 float16
#   3. API de traduction FastAPI           (port 8000)
#
# ⚠️ Non testé hors GPU. À débugger sur le pod (les commandes vLLM/transformers
#    peuvent demander un ajustement selon les versions installées).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "── 0/3 : libs CUDA 12 pour CTranslate2 (MADLAD) ──"
# CTranslate2 est buildé pour CUDA 12 et exige libcublas.so.12 / libcudnn.
# Le pod tourne en CUDA 13 → on fournit les libs cu12 via les wheels nvidia.
python -m pip install --quiet nvidia-cublas-cu12 nvidia-cudnn-cu12
CT2_LIBS=$(python -c "import nvidia.cublas, nvidia.cudnn, os; print(os.path.dirname(nvidia.cublas.__file__)+'/lib:'+os.path.dirname(nvidia.cudnn.__file__)+'/lib')")
export LD_LIBRARY_PATH="${CT2_LIBS}:${LD_LIBRARY_PATH:-}"
echo "   LD_LIBRARY_PATH += ${CT2_LIBS}"

echo "── 1/3 : vLLM Omni (Voxtral TTS) sur :8001 ──"
# vllm-omni sert l'endpoint OpenAI-compatible /v1/audio/speech.
vllm serve "${TTS_MODEL:-mistralai/Voxtral-4B-TTS-2603}" \
  --port 8001 \
  --gpu-memory-utilization 0.35 \
  > /tmp/vllm_tts.log 2>&1 &
VLLM_PID=$!
echo "   vLLM PID $VLLM_PID (logs: /tmp/vllm_tts.log)"

echo "── 2/3 : download STT + conversion MADLAD ──"
python scripts/download_models.py

echo "   attente vLLM TTS prêt..."
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then echo "   vLLM ready"; break; fi
  sleep 5
done

echo "── 3/3 : API traduction FastAPI sur :8000 ──"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
