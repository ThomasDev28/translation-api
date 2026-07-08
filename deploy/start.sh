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

# Répartition multi-GPU (2× RTX 4090). TTS (vLLM omni, 2 stages, ~16 Go) sur un
# GPU dédié ; STT+MT (uvicorn, ~20 Go avec MADLAD-7B) sur l'autre. Sans ce split
# les deux ciblent cuda:0 → OOM. Sur une seule carte, mets TTS_GPU=0 APP_GPU=0.
# Défaut mono-GPU : les 3 modèles sur GPU 0. Pour 2 cartes : APP_GPU=1.
TTS_GPU="${TTS_GPU:-0}"
APP_GPU="${APP_GPU:-0}"
# Fraction VRAM pour vLLM TTS. Mono-GPU partagé avec STT+MT → 0.45 (≈21 Go sur
# 48 Go) leur laisse ~27 Go. Si TTS seul sur son GPU (2× cartes), monte à 0.85.
TTS_MEM_UTIL="${TTS_MEM_UTIL:-0.45}"
# TTS_ENABLED=false : serveur sous-titres only — skip vLLM Omni entièrement
# (~21 Go VRAM économisés, seuls STT+MT chargés). L'app sert le texte traduit.
TTS_ENABLED="${TTS_ENABLED:-true}"

echo "── 0/3 : libs CUDA 12 pour CTranslate2 (MADLAD) ──"
# CTranslate2 est buildé pour CUDA 12 et exige libcublas.so.12 / libcudnn.
# Le pod tourne en CUDA 13 → on fournit les libs cu12 via les wheels nvidia.
python -m pip install --quiet nvidia-cublas-cu12 nvidia-cudnn-cu12
CT2_LIBS=$(python -c "import nvidia.cublas, nvidia.cudnn; print(list(nvidia.cublas.__path__)[0]+'/lib:'+list(nvidia.cudnn.__path__)[0]+'/lib')")
export LD_LIBRARY_PATH="${CT2_LIBS}:${LD_LIBRARY_PATH:-}"
echo "   LD_LIBRARY_PATH += ${CT2_LIBS}"

if [ "${TTS_ENABLED}" = "true" ]; then
  echo "── 1/3 : vLLM Omni (Voxtral TTS) sur :8001 ──"
  # Voxtral-4B-TTS (model_type=voxtral_tts) n'est PAS supporté par vLLM stock :
  # il a un module acoustic_transformer que MistralForCausalLM ignore. Le support
  # vient du plugin vllm-omni, qui enregistre l'archi et ajoute le flag --omni.
  # vllm-omni exige une vLLM de MÊME mineure : omni 0.22.0 ↔ vllm 0.22.x. Le pod
  # arrive en vllm 0.24.0 (ImportError supports_xccl) → on pin la paire 0.22.
  python -m pip install --quiet "vllm==0.22.*" "vllm-omni==0.22.0"
  # vllm-omni sert l'endpoint OpenAI-compatible /v1/audio/speech.
  # Seul sur son GPU (TTS_GPU) → 0.85 : large pour les poids 4B + le KV-cache
  # des 2 stages omni (LM + acoustic). STT/MT vivent sur APP_GPU, pas de partage.
  CUDA_VISIBLE_DEVICES="${TTS_GPU}" vllm serve "${TTS_MODEL:-mistralai/Voxtral-4B-TTS-2603}" \
    --omni \
    --port 8001 \
    --gpu-memory-utilization "${TTS_MEM_UTIL}" \
    > /tmp/vllm_tts.log 2>&1 &
  VLLM_PID=$!
  echo "   vLLM PID $VLLM_PID (logs: /tmp/vllm_tts.log)"
else
  echo "── 1/3 : vLLM Omni SKIPPÉ (TTS_ENABLED=false — sous-titres only) ──"
fi

echo "── 2/3 : download STT + conversion MADLAD ──"
python scripts/download_models.py

if [ "${TTS_ENABLED}" = "true" ]; then
  echo "   attente vLLM TTS prêt..."
  for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then echo "   vLLM ready"; break; fi
    sleep 5
  done
fi

echo "── 3/3 : API traduction FastAPI sur :8000 (GPU ${APP_GPU}) ──"
exec env CUDA_VISIBLE_DEVICES="${APP_GPU}" TTS_ENABLED="${TTS_ENABLED}" \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
