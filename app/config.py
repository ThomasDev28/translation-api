"""Configuration centrale — surchargeable par variables d'environnement.

Cible : GPU NVIDIA (RunPod RTX 4090), modèles full-precision (bf16/fp16), pas de
quantization. ~20 GB VRAM → tient sur 24 GB.
"""

import os

# --- Modèles (poids HuggingFace standards, full precision) ---
# STT : Voxtral Mini 3B (transcription batch, API transformers documentée).
STT_MODEL = os.getenv("STT_MODEL", "mistralai/Voxtral-Mini-3B-2507")
# TTS : Voxtral 4B TTS, servi par vLLM Omni (endpoint OpenAI-compatible).
TTS_MODEL = os.getenv("TTS_MODEL", "mistralai/Voxtral-4B-TTS-2603")
TTS_VLLM_URL = os.getenv("TTS_VLLM_URL", "http://127.0.0.1:8001/v1")
# MT : MADLAD-400-3B, poids normaux, converti CTranslate2 float16 au déploiement.
MT_MODEL = os.getenv("MT_MODEL", "google/madlad400-3b-mt")
MT_CT2_DIR = os.getenv("MT_CT2_DIR", "/models/madlad400-3b-ct2-fp16")
MT_DEVICE = os.getenv("MT_DEVICE", "cuda")
MT_COMPUTE_TYPE = os.getenv("MT_COMPUTE_TYPE", "float16")

# Device torch pour STT.
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cuda")

# --- Audio ---
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))  # PCM16 mono, aligné sur le worklet client
STT_RATE = 16000  # Voxtral attend du 16 kHz mono

# --- Voix TTS par défaut (Voxtral) ---
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "neutral_female")

# --- Garde-fous segment ---
MAX_SEGMENT_SECONDS = float(os.getenv("MAX_SEGMENT_SECONDS", "15"))  # coupe les segments trop longs
