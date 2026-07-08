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
# MT : MADLAD-400-7B-bt (T5), chargé DIRECT par transformers (app/mt.py), ~14 Go
# VRAM bf16. Variante backtranslation : quasi la qualité du 10B sur les paires
# majeures (fr/en/de/es) et plus robuste aux entrées bruitées — or nos entrées
# SONT du texte STT (erreurs de transcription, oral mal formé). Sur 48 Go sans
# TTS : STT 7 + MT 14 + inférence ≈ 32 Go pic, marge large.
# ⚠️ 24 Go (TTS actif ou pas) → madlad400-3b-mt (sinon OOM sous charge).
MT_MODEL = os.getenv("MT_MODEL", "google/madlad400-7b-mt-bt")
MT_DEVICE = os.getenv("MT_DEVICE", "cuda")
# bfloat16 OBLIGATOIRE : MADLAD est un T5 (embeddings std ~13) qui overflow en
# float16 → charabia. bf16 (ou float32) donne des traductions OK.
MT_COMPUTE_TYPE = os.getenv("MT_COMPUTE_TYPE", "bfloat16")

# Device torch pour STT.
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cuda")

# --- Audio ---
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))  # PCM16 mono, aligné sur le worklet client
STT_RATE = 16000  # Voxtral attend du 16 kHz mono

# --- Voix TTS par défaut (Voxtral) ---
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "neutral_female")

# --- Serveur sous-titres uniquement ---
# TTS_ENABLED=false : ne pas utiliser le TTS du tout (déploiement sous-titres
# only). Économise ~21 Go VRAM : le serveur vLLM Omni (port 8001) n'a pas
# besoin d'être lancé. Les sessions audio reçues sont servies en texte seul.
TTS_ENABLED = os.getenv("TTS_ENABLED", "true").lower() == "true"

# --- Garde-fous segment ---
MAX_SEGMENT_SECONDS = float(os.getenv("MAX_SEGMENT_SECONDS", "15"))  # coupe les segments trop longs
