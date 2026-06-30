# translation-local

API de **traduction vocale temps réel** (audio → audio) pour **GPU NVIDIA**
(RunPod RTX 4090, Scaleway L4/L40S...). Modèles full-precision, souveraine
(self-hostée, aucune dépendance cloud US).

Backend `LOCAL_TRANSLATION_URL` de la feature de traduction des meetings de Suite 366 :
même contrat WebSocket que l'ancien proxy OpenAI, mais self-hosted.

## Pipeline

```
audio src ──► [Voxtral STT] ──► texte src ──► [MADLAD-400] ──► texte tgt ──► [Voxtral TTS] ──► audio tgt
              transformers bf16              CTranslate2 fp16              vLLM Omni (fp16)
```

| Brique | Modèle | Précision | ~VRAM | Serving |
| ------ | ------ | --------- | ----- | ------- |
| STT | `mistralai/Voxtral-Mini-3B-2507` | bf16 | ~6 GB | transformers (in-process) |
| MT  | `google/madlad400-3b-mt` | fp16 | ~6 GB | CTranslate2 CUDA (in-process) |
| TTS | `mistralai/Voxtral-4B-TTS-2603` | fp16 | ~8 GB | vLLM Omni (port 8001) |

**Total ~20 GB VRAM** → tient sur une RTX 4090 (24 GB). Sur ce hardware, RTF ~0.3
→ paragraphe continu fluide (vs ~2.7 sur Mac, qui divergeait).

## Pré-requis

- GPU NVIDIA ≥ 24 GB VRAM, CUDA 12.x.
- Python 3.11+.
- Sur RunPod : prendre un template **PyTorch / CUDA**.

## Déploiement sur un pod RunPod

```bash
git clone <repo> && cd translation-local
bash deploy/deploy.sh          # install deps + lance vLLM TTS + download + API
```

`deploy.sh` enchaîne :
1. `nvidia-smi` (vérif GPU)
2. `pip install -e .` + `vllm`
3. `deploy/start.sh` :
   - serveur **vLLM Omni** (Voxtral TTS) sur `:8001`
   - download STT + conversion **MADLAD → CTranslate2 fp16**
   - API **FastAPI** sur `:8000`

Premier lancement = download (~15-20 GB) + conversion MADLAD → quelques minutes.

## Tester

```bash
python test_client.py sample.wav --src fr --tgt en --out out.wav
```

## Brancher à Suite 366

Le pod RunPod expose un port public (proxy TCP RunPod). Dans `serveur/.env` :

```bash
MEETING_TRANSLATION_ENABLED=true
LOCAL_TRANSLATION_URL=ws://<IP_OU_PROXY_RUNPOD>:8000/translate
```

Le proxy (`serveur/src/lib/translation-ws-proxy.ts`) route vers cette API.
Pooling / feeder-subscriber / fan-out room inchangés.

## Contrat WebSocket (`/translate`)

```
Client → serveur :
  {"type":"start","sourceLang":"fr","targetLang":"en","voice":"neutral_female"}
  <binary PCM16 mono 24kHz>          # frames audio
  {"type":"commit"}                  # fin de segment (forced commit côté client)

Serveur → client :
  {"type":"partial_text","stage":"stt"|"mt","text":"..."}   # debug
  <binary PCM16 mono 24kHz>          # audio traduit
  {"type":"done"}
```

## Voix Voxtral TTS

`neutral_male`, `casual_male`, `neutral_female`, `casual_female`, `cheerful_female`
(+ voix par-langue `fr_*`, `de_*`, `es_*`... côté modèle). Alignées sur l'UI Suite 366.

## ⚠️ Honnêteté

- **Non testé hors GPU** : ce code cible CUDA, il n'a pas tourné sur Mac. Les appels
  transformers (STT) / vLLM (TTS) peuvent demander un ajustement selon les versions —
  les points d'intégration sont isolés dans `app/stt.py` et `app/tts.py`.
- **PoC séquentiel** (STT → MT → TTS en série). Suffisant pour valider qualité +
  latence réelle sur GPU.
- **Sécurité** : l'API n'a pas d'auth → ne pas exposer une IP publique sans token /
  firewall. OK pour un test sur port éphémère RunPod, à durcir pour la prod.
