"""FastAPI + WebSocket — backend `TRANSLATION_BACKEND=local` de Suite 366.

Contrat WebSocket identique côté forme au proxy OpenAI : start / frames PCM16 / commit,
réponses en frames PCM16 + events JSON.

Traitement pipeliné : la boucle de réception ne fait que bufferiser l'audio et
snapshotter les segments au commit. L'inférence (STT/MT/TTS, bloquante) tourne
dans un worker par session via un thread — les commits suivants s'empilent dans
une queue au lieu de bloquer la réception (sinon, sur un long discours, les
phrases fusionnent et les sorties arrivent en rafale).
"""

import asyncio
import json
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .pipeline import TranslationPipeline

app = FastAPI(title="translation-local")

# Chargé une fois au démarrage (les 3 modèles en RAM).
pipeline: TranslationPipeline | None = None

# Sérialise l'inférence GPU entre sessions : les modèles sont partagés et les
# appels generate() concurrents risquent l'OOM (pics d'activations cumulés).
# Pris stage par stage → deux sessions peuvent s'entrelacer entre les étapes.
GPU_LOCK = threading.Lock()


def _next_item(gen):
    """Avance le générateur d'un stage sous lock GPU (exécuté dans un thread)."""
    with GPU_LOCK:
        return next(gen, None)


@app.on_event("startup")
def _load() -> None:
    global pipeline
    pipeline = TranslationPipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if pipeline else "loading"}


@app.websocket("/translate")
async def translate(ws: WebSocket) -> None:
    await ws.accept()
    session = None
    # Segments snapshottés en attente de traitement (None = fin de session).
    segments: asyncio.Queue[bytes | None] = asyncio.Queue()
    worker_task: asyncio.Task | None = None

    async def worker() -> None:
        """Consomme les segments dans l'ordre, streame les résultats.

        Un send sur socket fermée lève : on laisse remonter pour terminer la
        task — la boucle de réception gère le disconnect de son côté.
        """
        while True:
            pcm = await segments.get()
            if pcm is None:
                return
            gen = session.process(pcm)
            while True:
                # Chaque next() = un stage (STT, MT, chunk TTS) dans un thread :
                # l'event loop reste libre pour recevoir l'audio suivant.
                item = await asyncio.to_thread(_next_item, gen)
                if item is None:
                    break
                if item[0] == "text":
                    await ws.send_text(
                        json.dumps({"type": "partial_text", "stage": item[1], "text": item[2]})
                    )
                else:  # audio
                    await ws.send_bytes(item[1])
            await ws.send_text(json.dumps({"type": "done"}))

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            # --- Message de contrôle (JSON texte) ---
            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                kind = data.get("type")

                if kind == "start":
                    if session:
                        session.close()
                    session = pipeline.new_session(
                        src=data["sourceLang"],
                        tgt=data["targetLang"],
                        voice=data.get("voice"),
                        text_only=data.get("textOnly", False),
                    )
                    if worker_task is None:
                        worker_task = asyncio.create_task(worker())

                elif kind == "commit" and session:
                    # Snapshot immédiat : fige la frontière du segment, le
                    # traitement part dans le worker.
                    pcm = session.take()
                    if pcm:
                        segments.put_nowait(pcm)

                elif kind == "stop":
                    break

            # --- Frame audio (binaire PCM16) ---
            elif msg.get("bytes") is not None and session:
                session.feed(msg["bytes"])

    except WebSocketDisconnect:
        pass
    finally:
        if worker_task:
            worker_task.cancel()
        if session:
            session.close()
