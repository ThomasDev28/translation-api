"""FastAPI + WebSocket — backend `TRANSLATION_BACKEND=local` de Suite 366.

Contrat WebSocket identique côté forme au proxy OpenAI : start / frames PCM16 / commit,
réponses en frames PCM16 + events JSON.
"""

import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .pipeline import TranslationPipeline

app = FastAPI(title="translation-local")

# Chargé une fois au démarrage (les 3 modèles en RAM).
pipeline: TranslationPipeline | None = None


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

                elif kind == "commit" and session:
                    for item in session.flush():
                        if item[0] == "text":
                            await ws.send_text(
                                json.dumps(
                                    {"type": "partial_text", "stage": item[1], "text": item[2]}
                                )
                            )
                        else:  # audio
                            await ws.send_bytes(item[1])
                    await ws.send_text(json.dumps({"type": "done"}))

                elif kind == "stop":
                    break

            # --- Frame audio (binaire PCM16) ---
            elif msg.get("bytes") is not None and session:
                session.feed(msg["bytes"])

    except WebSocketDisconnect:
        pass
    finally:
        if session:
            session.close()
