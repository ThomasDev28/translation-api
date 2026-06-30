"""Client de test WebSocket pour /translate.

Usage:
    pip install websockets soundfile numpy
    python scripts/test_ws.py input.wav fr en
Lit un wav, le renvoie en PCM16 16 kHz mono, affiche le texte, écrit out_tts.wav.
"""
import asyncio
import json
import sys

import numpy as np
import soundfile as sf
import websockets

URL = "ws://127.0.0.1:9003/translate"
CHUNK = 3200  # ~100 ms @ 16 kHz


async def main(path: str, src: str, tgt: str) -> None:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        n = round(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype("float32")
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()

    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps(
            {"type": "start", "sourceLang": src, "targetLang": tgt, "voice": None}))
        for i in range(0, len(pcm), CHUNK):
            await ws.send(pcm[i:i + CHUNK])
        await ws.send(json.dumps({"type": "commit"}))

        out = bytearray()
        while True:
            m = await ws.recv()
            if isinstance(m, bytes):
                out += m
            else:
                e = json.loads(m)
                if e.get("type") == "partial_text":
                    print(f"[{e['stage']}] {e['text']}")
                elif e.get("type") == "done":
                    break
        if out:
            pcm16 = np.frombuffer(bytes(out), dtype="<i2").astype("float32") / 32767
            sf.write("out_tts.wav", pcm16, 16000)
            print(f"TTS écrit: out_tts.wav ({len(pcm16)/16000:.1f}s)")
        else:
            print("Pas d'audio TTS reçu")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3]))
