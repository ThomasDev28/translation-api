"""Client de test : envoie un WAV, récupère l'audio traduit.

Usage:
    python test_client.py samples/bonjour_fr.wav --src fr --tgt en [--out out.wav]
"""

import argparse
import asyncio
import json

import numpy as np
import soundfile as sf
import websockets

SAMPLE_RATE = 24000
URL = "ws://127.0.0.1:8000/translate"


def load_pcm16(path: str) -> bytes:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:  # stéréo → mono
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:  # rééchantillonnage linéaire simple (test only)
        n = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


async def run(path: str, src: str, tgt: str, out: str) -> None:
    pcm = load_pcm16(path)
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "sourceLang": src, "targetLang": tgt, "voice": None}))

        # Envoie l'audio par paquets de ~85ms (comme le worklet client).
        frame = int(SAMPLE_RATE * 0.085) * 2
        for i in range(0, len(pcm), frame):
            await ws.send(pcm[i : i + frame])
        await ws.send(json.dumps({"type": "commit"}))

        out_pcm = bytearray()
        async for msg in ws:
            if isinstance(msg, bytes):
                out_pcm.extend(msg)
            else:
                evt = json.loads(msg)
                if evt.get("type") == "partial_text":
                    print(f"[{evt['stage']}] {evt['text']}")
                elif evt.get("type") == "done":
                    break

    if out_pcm:
        audio = np.frombuffer(bytes(out_pcm), dtype="<i2").astype(np.float32) / 32768.0
        sf.write(out, audio, SAMPLE_RATE)
        print(f"\n✅ audio traduit écrit → {out} ({len(audio)/SAMPLE_RATE:.1f}s)")
    else:
        print("\n⚠️ aucun audio reçu")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--src", default="fr")
    p.add_argument("--tgt", default="en")
    p.add_argument("--out", default="out.wav")
    a = p.parse_args()
    asyncio.run(run(a.wav, a.src, a.tgt, a.out))
