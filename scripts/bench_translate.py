"""Bench qualité MADLAD : matrice de traduction texte→texte (+ audio TTS).

À lancer SUR LE POD GPU (où MADLAD CT2 + vLLM TTS tournent).

Pour chaque langue source, traduit vers les 3 autres langues cibles, imprime le
texte traduit, et (option --audio) synthétise le WAV via vLLM Omni.

Usage:
    python scripts/bench_translate.py                     # texte seul
    python scripts/bench_translate.py --audio             # + WAV dans bench_out/
    python scripts/bench_translate.py --langs fr,en       # sous-ensemble sources
    python scripts/bench_translate.py --texts scripts/bench_texts.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mt import MadladMT  # noqa: E402

LANG_NAMES = {"fr": "Français", "en": "English", "es": "Español", "de": "Deutsch"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--texts", default=str(Path(__file__).parent / "bench_texts.json"))
    p.add_argument("--langs", default="fr,en,es,de", help="langues sources, CSV")
    p.add_argument("--audio", action="store_true", help="génère aussi les WAV (TTS)")
    p.add_argument("--outdir", default="bench_out")
    p.add_argument("--voice", default=None)
    a = p.parse_args()

    texts = json.loads(Path(a.texts).read_text(encoding="utf-8"))
    langs = [l.strip() for l in a.langs.split(",") if l.strip()]

    mt = MadladMT()

    tts = None
    outdir = Path(a.outdir)
    if a.audio:
        import soundfile as sf  # noqa: F401
        from app.tts import VoxtralTTS
        from app.config import SAMPLE_RATE
        import numpy as np

        tts = VoxtralTTS()
        outdir.mkdir(parents=True, exist_ok=True)

    for src in langs:
        if src not in texts:
            print(f"⚠️  pas de texte source pour '{src}', skip")
            continue
        print("\n" + "=" * 78)
        print(f"SOURCE [{src}] {LANG_NAMES.get(src, src)}")
        print("=" * 78)
        print(texts[src])

        for tgt in langs:
            if tgt == src:
                continue
            out = mt.translate(texts[src], src, tgt)
            print(f"\n--- {src} → {tgt} [{LANG_NAMES.get(tgt, tgt)}] ---")
            print(out)

            if tts is not None:
                import numpy as np
                import soundfile as sf
                from app.config import SAMPLE_RATE

                chunks = list(tts.synthesize(out, lang=tgt, voice=a.voice))
                if chunks:
                    audio = np.concatenate(chunks)
                    wav = outdir / f"{src}2{tgt}.wav"
                    sf.write(str(wav), audio, SAMPLE_RATE)
                    print(f"    🔊 {wav}  ({len(audio)/SAMPLE_RATE:.1f}s)")
                else:
                    print("    ⚠️ aucun audio")

    print("\n✅ terminé")


if __name__ == "__main__":
    main()
