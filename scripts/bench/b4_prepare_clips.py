#!/usr/bin/env python3
"""B4 data prep: extract real speech clips (LibriSpeech dummy, 16 kHz) to WAV.

Source: HF dataset hf-internal-testing/librispeech_asr_dummy (config "clean",
split "validation", 73 LibriSpeech dev-clean utterances). First 20 clips ->
<out>/wm/, next 50 -> <out>/clean/ (disjoint sets).

Run:  python b4_prepare_clips.py --out DIR
Deps: pip install datasets huggingface_hub soundfile numpy
"""

import argparse
import io
import pathlib

import soundfile as sf
from datasets import Audio, load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-wm", type=int, default=20)
    ap.add_argument("--n-clean", type=int, default=50)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    for i, row in enumerate(ds):
        if i >= a.n_wm + a.n_clean:
            break
        sub = "wm" if i < a.n_wm else "clean"
        (out / sub).mkdir(parents=True, exist_ok=True)
        data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        sf.write(out / sub / f"clip{i:03d}.wav", data, sr)
    print("done", out)


if __name__ == "__main__":
    main()
