"""f5say: speak English text in your voice with F5-TTS (MLX), then cyv convert.

Experiment, not part of cyv. Runs with the F5 venv's python; see README.md.

    f5say "Hello there." -o hello.wav
    f5say "Line one." "Line two." -o out/        # one file per line
    f5say --ref casual "Hi!" -o hi.wav           # ref/casual.wav + casual.txt

Settings that sounded right in B7 (docs/benchmarks/B7-clone-tts-m1.md):
32 euler steps, duration from a text-length ratio, cfg 2.0, sway -1.0.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

HOME = Path(os.environ.get("F5SAY_HOME", Path.home() / ".cache/tts-spike"))
SR = 24_000
HOP = 256
TARGET_RMS = 0.1


def die(msg: str) -> None:
    sys.exit(f"f5say: {msg}")


def load_ref(name: str):
    import numpy as np
    import soundfile as sf

    wav, txt = HOME / "ref" / f"{name}.wav", HOME / "ref" / f"{name}.txt"
    if not wav.exists() or not txt.exists():
        die(f"need {wav} and {txt} (see README: 'Use a different reference clip')")
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim != 1 or sr != SR:
        die(f"{wav.name} must be mono {SR} Hz (convert with ffmpeg -ac 1 -ar {SR})")
    text = txt.read_text(encoding="utf-8").strip()
    if not text:
        die(f"{txt.name} is empty: it must hold the exact words spoken in {wav.name}")
    return np.asarray(audio), text


def synth(model, audio, ref_text: str, line: str, opt: argparse.Namespace):
    import mlx.core as mx
    import numpy as np
    from f5_tts_mlx.utils import convert_char_to_pinyin

    a = mx.array(audio)
    rms = mx.sqrt(mx.mean(mx.square(a)))
    if rms < TARGET_RMS:
        a = a * TARGET_RMS / rms
    # F5's own duration predictor cut every clip to ~2.8 s (B7), so estimate
    # the length from the reference's speaking rate instead.
    n = a.shape[0] // HOP
    dur = n + int(n / len(ref_text.encode()) * len(line.encode()) / opt.speed)
    text = convert_char_to_pinyin([ref_text + " " + line])
    wave, _ = model.sample(
        mx.expand_dims(a, axis=0),
        text=text,
        duration=dur,
        steps=opt.steps,
        method=opt.method,
        cfg_strength=opt.cfg,
        sway_sampling_coef=opt.sway,
        seed=opt.seed,
    )
    wave = wave[a.shape[0] :]
    mx.eval(wave)
    return np.array(wave)


def main() -> int:
    ap = argparse.ArgumentParser(prog="f5say", description=__doc__.split("\n")[0])
    ap.add_argument("text", nargs="+", help="English text; several arguments = several files")
    ap.add_argument(
        "-o", "--output", required=True, help="output .wav, or a folder for several lines"
    )
    ap.add_argument("--voice", default="sachinkoli2", help="cyv voice for the convert step")
    ap.add_argument(
        "--ref", default="ref_11s", help="reference clip name in ref/ (default ref_11s)"
    )
    ap.add_argument(
        "--steps", type=int, default=32, help="32 sounded right; 16 is 2x faster but worse"
    )
    ap.add_argument("--seed", type=int, default=42, help="change for a different take")
    ap.add_argument(
        "--speed", type=float, default=1.0, help="speaking rate vs the reference (1.1 = faster)"
    )
    ap.add_argument(
        "--cfg", type=float, default=2.0, help="how hard to follow text + voice (F5 default 2.0)"
    )
    ap.add_argument(
        "--sway", type=float, default=-1.0, help="sway sampling coefficient (F5 default -1.0)"
    )
    ap.add_argument(
        "--method",
        default="euler",
        choices=["euler", "midpoint", "rk4"],
        help="solver; midpoint is 2x and rk4 4x the work per step",
    )
    ap.add_argument("--raw", action="store_true", help="skip cyv convert (F5 output only)")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    a = ap.parse_args()

    out = Path(a.output).expanduser()
    many = len(a.text) > 1
    targets = [out / f"line{i}.wav" for i in range(1, len(a.text) + 1)] if many else [out]
    if many:
        out.mkdir(parents=True, exist_ok=True)
    for t in targets:
        if t.suffix.lower() != ".wav":
            die(f"{t} must end in .wav")
        if t.exists() and not a.force:
            die(f"{t} exists (use --force)")

    import mlx.core as mx
    import soundfile as sf
    from f5_tts_mlx.cfm import F5TTS

    mx.set_cache_limit(256 * 1024 * 1024)  # keeps swap down on 8 GB
    audio, ref_text = load_ref(a.ref)
    print("loading F5 ...", file=sys.stderr, flush=True)
    model = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")

    with tempfile.TemporaryDirectory() as tmp:
        for line, target in zip(a.text, targets, strict=True):
            t0 = time.time()
            print(
                f"speaking ({a.steps} steps, ~70 s on an M1): {line[:60]}",
                file=sys.stderr,
                flush=True,
            )
            wave = synth(model, audio, ref_text, line, a)
            raw = target if a.raw else Path(tmp) / "f5.wav"
            sf.write(raw, wave, SR)
            if not a.raw:
                r = subprocess.run(
                    [
                        "cyv",
                        "convert",
                        str(raw),
                        "--voice",
                        a.voice,
                        "--preset",
                        "master",
                        "-o",
                        str(target),
                        "--force",
                    ],
                    capture_output=True,
                    text=True,
                )
                if r.returncode != 0:
                    die(f"cyv convert failed: {r.stderr.strip()}")
            print(
                f"wrote {target}  ({len(wave) / SR:.1f} s audio, {time.time() - t0:.0f} s)",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
