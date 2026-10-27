#!/usr/bin/env python3
"""Local half of the voice-cloning TTS test.

Put tts_out.zip (from Colab) next to this file, then run:
    python3 run_local.py

It makes, for each of the 3 sentences:
  - f5_raw, cb_raw        : the cloning TTS output as-is
  - f5_conv, cb_conv      : that output run through cyv convert (sachinkoli2)
  - kokoro_say            : today's cyv say at pitch -6 (the baseline)

Then it copies everything into blind/ with shuffled names (s1_A.wav ...),
and writes the answer key to key.txt. Do not open key.txt until you've ranked.
"""

import json
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICE = "sachinkoli2"
SAY_PITCH = -6
SENTENCES = [
    "Hello, this is a test of my cloned voice. How does this sound to you?",
    "I was thinking we could grab lunch tomorrow, maybe around one, if that works for you.",
    "Honestly, I did not expect the results to be this good. Let me know what you think.",
]


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stderr.strip()}")


def unpack(zpath: Path, dest: Path) -> list[Path]:
    dest.mkdir(exist_ok=True)
    got = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            # only plain out/<model>_s<n>.wav files, nothing else
            if info.is_dir() or not name.endswith(".wav") or "/" in name or ".." in name:
                continue
            if not (name.startswith(("f5_s", "cb_s")) and info.file_size < 50 << 20):
                continue
            target = dest / name
            target.write_bytes(z.read(info))
            got.append(target)
    return sorted(got)


def main() -> None:
    zpath = HERE / "tts_out.zip"
    if not zpath.exists():
        sys.exit(f"put tts_out.zip in {HERE} first")

    work = HERE / "work"
    if work.exists():
        shutil.rmtree(work)
    raw = unpack(zpath, work)
    if not raw:
        sys.exit("no f5_s*.wav or cb_s*.wav files found in the zip")
    print(f"found {len(raw)} files from Colab")

    clips: dict[int, dict[str, Path]] = {}
    for f in raw:
        model, sn = f.stem.split("_s")  # f5_s1 -> f5, 1
        n = int(sn)
        clips.setdefault(n, {})[f"{model}_raw"] = f
        conv = work / f"{model}_conv_s{n}.wav"
        print(f"  converting {f.name} ...")
        run(
            [
                "cyv",
                "convert",
                str(f),
                "--voice",
                VOICE,
                "--preset",
                "master",
                "-o",
                str(conv),
                "--force",
            ]
        )
        clips[n][f"{model}_conv"] = conv

    for n in sorted(clips):
        out = work / f"kokoro_say_s{n}.wav"
        print(f"  kokoro baseline s{n} ...")
        run(
            [
                "cyv",
                "say",
                SENTENCES[n - 1],
                "--voice",
                VOICE,
                "--pitch",
                str(SAY_PITCH),
                "--preset",
                "master",
                "-o",
                str(out),
                "--force",
            ]
        )
        clips[n]["kokoro_say"] = out

    blind = HERE / "blind"
    if blind.exists():
        shutil.rmtree(blind)
    blind.mkdir()
    key = {}
    rng = random.Random()  # noqa: S311 (shuffles a listening test, not security)
    for n in sorted(clips):
        items = list(clips[n].items())
        rng.shuffle(items)
        for letter, (label, path) in zip("ABCDE", items, strict=False):
            name = f"s{n}_{letter}.wav"
            shutil.copyfile(path, blind / name)
            key[name] = label

    (HERE / "key.txt").write_text(json.dumps(key, indent=2) + "\n")
    print(f"\ndone: {len(key)} files in {blind}")
    print("listen per sentence (s1_A ... s1_E), rank them, THEN open key.txt")


if __name__ == "__main__":
    main()
