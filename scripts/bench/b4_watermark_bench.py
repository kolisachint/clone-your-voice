#!/usr/bin/env python3
"""B4: AudioSeal vs Resemble Perth watermark robustness (CPU).

Pipeline per clip: embed at the method's native SR (AudioSeal 16 kHz, Perth
32 kHz) -> resample to 48 kHz (the "delivered" WAV, as in DESIGN 6.3) ->
degrade through ffmpeg/resampling -> back to native SR -> detect.
Detection: AudioSeal prob>0.5 (also bit accuracy of the 16-bit message);
Perth get_watermark confidence >0.5. FPR measured on unwatermarked clips
(passed through the same "none" path, plus every attack, for completeness).

Setup (own venv, CPU torch):
  uv venv --python 3.12 venv && . venv/bin/activate
  uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  uv pip install audioseal resemble-perth soundfile numpy scipy librosa datasets
Run (from a directory that is NOT a torch checkout):
  python b4_prepare_clips.py --out DATA
  python b4_watermark_bench.py --data DATA --out results.json
  python b4_watermark_bench.py --data DATA --extra --n-wm 10 --n-clean 20 --out extra.json  # harsher attacks
Needs ffmpeg with libopus and libmp3lame on PATH.
"""

import argparse
import json
import pathlib
import subprocess
import tempfile
import time
import warnings

import numpy as np
import soundfile as sf
import soxr
import torch

warnings.filterwarnings("ignore")

SR_OUT = 48000
ATTACKS = {
    "none": None,
    "resample_48-16-48": "resample",
    "mp3_192k": ["-c:a", "libmp3lame", "-b:a", "192k", ".mp3"],
    "mp3_64k": ["-c:a", "libmp3lame", "-b:a", "64k", ".mp3"],
    "opus_32k_voip_mono": [
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        "32k",
        "-application",
        "voip",
        ".ogg",
    ],
    "opus_16k_voip_mono": [
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        "16k",
        "-application",
        "voip",
        ".ogg",
    ],
}


EXTRA = {
    "opus_8k_voip_mono": [
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        "8k",
        "-application",
        "voip",
        ".ogg",
    ],
    "mp3_32k": ["-c:a", "libmp3lame", "-b:a", "32k", ".mp3"],
    "amr_nb_like_resample_8k": "resample8k",
    "white_noise_20dB": "noise",
}


def rs(x, a, b):
    return x if a == b else soxr.resample(x, a, b)


def attack(x48, name, tmp):
    spec = ATTACKS[name]
    if spec is None:
        return x48
    if spec == "resample":
        return rs(rs(x48, 48000, 16000), 16000, 48000)
    if spec == "resample8k":
        return rs(rs(x48, 48000, 8000), 8000, 48000)
    if spec == "noise":
        r = np.random.default_rng(0)
        return x48 + r.standard_normal(len(x48)).astype("float32") * np.sqrt(np.mean(x48**2) / 100)
    ext = spec[-1]
    src, dst = tmp / "a.wav", tmp / "b.wav"
    enc = tmp / ("e" + ext)
    sf.write(src, x48, 48000, subtype="FLOAT")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *spec[:-1], str(enc)], check=True
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(enc),
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            str(dst),
        ],
        check=True,
    )
    y, _ = sf.read(dst, dtype="float32")
    n = len(x48)  # codec delay/padding: trim/pad to original length
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


class AudioSealM:
    name, sr = "audioseal", 16000

    def __init__(self):
        from audioseal import AudioSeal

        self.gen = AudioSeal.load_generator("audioseal_wm_16bits")
        self.det = AudioSeal.load_detector("audioseal_detector_16bits")
        g = torch.Generator().manual_seed(1234)
        self.msg = torch.randint(0, 2, (1, 16), generator=g)

    def embed(self, x):
        with torch.no_grad():
            t = torch.from_numpy(x)[None, None]
            return self.gen(t, sample_rate=self.sr, message=self.msg)[0, 0].numpy()

    def detect(self, x):
        with torch.no_grad():
            p, m = self.det.detect_watermark(torch.from_numpy(x)[None, None], sample_rate=self.sr)
        acc = float((m.round().long() == self.msg).float().mean())
        return float(p[0]), acc


class PerthM:
    name, sr = "perth", 32000

    def __init__(self):
        import perth

        self.m = perth.PerthImplicitWatermarker()

    def embed(self, x):
        return np.asarray(self.m.apply_watermark(x, sample_rate=self.sr), dtype="float32")

    def detect(self, x):
        return float(self.m.get_watermark(x, sample_rate=self.sr, round=False)), None


def load(d):
    return [(p.name, sf.read(p, dtype="float32")[0]) for p in sorted(pathlib.Path(d).glob("*.wav"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="b4_results.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--extra",
        action="store_true",
        help="add harsher attacks (opus 8k, mp3 32k, 8k resample, 20 dB noise)",
    )
    ap.add_argument("--n-wm", type=int, default=None)
    ap.add_argument("--n-clean", type=int, default=None)
    a = ap.parse_args()
    if a.extra:
        ATTACKS.update(EXTRA)
    torch.set_num_threads(a.threads)
    wm_clips, clean_clips = load(pathlib.Path(a.data) / "wm"), load(pathlib.Path(a.data) / "clean")
    wm_clips, clean_clips = wm_clips[: a.n_wm], clean_clips[: a.n_clean]
    tmp = pathlib.Path(tempfile.mkdtemp())
    res = {"env": {"torch": torch.__version__, "threads": a.threads}, "methods": {}}
    for M in (AudioSealM, PerthM):
        m = M()
        det = {k: [] for k in ATTACKS}
        acc = {k: [] for k in ATTACKS}
        fp = {k: [] for k in ATTACKS}
        snr, t_emb, t_det, dur = [], 0.0, 0.0, 0.0
        for _nm, x16 in wm_clips:
            x = rs(x16, 16000, m.sr)
            d = len(x) / m.sr
            dur += d
            t = time.perf_counter()
            y = m.embed(x)
            t_emb += time.perf_counter() - t
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
            snr.append(10 * np.log10(np.sum(x**2) / (np.sum((y - x) ** 2) + 1e-12)))
            y48 = rs(y, m.sr, SR_OUT)
            for k in ATTACKS:
                z = rs(attack(y48, k, tmp), SR_OUT, m.sr)
                t = time.perf_counter()
                p, b = m.detect(z)
                t_det += time.perf_counter() - t if k == "none" else 0
                det[k].append(p > 0.5)
                if b is not None:
                    acc[k].append(b)
        for _nm, x16 in clean_clips:
            x48 = rs(rs(x16, 16000, m.sr), m.sr, SR_OUT)
            for k in ATTACKS:
                z = rs(attack(x48, k, tmp), SR_OUT, m.sr)
                fp[k].append(m.detect(z)[0] > 0.5)
        res["methods"][m.name] = {
            "n_wm": len(wm_clips),
            "n_clean": len(clean_clips),
            "audio_s": dur,
            "detect_rate": {k: float(np.mean(v)) for k, v in det.items()},
            "bit_acc": {k: (float(np.mean(v)) if v else None) for k, v in acc.items()},
            "false_positive_rate": {k: float(np.mean(v)) for k, v in fp.items()},
            "snr_db_mean": float(np.mean(snr)),
            "snr_db_min": float(np.min(snr)),
            "embed_s_per_10s": t_emb / dur * 10,
            "detect_s_per_10s": t_det / dur * 10,
        }
        print(json.dumps({m.name: res["methods"][m.name]}, indent=1))
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
