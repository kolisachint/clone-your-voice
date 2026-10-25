"""Convert, normalize, check and package a folder of recordings into dataset.zip."""

from __future__ import annotations

import json
import os
import re
import tempfile
import wave
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cyv import ffmpeg

ATTESTATION_VERSION = 1
SAMPLE_RATE = 48000
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".aif", ".aiff", ".wma"}
MIN_SECONDS = 60.0
WARN_SECONDS = 300.0
CLIP_WARN = 0.001
CLIP_REJECT = 0.01
SILENCE_DB = -50.0
LOW_LEVEL_DB = -40.0
MOSTLY_SILENCE = 0.5
MIN_ATTEST_LEN = 10

# Kokoro voice prefix -> language.
_LANGS = {
    "a": "en-US", "b": "en-GB", "e": "es", "f": "fr",
    "h": "hi", "i": "it", "j": "ja", "p": "pt-BR", "z": "zh",
}  # fmt: skip

_FILTER = (
    "silenceremove=start_periods=1:start_threshold=-50dB:"
    "stop_periods=-1:stop_duration=0.6:stop_threshold=-50dB,"
    "loudnorm=I=-20:TP=-2:LRA=11"
)


class DatasetError(Exception):
    pass


@dataclass
class FileStats:
    seconds: float
    clipping_ratio: float
    silence_ratio: float
    speech_rms_db: float


@dataclass
class DatasetReport:
    files: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    raw_seconds: float = 0.0
    seconds: float = 0.0  # after silence trimming
    clipping_ratio: float = 0.0
    silence_ratio: float = 0.0
    speech_rms_db: float = float("-inf")
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"files used:        {len(self.files)}"
            + (f" ({len(self.skipped)} skipped)" if self.skipped else ""),
            f"raw duration:      {self.raw_seconds / 60:.1f} min",
            f"usable duration:   {self.seconds / 60:.1f} min (after silence trimming)",
            f"clipping:          {self.clipping_ratio * 100:.3f}% of samples",
            f"silence:           {self.silence_ratio * 100:.0f}% of raw audio",
            f"speech level:      {self.speech_rms_db:.1f} dBFS RMS",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        lines += [f"ERROR: {e}" for e in self.errors]
        return "\n".join(lines)


def find_audio_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Audio files under root, never following symlinks. Returns (files, skipped_symlinks)."""
    root = root.resolve()
    files: list[Path] = []
    skipped: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for d in list(dirnames):
            if Path(dirpath, d).is_symlink():
                skipped.append(Path(dirpath, d))
                dirnames.remove(d)
        for fn in sorted(filenames):
            p = Path(dirpath, fn)
            if p.suffix.lower() not in AUDIO_EXTS:
                continue
            if p.is_symlink():
                skipped.append(p)
            elif p.is_file():
                files.append(p)
    return files, skipped


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit WAV as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise DatasetError(f"unexpected wav format in {path.name}")
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, rate


def analyze_samples(x: np.ndarray, rate: int) -> FileStats:
    if x.size == 0:
        return FileStats(0.0, 0.0, 1.0, float("-inf"))
    clip = float(np.mean(np.abs(x) >= 0.999))
    frame = max(1, int(rate * 0.05))
    n = x.size // frame
    if n == 0:
        rms = np.array([float(np.sqrt(np.mean(x**2)))])
    else:
        rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-9))
    silent = db < SILENCE_DB
    speech = db[~silent]
    speech_db = float("-inf")
    if speech.size:
        speech_db = float(10 * np.log10(max(float(np.mean(10 ** (speech / 10))), 1e-12)))
    return FileStats(
        seconds=x.size / rate,
        clipping_ratio=clip,
        silence_ratio=float(np.mean(silent)),
        speech_rms_db=speech_db,
    )


def default_name(root: Path) -> str:
    name = re.sub(r"[^a-z0-9_-]+", "-", root.resolve().name.lower()).strip("-")
    return name[:32] or "voice"


def valid_name(name: str) -> bool:
    return re.fullmatch(r"[a-z0-9_-]{1,32}", name) is not None


def lang_for(base_voice: str) -> str:
    """Kokoro lang code, the voice id's first letter (am_adam -> "a"), as voice.json stores it."""
    if base_voice[:1] not in _LANGS:
        raise DatasetError(f"unknown Kokoro language in --base-voice {base_voice!r}")
    return base_voice[:1]


def process(root: Path, workdir: Path) -> tuple[list[Path], DatasetReport]:
    """Convert every source file to normalized wavs in workdir and analyze them."""
    sources, skipped = find_audio_files(root)
    report = DatasetReport(skipped=[str(p) for p in skipped])
    if not sources:
        raise DatasetError(f"no audio files found in {root}")
    outs: list[Path] = []
    clip_w = sil_w = 0.0
    speech_pow = []
    for i, src in enumerate(sources, 1):
        raw, out = workdir / f"raw{i:04d}.wav", workdir / f"{i:04d}.wav"
        try:
            ffmpeg.run(["-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                        "-c:a", "pcm_s16le", str(raw)])  # fmt: skip
            x, rate = read_wav(raw)
            st = analyze_samples(x, rate)
            ffmpeg.run(["-i", str(raw), "-af", _FILTER, "-ar", str(SAMPLE_RATE), "-ac", "1",
                        "-c:a", "pcm_s16le", str(out)])  # fmt: skip
            trimmed, _ = read_wav(out)
        except ffmpeg.FfmpegError as e:
            if "not found" in str(e):
                raise DatasetError(str(e)) from e
            report.skipped.append(f"{src} ({e})")
            continue
        finally:
            raw.unlink(missing_ok=True)
        report.files.append(str(src.relative_to(root.resolve())))
        outs.append(out)
        report.raw_seconds += st.seconds
        report.seconds += trimmed.size / SAMPLE_RATE
        clip_w += st.clipping_ratio * st.seconds
        sil_w += st.silence_ratio * st.seconds
        if np.isfinite(st.speech_rms_db):
            speech_pow.append((st.seconds, 10 ** (st.speech_rms_db / 10)))
    if not outs:
        raise DatasetError("none of the audio files could be decoded")
    if report.raw_seconds > 0:
        report.clipping_ratio = clip_w / report.raw_seconds
        report.silence_ratio = sil_w / report.raw_seconds
    if speech_pow:
        tot = sum(s for s, _ in speech_pow)
        report.speech_rms_db = float(10 * np.log10(sum(s * p for s, p in speech_pow) / tot))
    _judge(report)
    return outs, report


def _judge(r: DatasetReport) -> None:
    if r.seconds < MIN_SECONDS:
        r.errors.append(
            f"only {r.seconds:.0f} s of usable speech; at least {MIN_SECONDS:.0f} s is required"
        )
    elif r.seconds < WARN_SECONDS:
        r.warnings.append(
            f"under 5 minutes of speech ({r.seconds / 60:.1f} min); quality will likely suffer"
        )
    if r.clipping_ratio >= CLIP_REJECT:
        r.errors.append(
            f"heavy clipping ({r.clipping_ratio * 100:.2f}% of samples); re-record at a lower gain"
        )
    elif r.clipping_ratio >= CLIP_WARN:
        r.warnings.append(f"some clipping ({r.clipping_ratio * 100:.3f}% of samples)")
    if r.speech_rms_db < LOW_LEVEL_DB:
        r.warnings.append(
            f"very low recording level ({r.speech_rms_db:.0f} dBFS); raise input gain"
        )
    if r.silence_ratio > MOSTLY_SILENCE:
        r.warnings.append(f"{r.silence_ratio * 100:.0f}% of the audio is silence")
    if r.skipped:
        r.warnings.append(f"{len(r.skipped)} file(s) skipped (symlinks or undecodable)")


def prepare_dataset(
    root: Path,
    out_zip: Path,
    *,
    base_voice: str = "am_adam",
    name: str | None = None,
    attest: Callable[[DatasetReport], tuple[str, str]],
    force: bool = False,
    on_report: Callable[[DatasetReport], None] | None = None,
) -> DatasetReport:
    """Build out_zip from the audio in root.

    `attest(report)` returns (owner_name, attestation) after showing the report to the user.
    """
    if not root.is_dir():
        raise DatasetError(f"not a directory: {root}")
    name = name or default_name(root)
    if not valid_name(name):
        raise DatasetError("--name must match ^[a-z0-9_-]{1,32}$")
    if not re.fullmatch(r"[a-z]{2}_[a-z0-9]{1,24}", base_voice):
        raise DatasetError(f"bad --base-voice {base_voice!r} (expected e.g. am_adam)")
    lang = lang_for(base_voice)
    if out_zip.exists() and not force:
        raise DatasetError(f"{out_zip} exists; use --force to overwrite")

    with tempfile.TemporaryDirectory(prefix="cyv-ds-") as td:
        outs, report = process(root, Path(td))
        if on_report:
            on_report(report)
        if report.errors:
            raise DatasetError("dataset rejected: " + "; ".join(report.errors))
        owner, attestation = attest(report)
        owner, attestation = owner.strip(), attestation.strip()
        if not owner:
            raise DatasetError("owner name is required")
        if len(attestation) < MIN_ATTEST_LEN:
            raise DatasetError("a typed consent attestation is required")
        consent = {
            "attestation": attestation,
            "attestation_version": ATTESTATION_VERSION,
            "owner_name": owner,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        meta = {
            "name": name,
            "base_voice": base_voice,
            "lang": lang,
            "dataset_seconds": round(report.seconds, 1),
        }
        tmp_zip = Path(td) / "dataset.zip"
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for i, wav in enumerate(outs, 1):
                z.write(wav, f"wavs/{i:04d}.wav")
            z.writestr("consent.json", json.dumps(consent, indent=2))
            z.writestr("dataset_meta.json", json.dumps(meta, indent=2))
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        os.replace(_move_beside(tmp_zip, out_zip), out_zip)
    return report


def _move_beside(src: Path, dst: Path) -> Path:
    """Copy src next to dst so the final rename is atomic (same filesystem)."""
    part = dst.with_name(dst.name + ".part")
    part.write_bytes(src.read_bytes())
    return part
