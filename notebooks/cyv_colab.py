"""Helpers for notebooks/train_colab.ipynb (pure standard library, except numpy in analyze_wav).

The notebook downloads/copies this file next to itself and imports it. Everything that can
be tested without a GPU lives here, so the notebook cells stay short and readable:

    validate_dataset_zip   check the zip made by `cyv dataset prepare`
    extract_dataset        unpack the wavs into Applio's dataset folder
    find_final_weights     locate the small inference model Applio writes
    build_voice_json       make voice.json (matches src/cyv/packs/voice_schema.json)
    write_pack             write <name>.cyvpack.zip with exactly 4 members
    run_and_sync           run an Applio command and mirror checkpoints to Drive
    save_prepared / restore_from_drive   resume after a disconnect
    analyze_wav            heuristics for the PASS/FAIL self-test
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import threading
import time
import wave
import zipfile
from pathlib import Path, PurePosixPath

NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
VOICE_RE = re.compile(r"^[a-z]{2}_[a-z0-9]{1,24}$")
CONSENT_KEYS = ("attestation", "attestation_version", "owner_name", "created_at")

MIN_DATASET_SECONDS = 60.0  # below this a clone is hopeless; refuse
WARN_DATASET_SECONDS = 180.0  # below this, warn
MAX_ZIP_BYTES = 2 * 1024**3  # sanity cap on total uncompressed size


class DatasetError(Exception):
    """The dataset zip is not usable. The message is written for humans."""


class PackError(Exception):
    """Could not build the voice pack."""


# --------------------------------------------------------------------------- dataset


def wav_seconds(fileobj) -> float:
    with wave.open(fileobj, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def validate_dataset_zip(zip_path) -> dict:
    """Validate the zip written by `cyv dataset prepare`.

    Expected members: wavs/*.wav, consent.json, dataset_meta.json (nothing else).
    Returns {"wav_names", "total_seconds", "consent", "meta", "warnings"}.
    Raises DatasetError with a plain-English reason otherwise.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise DatasetError(f"{zip_path} does not exist.")
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise DatasetError(f"{zip_path.name} is not a valid zip file ({exc}).") from exc

    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if sum(i.file_size for i in infos) > MAX_ZIP_BYTES:
            raise DatasetError("The zip is unreasonably large (over 2 GB uncompressed).")
        wav_infos = []
        for i in infos:
            p = PurePosixPath(i.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in i.filename:
                raise DatasetError(f"Unsafe path inside the zip: {i.filename!r}")
            if i.filename in ("consent.json", "dataset_meta.json"):
                continue
            if len(p.parts) == 2 and p.parts[0] == "wavs" and p.suffix.lower() == ".wav":
                wav_infos.append(i)
                continue
            raise DatasetError(
                f"Unexpected file in the zip: {i.filename!r}. "
                "Create the zip with `cyv dataset prepare` and do not add files."
            )

        names = {i.filename for i in infos}
        if "consent.json" not in names:
            raise DatasetError(
                "consent.json is missing. `cyv dataset prepare` writes it after you confirm "
                "the recordings are your own voice (or you have permission). Re-run it."
            )
        if "dataset_meta.json" not in names:
            raise DatasetError("dataset_meta.json is missing. Re-run `cyv dataset prepare`.")
        if not wav_infos:
            raise DatasetError("There are no .wav files under wavs/ in the zip.")

        try:
            consent = json.loads(zf.read("consent.json"))
            meta = json.loads(zf.read("dataset_meta.json"))
        except ValueError as exc:
            raise DatasetError(
                f"consent.json or dataset_meta.json is not valid JSON ({exc})."
            ) from exc
        if not isinstance(consent, dict) or not isinstance(meta, dict):
            raise DatasetError("consent.json and dataset_meta.json must be JSON objects.")
        for k in CONSENT_KEYS:
            if k == "attestation_version":
                v = consent.get(k)
                if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                    raise DatasetError(
                        "consent.json: 'attestation_version' must be a positive integer."
                    )
            elif not isinstance(consent.get(k), str) or not consent[k].strip():
                raise DatasetError(f"consent.json: field {k!r} is missing or empty.")
        base_voice = meta.get("base_voice")
        if not isinstance(base_voice, str) or not VOICE_RE.match(base_voice):
            raise DatasetError(
                f"dataset_meta.json: base_voice {base_voice!r} is not a Kokoro voice id like 'am_adam'."
            )

        total = 0.0
        for i in wav_infos:
            try:
                with zf.open(i) as fh:
                    total += wav_seconds(fh)
            except (wave.Error, EOFError) as exc:
                raise DatasetError(f"{i.filename} is not a readable PCM .wav ({exc}).") from exc

    warnings = []
    if total < MIN_DATASET_SECONDS:
        raise DatasetError(
            f"Only {total:.0f} s of audio. Record at least {MIN_DATASET_SECONDS:.0f} s "
            "(5 to 10 minutes is much better)."
        )
    if total < WARN_DATASET_SECONDS:
        warnings.append(
            f"Only {total / 60:.1f} min of audio; the clone may sound thin. 5 to 10 min is better."
        )
    return {
        "wav_names": sorted(i.filename for i in wav_infos),
        "total_seconds": round(total, 2),
        "consent": consent,
        "meta": meta,
        "warnings": warnings,
    }


def extract_dataset(zip_path, wav_dest, meta_dest) -> None:
    """Copy wavs/*.wav (flattened) into wav_dest and the two json files into meta_dest.

    Call validate_dataset_zip first. Applio's preprocess walks the whole dataset folder, so
    wav_dest must contain nothing but the wavs.
    """
    wav_dest, meta_dest = Path(wav_dest), Path(meta_dest)
    if wav_dest.exists():
        shutil.rmtree(wav_dest)
    wav_dest.mkdir(parents=True)
    meta_dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for i in zf.infolist():
            if i.is_dir():
                continue
            if i.filename in ("consent.json", "dataset_meta.json"):
                (meta_dest / i.filename).write_bytes(zf.read(i))
            else:
                (wav_dest / PurePosixPath(i.filename).name).write_bytes(zf.read(i))


# --------------------------------------------------------------------------- Applio outputs


def find_final_weights(exp_dir, name):
    """Return (path, epoch) of the highest-epoch small inference model.

    At the pinned Applio commit, rvc/train/train.py writes `<name>_<epoch>e_<step>s.pth`
    into logs/<name>/ (this is the ~55 MB half-precision model; G_*.pth / D_*.pth are the
    big resumable checkpoints and must NOT be shipped). Returns (None, 0) if there is none.
    """
    rx = re.compile(rf"^{re.escape(name)}_(\d+)e_(\d+)s\.pth$")
    best, best_epoch, best_step = None, 0, -1
    for p in Path(exp_dir).glob("*.pth"):
        m = rx.match(p.name)
        if m and (int(m[1]), int(m[2])) > (best_epoch, best_step):
            best, best_epoch, best_step = p, int(m[1]), int(m[2])
    return (best, best_epoch) if best else (None, 0)


def find_index(exp_dir, name):
    """The FAISS index. At the pinned commit it is logs/<name>/<name>.index (older
    Applio/RVC versions wrote added_*.index, so accept that too)."""
    exp_dir = Path(exp_dir)
    p = exp_dir / f"{name}.index"
    if p.is_file():
        return p
    added = sorted(exp_dir.glob("added_*.index"), key=lambda f: f.stat().st_mtime)
    return added[-1] if added else None


def latest_loss_lines(log_text: str, n: int = 5) -> list[str]:
    """Last n per-epoch progress lines plus the 'Training has been successfully completed' lines."""
    lines = [ln.strip() for ln in log_text.splitlines()]
    epoch = [ln for ln in lines if " | epoch=" in ln]
    done = [
        ln
        for ln in lines
        if ln.startswith(("Training has been successfully", "Lowest generator loss"))
    ]
    return epoch[-n:] + done


# --------------------------------------------------------------------------- voice pack


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lang_from_voice(voice: str) -> str:
    """Kokoro voice ids start with the language letter: am_adam -> 'a' (American English)."""
    if not isinstance(voice, str) or not VOICE_RE.match(voice):
        raise PackError(f"{voice!r} is not a Kokoro voice id like 'am_adam'.")
    return voice[0]


def build_voice_json(
    *,
    name: str,
    engine_version: str,
    sample_rate: int,
    base_voice: str,
    pth_sha256: str,
    index_sha256: str,
    epochs: int | None = None,
    dataset_seconds: float | None = None,
    pitch_shift: int = 0,
) -> dict:
    """Build voice.json exactly as src/cyv/packs/voice_schema.json demands."""
    if not NAME_RE.match(name):
        raise PackError(f"Invalid voice name {name!r}: use a-z, 0-9, _ or -, 1 to 32 characters.")
    voice = {
        "schema_version": 1,
        "name": name,
        "engine": "applio-rvc",
        "engine_version": engine_version,
        "rvc_version": "v2",
        "sample_rate": int(sample_rate),
        "embedder": "contentvec",
        "f0_method": "rmvpe",
        "vocoder": "HiFi-GAN",
        "tts_base": {"voice": base_voice, "lang": lang_from_voice(base_voice)},
        "pitch_shift": int(pitch_shift),
        "sha256": {"model.pth": pth_sha256, "model.index": index_sha256},
    }
    if epochs is not None:
        voice["epochs"] = int(epochs)
    if dataset_seconds is not None:
        voice["dataset_seconds"] = float(dataset_seconds)
    return voice


def write_pack(out_zip, *, pth_path, index_path, voice: dict, consent_bytes: bytes) -> Path:
    """Write <name>.cyvpack.zip with exactly: model.pth, model.index, voice.json, consent.json.

    consent_bytes is the original consent.json, copied verbatim. voice's sha256 values must
    match the files being packed (checked here so a mismatch can never be shipped).
    """
    out_zip = Path(out_zip)
    for fname, p in (("model.pth", pth_path), ("model.index", index_path)):
        if sha256_file(p) != voice["sha256"][fname]:
            raise PackError(f"sha256 in voice.json does not match {fname}.")
    tmp = out_zip.with_suffix(out_zip.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pth_path, "model.pth")
        zf.write(index_path, "model.index")
        zf.writestr("voice.json", json.dumps(voice, indent=2) + "\n")
        zf.writestr("consent.json", consent_bytes)
    os.replace(tmp, out_zip)
    return out_zip


# --------------------------------------------------------------------------- Drive sync / resume

# Only these are worth keeping on Drive: the resumable checkpoints and the model weights.
SYNC_PATTERNS = ("G_*.pth", "D_*.pth", "*e_*s.pth", "*.index")


_WEIGHT_RX = re.compile(r"_(\d+)e_\d+s\.pth$")


def _newest_weights(folder, keep):
    """Small inference weights in folder, oldest first (or only the newest `keep`)."""
    found = sorted(
        (p for p in Path(folder).glob("*e_*s.pth") if _WEIGHT_RX.search(p.name)),
        key=lambda p: int(_WEIGHT_RX.search(p.name).group(1)),
    )
    return found[-keep:] if keep else found


def sync_dir(local_dir, drive_dir, patterns=SYNC_PATTERNS, keep_weights=2) -> int:
    """Copy new/changed files matching patterns from local_dir to drive_dir.

    Returns the number of files copied. Keeps only the newest `keep_weights` small
    `<name>_<epoch>e_<step>s.pth` files on Drive so the folder does not grow by 55 MB per save.
    Copies via a temp name so a disconnect mid-copy never leaves a half-written checkpoint.
    """
    local_dir, drive_dir = Path(local_dir), Path(drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    seen = set()
    newest = set(_newest_weights(local_dir, keep_weights)) if keep_weights else None
    for pat in patterns:
        for src in local_dir.glob(pat):
            if src in seen or not src.is_file():
                continue
            if newest is not None and _WEIGHT_RX.search(src.name) and src not in newest:
                continue  # old weights: not worth 55 MB of Drive space
            seen.add(src)
            dst = drive_dir / src.name
            if (
                dst.exists()
                and dst.stat().st_size == src.stat().st_size
                and dst.stat().st_mtime >= src.stat().st_mtime
            ):
                continue
            tmp = dst.with_name(dst.name + ".part")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            copied += 1
    if keep_weights:
        for old in _newest_weights(drive_dir, None)[:-keep_weights]:
            old.unlink(missing_ok=True)
    return copied


PREPARED_TAR = "prepared.tar"
_CKPT_RX = re.compile(r"^[GD]_\d+\.pth$|.*e_\d+s\.pth$")


def save_prepared(exp_dir, drive_dir) -> Path:
    """Pack the preprocess+extract output (everything in logs/<name>/ except big checkpoints
    and weights) into one tar on Drive. One big file is far faster on Drive than thousands
    of tiny feature files, and it lets a resumed session skip preprocess and extract."""
    import tarfile

    exp_dir, drive_dir = Path(exp_dir), Path(drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)
    dst = drive_dir / PREPARED_TAR
    tmp = dst.with_name(dst.name + ".part")
    with tarfile.open(tmp, "w") as tf:
        for p in sorted(exp_dir.iterdir()):
            if not _CKPT_RX.match(p.name) and not p.name.endswith(".index"):
                tf.add(p, arcname=p.name)
    os.replace(tmp, dst)
    return dst


def restore_from_drive(drive_dir, local_dir) -> dict:
    """After a disconnect: put prepared data and checkpoints back into logs/<name>/.

    Returns {"prepared": bool, "checkpoint": bool}. With both True, Applio's train command
    finds G_*.pth / D_*.pth in the folder and continues from that epoch by itself.
    """
    import tarfile

    drive_dir, local_dir = Path(drive_dir), Path(local_dir)
    state = {"prepared": False, "checkpoint": False}
    tar = drive_dir / PREPARED_TAR
    if not tar.is_file():
        return state
    local_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as tf:
        tf.extractall(local_dir, filter="data")  # "data" filter refuses absolute paths, links, ..
    state["prepared"] = True
    if list(drive_dir.glob("G_*.pth")) and list(drive_dir.glob("D_*.pth")):
        for src in drive_dir.iterdir():
            if src.is_file() and not src.name.endswith(".part") and src.name != PREPARED_TAR:
                shutil.copy2(src, local_dir / src.name)
        state["checkpoint"] = True
    return state


def run_and_sync(
    cmd, *, cwd, log_path, local_dir=None, drive_dir=None, interval=120, env=None
) -> int:
    """Run cmd, stream its output into the notebook and a log file, and (optionally) copy
    checkpoints from local_dir to drive_dir every `interval` seconds while it runs.
    Returns the exit code."""
    stop = threading.Event()

    def _sync_loop():
        while not stop.wait(interval):
            try:
                sync_dir(local_dir, drive_dir)
            except OSError as exc:  # Drive hiccup: try again next round
                print(f"[sync] warning: {exc}", flush=True)

    t = None
    if local_dir and drive_dir:
        t = threading.Thread(target=_sync_loop, daemon=True)
        t.start()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [str(c) for c in cmd],
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            last_print = 0.0
            for line in proc.stdout:
                log.write(line)
                # Epoch lines come every few seconds; print only some to keep the page light.
                if " | epoch=" in line and time.time() - last_print < 30:
                    continue
                last_print = time.time()
                print(line.rstrip(), flush=True)
            rc = proc.wait()
    finally:
        stop.set()
        if t:
            t.join(timeout=5)
        if local_dir and drive_dir:
            try:
                sync_dir(local_dir, drive_dir)
            except OSError as exc:
                print(f"[sync] final sync failed: {exc}", flush=True)
    return rc


# --------------------------------------------------------------------------- self-test


def read_wav_samples(path):
    """Return (float32 numpy array mono, sample_rate). Handles PCM 16/24/32 and float32 WAV."""
    import numpy as np

    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, fmt_body, pcm = 12, None, b"", None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            fmt_body = body
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("missing fmt/data chunk")
    tag, ch, sr, _, _, bits = fmt
    if (
        tag == 0xFFFE and len(fmt_body) >= 26
    ):  # WAVE_FORMAT_EXTENSIBLE: real tag is in the sub-format
        tag = struct.unpack("<H", fmt_body[24:26])[0]
    if tag == 3 and bits == 32:
        x = np.frombuffer(pcm[: len(pcm) // 4 * 4], dtype="<f4").astype(np.float32)
    elif tag == 1 and bits == 16:
        x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(pcm[: len(pcm) // 4 * 4], dtype="<i4").astype(np.float32) / 2147483648.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(pcm[: len(pcm) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - (1 << 24), v)
        x = v.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported WAV format tag={tag} bits={bits}")
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return x, sr


def analyze_wav(path, expected_seconds=None) -> dict:
    """Heuristic checks used for the PASS/FAIL line. Returns {"checks": {name: (ok, detail)}, "passed": bool}."""
    import numpy as np

    checks = {}
    try:
        x, sr = read_wav_samples(path)
    except Exception as exc:  # noqa: BLE001 - report any parse failure as a failed check
        return {"checks": {"readable": (False, str(exc))}, "passed": False}
    dur = len(x) / sr if sr else 0.0
    checks["readable"] = (True, f"{dur:.2f} s at {sr} Hz")
    checks["no_nan_inf"] = (
        bool(np.isfinite(x).all()),
        "all samples finite" if np.isfinite(x).all() else "NaN/Inf found",
    )
    xf = np.nan_to_num(x)
    rms = float(np.sqrt(np.mean(xf**2))) if len(xf) else 0.0
    peak = float(np.abs(xf).max()) if len(xf) else 0.0
    checks["not_silent"] = (rms > 0.003 and peak > 0.02, f"rms={rms:.4f} peak={peak:.3f}")
    clipped = float(np.mean(np.abs(xf) >= 0.999)) if len(xf) else 0.0
    checks["not_clipped"] = (clipped < 0.01, f"{clipped * 100:.2f}% of samples at full scale")
    if expected_seconds:
        ok = 0.5 * expected_seconds <= dur <= 2.0 * expected_seconds
        checks["duration_sane"] = (ok, f"{dur:.2f} s vs input {expected_seconds:.2f} s")
    else:
        checks["duration_sane"] = (0.5 <= dur <= 60, f"{dur:.2f} s")
    return {"checks": checks, "passed": all(ok for ok, _ in checks.values())}


def pick_reference_clip(zip_path, max_seconds=12.0, min_seconds=3.0):
    """Choose one of the user's dataset wavs for the self-test: the first (by name) whose
    length is within [min_seconds, max_seconds]; else the shortest. Returns the member name."""
    best = None
    with zipfile.ZipFile(zip_path) as zf:
        for i in sorted(zf.infolist(), key=lambda i: i.filename):
            if i.is_dir() or not i.filename.startswith("wavs/"):
                continue
            with zf.open(i) as fh:
                s = wav_seconds(fh)
            if min_seconds <= s <= max_seconds:
                return i.filename
            if best is None or s < best[1]:
                best = (i.filename, s)
    if best is None:
        raise DatasetError("no wavs in zip")
    return best[0]
