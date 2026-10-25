"""`cyv` entry point. Exit codes: 0 ok, 1 user error, 2 usage, 3 daemon/setup missing."""

from __future__ import annotations

import argparse
import io
import os
import re
import struct
import sys
import wave
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from cyv import __version__

EXIT_OK, EXIT_USER, EXIT_USAGE, EXIT_SETUP = 0, 1, 2, 3
HINT_SETUP = "run `cyv setup` first"


class CliError(Exception):
    def __init__(self, message: str, code: int = EXIT_USER) -> None:
        super().__init__(message)
        self.code = code


# --- output helpers ----------------------------------------------------------


def default_output_name(voice: str | None, ext: str, now: datetime | None = None) -> str:
    safe = re.sub(r"[^a-z0-9_-]+", "-", (voice or "default").lower()).strip("-")[:32] or "default"
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"cyv-{safe}-{stamp}.{ext}"


def resolve_output(out: str | None, voice: str | None, ext: str, force: bool) -> Path | None:
    """Return the target path, or None for stdout. Fails early if it would be overwritten."""
    if out == "-":
        return None
    path = Path(out) if out else Path(default_output_name(voice, ext))
    if path.is_dir():
        raise CliError(f"{path} is a directory")
    if path.exists() and not force:
        raise CliError(f"{path} already exists; use --force to overwrite")
    return path


def write_output(data: bytes, path: Path | None, force: bool) -> None:
    if path is None:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if force else os.O_EXCL)
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError:
        raise CliError(f"{path} already exists; use --force to overwrite") from None
    except OSError as e:
        raise CliError(f"cannot write {path}: {e.strerror}") from e
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    print(f"wrote {path}", file=sys.stderr)


def _client() -> Any:
    from cyv.client import Client

    return Client()


def _is_pcm16_mono_wav(data: bytes) -> bool:
    """True only when `data` is verifiably a 16-bit PCM mono WAV (what the daemon accepts)."""
    if not (data[:4] == b"RIFF" and data[8:12] == b"WAVE"):
        return False
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getcomptype() == "NONE"
    except (wave.Error, EOFError, struct.error):
        return False


def _notify_daemon_reload() -> None:
    """Best-effort: tell an already-running daemon to reload voices. Never spawns one."""
    try:
        from cyv.client import Client

        client = Client()
        if client.status().get("running"):
            client.reload()
    except Exception:  # noqa: BLE001, S110 - convenience only; the next daemon start reloads
        pass


def _finish(wav: bytes, preset_name: str, path: Path | None, force: bool) -> None:
    from cyv import ffmpeg
    from cyv.presets import get_preset

    preset = get_preset(preset_name)
    try:
        data = ffmpeg.encode(wav, preset)
    except ffmpeg.FfmpegError as e:
        raise CliError(str(e)) from e
    write_output(data, path, force)


def _preflight_preset(name: str) -> str:
    from cyv.presets import PresetError, get_preset

    try:
        return get_preset(name).ext
    except PresetError as e:
        raise CliError(str(e), EXIT_USAGE) from e


# --- commands ----------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> int:
    from cyv import setup

    try:
        return setup.run_setup(dry_run=args.dry_run, force=args.force)
    except setup.SetupError as e:
        raise CliError(str(e), EXIT_SETUP) from e


def cmd_doctor(args: argparse.Namespace) -> int:
    from cyv import doctor

    return doctor.run_doctor()


def cmd_dataset_prepare(args: argparse.Namespace) -> int:
    from cyv.dataset import DatasetError, prepare_dataset

    def attest(_report: Any) -> tuple[str, str]:
        owner, text = args.owner, args.attest
        if owner is None or text is None:
            if not sys.stdin.isatty():
                raise CliError("non-interactive: pass --owner and --attest", EXIT_USAGE)
            print(
                "\nConsent: train only on your own voice, or one you have explicit "
                "permission to clone.\nType a one-line attestation (e.g. 'This is my own voice')."
            )
            if owner is None:
                owner = input("Your full name: ")
            if text is None:
                text = input("Attestation: ")
        return owner, text

    out = Path(args.output)
    try:
        report = prepare_dataset(
            Path(args.dir),
            out,
            base_voice=args.base_voice,
            name=args.name,
            attest=attest,
            force=args.force,
            on_report=lambda r: print(r.render()),
        )
    except DatasetError as e:
        raise CliError(str(e)) from e
    print(f"\nwrote {out} ({report.seconds / 60:.1f} min of speech)")
    return EXIT_OK


def cmd_voices(args: argparse.Namespace) -> int:
    from cyv import packs

    try:
        if args.voices_cmd == "import":
            info = packs.import_pack(Path(args.pack), force=args.force)
            print(f"imported voice {info.name!r} -> {info.dir}")
            for warning in getattr(info, "warnings", ()):
                print(f"warning: {warning}", file=sys.stderr)
            _notify_daemon_reload()
        elif args.voices_cmd == "list":
            default = packs.get_default()
            voices = packs.list_voices()
            if not voices:
                print("no voices installed; import one with `cyv voices import <pack>`")
            for v in voices:
                print(f"{'*' if v.name == default else ' '} {v.name}")
        elif args.voices_cmd == "remove":
            packs.remove_voice(args.name)
            print(f"removed voice {args.name!r}")
            _notify_daemon_reload()
        elif args.voices_cmd == "default":
            packs.set_default(args.name)
            print(f"default voice is now {args.name!r}")
            _notify_daemon_reload()
    except packs.PackError as e:
        raise CliError(str(e)) from e
    return EXIT_OK


def _daemon_call(fn: Any) -> Any:
    from cyv.client import DaemonError

    try:
        return fn()
    except DaemonError as e:
        code = getattr(e, "code", None)
        if code == "no_voice":
            raise CliError(
                "no default voice; run `cyv voices import <pack>` / `cyv voices default <name>`"
            ) from e
        if code:  # the daemon answered: this is a problem with the request, not the install
            raise CliError(str(e)) from e
        raise CliError(f"{e}\n{HINT_SETUP} (or see `cyv doctor`)", EXIT_SETUP) from e


def cmd_say(args: argparse.Namespace) -> int:
    text = sys.stdin.read() if args.text == "-" else args.text
    if not text.strip():
        raise CliError("nothing to say: text is empty")
    ext = _preflight_preset(args.preset)
    path = resolve_output(args.output, args.voice, ext, args.force)
    if args.no_watermark:
        print("note: watermark disabled for this output", file=sys.stderr)
    client = _client()

    def go() -> Any:
        client.ensure_daemon()
        return client.say(text, voice=args.voice, pitch=args.pitch, watermark=not args.no_watermark)

    _header, wav = _daemon_call(go)
    _finish(wav, args.preset, path, args.force)
    return EXIT_OK


def cmd_convert(args: argparse.Namespace) -> int:
    from cyv import ffmpeg

    ext = _preflight_preset(args.preset)
    path = resolve_output(args.output, args.voice, ext, args.force)
    try:
        data = sys.stdin.buffer.read() if args.input == "-" else Path(args.input).read_bytes()
    except OSError as e:
        raise CliError(f"cannot read {args.input}: {e.strerror}") from e
    if not _is_pcm16_mono_wav(data):
        try:
            data = ffmpeg.to_wav(data)
        except ffmpeg.FfmpegError as e:
            raise CliError(f"cannot decode input audio: {e}") from e
    if args.no_watermark:
        print("note: watermark disabled for this output", file=sys.stderr)
    client = _client()

    def go() -> Any:
        client.ensure_daemon()
        return client.convert(
            data, voice=args.voice, pitch=args.pitch, watermark=not args.no_watermark
        )

    _header, wav = _daemon_call(go)
    _finish(wav, args.preset, path, args.force)
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    client = _client()
    _daemon_call(client.ensure_daemon)
    print("daemon running")
    return EXIT_OK


def cmd_stop(args: argparse.Namespace) -> int:
    from cyv.client import DaemonError

    try:
        was_running = _client().shutdown()
    except DaemonError:
        was_running = False
    print("daemon stopped" if was_running else "daemon was not running")
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    from cyv.client import DaemonError

    try:
        status = _client().status()
    except DaemonError:
        status = {"running": False}
    if not status.get("running"):
        print("daemon not running")
        return EXIT_SETUP
    for k, v in status.items():
        print(f"{k}: {v}")
    return EXIT_OK


# --- parser ------------------------------------------------------------------


def _synth_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--voice", help="voice name (default: your default voice)")
    p.add_argument("--preset", default="master", help="master (default), telegram, whatsapp, mp3")
    p.add_argument("--pitch", type=int, default=0, help="semitone shift")
    p.add_argument("-o", "--output", help="output file, or - for stdout")
    p.add_argument("--force", action="store_true", help="overwrite an existing output file")
    p.add_argument("--no-watermark", action="store_true", help="disable the audio watermark")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cyv", description="Clone your own voice, locally.")
    p.add_argument("--version", action="version", version=f"cyv {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>", required=True)

    s = sub.add_parser("setup", help="install the daemon environment and models")
    s.add_argument("--dry-run", action="store_true", help="print the plan only")
    s.add_argument("--force", action="store_true", help="proceed on unsupported platforms")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("doctor", help="diagnose the installation")
    s.set_defaults(func=cmd_doctor)

    ds = sub.add_parser("dataset", help="dataset tools").add_subparsers(
        dest="dataset_cmd", metavar="<subcommand>", required=True
    )
    s = ds.add_parser("prepare", help="build dataset.zip from a folder of recordings")
    s.add_argument("dir", help="folder with your recordings")
    s.add_argument("-o", "--output", default="dataset.zip")
    s.add_argument("--base-voice", default="am_adam", help="Kokoro base voice (default am_adam)")
    s.add_argument("--name", help="voice name ^[a-z0-9_-]{1,32}$ (default: folder name)")
    s.add_argument("--owner", help="your name for the consent record (else prompted)")
    s.add_argument("--attest", help="consent attestation text (else prompted)")
    s.add_argument("--force", action="store_true", help="overwrite an existing output")
    s.set_defaults(func=cmd_dataset_prepare)

    vs = sub.add_parser("voices", help="manage voices").add_subparsers(
        dest="voices_cmd", metavar="<subcommand>", required=True
    )
    s = vs.add_parser("import", help="import a .cyvpack.zip (only from people you trust)")
    s.add_argument("pack")
    s.add_argument("--force", action="store_true", help="replace an existing voice")
    vs.add_parser("list", help="list installed voices")
    s = vs.add_parser("remove", help="remove a voice")
    s.add_argument("name")
    s = vs.add_parser("default", help="set the default voice")
    s.add_argument("name")
    for sp in vs.choices.values():
        sp.set_defaults(func=cmd_voices)

    s = sub.add_parser("say", help="synthesize text in your voice")
    s.add_argument("text", help='text to speak, or "-" to read stdin')
    _synth_opts(s)
    s.set_defaults(func=cmd_say)

    s = sub.add_parser("convert", help="convert a recording to your voice")
    s.add_argument("input", help="input audio file, or - for stdin")
    _synth_opts(s)
    s.set_defaults(func=cmd_convert)

    for name, fn, helptext in (
        ("serve", cmd_serve, "start the daemon"),
        ("stop", cmd_stop, "stop the daemon"),
        ("status", cmd_status, "show daemon status"),
    ):
        sub.add_parser(name, help=helptext).set_defaults(func=fn)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else EXIT_USAGE
    try:
        return int(args.func(args))
    except CliError as e:
        print(f"cyv: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
