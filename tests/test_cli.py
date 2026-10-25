import io
import subprocess
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

import cyv
from cyv.cli import main as cli

needs_ffmpeg = pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="needs ffmpeg",
)


def make_wav(seconds=0.5, rate=24000) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    x = (0.3 * np.sin(2 * np.pi * 330 * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return buf.getvalue()


class DaemonError(Exception):
    pass


class FakeClient:
    calls: list = []
    down = False

    def ensure_daemon(self):
        if FakeClient.down:
            raise DaemonError("daemon venv missing")

    def say(self, text, voice=None, pitch=0, watermark=True):
        FakeClient.calls.append(("say", text, voice, pitch, watermark))
        return {"ok": True}, make_wav()

    def convert(self, wav, voice=None, pitch=0, watermark=True):
        FakeClient.calls.append(("convert", wav[:4], voice, pitch, watermark))
        return {"ok": True}, make_wav()

    def shutdown(self):
        if FakeClient.down:
            raise DaemonError("not running")

    def status(self):
        if FakeClient.down:
            raise DaemonError("not running")
        return {"device": "cpu"}


@pytest.fixture
def fake_client(monkeypatch, tmp_path):
    FakeClient.calls = []
    FakeClient.down = False
    mod = types.ModuleType("cyv.client")
    mod.Client = FakeClient
    mod.DaemonError = DaemonError
    monkeypatch.setitem(sys.modules, "cyv.client", mod)
    # `from cyv import client` reads the package attribute if the real module was imported.
    monkeypatch.setattr(cyv, "client", mod, raising=False)
    monkeypatch.chdir(tmp_path)
    return FakeClient


def test_help_is_light():
    code = (
        "import sys\nfrom cyv.cli.main import main\n"
        "assert main(['--help']) == 0\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'cyv.client' not in sys.modules and 'cyv.packs' not in sys.modules\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert "cyv" in capsys.readouterr().out


def test_usage_errors():
    assert cli.main([]) == 2
    assert cli.main(["bogus"]) == 2
    assert cli.main(["say"]) == 2


def test_default_output_name_is_safe():
    name = cli.default_output_name("../Evil\n voice", "ogg")
    assert "/" not in name and "\n" not in name and name.endswith(".ogg")
    assert name.startswith("cyv-")


@needs_ffmpeg
@pytest.mark.parametrize(
    ("preset", "ext"),
    [("master", "wav"), ("telegram", "ogg"), ("whatsapp", "ogg"), ("mp3", "mp3")],
)
def test_say_presets(fake_client, preset, ext, tmp_path):
    out = tmp_path / f"o.{ext}"
    rc = cli.main(["say", "hello", "--voice", "me", "--preset", preset, "-o", str(out)])
    assert rc == 0
    assert out.stat().st_size > 100
    assert fake_client.calls == [("say", "hello", "me", 0, True)]


@needs_ffmpeg
def test_say_default_name_and_no_overwrite(fake_client, tmp_path, capsys):
    assert cli.main(["say", "hi", "--voice", "me"]) == 0
    files = list(tmp_path.glob("cyv-me-*.wav"))
    assert len(files) == 1
    existing = tmp_path / "x.wav"
    existing.write_bytes(b"keep")
    assert cli.main(["say", "hi", "-o", str(existing)]) == 1
    assert existing.read_bytes() == b"keep"
    assert "--force" in capsys.readouterr().err
    assert cli.main(["say", "hi", "-o", str(existing), "--force"]) == 0
    assert existing.read_bytes() != b"keep"


@needs_ffmpeg
def test_say_stdout(fake_client, monkeypatch, capsysbinary):
    assert cli.main(["say", "hi", "-o", "-", "--preset", "telegram"]) == 0
    assert capsysbinary.readouterr().out.startswith(b"OggS")


@needs_ffmpeg
def test_no_watermark_notice(fake_client, tmp_path, capsys):
    assert cli.main(["say", "hi", "--no-watermark", "-o", str(tmp_path / "a.wav")]) == 0
    assert "watermark disabled" in capsys.readouterr().err
    assert fake_client.calls[0][-1] is False


def test_bad_preset_is_usage_error(fake_client, capsys):
    assert cli.main(["say", "hi", "--preset", "nope"]) == 2
    assert fake_client.calls == []


def test_daemon_missing_exit_3(fake_client, capsys):
    fake_client.down = True
    assert cli.main(["say", "hi"]) == 3
    assert "cyv setup" in capsys.readouterr().err
    assert cli.main(["status"]) == 3
    assert cli.main(["stop"]) == 0


@needs_ffmpeg
def test_convert(fake_client, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(make_wav())
    out = tmp_path / "out.ogg"
    rc = cli.main(["convert", str(src), "--preset", "telegram", "--pitch", "2", "-o", str(out)])
    assert rc == 0 and out.read_bytes().startswith(b"OggS")
    assert fake_client.calls == [("convert", b"RIFF", None, 2, True)]


def test_convert_missing_input(fake_client, tmp_path):
    assert cli.main(["convert", str(tmp_path / "nope.wav")]) == 1


def test_serve_status_stop(fake_client, capsys):
    assert cli.main(["serve"]) == 0
    assert cli.main(["status"]) == 0
    assert "device: cpu" in capsys.readouterr().out
    assert cli.main(["stop"]) == 0


@pytest.fixture
def fake_packs(monkeypatch):
    class PackError(Exception):
        pass

    state = {"voices": ["alice", "bob"], "default": "alice", "log": []}

    def info(name):
        return types.SimpleNamespace(name=name, dir=Path("/v") / name, manifest={})

    mod = types.ModuleType("cyv.packs")
    mod.PackError = PackError
    mod.list_voices = lambda: [info(n) for n in state["voices"]]
    mod.get_default = lambda: state["default"]

    def import_pack(path, *, daemon_python=None, force=False):
        if "bad" in str(path):
            raise PackError("bad pack")
        state["log"].append(("import", str(path), force))
        return info("carol")

    mod.import_pack = import_pack
    mod.remove_voice = lambda n: state["log"].append(("remove", n))
    mod.set_default = lambda n: state["log"].append(("default", n))
    monkeypatch.setitem(sys.modules, "cyv.packs", mod)
    # `from cyv import packs` reads the package attribute if the real module was imported.
    monkeypatch.setattr(cyv, "packs", mod, raising=False)
    return state


def test_voices_commands(fake_packs, capsys):
    assert cli.main(["voices", "list"]) == 0
    out = capsys.readouterr().out
    assert "* alice" in out and "  bob" in out
    assert cli.main(["voices", "import", "p.cyvpack.zip", "--force"]) == 0
    assert cli.main(["voices", "remove", "bob"]) == 0
    assert cli.main(["voices", "default", "bob"]) == 0
    assert fake_packs["log"] == [
        ("import", "p.cyvpack.zip", True),
        ("remove", "bob"),
        ("default", "bob"),
    ]


def test_voices_pack_error(fake_packs, capsys):
    assert cli.main(["voices", "import", "bad.zip"]) == 1
    assert "bad pack" in capsys.readouterr().err
