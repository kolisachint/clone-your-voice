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
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class FakeClient:
    calls: list = []
    down = False
    fail_code = None
    reloads = 0
    detect_result = {"ok": True, "prob": 1.0, "message": 0x1177, "crc_ok": True, "backend": "fake"}

    def ensure_daemon(self):
        if FakeClient.fail_code:
            raise DaemonError("daemon says no", FakeClient.fail_code)
        if FakeClient.down:
            raise DaemonError("daemon venv missing")

    def say(self, text, voice=None, pitch=0, watermark=True):
        FakeClient.calls.append(("say", text, voice, pitch, watermark))
        return {"ok": True}, make_wav()

    def convert(self, wav, voice=None, pitch=0, watermark=True):
        FakeClient.calls.append(("convert", wav[:4], voice, pitch, watermark))
        return {"ok": True}, make_wav()

    def detect(self, wav):
        FakeClient.calls.append(("detect", wav[:4]))
        return dict(FakeClient.detect_result)

    def shutdown(self):
        return not FakeClient.down

    def reload(self, voice=None):
        FakeClient.reloads += 1

    def status(self):
        if FakeClient.down:
            return {"running": False}
        return {"running": True, "device": "cpu"}


@pytest.fixture
def fake_client(monkeypatch, tmp_path):
    FakeClient.calls = []
    FakeClient.down = False
    FakeClient.fail_code = None
    FakeClient.reloads = 0
    FakeClient.detect_result = {
        "ok": True,
        "prob": 1.0,
        "message": 0x1177,
        "crc_ok": True,
        "backend": "fake",
    }
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
    assert "not running" in capsys.readouterr().out
    assert cli.main(["stop"]) == 0
    assert "daemon was not running" in capsys.readouterr().out


@pytest.mark.parametrize(
    "code,needle",
    [
        ("no_voice", "voices default"),
        ("bad_voice", "daemon says no"),
        ("voice_not_found", "daemon says no"),
        ("bad_audio", "daemon says no"),
    ],
)
def test_daemon_error_codes_are_user_errors(fake_client, capsys, code, needle):
    fake_client.fail_code = code
    assert cli.main(["say", "hi"]) == 1
    err = capsys.readouterr().err
    assert needle in err and "cyv setup" not in err


@needs_ffmpeg
def test_convert(fake_client, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(make_wav())
    out = tmp_path / "out.ogg"
    rc = cli.main(["convert", str(src), "--preset", "telegram", "--pitch", "2", "-o", str(out)])
    assert rc == 0 and out.read_bytes().startswith(b"OggS")
    assert fake_client.calls == [("convert", b"RIFF", None, 2, True)]


def _wav_with(width, rate=24000, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x01" * (width * channels * 2400))
    return buf.getvalue()


def test_pcm16_detection():
    assert cli._is_pcm16_mono_wav(make_wav())
    assert not cli._is_pcm16_mono_wav(_wav_with(3))
    assert not cli._is_pcm16_mono_wav(_wav_with(2, channels=2))
    assert not cli._is_pcm16_mono_wav(b"RIFF\x00\x00\x00\x00WAVEjunk")
    assert not cli._is_pcm16_mono_wav(b"RIFFgarbage")
    assert not cli._is_pcm16_mono_wav(b"")


@needs_ffmpeg
def test_convert_transcodes_master_preset_24bit(fake_client, tmp_path):
    from cyv import ffmpeg
    from cyv.presets import get_preset

    master = ffmpeg.encode(make_wav(), get_preset("master"))
    with wave.open(io.BytesIO(master)) as w:
        assert w.getsampwidth() == 3
    src = tmp_path / "master.wav"
    src.write_bytes(master)
    seen = {}
    orig = FakeClient.convert

    def spy(self, wav, **kw):
        seen["wav"] = wav
        return orig(self, wav, **kw)

    FakeClient.convert = spy
    try:
        assert cli.main(["convert", str(src), "-o", str(tmp_path / "o.wav")]) == 0
    finally:
        FakeClient.convert = orig
    assert cli._is_pcm16_mono_wav(seen["wav"])


def test_convert_missing_input(fake_client, tmp_path):
    assert cli.main(["convert", str(tmp_path / "nope.wav")]) == 1


def test_watermark_detect(fake_client, tmp_path, capsys):
    src = tmp_path / "in.wav"
    src.write_bytes(make_wav())
    assert cli.main(["watermark", "detect", str(src)]) == 0
    assert capsys.readouterr().out.strip() == "watermarked prob=1.00 message=0x1177 (crc ok)"
    assert fake_client.calls == [("detect", b"RIFF")]


def test_watermark_detect_no_watermark(fake_client, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        FakeClient,
        "detect",
        lambda self, wav: {"ok": True, "prob": 0.03, "message": None, "crc_ok": False},
    )
    src = tmp_path / "in.wav"
    src.write_bytes(make_wav())
    assert cli.main(["watermark", "detect", str(src)]) == 0
    assert capsys.readouterr().out.strip() == "no watermark (prob=0.03)"


def test_watermark_detect_json(fake_client, tmp_path, capsys):
    import json

    src = tmp_path / "in.wav"
    src.write_bytes(make_wav())
    assert cli.main(["watermark", "detect", str(src), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "watermarked": True,
        "prob": 1.0,
        "message": 0x1177,
        "crc_ok": True,
        "backend": "fake",
    }


def test_watermark_detect_stdin(fake_client, capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(make_wav())))
    assert cli.main(["watermark", "detect", "-"]) == 0
    assert "watermarked prob=1.00" in capsys.readouterr().out
    assert fake_client.calls == [("detect", b"RIFF")]


def test_watermark_detect_missing_input(fake_client, tmp_path, capsys):
    assert cli.main(["watermark", "detect", str(tmp_path / "nope.wav")]) == 1
    assert "cannot read" in capsys.readouterr().err
    assert fake_client.calls == []


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


def test_voices_commands(fake_packs, fake_client, capsys):
    assert cli.main(["voices", "list"]) == 0
    out = capsys.readouterr().out
    assert "* alice" in out and "  bob" in out
    assert cli.main(["voices", "import", "p.cyvpack.zip", "--force"]) == 0
    assert cli.main(["voices", "remove", "bob"]) == 0
    assert cli.main(["voices", "default", "bob"]) == 0
    assert fake_client.reloads == 3
    assert fake_packs["log"] == [
        ("import", "p.cyvpack.zip", True),
        ("remove", "bob"),
        ("default", "bob"),
    ]


def test_voices_no_reload_when_daemon_down(fake_packs, fake_client):
    fake_client.down = True
    assert cli.main(["voices", "default", "bob"]) == 0
    assert fake_client.reloads == 0


def test_voices_reload_failure_is_ignored(fake_packs, fake_client, monkeypatch):
    def boom(self, voice=None):
        raise RuntimeError("x")

    monkeypatch.setattr(FakeClient, "reload", boom)
    assert cli.main(["voices", "remove", "bob"]) == 0


def test_voices_pack_error(fake_packs, capsys):
    assert cli.main(["voices", "import", "bad.zip"]) == 1
    assert "bad pack" in capsys.readouterr().err
