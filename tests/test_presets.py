import json
import shutil
import subprocess
import wave

import numpy as np
import pytest

from cyv import ffmpeg
from cyv.presets import PresetError, get_preset, load_presets

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def sine_wav(tmp_path, rate=24000, seconds=1.0):
    t = np.arange(int(rate * seconds)) / rate
    x = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return p.read_bytes()


def probe(tmp_path, data, ext):
    p = tmp_path / f"o.{ext}"
    p.write_bytes(data)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,channels,sample_rate,bits_per_raw_sample,bit_rate",
         "-show_entries", "format=format_name", "-of", "json", str(p)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return json.loads(out.stdout)


def test_all_presets_load():
    assert set(load_presets()) == {"master", "telegram", "whatsapp", "mp3"}


def test_unknown_preset():
    with pytest.raises(PresetError, match="unknown preset"):
        get_preset("nope; rm -rf /")


def test_args_from_data():
    args = get_preset("telegram").ffmpeg_args()
    assert args[args.index("-c:a") + 1] == "libopus"
    assert args[args.index("-application") + 1] == "voip"
    assert args[args.index("-b:a") + 1] == "32k"


def test_master(tmp_path):
    info = probe(tmp_path, ffmpeg.encode(sine_wav(tmp_path), get_preset("master")), "wav")
    s = info["streams"][0]
    assert s["codec_name"] == "pcm_s24le"
    assert int(s["sample_rate"]) == 48000


@pytest.mark.parametrize("name", ["telegram", "whatsapp"])
def test_voice_note(tmp_path, name):
    data = ffmpeg.encode(sine_wav(tmp_path), get_preset(name))
    info = probe(tmp_path, data, "ogg")
    s = info["streams"][0]
    assert s["codec_name"] == "opus"
    assert s["channels"] == 1
    assert int(s["sample_rate"]) == 48000
    assert "ogg" in info["format"]["format_name"]


def test_mp3(tmp_path):
    info = probe(tmp_path, ffmpeg.encode(sine_wav(tmp_path), get_preset("mp3")), "mp3")
    assert info["streams"][0]["codec_name"] == "mp3"


def test_ffmpeg_failure_is_error(tmp_path):
    with pytest.raises(ffmpeg.FfmpegError):
        ffmpeg.encode(b"not a wav", get_preset("master"))
