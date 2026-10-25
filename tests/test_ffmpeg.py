import shutil

import pytest

from cyv import ffmpeg

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _wav():
    import io
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(struct.pack("<hh", 1000, -1000) * 2400)
    return buf.getvalue()


def test_to_wav_is_mono_16bit():
    import io
    import wave

    with wave.open(io.BytesIO(ffmpeg.to_wav(_wav()))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)


def test_to_wav_rejects_playlist_referencing_local_file(tmp_path):
    secret = tmp_path / "secret.wav"
    secret.write_bytes(_wav())
    hls = f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n{secret}\n#EXT-X-ENDLIST\n".encode()
    concat = f"ffconcat version 1.0\nfile {secret}\n".encode()
    for payload in (hls, concat):
        with pytest.raises(ffmpeg.FfmpegError):
            ffmpeg.to_wav(payload)
