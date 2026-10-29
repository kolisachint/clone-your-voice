from pathlib import Path

import pytest

from cyv.dataset.batch import (
    BatchError,
    BatchLine,
    f5_metadata,
    format_manifest,
    match_recordings,
    parse_manifest,
    read_manifest,
    split_by_model,
)
from cyv.languages import tts_engine

REPO = Path(__file__).resolve().parents[1]
HEADER = "file\tlang\tstyle\ttext\n"


def _lines() -> list[BatchLine]:
    return [
        BatchLine("en_conv_001", "en", "conv", "Hey, are you free tonight?"),
        BatchLine("hi_conv_001", "hi", "conv", "यार, आज मीटिंग बहुत लंबी चली।"),
        BatchLine("mr_read_001", "mr", "read", "आमच्या गावाजवळ एक जुनी विहीर आहे."),
        BatchLine("hi_sing_001", "hi", "sing", "सा रे ग म प"),
    ]


def test_routing_by_language() -> None:
    assert tts_engine("en") == "f5"
    assert tts_engine("hi") == tts_engine("mr") == "indicf5"
    assert tts_engine("hi", "sing") is None  # no speech model learns to sing
    with pytest.raises(ValueError, match="unsupported language"):
        tts_engine("ta")


def test_manifest_round_trip_and_split() -> None:
    lines = _lines()
    assert parse_manifest(format_manifest(lines)) == lines
    split = split_by_model(lines)
    assert split["rvc"] == lines  # every clip trains the voice model
    assert [x.file for x in split["f5"]] == ["en_conv_001"]
    assert [x.file for x in split["indicf5"]] == ["hi_conv_001", "mr_read_001"]


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ("en_conv_1\ten\tconv\tHello there.", "look like"),
        ("ta_conv_001\tta\tconv\tவணக்கம்", "language must be"),
        ("en_conv_001\ten\tread\tHello there.", "does not match"),
        ("en_conv_001\ten\tconv\t  ", "empty"),
        ("en_conv_001\ten\tconv\tनमस्ते", "Devanagari"),
        ("hi_conv_001\thi\tconv\tnamaste dost", "must be in Devanagari"),
        ("en_conv_001\ten\tconv\t" + "a" * 301, "longer"),
        ("en_conv_001\ten\tconv", "columns"),
    ],
)
def test_bad_rows_rejected(row: str, match: str) -> None:
    with pytest.raises(BatchError, match=match):
        parse_manifest(HEADER + row + "\n")


def test_header_duplicates_and_empty() -> None:
    with pytest.raises(BatchError, match="header"):
        parse_manifest("en_conv_001\ten\tconv\tHi.\n")
    row = "en_conv_001\ten\tconv\tHi there.\n"
    with pytest.raises(BatchError, match="duplicate"):
        parse_manifest(HEADER + row + row)
    with pytest.raises(BatchError, match="no lines"):
        parse_manifest(HEADER)


def test_format_rejects_tabs() -> None:
    with pytest.raises(BatchError, match="tabs"):
        format_manifest([BatchLine("en_conv_001", "en", "conv", "a\tb")])


def test_match_recordings(tmp_path: Path) -> None:
    lines = _lines()
    for name in ("en_conv_001.m4a", "hi_conv_001.wav", "hi_conv_001.mp3", "extra_001.wav"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not audio")
    r = match_recordings(lines, tmp_path)
    assert r.found == {"en_conv_001": tmp_path / "en_conv_001.m4a"}
    assert r.duplicates == ["hi_conv_001"]
    assert r.missing == ["mr_read_001", "hi_sing_001"]
    assert r.unexpected == ["extra_001.wav"]
    assert not r.complete


def test_f5_metadata() -> None:
    meta = f5_metadata(split_by_model(_lines())["f5"])
    assert meta == "audio_file|text\nwavs/en_conv_001.wav|Hey, are you free tonight?\n"
    with pytest.raises(BatchError, match=r"\|"):
        f5_metadata([BatchLine("en_conv_001", "en", "conv", "a | b")])


def test_committed_batches_are_valid() -> None:
    batches = sorted((REPO / "docs" / "guides" / "batches").glob("batch-*.tsv"))
    assert batches
    for b in batches:
        assert read_manifest(b)


def test_takes_and_reading_sheet() -> None:
    from cyv.dataset.batch import reading_sheet, take_name, takes

    t = takes(_lines())
    assert {k: [x.file for x in v] for k, v in t.items()} == {
        "en": ["en_conv_001"],
        "hi": ["hi_conv_001"],
        "mr": ["mr_read_001"],
        "sing": ["hi_sing_001"],
    }
    assert take_name("001", "hi") == "batch-001_hi"
    with pytest.raises(BatchError, match="3 digits"):
        take_name("1", "hi")
    sheet = reading_sheet("001", _lines())
    assert "## `batch-001_sing`" in sheet and "1. *(sing)* सा रे ग म प" in sheet
