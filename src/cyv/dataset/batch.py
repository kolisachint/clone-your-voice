"""Recording batches: the script you record, and the files you upload (docs/adr/0008).

A batch is a tab-separated file with one line per clip:

    file<TAB>lang<TAB>style<TAB>text
    en_conv_001<TAB>en<TAB>conv<TAB>Hey, are you free this evening?

The file name is `<lang>_<style>_<NNN>`. Every clip trains the voice model. The language
decides which speech model it also trains (`cyv.languages`).

You can upload clips two ways:

- one file per line, named after the line (`en_conv_001.m4a`), or
- one long recording per take (`batch-001_hi.m4a`, see `takes`), read in order, with the line
  number spoken in English before each line. The training notebook cuts it into clips.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from cyv.dataset.prepare import AUDIO_EXTS
from cyv.languages import RVC_ONLY_STYLES, TTS_FOR_LANG, tts_engine

HEADER = ("file", "lang", "style", "text")
NAME_RE = re.compile(r"^(?P<lang>[a-z]{2})_(?P<style>[a-z]{2,12})_(?P<num>[0-9]{3})$")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
MAX_TEXT_CHARS = 300  # about 20 s of speech; longer clips are hard to record cleanly


class BatchError(Exception):
    pass


@dataclass(frozen=True)
class BatchLine:
    file: str
    lang: str
    style: str
    text: str

    @property
    def engine(self) -> str | None:
        """The speech model this clip trains, or None for voice-model-only clips."""
        return tts_engine(self.lang, self.style)


def _check_line(n: int, line: BatchLine) -> None:
    where = f"line {n} ({line.file!r})"
    m = NAME_RE.fullmatch(line.file)
    if not m:
        raise BatchError(f"{where}: file must look like en_conv_001")
    if line.lang not in TTS_FOR_LANG:
        raise BatchError(f"{where}: language must be one of {', '.join(TTS_FOR_LANG)}")
    if (m["lang"], m["style"]) != (line.lang, line.style):
        raise BatchError(
            f"{where}: file name does not match lang {line.lang!r} / style {line.style!r}"
        )
    text = line.text.strip()
    if not text:
        raise BatchError(f"{where}: text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise BatchError(f"{where}: text is longer than {MAX_TEXT_CHARS} characters")
    # A clip in the wrong script would train the wrong speech model.
    if line.lang == "en" and _DEVANAGARI.search(text):
        raise BatchError(f"{where}: English text contains Devanagari")
    if line.lang in ("hi", "mr") and not _DEVANAGARI.search(text):
        raise BatchError(f"{where}: Hindi and Marathi text must be in Devanagari")


def parse_manifest(content: str) -> list[BatchLine]:
    rows = list(csv.reader(io.StringIO(content), delimiter="\t", quoting=csv.QUOTE_NONE))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows or tuple(c.strip() for c in rows[0]) != HEADER:
        raise BatchError("first line must be the header: " + "\\t".join(HEADER))
    lines: list[BatchLine] = []
    seen: set[str] = set()
    for n, row in enumerate(rows[1:], start=2):
        if len(row) != len(HEADER):
            raise BatchError(f"line {n}: expected {len(HEADER)} tab-separated columns")
        line = BatchLine(*(c.strip() for c in row))
        _check_line(n, line)
        if line.file in seen:
            raise BatchError(f"line {n}: duplicate file {line.file!r}")
        seen.add(line.file)
        lines.append(line)
    if not lines:
        raise BatchError("the batch has no lines")
    return lines


def read_manifest(path: Path) -> list[BatchLine]:
    return parse_manifest(Path(path).read_text(encoding="utf-8"))


def format_manifest(lines: list[BatchLine]) -> str:
    for n, line in enumerate(lines, start=2):
        _check_line(n, line)
        if "\t" in line.text or "\n" in line.text:
            raise BatchError(f"line {n}: text must not contain tabs or newlines")
    out = ["\t".join(HEADER)] + ["\t".join((x.file, x.lang, x.style, x.text)) for x in lines]
    return "\n".join(out) + "\n"


@dataclass
class MatchReport:
    found: dict[str, Path] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not (self.missing or self.duplicates)


def match_recordings(lines: list[BatchLine], folder: Path) -> MatchReport:
    """Pair each batch line with its uploaded recording (same name, any audio extension)."""
    wanted = {x.file for x in lines}
    by_stem: dict[str, list[Path]] = {}
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in AUDIO_EXTS:
            by_stem.setdefault(p.stem, []).append(p)
    report = MatchReport()
    for stem, paths in by_stem.items():
        if stem not in wanted:
            report.unexpected.extend(p.name for p in paths)
        elif len(paths) > 1:
            report.duplicates.append(stem)
        else:
            report.found[stem] = paths[0]
    report.missing = [x.file for x in lines if x.file not in by_stem]
    return report


def take_of(line: BatchLine) -> str:
    """Which long recording a line belongs in: its language, or `sing` for singing."""
    return "sing" if line.style in RVC_ONLY_STYLES else line.lang


def takes(lines: list[BatchLine]) -> dict[str, list[BatchLine]]:
    """Lines grouped into long recordings, in reading order. Line numbers start at 1 per take."""
    out: dict[str, list[BatchLine]] = {}
    for x in lines:
        out.setdefault(take_of(x), []).append(x)
    return out


def take_name(batch: str, take: str) -> str:
    """File name (without extension) of one long recording, for example `batch-001_hi`."""
    if not re.fullmatch(r"[0-9]{3}", batch):
        raise BatchError(f"batch id must be 3 digits, got {batch!r}")
    return f"batch-{batch}_{take}"


def reading_sheet(batch: str, lines: list[BatchLine]) -> str:
    """A Markdown sheet to read from: one section per long recording, lines numbered."""
    out = [f"# Batch {batch}: reading sheet", ""]
    for take, group in takes(lines).items():
        out += [f"## `{take_name(batch, take)}`", ""]
        for n, x in enumerate(group, start=1):
            out.append(f"{n}. *({x.style})* {x.text}")
        out.append("")
    return "\n".join(out)


def split_by_model(lines: list[BatchLine]) -> dict[str, list[BatchLine]]:
    """Which clips train which model. Every clip trains `rvc`."""
    out: dict[str, list[BatchLine]] = {"rvc": list(lines)}
    for engine in sorted(set(TTS_FOR_LANG.values())):
        out[engine] = [x for x in lines if x.engine == engine]
    return out


def f5_metadata(lines: list[BatchLine], wav_dir: str = "wavs") -> str:
    """`metadata.csv` in F5-TTS's fine-tuning format (`audio_file|text`, one clip per line)."""
    out = ["audio_file|text"]
    for x in lines:
        if "|" in x.text:
            raise BatchError(f"{x.file}: text must not contain '|'")
        out.append(f"{wav_dir}/{x.file}.wav|{x.text}")
    return "\n".join(out) + "\n"
