"""Which text-to-speech model handles which language (docs/adr/0008).

Every clip trains the voice model (Applio RVC). Only one speech model learns from it,
picked by the clip's language: F5-TTS for English, IndicF5 for Hindi and Marathi.
"""

from __future__ import annotations

F5 = "f5"
INDICF5 = "indicf5"

# Language code -> the speech model that learns from (and later speaks) that language.
TTS_FOR_LANG = {"en": F5, "hi": INDICF5, "mr": INDICF5}

# Styles that train only the voice model. No speech model can sing.
RVC_ONLY_STYLES = frozenset({"sing"})


def tts_engine(lang: str, style: str = "") -> str | None:
    """The speech model a clip trains, or None if it trains the voice model only."""
    if lang not in TTS_FOR_LANG:
        raise ValueError(f"unsupported language {lang!r} (supported: {', '.join(TTS_FOR_LANG)})")
    if style in RVC_ONLY_STYLES:
        return None
    return TTS_FOR_LANG[lang]
