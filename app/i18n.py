"""Single-source language module for the EN/HI/ES kiosk.

Every subsystem that needs the supported-language set or the per-language
voice / G2P mapping imports it from here, so that mapping lives in exactly one
place. Settings mirrors ``SUPPORTED`` for ops visibility, but this module is the
authority.
"""
from __future__ import annotations

from dataclasses import dataclass

SUPPORTED: tuple[str, ...] = ("en", "hi", "es")
DEFAULT: str = "en"


@dataclass(frozen=True)
class LangProfile:
    """Per-language voice/G2P mapping used by TTS, STT and the clip caches."""

    code: str                 # "en" | "hi" | "es"
    kokoro_voice: str         # Kokoro voice id, e.g. "af_heart"
    kokoro_lang_code: str     # Kokoro lang_code: "a" | "h" | "e"
    whisper_hint: str | None  # ISO code passed to Whisper, or None for auto-detect


_PROFILES: dict[str, LangProfile] = {
    "en": LangProfile("en", "af_heart", "a", "en"),
    "hi": LangProfile("hi", "hf_alpha", "h", "hi"),
    "es": LangProfile("es", "ef_dora", "e", "es"),
}


def is_supported(code: str | None) -> bool:
    """True only for the two-letter codes in ``SUPPORTED`` (case-insensitive)."""
    if not code:
        return False
    return code.strip().lower() in SUPPORTED


def normalize(code: str | None) -> str:
    """Normalize a language tag to a supported two-letter code.

    None/empty -> DEFAULT; lower-cased; a trailing ``-REGION`` is stripped
    (e.g. ``en-US`` -> ``en``); anything outside ``SUPPORTED`` -> DEFAULT.
    """
    if not code:
        return DEFAULT
    base = code.strip().lower().split("-", 1)[0]
    return base if base in SUPPORTED else DEFAULT


def profile(code: str | None) -> LangProfile:
    """Return the ``LangProfile`` for ``code`` (normalized; DEFAULT on miss)."""
    return _PROFILES[normalize(code)]


def kokoro_voice_for(code: str | None) -> str:
    """Kokoro voice id for ``code`` (e.g. ``af_heart`` for ``en``)."""
    return profile(code).kokoro_voice


def kokoro_lang_code_for(code: str | None) -> str:
    """Kokoro lang_code for ``code`` (``a``/``h``/``e`` for en/hi/es)."""
    return profile(code).kokoro_lang_code


def whisper_hint_for(code: str | None) -> str | None:
    """ISO hint passed to Whisper for ``code``, or None for auto-detect."""
    return profile(code).whisper_hint
