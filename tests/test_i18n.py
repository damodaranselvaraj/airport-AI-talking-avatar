"""Unit tests for the single-source language module (no models)."""
from __future__ import annotations

import pytest

from app import i18n


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("EN-us", "en"),
        ("fr", "en"),
        (None, "en"),
        ("", "en"),
        ("hi", "hi"),
        ("es-ES", "es"),
        ("  HI  ", "hi"),
    ],
)
def test_normalize(raw, expected):
    assert i18n.normalize(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("en", True),
        ("hi", True),
        ("es", True),
        ("fr", False),
        ("", False),
        (None, False),
    ],
)
def test_is_supported(raw, expected):
    assert i18n.is_supported(raw) is expected


@pytest.mark.parametrize(
    "code,lang_code",
    [("en", "a"), ("hi", "h"), ("es", "e")],
)
def test_kokoro_lang_code_for(code, lang_code):
    assert i18n.kokoro_lang_code_for(code) == lang_code


@pytest.mark.parametrize(
    "code,voice",
    [("en", "af_heart"), ("hi", "hf_alpha"), ("es", "ef_dora")],
)
def test_kokoro_voice_for(code, voice):
    assert i18n.kokoro_voice_for(code) == voice


def test_whisper_hint_for():
    assert i18n.whisper_hint_for("en") == "en"
    assert i18n.whisper_hint_for("hi") == "hi"
    assert i18n.whisper_hint_for("es") == "es"


def test_profile_falls_back_to_default():
    assert i18n.profile("fr").code == "en"
    assert i18n.profile(None).code == "en"
    assert i18n.profile("hi").code == "hi"


def test_module_surface():
    assert i18n.SUPPORTED == ("en", "hi", "es")
    assert i18n.DEFAULT == "en"
