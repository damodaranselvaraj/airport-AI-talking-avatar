"""Unit tests for the lingua typed-text detector (real lingua, offline)."""
from __future__ import annotations

import pytest

from app.language.detector import DetectionResult, LinguaLanguageDetector


@pytest.fixture(scope="module")
def detector() -> LinguaLanguageDetector:
    return LinguaLanguageDetector()


def test_detects_hindi(detector):
    result = detector.detect("गेट C22 कहाँ है और सुरक्षा जाँच कितने बजे खुलती है")
    assert result.code == "hi"
    assert result.is_supported is True


def test_detects_spanish(detector):
    result = detector.detect("¿Dónde está la puerta de embarque y a qué hora abre seguridad?")
    assert result.code == "es"
    assert result.is_supported is True


def test_detects_english(detector):
    result = detector.detect("Where is the boarding gate and what time does security open?")
    assert result.code == "en"
    assert result.is_supported is True


def test_empty_string_is_unsupported(detector):
    result = detector.detect("")
    assert isinstance(result, DetectionResult)
    assert result.is_supported is False


def test_whitespace_is_unsupported(detector):
    result = detector.detect("   \t  \n ")
    assert result.is_supported is False


def test_short_gibberish_below_floor(detector):
    # Digits/punctuation carry no language signal, so lingua reports zero
    # confidence -> below the floor -> unsupported refusal.
    result = detector.detect("123 456 !!! ???")
    assert result.code == "en"
    assert result.is_supported is False
