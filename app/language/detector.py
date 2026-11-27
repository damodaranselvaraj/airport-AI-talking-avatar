"""Offline typed-text language detection restricted to EN/HI/ES (lingua).

The detector is the authoritative supported-set gate for typed text at
``/api/ask``: it decides the governing language of a turn. It is built once
(model load is non-trivial) and injected. ``detect`` never raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lingua import Language, LanguageDetectorBuilder

from app import i18n


@dataclass(frozen=True)
class DetectionResult:
    code: str          # normalized to en|hi|es (DEFAULT when unsupported)
    is_supported: bool  # False for out-of-set OR below the confidence floor


class LanguageDetector(Protocol):
    def detect(self, text: str) -> DetectionResult: ...


class LinguaLanguageDetector:
    """Typed-text detector restricted to EN, HI and ES.

    Restricting the builder to the three languages is the primary isolation
    mechanism: the model can only ever return one of them. A top-candidate
    confidence below ``min_confidence`` is treated as unsupported so the caller
    can refuse politely.
    """

    def __init__(self, min_confidence: float = 0.60) -> None:
        self._min_confidence = min_confidence
        self._detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.HINDI, Language.SPANISH
        ).build()

    def detect(self, text: str) -> DetectionResult:
        if not text or not text.strip():
            return DetectionResult(i18n.DEFAULT, False)
        try:
            values = self._detector.compute_language_confidence_values(text)
        except Exception:
            return DetectionResult(i18n.DEFAULT, False)
        if not values:
            return DetectionResult(i18n.DEFAULT, False)
        top = values[0]
        if top.value < self._min_confidence:
            return DetectionResult(i18n.DEFAULT, False)
        code = i18n.normalize(top.language.iso_code_639_1.name)
        return DetectionResult(code, i18n.is_supported(code))
