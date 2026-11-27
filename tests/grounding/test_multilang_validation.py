"""FEAT-003: CitationOverlapValidator across languages.

The validator's digit/code exact-match rule (step 1) is safety critical and must
hold for EVERY language: a kiosk must never speak an invented gate code, time or
phone number. The lexical word-overlap rule (step 2) is Latin-script only, so it
is skipped for Devanagari sentences (which carry no [a-z0-9] content words) while
Spanish (Latin script) keeps full overlap checking. These tests pin that contract.
"""
from __future__ import annotations

from app.workflow.contracts import RetrievedPassage
from app.workflow.validation import CitationOverlapValidator

V = CitationOverlapValidator()

# A Hindi passage that states Gate C22 is in Terminal 2 (the code stays ASCII).
GATE_C22_HI = RetrievedPassage(
    chunk_id="gates-hi-c22",
    source="data/hi/gates-hi.md",
    score=0.93,
    text="गेट सी22 टर्मिनल 2, कॉनकोर्स सी, लेवल 2 में है। सुरक्षा जांच से 12 मिनट पैदल।",
)

# A Spanish passage (Latin script) that mirrors the English GATE_C22 shape.
GATE_C22_ES = RetrievedPassage(
    chunk_id="gates-es-c22",
    source="data/es/gates-es.md",
    score=0.93,
    text="La puerta C22 está en la Terminal 2, Concourse C, Nivel 2. Camine 12 minutos.",
)


def test_hindi_answer_with_correct_gate_code_validates():
    """Devanagari answer, correct code present verbatim in the cited passage → supported.

    The word-overlap check is skipped (no Latin content words); the digit/code
    guard still ran and passed because 2 and 12 appear in the passage.
    """
    answer = "गेट सी22 टर्मिनल 2, लेवल 2 में है। 12 मिनट पैदल चलें।"
    assert V.validate(answer, [GATE_C22_HI]).supported


def test_hindi_answer_with_invented_code_is_rejected():
    """Devanagari answer, wrong/invented level (3) not in the passage → rejected.

    Step 1 (digit/code exact match) runs for all languages, so the invented '3'
    fails even though the lexical overlap check was skipped for Devanagari.
    """
    answer = "गेट सी22 टर्मिनल 2, लेवल 3 में है।"
    assert not V.validate(answer, [GATE_C22_HI]).supported


def test_hindi_answer_with_invented_phone_is_rejected():
    answer = "मदद के लिए +971-0-000-9999 पर कॉल करें।"
    assert not V.validate(answer, [GATE_C22_HI]).supported


def test_spanish_answer_keeps_full_word_overlap_like_english():
    """Spanish is Latin script → the word-overlap check runs (behaves like English)."""
    faithful = "La puerta C22 está en la Terminal 2, Nivel 2."
    assert V.validate(faithful, [GATE_C22_ES]).supported

    # Drifts to Latin content words absent from the passage → overlap check rejects it.
    drift = "La puerta C22 tiene un cine y una piscina enorme."
    assert not V.validate(drift, [GATE_C22_ES]).supported


def test_spanish_invented_code_is_rejected():
    answer = "La puerta C22 está en la Terminal 2, Nivel 3."
    assert not V.validate(answer, [GATE_C22_ES]).supported
