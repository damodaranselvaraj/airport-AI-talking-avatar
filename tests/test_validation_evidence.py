"""Task 3-4: evidence tiers, validator, generator parsing."""
import json

import pytest

from app.workflow.answer import OpenAIAnswerGenerator, build_user_message, parse_generated
from app.workflow.contracts import EvidenceTier
from app.workflow.evidence import EvidenceAssessor, EvidenceThresholds
from app.workflow.validation import CitationOverlapValidator
from tests.fakes import GATE_C22, PRAYER


def passage(score):
    return GATE_C22.model_copy(update={"score": score})


@pytest.mark.parametrize(
    "score,tier",
    [(0.9, EvidenceTier.STRONG), (0.5, EvidenceTier.STRONG), (0.49, EvidenceTier.MEDIUM),
     (0.2, EvidenceTier.MEDIUM), (0.19, EvidenceTier.WEAK), (0.0, EvidenceTier.WEAK)],
)
def test_tier_boundaries(score, tier):
    a = EvidenceAssessor(EvidenceThresholds(0.5, 0.2, calibrated=True))
    assert a.assess([passage(score)]) is tier


def test_top_score_decides_not_first_passage():
    a = EvidenceAssessor(EvidenceThresholds(0.5, 0.2, calibrated=True))
    assert a.assess([passage(0.1), passage(0.8)]) is EvidenceTier.STRONG


def test_no_passages_is_weak_and_rerank_failure_caps_at_medium():
    a = EvidenceAssessor(EvidenceThresholds(0.5, 0.2, calibrated=True))
    assert a.assess([]) is EvidenceTier.WEAK
    assert a.assess([passage(0.9)], rerank_applied=False) is EvidenceTier.MEDIUM
    # Raw RRF scores are tiny (~0.03); that must read as "degraded", not "weak evidence".
    assert a.assess([passage(0.03)], rerank_applied=False) is EvidenceTier.MEDIUM
    assert a.assess([], rerank_applied=False) is EvidenceTier.WEAK


def test_retrieval_degraded_caps_at_medium():
    """When retrieval_degraded=True, always MEDIUM (single-source can never be STRONG)."""
    a = EvidenceAssessor(EvidenceThresholds(0.5, 0.2, calibrated=True))
    # Would be STRONG without degradation → MEDIUM
    assert a.assess([passage(0.9)], retrieval_degraded=True) is EvidenceTier.MEDIUM
    # Would be MEDIUM → MEDIUM
    assert a.assess([passage(0.3)], retrieval_degraded=True) is EvidenceTier.MEDIUM
    # Would be WEAK → MEDIUM (retrieved something, can't confirm relevance)
    assert a.assess([passage(0.1)], retrieval_degraded=True) is EvidenceTier.MEDIUM
    # Empty stays WEAK
    assert a.assess([], retrieval_degraded=True) is EvidenceTier.WEAK


def test_both_rerank_and_retrieval_degraded():
    """Both degraded flags set → always MEDIUM (non-empty) or WEAK (empty)."""
    a = EvidenceAssessor(EvidenceThresholds(0.5, 0.2, calibrated=True))
    assert a.assess([passage(0.9)], rerank_applied=False, retrieval_degraded=True) is EvidenceTier.MEDIUM
    assert a.assess([passage(0.1)], rerank_applied=False, retrieval_degraded=True) is EvidenceTier.MEDIUM
    assert a.assess([], rerank_applied=False, retrieval_degraded=True) is EvidenceTier.WEAK


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        EvidenceThresholds(strong=0.2, medium=0.5)


def test_uncalibrated_placeholder_logs_warning(caplog):
    with caplog.at_level("WARNING"):
        EvidenceAssessor(EvidenceThresholds.from_file("config/evidence_thresholds.json"))
    assert "uncalibrated" in caplog.text


V = CitationOverlapValidator()


def test_faithful_answer_passes():
    assert V.validate("Gate C22 is in Terminal 2, Concourse C, Level 2. Allow 12 minutes.", [GATE_C22]).supported


@pytest.mark.parametrize(
    "answer",
    [
        "Gate C22 is in Terminal 2, Concourse C, Level 3.",          # wrong number
        "Gate C22 closes at 23:30.",                                 # invented time
        "Call +971-0-000-9999 for help.",                            # invented phone
        "Gate C22 has a spa, a cinema and a swimming pool.",         # drifts from the passage
        "",                                                          # empty
    ],
)
def test_unsupported_answers_fail(answer):
    assert not V.validate(answer, [GATE_C22]).supported


def test_no_cited_passages_fails():
    assert not V.validate("Gate C22 is in Terminal 2.", []).supported


def test_validator_only_trusts_cited_passages():
    assert not V.validate("The prayer room is on Level 1.", [GATE_C22]).supported
    assert V.validate("The main prayer room is on Terminal 2, Level 1.", [PRAYER]).supported


def test_time_ranges_with_to_are_supported():
    p = GATE_C22.model_copy(update={"text": "Date Palm Café is open 05:00–22:00."})
    assert V.validate("Date Palm Café is open 05:00 to 22:00.", [p]).supported


def test_parse_generated_variants():
    ok = parse_generated(json.dumps({"can_answer": True, "answer": "Gate C22 is in T2 [1].", "used_passages": [1, 1, 9, "x", True]}), 2)
    assert ok.can_answer and ok.text == "Gate C22 is in T2 ." and ok.used == [0]
    assert not parse_generated(json.dumps({"can_answer": False, "answer": "x", "used_passages": [1]}), 2).can_answer
    assert not parse_generated(json.dumps({"can_answer": True, "answer": "", "used_passages": [1]}), 2).can_answer
    assert not parse_generated("not json", 2).can_answer
    assert not parse_generated("[1,2]", 2).can_answer
    assert not parse_generated(json.dumps({"can_answer": "yes", "answer": "x"}), 2).can_answer


def test_user_message_numbers_passages_and_includes_source():
    msg = build_user_message("Where?", [GATE_C22, PRAYER])
    assert "[1] (source: data/gates.md)" in msg and "[2] (source: data/faq.md)" in msg
    assert msg.endswith("Question: Where?")


class _Client:
    """Minimal OpenAI client stand-in recording the request."""

    def __init__(self, content):
        self.kwargs = None
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.kwargs = kwargs
                msg = type("M", (), {"content": content})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_openai_generator_uses_json_mode_temperature_zero():
    client = _Client(json.dumps({"can_answer": True, "answer": "Gate C22 is in Terminal 2.", "used_passages": [1]}))
    gen = OpenAIAnswerGenerator("k", "gpt-4o", 400, client=client)
    out = gen.generate("Where is C22?", [GATE_C22])
    assert out.can_answer and out.used == [0]
    assert client.kwargs["temperature"] == 0
    assert client.kwargs["response_format"] == {"type": "json_object"}
    assert client.kwargs["max_tokens"] == 400


def test_openai_generator_skips_call_without_passages():
    client = _Client("{}")
    assert not OpenAIAnswerGenerator("k", "gpt-4o", 100, client=client).generate("q", []).can_answer
    assert client.kwargs is None
