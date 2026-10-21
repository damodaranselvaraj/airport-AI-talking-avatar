"""Validation agent, fallback node and error node."""
from app.agents.validation_agent import make_error_node, make_fallback_node, make_validation_node
from app.workflow.answer import GeneratedAnswer
from app.workflow.contracts import EvidenceTier, KioskRequest, ResponseStatus
from app.workflow.routing import ResponseBook
from app.workflow.validation import CitationOverlapValidator
from tests.agents.helpers import RESPONSES
from tests.fakes import GATE_C22

BOOK = ResponseBook.from_file(RESPONSES)


def base_state(**extra) -> dict:
    """Minimal state with a request, plus any extra keys the test needs."""
    return {"request": KioskRequest(session_id="s1", text="Where is gate C22?")} | extra


def test_supported_answer_passes_validation():
    state = base_state(
        generated=GeneratedAnswer(True, "Gate C22 is in Terminal 2, Concourse C, Level 2.", [0]),
        cited=[GATE_C22],
    )
    out = make_validation_node(CitationOverlapValidator())(state)
    assert out["validation"].supported is True


def test_invented_detail_fails_validation():
    state = base_state(
        generated=GeneratedAnswer(True, "Gate C22 is in Terminal 3, Concourse D, gate D9.", [0]),
        cited=[GATE_C22],
    )
    out = make_validation_node(CitationOverlapValidator())(state)
    assert out["validation"].supported is False


def test_answer_with_no_cited_passages_fails_validation():
    state = base_state(generated=GeneratedAnswer(True, "Gate C22 is in Terminal 2.", []), cited=[])
    assert make_validation_node(CitationOverlapValidator())(state)["validation"].supported is False


def test_fallback_is_spoken_and_keeps_the_tier():
    result = make_fallback_node(BOOK)(base_state(tier=EvidenceTier.WEAK))["result"]
    assert result.status is ResponseStatus.FALLBACK
    assert result.spoken is True
    assert result.evidence_tier is EvidenceTier.WEAK
    assert result.text == BOOK.fallback()
    assert result.citations == []


def test_error_is_text_only():
    result = make_error_node(BOOK)(base_state())["result"]
    assert result.status is ResponseStatus.ERROR
    assert result.spoken is False
    assert result.text == BOOK.error()
