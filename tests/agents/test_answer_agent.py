"""Answer agent (gpt-4o via a fake generator) and the finalize node."""
import pytest

from app.agents.answer_agent import make_answer_node, make_finalize_node
from app.agents.safety import TimeoutRunner
from app.workflow.answer import GeneratedAnswer
from app.workflow.contracts import EvidenceTier, KioskRequest, ResponseStatus
from tests.fakes import GATE_C22, PRAYER, FakeGenerator


def base_state(**extra) -> dict:
    """Minimal state with a request, plus any extra keys the test needs."""
    return {"request": KioskRequest(session_id="s1", text="Where is gate C22?")} | extra


def test_used_indices_map_to_cited_passages():
    gen = FakeGenerator(GeneratedAnswer(True, "Gate C22 is in Terminal 2.", [1]))
    out = make_answer_node(gen, TimeoutRunner(5))(base_state(passages=[GATE_C22, PRAYER]))
    assert out["cited"] == [PRAYER]  # index 1, not index 0
    assert out["generated"].text == "Gate C22 is in Terminal 2."
    assert gen.calls == 1


def test_out_of_range_indices_are_ignored():
    gen = FakeGenerator(GeneratedAnswer(True, "Gate C22 is in Terminal 2.", [0, 7, -1]))
    out = make_answer_node(gen, TimeoutRunner(5))(base_state(passages=[GATE_C22]))
    assert out["cited"] == [GATE_C22]


def test_generator_receives_question_passages_and_language():
    seen = {}

    class Spy:
        """Records what the generator is called with."""

        def generate(self, question, passages, language="en"):
            """Capture the arguments and decline to answer."""
            seen.update(question=question, passages=passages, language=language)
            return GeneratedAnswer(False)

    make_answer_node(Spy(), TimeoutRunner(5))(base_state(passages=[GATE_C22]))
    assert seen == {"question": "Where is gate C22?", "passages": [GATE_C22], "language": "en"}


def test_generator_timeout_propagates_as_an_exception():
    # safe_node (in the graph) turns this into state["error"]; the raw node must raise.
    node = make_answer_node(FakeGenerator(delay=1.0), TimeoutRunner(0.1))
    with pytest.raises(Exception):
        node(base_state(passages=[GATE_C22]))


def test_finalize_dedupes_citations_and_is_spoken():
    state = base_state(
        generated=GeneratedAnswer(True, "Gate C22 is in Terminal 2.", [0, 1]),
        cited=[GATE_C22, GATE_C22, PRAYER],
        tier=EvidenceTier.STRONG,
    )
    result = make_finalize_node()(state)["result"]
    assert result.status is ResponseStatus.ANSWERED
    assert result.spoken is True
    assert result.evidence_tier is EvidenceTier.STRONG
    assert [(c.source, c.chunk_id) for c in result.citations] == [
        ("data/gates.md", "gates-c22"),
        ("data/faq.md", "faq-prayer"),
    ]
    assert result.text == "Gate C22 is in Terminal 2."
