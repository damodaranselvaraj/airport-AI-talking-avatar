"""RAG sub-graph: retrieve -> evidence_evaluator, invoked on its own (no main graph)."""
import pytest

from app.agents.rag_subgraph import build_rag_subgraph
from app.agents.safety import TimeoutRunner
from app.agents.tools import make_search_tool
from app.retrieval.search import AirportSearch
from app.workflow.contracts import ErrorState, EvidenceTier, KioskRequest, RetrievedPassage
from app.workflow.evidence import EvidenceAssessor, EvidenceThresholds
from tests.fakes import GATE_C22, FakeRetriever


def build(retriever: FakeRetriever, timeout: float = 5.0):
    """Compile the sub-graph around a fake retriever."""
    assessor = EvidenceAssessor(EvidenceThresholds(0.5, 0.2, calibrated=True))
    tool = make_search_tool(AirportSearch(retriever, top_k=5))
    return build_rag_subgraph(assessor, TimeoutRunner(timeout), tool)


def request(text: str = "Where is gate C22?") -> dict:
    """Initial sub-graph state holding only the request."""
    return {"request": KioskRequest(session_id="s1", text=text)}


def passage(chunk_id: str, score: float) -> RetrievedPassage:
    """A passage with the given id and rerank score."""
    return RetrievedPassage(chunk_id=chunk_id, text=f"text {chunk_id}", source="data/gates.md", score=score)


def test_three_strong_passages_give_strong_tier():
    retriever = FakeRetriever([passage("a", 0.9), passage("b", 0.8), passage("c", 0.7)])
    out = build(retriever).invoke(request())
    assert [p.chunk_id for p in out["passages"]] == ["a", "b", "c"]
    assert all(isinstance(p, RetrievedPassage) for p in out["passages"])  # typed again after the tool
    assert out["rerank_applied"] is True
    assert out["tier"] is EvidenceTier.STRONG
    assert set(out["timings"]) == {"retrieve", "evidence_evaluator"}
    assert retriever.calls == ["Where is gate C22?"]


def test_weak_scores_give_weak_tier():
    out = build(FakeRetriever([passage("a", 0.05)])).invoke(request())
    assert out["tier"] is EvidenceTier.WEAK


def test_medium_scores_give_medium_tier():
    out = build(FakeRetriever([passage("a", 0.3)])).invoke(request())
    assert out["tier"] is EvidenceTier.MEDIUM


def test_empty_result_is_weak():
    out = build(FakeRetriever([])).invoke(request())
    assert out["passages"] == []
    assert out["tier"] is EvidenceTier.WEAK


def test_rerank_failure_caps_tier_at_medium():
    err = ErrorState(stage="rerank", message="down")
    out = build(FakeRetriever([GATE_C22], error=err)).invoke(request())
    assert out["rerank_applied"] is False
    assert out["tier"] is EvidenceTier.MEDIUM


def test_retrieval_failure_sets_error_and_skips_grading():
    out = build(FakeRetriever(raises=RuntimeError("chroma exploded: sk-SECRET"))).invoke(request())
    assert out["error"].stage == "retrieve"
    assert out["error"].message == "RuntimeError"  # type only, never the exception text
    assert "tier" not in out


def test_slow_retrieval_times_out_into_an_error():
    class Slow(FakeRetriever):
        """A retriever that sleeps longer than the timeout."""

        def retrieve_with_status(self, query, top_k):
            """Sleep, then behave like the normal fake."""
            import time

            time.sleep(1.0)
            return super().retrieve_with_status(query, top_k)

    out = build(Slow([GATE_C22]), timeout=0.1).invoke(request())
    assert out["error"].stage == "retrieve"


def test_subgraph_can_be_invoked_with_the_module_level_tool():
    from app.agents.tools import configure_search

    configure_search(AirportSearch(FakeRetriever([GATE_C22]), top_k=5))
    try:
        assessor = EvidenceAssessor(EvidenceThresholds(0.5, 0.2, calibrated=True))
        out = build_rag_subgraph(assessor, TimeoutRunner(5.0)).invoke(request())
        assert out["tier"] is EvidenceTier.STRONG
    finally:
        configure_search(None)


def test_degraded_retrieval_caps_tier_at_medium():
    """Retrieval degraded (one retriever failed) → tier capped at MEDIUM."""
    retriever = FakeRetriever([passage("a", 0.9)], degraded=["dense"])
    out = build(retriever).invoke(request())
    assert out["passages"] == [passage("a", 0.9)]
    assert out["rerank_applied"] is True  # rerank succeeded
    assert out["retrieval_degraded"] is True
    assert out["tier"] is EvidenceTier.MEDIUM  # capped despite high score


def test_healthy_retrieval_no_degradation():
    """Healthy retrieval (no degradation) → retrieval_degraded=False, tier uses normal thresholds."""
    retriever = FakeRetriever([passage("a", 0.9)], degraded=[])
    out = build(retriever).invoke(request())
    assert out["rerank_applied"] is True
    assert out["retrieval_degraded"] is False
    assert out["tier"] is EvidenceTier.STRONG  # not capped


def test_both_rerank_and_retrieval_degraded():
    """Both rerank and retrieval degraded → tier is MEDIUM."""
    err = ErrorState(stage="rerank", message="down")
    retriever = FakeRetriever([passage("a", 0.9)], error=err, degraded=["sparse"])
    out = build(retriever).invoke(request())
    assert out["rerank_applied"] is False
    assert out["retrieval_degraded"] is True
    assert out["tier"] is EvidenceTier.MEDIUM
