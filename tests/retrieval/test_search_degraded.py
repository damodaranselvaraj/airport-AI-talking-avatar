"""Tests for retrieval degradation signaling through the service layer."""
from app.retrieval.search import AirportSearch
from app.workflow.contracts import ErrorState, RetrievedPassage
from tests.fakes import GATE_C22


class FakeRetrieverWithDegradation:
    """Test double that reports degraded stages via last_degraded."""

    def __init__(self, passages, error=None, degraded=None):
        self.passages = passages
        self.error = error
        self.last_degraded = degraded if degraded is not None else []
        self.calls = []

    def retrieve_with_status(self, query, top_k):
        self.calls.append(query)
        return list(self.passages)[:top_k], self.error


def test_degraded_retrieval_signals_correctly():
    """A retriever reporting a degraded stage should populate retrieval_degraded and degraded_retrievers."""
    retriever = FakeRetrieverWithDegradation(
        passages=[GATE_C22],
        error=None,  # rerank succeeded
        degraded=["dense"],
    )
    search = AirportSearch(retriever, top_k=5)

    result = search.search_airport("Where is gate C22?")

    assert result.passages == [GATE_C22]
    assert result.rerank_applied is True  # error is None
    assert result.error is None
    assert result.retrieval_degraded is True
    assert result.degraded_retrievers == ["dense"]


def test_healthy_retrieval_no_degradation():
    """A healthy retrieval should have retrieval_degraded=False and empty degraded_retrievers."""
    retriever = FakeRetrieverWithDegradation(
        passages=[GATE_C22],
        error=None,
        degraded=[],  # no failures
    )
    search = AirportSearch(retriever, top_k=5)

    result = search.search_airport("Where is gate C22?")

    assert result.passages == [GATE_C22]
    assert result.rerank_applied is True
    assert result.error is None
    assert result.retrieval_degraded is False
    assert result.degraded_retrievers == []


def test_both_degraded_and_rerank_failure():
    """Degradation and rerank failure are tracked independently."""
    retriever = FakeRetrieverWithDegradation(
        passages=[GATE_C22],
        error=ErrorState(stage="rerank", message="rerank failed", recoverable=True),
        degraded=["sparse"],
    )
    search = AirportSearch(retriever, top_k=5)

    result = search.search_airport("Where is gate C22?")

    assert result.passages == [GATE_C22]
    assert result.rerank_applied is False  # error is not None
    assert result.error is not None
    assert result.error.stage == "rerank"
    assert result.retrieval_degraded is True
    assert result.degraded_retrievers == ["sparse"]


def test_both_retrievers_failed():
    """Both retrievers failing should show both in degraded_retrievers."""
    retriever = FakeRetrieverWithDegradation(
        passages=[],
        error=None,
        degraded=["dense", "sparse"],
    )
    search = AirportSearch(retriever, top_k=5)

    result = search.search_airport("Where is gate C22?")

    assert result.passages == []
    assert result.rerank_applied is True  # no error from rerank (it wasn't called)
    assert result.error is None
    assert result.retrieval_degraded is True
    assert set(result.degraded_retrievers) == {"dense", "sparse"}
