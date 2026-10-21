"""Tests for parallel dense/sparse retrieval with fault tolerance."""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from unittest.mock import Mock

import pytest

from app.retrieval.retrievers import HybridRetriever
from app.workflow.contracts import ErrorState, RetrievedPassage


class FakeDenseRetriever:
    """Test double for DenseRetriever."""

    def __init__(
        self,
        results: Optional[list[tuple[str, float]]] = None,
        raises: Optional[Exception] = None,
        delay: float = 0.0,
    ) -> None:
        self.results = results if results is not None else [("dense-1", 0.9), ("dense-2", 0.8)]
        self.raises = raises
        self.delay = delay
        self.calls = 0
        self.call_time: Optional[float] = None

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        self.calls += 1
        self.call_time = time.time()
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        return self.results[:top_k]


class FakeSparseRetriever:
    """Test double for SparseRetriever (BM25)."""

    def __init__(
        self,
        results: Optional[list[tuple[str, float]]] = None,
        raises: Optional[Exception] = None,
        delay: float = 0.0,
    ) -> None:
        self.results = results if results is not None else [("sparse-1", 0.7), ("sparse-2", 0.6)]
        self.raises = raises
        self.delay = delay
        self.calls = 0
        self.call_time: Optional[float] = None

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        self.calls += 1
        self.call_time = time.time()
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        return self.results[:top_k]


def test_both_retrievers_execute_concurrently():
    """Both retrievers should be called, and timing proves they overlap."""
    dense = FakeDenseRetriever(delay=0.1)
    sparse = FakeSparseRetriever(delay=0.1)

    # The helper we'll implement on HybridRetriever
    def _run_dense_sparse(query: str, dense_k: int, sparse_k: int):
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(dense.retrieve, query, dense_k)
            sparse_future = executor.submit(sparse.retrieve, query, sparse_k)
            
            dense_outcome = None
            sparse_outcome = None
            
            try:
                dense_outcome = dense_future.result()
            except Exception as exc:
                dense_outcome = exc
            
            try:
                sparse_outcome = sparse_future.result()
            except Exception as exc:
                sparse_outcome = exc
            
            return dense_outcome, sparse_outcome

    start = time.time()
    dense_out, sparse_out = _run_dense_sparse("test query", 5, 5)
    elapsed = time.time() - start

    # Both were called
    assert dense.calls == 1
    assert sparse.calls == 1
    
    # Results are returned correctly
    assert dense_out == [("dense-1", 0.9), ("dense-2", 0.8)]
    assert sparse_out == [("sparse-1", 0.7), ("sparse-2", 0.6)]
    
    # Timing proves concurrency: 0.1 + 0.1 would be ~0.2 sequential, but should be ~0.1 parallel
    assert elapsed < 0.15, f"Expected ~0.1s (parallel), got {elapsed:.3f}s (sequential?)"
    
    # Call times should be nearly simultaneous
    assert dense.call_time is not None and sparse.call_time is not None
    time_diff = abs(dense.call_time - sparse.call_time)
    assert time_diff < 0.05, f"Calls should start at nearly the same time, got {time_diff:.3f}s apart"


def test_exception_is_captured_not_raised():
    """When one retriever raises, it should be captured as an outcome, not propagate."""
    dense = FakeDenseRetriever(raises=ValueError("dense failed"))
    sparse = FakeSparseRetriever()

    def _run_dense_sparse(query: str, dense_k: int, sparse_k: int):
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(dense.retrieve, query, dense_k)
            sparse_future = executor.submit(sparse.retrieve, query, sparse_k)
            
            dense_outcome = None
            sparse_outcome = None
            
            try:
                dense_outcome = dense_future.result()
            except Exception as exc:
                dense_outcome = exc
            
            try:
                sparse_outcome = sparse_future.result()
            except Exception as exc:
                sparse_outcome = exc
            
            return dense_outcome, sparse_outcome

    dense_out, sparse_out = _run_dense_sparse("test query", 5, 5)

    # Dense failure captured as exception object
    assert isinstance(dense_out, ValueError)
    assert "dense failed" in str(dense_out)
    
    # Sparse succeeded normally
    assert sparse_out == [("sparse-1", 0.7), ("sparse-2", 0.6)]
    
    # Both were invoked despite one failing
    assert dense.calls == 1
    assert sparse.calls == 1


def test_both_retrievers_fail():
    """When both raise, both exceptions should be captured."""
    dense = FakeDenseRetriever(raises=ValueError("dense boom"))
    sparse = FakeSparseRetriever(raises=RuntimeError("sparse boom"))

    def _run_dense_sparse(query: str, dense_k: int, sparse_k: int):
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(dense.retrieve, query, dense_k)
            sparse_future = executor.submit(sparse.retrieve, query, sparse_k)
            
            dense_outcome = None
            sparse_outcome = None
            
            try:
                dense_outcome = dense_future.result()
            except Exception as exc:
                dense_outcome = exc
            
            try:
                sparse_outcome = sparse_future.result()
            except Exception as exc:
                sparse_outcome = exc
            
            return dense_outcome, sparse_outcome

    dense_out, sparse_out = _run_dense_sparse("test query", 5, 5)

    assert isinstance(dense_out, ValueError)
    assert "dense boom" in str(dense_out)
    assert isinstance(sparse_out, RuntimeError)
    assert "sparse boom" in str(sparse_out)


def test_empty_results_returned_correctly():
    """Empty results (not exceptions) should pass through."""
    dense = FakeDenseRetriever(results=[])
    sparse = FakeSparseRetriever(results=[])

    def _run_dense_sparse(query: str, dense_k: int, sparse_k: int):
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(dense.retrieve, query, dense_k)
            sparse_future = executor.submit(sparse.retrieve, query, sparse_k)
            
            dense_outcome = None
            sparse_outcome = None
            
            try:
                dense_outcome = dense_future.result()
            except Exception as exc:
                dense_outcome = exc
            
            try:
                sparse_outcome = sparse_future.result()
            except Exception as exc:
                sparse_outcome = exc
            
            return dense_outcome, sparse_outcome

    dense_out, sparse_out = _run_dense_sparse("test query", 5, 5)

    assert dense_out == []
    assert sparse_out == []
    assert dense.calls == 1
    assert sparse.calls == 1


# --- Integration tests with HybridRetriever.retrieve_with_status ---


def _make_retriever(dense, sparse):
    """Build a HybridRetriever with fake dense/sparse components."""
    settings = Mock()
    settings.dense_top_k = 10
    settings.sparse_top_k = 10
    settings.rrf_k = 60
    settings.hybrid_top_k = 20
    settings.reranker_max_candidates = 50
    
    # Minimal fuser
    fuser = Mock()
    fuser.fuse = Mock(return_value=[("chunk-1", 0.5), ("chunk-2", 0.4)])
    
    # Minimal chroma
    chroma = Mock()
    chroma.get_texts = Mock(return_value={"chunk-1": "Text 1", "chunk-2": "Text 2"})
    chroma.metadata = Mock(side_effect=lambda cid: {"source": f"data/{cid}.md"})
    
    # Minimal reranker
    reranker = Mock()
    reranker.rerank = Mock(return_value=[("chunk-1", 0.9), ("chunk-2", 0.7)])
    
    retriever = HybridRetriever(settings, None, chroma, None, reranker)
    retriever._dense = dense
    retriever._sparse = sparse
    retriever._fuser = fuser
    
    return retriever


def test_retrieve_with_status_dense_fails_sparse_succeeds():
    """Dense fails → sparse-only results, last_degraded=['dense'], error is None."""
    dense = FakeDenseRetriever(raises=ValueError("dense boom"))
    sparse = FakeSparseRetriever(results=[("chunk-1", 0.7), ("chunk-2", 0.6)])
    retriever = _make_retriever(dense, sparse)
    
    passages, error = retriever.retrieve_with_status("test query", 5)
    
    assert len(passages) == 2
    assert error is None  # rerank succeeded
    assert retriever.last_degraded == ["dense"]
    assert dense.calls == 1
    assert sparse.calls == 1


def test_retrieve_with_status_sparse_fails_dense_succeeds():
    """Sparse fails → dense-only results, last_degraded=['sparse'], error is None."""
    dense = FakeDenseRetriever(results=[("chunk-1", 0.9), ("chunk-2", 0.8)])
    sparse = FakeSparseRetriever(raises=RuntimeError("sparse boom"))
    retriever = _make_retriever(dense, sparse)
    
    passages, error = retriever.retrieve_with_status("test query", 5)
    
    assert len(passages) == 2
    assert error is None
    assert retriever.last_degraded == ["sparse"]
    assert dense.calls == 1
    assert sparse.calls == 1


def test_retrieve_with_status_both_fail():
    """Both fail → empty passages, last_degraded lists both, error is None."""
    dense = FakeDenseRetriever(raises=ValueError("dense boom"))
    sparse = FakeSparseRetriever(raises=RuntimeError("sparse boom"))
    retriever = _make_retriever(dense, sparse)
    
    passages, error = retriever.retrieve_with_status("test query", 5)
    
    assert passages == []
    assert error is None  # no rerank attempted
    assert set(retriever.last_degraded) == {"dense", "sparse"}
    assert dense.calls == 1
    assert sparse.calls == 1


def test_retrieve_with_status_both_succeed():
    """Neither fails → normal behavior, last_degraded=[], error is None."""
    dense = FakeDenseRetriever(results=[("chunk-1", 0.9)])
    sparse = FakeSparseRetriever(results=[("chunk-2", 0.7)])
    retriever = _make_retriever(dense, sparse)
    
    passages, error = retriever.retrieve_with_status("test query", 5)
    
    assert len(passages) == 2
    assert error is None
    assert retriever.last_degraded == []
    assert dense.calls == 1
    assert sparse.calls == 1


def test_retrieve_with_status_rerank_failure_independent_of_retriever_failure():
    """Dense fails + rerank fails → sparse-only with fused order, last_degraded=['dense'], error.stage='rerank'."""
    dense = FakeDenseRetriever(raises=ValueError("dense boom"))
    sparse = FakeSparseRetriever(results=[("chunk-1", 0.7)])
    retriever = _make_retriever(dense, sparse)
    retriever._reranker.rerank = Mock(side_effect=Exception("rerank boom"))
    
    passages, error = retriever.retrieve_with_status("test query", 5)
    
    assert len(passages) == 2  # fused-order fallback
    assert error is not None
    assert error.stage == "rerank"
    assert retriever.last_degraded == ["dense"]  # retriever failure tracked separately

