"""Online retrieval components: dense, sparse, RRF fusion and Cohere reranking."""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

from app.retrieval.embedding import DenseEmbedder
from app.retrieval.errors import StartupValidationError
from app.retrieval.sparse import Bm25Index
from app.retrieval.store import ChromaStore
from app.security import scrub
from app.workflow.contracts import ErrorState, RetrievedPassage

if TYPE_CHECKING:
    from config.settings import Settings

_COHERE_TIMEOUT_SECONDS = 10.0

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, embedder: DenseEmbedder, store: ChromaStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Embed the query and return up to ``top_k`` (chunk_id, similarity), descending."""
        if top_k <= 0:
            return []
        ranked = self._store.query(self._embedder.embed_query(query), top_k)
        return sorted(ranked, key=lambda item: -item[1])[:top_k]


class SparseRetriever:
    def __init__(self, index: Bm25Index) -> None:
        self._index = index

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Up to ``top_k`` BM25 (chunk_id, score), descending; non-positive scores dropped."""
        if top_k <= 0:
            return []
        hits = [(cid, s) for cid, s in self._index.query(query, top_k) if s > 0]
        return sorted(hits, key=lambda item: -item[1])[:top_k]


class RrfFuser:
    def fuse(
        self,
        ranked_lists: list[list[tuple[str, float]]],
        rrf_k: int,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion; output descending by score, ties broken by chunk_id."""
        if rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")
        if top_k <= 0:
            return []
        terms: dict[str, list[float]] = {}
        for ranked in ranked_lists:
            for rank, (chunk_id, _score) in enumerate(ranked):  # rank starts at 0
                terms.setdefault(chunk_id, []).append(1.0 / (rrf_k + rank + 1))
        # fsum is exact, so the score does not depend on the order of the input lists.
        scores = {cid: math.fsum(parts) for cid, parts in terms.items()}
        fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return fused[:top_k]


class CohereReranker:
    """Cohere ``rerank-v3.5`` reranker. No local fallback exists by design."""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v3.5",
        client: Optional[Any] = None,
        timeout: float = _COHERE_TIMEOUT_SECONDS,
        max_candidates: int = 50,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client
        self._timeout = timeout
        self._max_candidates = max_candidates

    def _get_client(self) -> Any:
        if self._client is None:
            import cohere  # lazy: importing this module needs neither the SDK nor a key

            self._client = cohere.ClientV2(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def validate(self) -> None:
        """Startup probe against ``rerank-v3.5``; raises StartupValidationError on failure."""
        if not self._api_key:
            raise StartupValidationError("Cohere probe failed: COHERE_API_KEY is missing")
        try:
            response = self._get_client().rerank(
                model=self._model, query="ping", documents=["ping"], top_n=1
            )
            if not response.results:
                raise ValueError("probe returned no results")
        except Exception as exc:
            raise StartupValidationError(
                f"Cohere probe to {self._model} failed: {exc!r}"
            ) from exc

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]:
        """candidates = [(chunk_id, text)]; returns up to ``top_k`` (chunk_id, relevance)."""
        if len(candidates) > self._max_candidates:
            raise ValueError(
                f"At most {self._max_candidates} candidates accepted; got {len(candidates)}"
            )
        if not candidates or top_k <= 0:
            return []
        n = min(top_k, len(candidates))
        response = self._get_client().rerank(
            model=self._model,
            query=query,
            documents=[text for _cid, text in candidates],
            top_n=n,
        )
        scored = [
            (candidates[r.index][0], float(r.relevance_score)) for r in response.results
        ]
        # Stable sort keeps Cohere's order among equal scores.
        return sorted(scored, key=lambda item: -item[1])[:n]


@dataclass
class RetrievalTrace:
    """Per-stage results of one hybrid retrieval run (for diagnostics / CLI display).

    Each stage list holds ``(chunk_id, score)`` in rank order. ``errors`` maps a stage
    name ("dense", "sparse", "fuse", "rerank", "lookup") to the failure message.
    ``rerank_fallback`` is True when the fused order stands in for a failed rerank.
    """

    dense: list[tuple[str, float]] = field(default_factory=list)
    sparse: list[tuple[str, float]] = field(default_factory=list)
    fused: list[tuple[str, float]] = field(default_factory=list)
    reranked: list[tuple[str, float]] = field(default_factory=list)
    texts: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    rerank_fallback: bool = False


class HybridRetriever:
    """Dense + sparse -> RRF -> Cohere rerank -> ``RetrievedPassage`` list."""

    def __init__(
        self,
        settings: "Settings",
        embedder: DenseEmbedder,
        chroma: ChromaStore,
        bm25: Bm25Index,
        reranker: CohereReranker,
    ) -> None:
        self._settings = settings
        self._chroma = chroma
        self._dense = DenseRetriever(embedder, chroma)
        self._sparse = SparseRetriever(bm25)
        self._fuser = RrfFuser()
        self._reranker = reranker
        # Set by every retrieve() call: ErrorState(stage="rerank") when the Cohere
        # rerank failed and the fused order was returned instead, else None.
        self.last_error: Optional[ErrorState] = None
        # Set by every retrieve_with_status() call: list of failed stages ("dense", "sparse")
        # when a retriever raised or returned empty; empty list when both succeeded.
        self.last_degraded: list[str] = []

    @classmethod
    def from_settings(cls, settings: "Settings") -> "HybridRetriever":
        """Build the retriever from ``Settings`` (loads the persisted BM25 index)."""
        embedder = DenseEmbedder(
            settings.openai_embedding_model,
            settings.openai_embedding_dims,
            api_key=settings.openai_api_key,
        )
        chroma = ChromaStore(settings.chroma_dir, settings.chroma_collection)
        bm25 = Bm25Index.load(settings.bm25_index_path)
        reranker = CohereReranker(
            settings.cohere_api_key,
            settings.reranker_model,
            max_candidates=settings.reranker_max_candidates,
        )
        return cls(settings, embedder, chroma, bm25, reranker)

    def _run_dense_sparse(
        self, query: str
    ) -> tuple[Union[list[tuple[str, float]], Exception], Union[list[tuple[str, float]], Exception]]:
        """Run dense and sparse retrieval concurrently; return each as result-or-exception.

        Uses ThreadPoolExecutor(max_workers=2) to overlap the I/O-bound dense retrieval
        (OpenAI embed + Chroma) with the CPU-bound sparse retrieval (local BM25). Neither
        exception propagates; both outcomes are captured and returned so the caller can
        decide how to handle partial failure.
        
        Returns: (dense_outcome, sparse_outcome) where each is either a list[(chunk_id, score)]
                 or an Exception instance.
        """
        s = self._settings
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(self._dense.retrieve, query, s.dense_top_k)
            sparse_future = executor.submit(self._sparse.retrieve, query, s.sparse_top_k)

            dense_outcome: Union[list[tuple[str, float]], Exception]
            sparse_outcome: Union[list[tuple[str, float]], Exception]

            try:
                dense_outcome = dense_future.result()
            except Exception as exc:
                dense_outcome = exc

            try:
                sparse_outcome = sparse_future.result()
            except Exception as exc:
                sparse_outcome = exc

            return dense_outcome, sparse_outcome

    def retrieve(self, query: str, top_k: int) -> list[RetrievedPassage]:
        passages, error = self.retrieve_with_status(query, top_k)
        self.last_error = error
        return passages

    def retrieve_trace(self, query: str, top_k: int) -> RetrievalTrace:
        """Run the same pipeline as ``retrieve_with_status`` and keep every stage's output.

        ``top_k`` is the *final* answer count (passed to Cohere reranking). The
        intermediate stages (dense, sparse, fuse) use the wider Settings values for
        recall breadth, exactly as in ``retrieve_with_status``.

        Unlike ``retrieve_with_status`` this never raises: a failing stage is recorded in
        ``trace.errors`` and the pipeline continues with whatever the other stages gave
        (one retriever failing -> the other feeds RRF; rerank failing -> fused order).
        """
        trace = RetrievalTrace()
        if not query or not query.strip() or top_k <= 0:
            return trace
        s = self._settings
        try:
            trace.dense = self._dense.retrieve(query, s.dense_top_k)
        except Exception as exc:
            trace.errors["dense"] = repr(exc)
        try:
            trace.sparse = self._sparse.retrieve(query, s.sparse_top_k)
        except Exception as exc:
            trace.errors["sparse"] = repr(exc)
        try:
            trace.fused = self._fuser.fuse(
                [trace.dense, trace.sparse], s.rrf_k, s.hybrid_top_k
            )
        except Exception as exc:
            trace.errors["fuse"] = repr(exc)
        try:
            ids = list(dict.fromkeys(cid for stage in (trace.dense, trace.sparse, trace.fused)
                                     for cid, _ in stage))
            trace.texts = self._chroma.get_texts(ids)
            for cid in ids:
                try:
                    source = self._chroma.metadata(cid).get("source")
                except KeyError:
                    source = None
                if source:
                    trace.sources[cid] = str(source)
        except Exception as exc:
            trace.errors["lookup"] = repr(exc)

        candidates = [(cid, trace.texts[cid]) for cid, _ in trace.fused if cid in trace.texts]
        if not candidates:
            return trace
        try:
            trace.reranked = self._reranker.rerank(query, candidates, top_k)
        except Exception as exc:
            trace.errors["rerank"] = repr(exc)
            trace.rerank_fallback = True
            fused_scores = dict(trace.fused)
            trace.reranked = [(cid, fused_scores[cid]) for cid, _ in candidates][:top_k]
        return trace

    def retrieve_with_status(
        self, query: str, top_k: int
    ) -> tuple[list[RetrievedPassage], Optional[ErrorState]]:
        """Return up to ``top_k`` passages and a rerank-not-applied signal.

        Dense and sparse retrieval run concurrently. If one fails, the pipeline continues
        with the survivor; only when both fail (or both return empty) does this return [].
        Failed stages are recorded in ``self.last_degraded`` (reset at the start of each call).

        Pipeline stages use the wider Settings values for recall:
          - dense_top_k  → initial dense candidates (concurrent)
          - sparse_top_k → initial BM25 candidates (concurrent)
          - rrf_k / hybrid_top_k → fusion width

        ``top_k`` (this parameter) is the *final* count passed to Cohere reranking
        and returned to the caller. It always governs the output size; Settings
        values only control upstream recall breadth.
        """
        # Reset degradation tracking
        self.last_degraded = []
        
        if not query or not query.strip() or top_k <= 0:
            return [], None  # nothing is invoked for an empty/whitespace query
        
        s = self._settings
        
        # Run dense and sparse concurrently
        dense_outcome, sparse_outcome = self._run_dense_sparse(query)
        
        # Unpack outcomes: result or exception
        dense: list[tuple[str, float]] = []
        sparse: list[tuple[str, float]] = []
        
        if isinstance(dense_outcome, Exception):
            logger.warning("Dense retrieval failed: %s", scrub(repr(dense_outcome))[:200])
            self.last_degraded.append("dense")
        else:
            dense = dense_outcome
        
        if isinstance(sparse_outcome, Exception):
            logger.warning("Sparse (BM25) retrieval failed: %s", scrub(repr(sparse_outcome))[:200])
            self.last_degraded.append("sparse")
        else:
            sparse = sparse_outcome
        
        # Both failed or both empty → hard failure
        if not dense and not sparse:
            return [], None
        
        # Fuse whatever survived (RrfFuser handles empty inputs gracefully)
        fused = self._fuser.fuse([dense, sparse], s.rrf_k, s.hybrid_top_k)
        if not fused:
            return [], None

        texts = self._chroma.get_texts([cid for cid, _ in fused])
        candidates = [(cid, texts[cid]) for cid, _ in fused if cid in texts]
        if not candidates:
            return [], None

        # Rerank stage (unchanged, including its ErrorState fallback)
        error: Optional[ErrorState] = None
        try:
            ranked = self._reranker.rerank(query, candidates, top_k)
        except Exception as exc:
            # No local rerank fallback: keep the fused order and flag it to the caller.
            error = ErrorState(
                stage="rerank",
                message=f"Cohere rerank failed; returning fused order: {exc!r}",
                recoverable=True,
            )
            fused_scores = dict(fused)
            ranked = [(cid, fused_scores[cid]) for cid, _ in candidates][:top_k]

        passages: list[RetrievedPassage] = []
        for chunk_id, score in ranked:
            source = self._chroma.metadata(chunk_id).get("source")
            if not source:
                continue  # a passage without a citable source is not valid
            passages.append(
                RetrievedPassage(
                    chunk_id=chunk_id,
                    text=texts[chunk_id],
                    source=str(source),
                    score=score,
                )
            )
        return passages, error
