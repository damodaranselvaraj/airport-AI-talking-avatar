"""``search_airport``: the workflow-facing retrieval adapter.

Wraps ``HybridRetriever`` so the workflow gets passages plus an explicit
"was the Cohere rerank applied?" flag, which the evidence assessor needs
(fused RRF scores are on a different scale than rerank relevance scores).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import logging

from app.security import scrub
from app.workflow.contracts import ErrorState, RetrievedPassage

logger = logging.getLogger(__name__)


class _Retriever(Protocol):
    def retrieve_with_status(
        self, query: str, top_k: int
    ) -> tuple[list[RetrievedPassage], Optional[ErrorState]]: ...


@dataclass
class SearchResult:
    passages: list[RetrievedPassage] = field(default_factory=list)
    rerank_applied: bool = True
    error: Optional[ErrorState] = None


class AirportSearch:
    """Search the approved airport documents. Raises if dense/sparse retrieval fails.

    ``top_k`` controls the final number of passages returned to the caller. It is
    passed directly to ``HybridRetriever.retrieve_with_status``, which runs wider
    internal stages (dense_top_k → sparse_top_k → hybrid_top_k) before narrowing
    to this count via Cohere reranking. Always supply ``top_k`` explicitly from
    ``settings.answer_passages`` so the value is never silently defaulted.
    """

    def __init__(self, retriever: _Retriever, top_k: int) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def search_airport(self, query: str) -> SearchResult:
        passages, error = self._retriever.retrieve_with_status(query, self._top_k)
        if error is not None:
            # Diagnostic only: scrubbed, truncated, and never includes the question.
            logger.warning("rerank not applied (%s): %s", error.stage, scrub(error.message)[:200])
        return SearchResult(
            passages=list(passages),
            rerank_applied=error is None,
            error=error,
        )
