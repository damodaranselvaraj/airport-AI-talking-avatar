"""Tools available to the agents. Currently one: ``search_airport`` (RAG retrieval).

Where it sits:   called by the ``retrieve`` node of the Airport RAG sub-graph.
LLM call:        NONE directly. The tool runs the existing hybrid retriever, which itself makes
                 two outbound API calls: an OpenAI embedding of the query and a Cohere rerank.
                 No LLM is bound to this tool (the Answer agent has no tool binding by design);
                 it is a LangChain ``@tool`` so it has a typed schema, can be invoked uniformly
                 and can be bound to a model later without rewriting it.
Inputs:          ``query`` (the passenger's question text)
Outputs:         ``{"passages": [{chunk_id, text, source, score}, ...], "rerank_applied": bool, "retrieval_degraded": bool}``
                 Plain dicts, so the result is JSON-serializable and framework neutral.

Two ways to get the tool:

* ``search_airport``        module-level tool wired once at startup with ``configure_search``
                            (a ``@tool`` function cannot take constructor arguments).
* ``make_search_tool(s)``   a tool bound to one specific ``AirportSearch``. The workflow uses this
                            so two workflows in one process (e.g. in tests) never share state.
Both run exactly the same code (``_run_search``).
"""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool, tool

from app.retrieval.search import AirportSearch

# The AirportSearch used by the module-level tool. Set by ``configure_search`` at startup.
_search: Optional[AirportSearch] = None


def configure_search(search: Optional[AirportSearch]) -> None:
    """Inject (or clear, with ``None``) the retriever the module-level ``search_airport`` uses."""
    global _search
    _search = search


def _run_search(search: AirportSearch, query: str) -> dict:
    """Run the hybrid search and return plain, JSON-friendly dicts (shared by both tool forms)."""
    found = search.search_airport(query)
    return {
        "passages": [
            {"chunk_id": p.chunk_id, "text": p.text, "source": p.source, "score": p.score}
            for p in found.passages
        ],
        # Evidence grading needs this: without a rerank the scores are on a different scale.
        "rerank_applied": found.rerank_applied,
        # Evidence grading also needs this: single-source retrieval must be capped at MEDIUM.
        "retrieval_degraded": found.retrieval_degraded,
    }


@tool
def search_airport(query: str) -> dict:
    """Search the approved airport documents for passages relevant to the passenger's question.

    Returns the best reranked passages (chunk_id, text, source, score) and whether the Cohere
    rerank was applied. Raises RuntimeError if the retriever has not been configured.
    """
    if _search is None:
        # Fail loudly: a missing retriever is a wiring bug, not a runtime condition to hide.
        raise RuntimeError("search_airport is not configured; call configure_search() at startup")
    return _run_search(_search, query)


def make_search_tool(search: AirportSearch) -> BaseTool:
    """Return a ``search_airport`` tool bound to ``search`` (no module-level state involved)."""

    @tool("search_airport")
    def bound_search_airport(query: str) -> dict:
        """Search the approved airport documents for passages relevant to the passenger's question.

        Returns the best reranked passages (chunk_id, text, source, score) and whether the Cohere
        rerank was applied.
        """
        return _run_search(search, query)

    return bound_search_airport
