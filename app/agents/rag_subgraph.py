"""Airport RAG sub-graph: retrieve passages, then grade how well they support an answer.

Where it sits:   a compiled LangGraph sub-graph used as ONE node ("rag") in the main graph, entered
                 only for AIRPORT_QUESTION turns.

                     START -> retrieve -> evidence_evaluator -> END

LLM call:        NONE. Retrieval does call external APIs (OpenAI embeddings for the dense query,
                 Cohere for the rerank) but no chat model is involved. Evidence grading is pure
                 Python thresholds.
Reads:           ``request``
Writes:          ``passages``, ``rerank_applied`` (retrieve) and ``tier`` (evidence_evaluator).

The main graph reads ``tier`` after this sub-graph: STRONG goes on to the Answer agent, MEDIUM and
WEAK go to the fallback. Each inner node is wrapped with ``safe_node`` so a failure here leaves
``state["error"]`` set and the main graph routes to the error node.
"""
from __future__ import annotations

from typing import Callable, Optional

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from app.agents.safety import TimeoutRunner, safe_node
from app.agents.state import AgentState
from app.agents.tools import search_airport
from app.workflow.contracts import RetrievedPassage
from app.workflow.evidence import EvidenceAssessor


def make_retrieve_node(runner: TimeoutRunner, search_tool: BaseTool) -> Callable[[AgentState], dict]:
    """Build the retrieve node. ``runner`` enforces the per-node timeout on the slow network calls."""

    def retrieve_node(state: AgentState) -> dict:
        """Graph node: call the ``search_airport`` tool and rebuild typed passages from its dicts."""
        query = state["request"].text
        # ``.invoke`` goes through the tool's schema validation, exactly as an LLM tool call would.
        found = runner.run(search_tool.invoke, {"query": query})
        passages = [RetrievedPassage(**item) for item in found["passages"]]
        return {"passages": passages, "rerank_applied": bool(found["rerank_applied"])}

    return retrieve_node


def make_evidence_evaluator_node(assessor: EvidenceAssessor) -> Callable[[AgentState], dict]:
    """Build the evidence node around the calibrated threshold assessor (Python, no LLM)."""

    def evidence_evaluator_node(state: AgentState) -> dict:
        """Graph node: grade the passages STRONG / MEDIUM / WEAK from the top rerank score."""
        tier = assessor.assess(state.get("passages", []), state.get("rerank_applied", True))
        return {"tier": tier}

    return evidence_evaluator_node


def build_rag_subgraph(
    assessor: EvidenceAssessor,
    runner: TimeoutRunner,
    search_tool: Optional[BaseTool] = None,
):
    """Compile the sub-graph ``retrieve -> evidence_evaluator``. Usable on its own or as a node.

    ``search_tool`` defaults to the module-level ``search_airport`` (configured at startup).
    """
    g = StateGraph(AgentState)
    g.add_node("retrieve", safe_node("retrieve", make_retrieve_node(runner, search_tool or search_airport)))
    g.add_node(
        "evidence_evaluator",
        safe_node("evidence_evaluator", make_evidence_evaluator_node(assessor)),
    )
    g.add_edge(START, "retrieve")
    # If retrieve failed (error set) there are no passages to grade: skip straight to the end so
    # the main graph's guard sends the turn to the error node.
    g.add_conditional_edges(
        "retrieve",
        lambda s: END if s.get("error") else "evidence_evaluator",
        {END: END, "evidence_evaluator": "evidence_evaluator"},
    )
    g.add_edge("evidence_evaluator", END)
    return g.compile()
