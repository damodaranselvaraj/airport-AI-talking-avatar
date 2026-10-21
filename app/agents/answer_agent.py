"""Answer Generation agent (gpt-4o) and the Finalize node that approves its output.

Where it sits:   rag (tier STRONG) -> answer -> validate -> finalize | fallback

``answer_node``
    LLM call:  YES. One gpt-4o call through the existing ``OpenAIAnswerGenerator`` (JSON mode,
               temperature 0). The model sees ONLY the numbered retrieved passages and the
               question, and must answer from them. It has NO tool binding: retrieval already
               happened in the RAG sub-graph, so the agent cannot fetch anything else.
    Reads:     ``request``, ``passages``
    Writes:    ``generated`` (the model's structured answer) and ``cited`` (the passages the
               answer says it used).

``finalize_node``
    LLM call:  none. Runs only after the validation agent accepted the answer.
    Reads:     ``request``, ``generated``, ``cited``, ``tier``
    Writes:    ``result``: ``ANSWERED``, ``spoken=True``, with de-duplicated citations.

Unvalidated model text is never returned to the passenger: ``finalize_node`` is only reachable
through the ``validate`` node's "supported" edge (see ``app/agents/graph.py``).
"""
from __future__ import annotations

from typing import Callable

from app.agents.safety import TimeoutRunner
from app.agents.state import AgentState
from app.workflow.answer import AnswerGenerator
from app.workflow.contracts import AnswerResult, Citation, ResponseStatus


def make_answer_node(generator: AnswerGenerator, runner: TimeoutRunner) -> Callable[[AgentState], dict]:
    """Build the answer node. ``runner`` enforces the per-node timeout on the LLM call."""

    def answer_node(state: AgentState) -> dict:
        """Graph node: ask gpt-4o for a grounded answer and map its used-passage numbers to passages."""
        req, passages = state["request"], state["passages"]
        generated = runner.run(generator.generate, req.text, passages, req.language)
        # ``generated.used`` holds 0-based indices into ``passages``; ignore any out of range so a
        # misbehaving model cannot cite a passage it was never given.
        cited = [passages[i] for i in generated.used if 0 <= i < len(passages)]
        return {"generated": generated, "cited": cited}

    return answer_node


def make_finalize_node() -> Callable[[AgentState], dict]:
    """Build the finalize node (approves a validated answer)."""

    def finalize_node(state: AgentState) -> dict:
        """Graph node: package the validated answer with unique citations; spoken aloud."""
        req = state["request"]
        # Several cited passages can come from the same chunk; show each (source, chunk) once.
        seen: set[tuple[str, str]] = set()
        citations: list[Citation] = []
        for p in state.get("cited", []):
            key = (p.source, p.chunk_id)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(source=p.source, chunk_id=p.chunk_id))
        return {
            "result": AnswerResult(
                session_id=req.session_id,
                status=ResponseStatus.ANSWERED,
                text=state["generated"].text,
                citations=citations,
                evidence_tier=state.get("tier"),
                language=req.language,
                spoken=True,
            )
        }

    return finalize_node
