"""Answer Validation agent, plus the Fallback and Error terminal nodes.

Where it sits:   answer -> validate -> finalize (supported) | fallback (not supported)
                 rag    -> fallback (tier MEDIUM / WEAK)
                 any failing node -> error

``validation_node``
    LLM call:  none. Python rules (``CitationOverlapValidator``): every number / code in the
               answer must appear verbatim in the cited passages, and each sentence must be mostly
               made of words from them. (Jev, the intended primary validator, is deferred; to add
               it, implement the ``Validator`` protocol and pass it to the workflow.)
    Reads:     ``generated``, ``cited``
    Writes:    ``validation``

``fallback_node``
    LLM call:  none. Approved "I can't confirm that..." wording from ``ResponseBook``.
    Writes:    ``result``: ``FALLBACK``, ``spoken=True``, keeps the evidence tier for the trace.

``error_node``
    LLM call:  none. Safe wording when any stage failed. ``spoken=False``: text only, so a broken
               turn never produces a confident-sounding voice or avatar clip.
    Writes:    ``result``: ``ERROR``, ``spoken=False``.
"""
from __future__ import annotations

from typing import Callable

from app.agents.state import AgentState
from app.workflow.contracts import AnswerResult, ResponseStatus
from app.workflow.routing import ResponseBook
from app.workflow.validation import Validator


def make_validation_node(validator: Validator) -> Callable[[AgentState], dict]:
    """Build the validation node around any ``Validator`` implementation."""

    def validation_node(state: AgentState) -> dict:
        """Graph node: check the generated answer is supported by the passages it cited."""
        generated, cited = state["generated"], state.get("cited", [])
        return {"validation": validator.validate(generated.text, cited)}

    return validation_node


def make_fallback_node(responses: ResponseBook) -> Callable[[AgentState], dict]:
    """Build the fallback node (safe, predefined wording)."""

    def fallback_node(state: AgentState) -> dict:
        """Graph node: reply that the answer cannot be confirmed; spoken aloud, no citations."""
        req = state["request"]
        return {
            "result": AnswerResult(
                session_id=req.session_id,
                status=ResponseStatus.FALLBACK,
                text=responses.fallback(req.language),
                evidence_tier=state.get("tier"),
                language=req.language,
                spoken=True,
            )
        }

    return fallback_node


def make_error_node(responses: ResponseBook) -> Callable[[AgentState], dict]:
    """Build the terminal error node. It is not wrapped by ``safe_node``: it must not fail."""

    def error_node(state: AgentState) -> dict:
        """Graph node: reply with the safe error wording as text only (``spoken=False``)."""
        req = state["request"]
        return {
            "result": AnswerResult(
                session_id=req.session_id,
                status=ResponseStatus.ERROR,
                text=responses.error(req.language),
                evidence_tier=state.get("tier"),
                language=req.language,
                spoken=False,
            )
        }

    return error_node
