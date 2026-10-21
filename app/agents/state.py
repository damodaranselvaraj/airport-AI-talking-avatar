"""AgentState: the single shared state object that every agent in the graph reads and writes.

LangGraph passes this dict through the graph. Each node returns a PARTIAL dict containing only
the keys it changed; LangGraph merges that into the running state (that is why ``total=False``).

Which agent writes which key:

    request              caller (``MultiAgentKioskWorkflow.run``)   the passenger's question
    guardrail_status     input_guardrail                            "ALLOW" or "BLOCK"
    guardrail_reason     input_guardrail                            EMPTY_INPUT | INPUT_TOO_LONG | ...
    decision             supervisor                                 route (greeting/out_of_scope/airport)
    passages             RAG sub-graph: retrieve                    reranked document chunks
    rerank_applied       RAG sub-graph: retrieve                    False when Cohere rerank failed
    retrieval_degraded   RAG sub-graph: retrieve                    True when dense or sparse failed
    tier                 RAG sub-graph: evidence_evaluator          STRONG / MEDIUM / WEAK
    generated            answer agent                               gpt-4o's structured answer
    cited                answer agent                               the passages the answer relied on
    validation           validation agent                           supported / not supported
    result               greeting|deny|finalize|fallback|error|     the approved ``AnswerResult``
                         guardrail_result                           (what the kiosk shows / speaks)
    error                any node (via ``safe_node``)               set when a stage raised
    timings              any node (via ``safe_node``)               per-stage milliseconds

All value types are the existing contracts from ``app/workflow``; none are redefined here.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from app.workflow.answer import GeneratedAnswer
from app.workflow.contracts import (
    AnswerResult,
    ErrorState,
    EvidenceTier,
    KioskRequest,
    RetrievedPassage,
)
from app.workflow.routing import RouteDecision
from app.workflow.validation import ValidationResult

# Values of ``guardrail_status``.
GUARDRAIL_ALLOW = "ALLOW"
GUARDRAIL_BLOCK = "BLOCK"


class AgentState(TypedDict, total=False):
    """Shared per-turn state. Every key is optional because nodes fill it in as the turn advances."""

    request: KioskRequest
    guardrail_status: str
    guardrail_reason: Optional[str]
    decision: RouteDecision
    passages: list[RetrievedPassage]
    rerank_applied: bool
    retrieval_degraded: bool
    tier: EvidenceTier
    generated: GeneratedAnswer
    cited: list[RetrievedPassage]
    validation: ValidationResult
    result: AnswerResult
    error: ErrorState
    timings: dict[str, float]
