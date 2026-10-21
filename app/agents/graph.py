"""MultiAgentKioskWorkflow: wires every agent into one LangGraph graph and runs a turn.

Graph (each box is a node; "guard" means any captured error is routed to ``error`` instead):

    START -> guardrail --BLOCK--> guardrail_result (spoken=True) -> END
                 |
               ALLOW
                 v
             supervisor --GREETING-------> greeting -> END
                 |      --OUT_OF_SCOPE---> deny     -> END
                 |      --AIRPORT_QUESTION
                 v
            rag (sub-graph: retrieve -> evidence_evaluator)
                 |--MEDIUM/WEAK--> fallback -> END
                 |--STRONG
                 v
              answer (gpt-4o) --cannot answer--> fallback -> END
                 v
             validate --unsupported--> fallback -> END
                 |--supported--> finalize -> END
    any node error ---------------> error (spoken=False) -> END

LLM calls per turn: 0 (greeting / denied by policy / guardrail block), 1 (denied by classifier), or
2 (gpt-4o-mini intent classifier + gpt-4o answer) for a normal airport question. Retrieval adds an
OpenAI embedding call and a Cohere rerank call.

Statelessness: a ``MemorySaver`` checkpointer is wired (infrastructure for a later multi-turn
version) but each turn uses a fresh random ``thread_id`` and the thread is deleted when the turn
ends, so no state, history or question text survives between turns and none is put in prompts.

Never raises: node failures become ``state["error"]`` (``safe_node``), conditional edges send them
to the ``error`` node (``guard``), and ``run`` converts even a graph crash into a safe ERROR result.
The ``TurnTrace`` and the log line carry no question text.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.answer_agent import make_answer_node, make_finalize_node
from app.agents.guardrail import (
    GuardrailResponses,
    make_guardrail_result_node,
    make_input_guardrail_node,
)
from app.agents.rag_subgraph import build_rag_subgraph
from app.agents.safety import ERROR_NODE, TimeoutRunner, guard, safe_node
from app.agents.state import GUARDRAIL_BLOCK, AgentState
from app.agents.supervisor import (
    IntentClassifier,
    make_deny_node,
    make_greeting_node,
    make_supervisor_node,
)
from app.agents.tools import make_search_tool
from app.agents.validation_agent import make_error_node, make_fallback_node, make_validation_node
from app.retrieval.search import AirportSearch
from app.workflow.answer import AnswerGenerator
from app.workflow.contracts import (
    AnswerResult,
    ErrorState,
    EvidenceTier,
    KioskRequest,
    Route,
    TurnOutcome,
    TurnTrace,
)
from app.workflow.evidence import EvidenceAssessor
from app.workflow.routing import ResponseBook, Router
from app.workflow.validation import Validator

logger = logging.getLogger(__name__)


class MultiAgentKioskWorkflow:
    """Multi-agent kiosk workflow: guardrail, hybrid supervisor, RAG sub-graph, answer + validation."""

    # Reported by ``Services.health()`` so it is visible which workflow is running.
    graph_kind = "multi_agent"

    def __init__(
        self,
        router: Router,
        responses: ResponseBook,
        search: AirportSearch,
        assessor: EvidenceAssessor,
        generator: AnswerGenerator,
        validator: Validator,
        node_timeout_seconds: float = 30.0,
        classifier: Optional[IntentClassifier] = None,
        guardrail_responses: Optional[GuardrailResponses] = None,
        max_question_chars: int = 500,
    ) -> None:
        """Build and compile the graph.

        ``classifier=None`` disables the LLM step of the supervisor (rules only),
        ``guardrail_responses=None`` uses the built-in English wording for blocked inputs, and
        ``max_question_chars`` is the guardrail length limit.
        """
        self._assessor = assessor
        self._validator = validator
        self._runner = TimeoutRunner(node_timeout_seconds)
        # Stateless by design: see the module docstring. One checkpointer, one thread per turn.
        self._checkpointer = MemorySaver()

        # Retrieval is exposed to the RAG sub-graph as a LangChain tool bound to this search.
        rag = build_rag_subgraph(assessor, self._runner, make_search_tool(search))

        self._guardrail_responses = guardrail_responses or GuardrailResponses()
        self._graph = self._build(
            router=router,
            responses=responses,
            rag=rag,
            generator=generator,
            validator=validator,
            classifier=classifier,
            guardrail_responses=self._guardrail_responses,
            max_question_chars=max_question_chars,
        )
        self._error_node = make_error_node(responses)

    @property
    def evidence_calibrated(self) -> bool:
        """True when the evidence thresholds were calibrated (shown on /health)."""
        return self._assessor.calibrated

    # ------------------------------------------------------------------ graph
    def _build(
        self,
        *,
        router: Router,
        responses: ResponseBook,
        rag,
        generator: AnswerGenerator,
        validator: Validator,
        classifier: Optional[IntentClassifier],
        guardrail_responses: GuardrailResponses,
        max_question_chars: int,
    ):
        """Create all nodes and edges and compile the graph with the per-turn checkpointer."""
        g = StateGraph(AgentState)

        # --- nodes: every one except ``error`` is wrapped so exceptions become state["error"] ---
        g.add_node("guardrail", safe_node("guardrail", make_input_guardrail_node(max_question_chars)))
        g.add_node(
            "guardrail_result",
            safe_node("guardrail_result", make_guardrail_result_node(guardrail_responses)),
        )
        g.add_node("supervisor", safe_node("supervisor", make_supervisor_node(router, classifier, self._runner)))
        g.add_node("greeting", safe_node("greeting", make_greeting_node(responses)))
        g.add_node("deny", safe_node("deny", make_deny_node(responses)))
        # The compiled sub-graph is used directly as a node. Its inner nodes are already wrapped.
        g.add_node("rag", rag)
        g.add_node("answer", safe_node("answer", make_answer_node(generator, self._runner)))
        g.add_node("validate", safe_node("validate", make_validation_node(validator)))
        g.add_node("finalize", safe_node("finalize", make_finalize_node()))
        g.add_node("fallback", safe_node("fallback", make_fallback_node(responses)))
        g.add_node(ERROR_NODE, make_error_node(responses))

        # --- edges ---
        g.add_edge(START, "guardrail")
        g.add_conditional_edges(
            "guardrail",
            guard(lambda s: "guardrail_result" if s.get("guardrail_status") == GUARDRAIL_BLOCK else "supervisor"),
            {"guardrail_result": "guardrail_result", "supervisor": "supervisor", ERROR_NODE: ERROR_NODE},
        )
        g.add_conditional_edges(
            "supervisor",
            guard(self._after_supervisor),
            {"greeting": "greeting", "deny": "deny", "rag": "rag", ERROR_NODE: ERROR_NODE},
        )
        # Evidence gate: only STRONG evidence may reach the answer LLM.
        g.add_conditional_edges(
            "rag",
            guard(lambda s: "answer" if s["tier"] is EvidenceTier.STRONG else "fallback"),
            {"answer": "answer", "fallback": "fallback", ERROR_NODE: ERROR_NODE},
        )
        g.add_conditional_edges(
            "answer",
            guard(lambda s: "validate" if s["generated"].can_answer else "fallback"),
            {"validate": "validate", "fallback": "fallback", ERROR_NODE: ERROR_NODE},
        )
        # Validation gate: unsupported model text is discarded, never shown or spoken.
        g.add_conditional_edges(
            "validate",
            guard(lambda s: "finalize" if s["validation"].supported else "fallback"),
            {"finalize": "finalize", "fallback": "fallback", ERROR_NODE: ERROR_NODE},
        )
        for terminal in ("guardrail_result", "greeting", "deny", "finalize", "fallback", ERROR_NODE):
            g.add_edge(terminal, END)

        return g.compile(checkpointer=self._checkpointer)

    @staticmethod
    def _after_supervisor(state: AgentState) -> str:
        """Map the supervisor's route decision to the next node name."""
        return {
            Route.GREETING: "greeting",
            Route.OUT_OF_SCOPE: "deny",
            Route.AIRPORT_QUESTION: "rag",
        }[state["decision"].route]

    # -------------------------------------------------------------------- run
    def run(self, request: KioskRequest) -> TurnOutcome:
        """Run one turn. Never raises; failures become a safe ERROR result."""
        started = time.perf_counter()
        thread_id = uuid.uuid4().hex  # unique per turn so nothing leaks between turns
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state: AgentState = self._graph.invoke({"request": request}, config)
        except Exception as exc:  # noqa: BLE001
            logger.error("agent graph crashed: %s", type(exc).__name__)
            state = {
                "request": request,
                "error": ErrorState(stage="graph", message=type(exc).__name__),
            }
        finally:
            # Drop this turn's checkpoints: keeps memory flat on a long-running kiosk and means
            # no question text is retained after the turn.
            self._checkpointer.delete_thread(thread_id)
        if "result" not in state:  # a terminal node failed, or the graph crashed
            state.update(self._error_node(state))

        result: AnswerResult = state["result"]
        decision = state.get("decision")
        error = state.get("error")
        trace = TurnTrace(
            session_id=request.session_id,
            route=decision.route if decision else None,
            evidence_tier=state.get("tier"),
            status=result.status,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            jev_available=False,
            validator=self._validator.name if "validation" in state else None,
            error_stage=error.stage if error else None,
            stage_ms=state.get("timings", {}),
            guardrail_reason=state.get("guardrail_reason"),
        )
        logger.info(
            "turn session=%s route=%s tier=%s status=%s guardrail=%s latency_ms=%s stages=%s",
            trace.session_id,
            trace.route.value if trace.route else None,
            trace.evidence_tier.value if trace.evidence_tier else None,
            trace.status.value if trace.status else None,
            trace.guardrail_reason,
            trace.latency_ms,
            trace.stage_ms,
        )
        return TurnOutcome(result=result, trace=trace)

    def close(self) -> None:
        """Release the worker threads used for timeouts."""
        self._runner.close()
