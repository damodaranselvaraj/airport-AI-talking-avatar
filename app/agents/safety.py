"""Shared safety plumbing for every agent: timeouts, exception capture, error routing.

This is not an agent itself. It holds the three helpers that make the graph "never raise":

* ``TimeoutRunner``  runs a slow call (LLM, retrieval) in a worker thread with a deadline.
* ``safe_node``      wraps a node so an exception becomes ``state["error"]`` instead of crashing
                     the graph, and records how long the node took in ``state["timings"]``.
* ``guard``          wraps a conditional-edge chooser so any state carrying an error is sent to
                     the terminal ``error`` node before the normal routing logic runs.

No LLM calls happen here. Only the exception TYPE is logged, never its text (it can embed
request details).
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from app.agents.state import AgentState
from app.workflow.contracts import ErrorState

logger = logging.getLogger(__name__)

# Name of the terminal node that turns any captured error into a safe text-only result.
# It cannot be called "error": LangGraph forbids a node name that is also a state key.
ERROR_NODE = "error_handler"


class TimeoutRunner:
    """Run a blocking call with a deadline so one slow dependency cannot hang a turn."""

    def __init__(self, timeout_seconds: float, max_workers: int = 4) -> None:
        """``timeout_seconds`` is the deadline applied to every ``run`` call."""
        self._timeout = timeout_seconds
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agents")

    def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Call ``fn(*args)`` in a worker thread; raise ``TimeoutError`` after the deadline."""
        future = self._pool.submit(fn, *args)
        try:
            return future.result(timeout=self._timeout)
        except Exception:
            # Best effort: a call that is already running cannot be interrupted, but a queued one can.
            future.cancel()
            raise

    def close(self) -> None:
        """Stop accepting work (used when the workflow is torn down)."""
        self._pool.shutdown(wait=False, cancel_futures=True)


def safe_node(stage: str, fn: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
    """Wrap ``fn`` so it never raises and always records its duration under ``stage``."""

    def wrapper(state: AgentState) -> dict:
        """Run the node, time it, and turn any exception into ``{"error": ErrorState}``."""
        started = time.perf_counter()
        try:
            update = fn(state)
        except Exception as exc:  # noqa: BLE001 - every failure must degrade to a safe answer
            # Exception text may embed request details or secrets; log only the type.
            logger.warning("agent stage %r failed: %s", stage, type(exc).__name__)
            update = {"error": ErrorState(stage=stage, message=type(exc).__name__, recoverable=True)}
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        # Merge into the timings accumulated by earlier nodes (LangGraph replaces, not merges, dicts).
        update["timings"] = {**state.get("timings", {}), stage: elapsed}
        return update

    return wrapper


def guard(chooser: Callable[[AgentState], str]) -> Callable[[AgentState], str]:
    """Wrap a conditional-edge chooser: a state with ``error`` set always goes to the error node."""

    def route(state: AgentState) -> str:
        """Return the error node name when a stage failed, else whatever ``chooser`` decides."""
        return ERROR_NODE if state.get("error") else chooser(state)

    return route
