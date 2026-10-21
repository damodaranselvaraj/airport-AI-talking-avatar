"""LangSmith tracing bootstrap and span helpers.

All public functions are no-ops when tracing is disabled, so call sites need
no if/else guards and the kiosk behaves identically with or without tracing.

Architecture
------------
configure_tracing(settings)
    Called once from app/main.py lifespan, before build_services(). Sets the
    required env vars so LangGraph / LangChain auto-trace all LLM calls once
    the server is up. Also sets the module-level ``_active`` flag used by
    every helper below.

turn_trace(name, inputs, metadata)
    Context-manager for the top-level per-turn trace (``ask_turn`` in
    app/api.py ask() and ``transcribe_turn`` in transcribe()). Yields the
    LangSmith RunTree (or None when tracing is off). LangGraph/LangChain calls
    made *inside* the with-block automatically attach as children because
    langsmith.trace() sets the ambient context variable that LangSmith's
    LangChainTracer reads when it creates new runs.

    The RunTree object **stays valid after the with-block exits** — its id,
    trace_id and dotted_order remain in memory. Pass it to background tasks
    so child spans can use create_child() to get the correct dotted_order.

child_span(parent_run, name, run_type, inputs)
    Synchronous child span for components where we own start and end (Kokoro
    TTS, Whisper STT). Accepts the RunTree from turn_trace (may be None when
    tracing is off). Yields the child RunTree so the caller can record outputs
    before the context exits. Auto-closes on exit; explicit span.end(outputs=)
    before exit is honoured and prevents a double-close.

    Uses parent_run.create_child() which correctly prefixes the parent's
    dotted_order into the child's dotted_order — required by the LangSmith
    server for child run validation.

open_child_span / close_child_span
    Non-blocking pair for components with an async callback (Ditto avatar
    render). open_child_span() posts the run immediately via create_child();
    close_child_span() ends and patches it from whatever thread the callback
    fires on.

Failure contract
----------------
Every LangSmith SDK call is wrapped in try/except. A LangSmith outage, a bad
API key, or a network timeout logs a single WARNING and returns None/no-op.
The kiosk never raises to the caller because of observability.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)

# Module-level flag: True only after configure_tracing() succeeds.
_active: bool = False
_project: str = "dam-kiosk"


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def configure_tracing(settings: "Settings") -> None:
    """Set LangSmith env vars from settings. Call once at startup before building services.

    Sets both the legacy LANGCHAIN_* names and the newer LANGSMITH_* names so
    every version of langchain-core picks them up correctly.
    """
    global _active, _project
    if not settings.tracing_enabled:
        logger.info(
            "LangSmith tracing disabled "
            "(LANGSMITH_TRACING=false or LANGSMITH_API_KEY not set)."
        )
        _active = False
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    # Both key names are checked by different langsmith/langchain versions.
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    _active = True
    _project = settings.langsmith_project
    logger.info(
        "LangSmith tracing enabled → project=%s endpoint=%s",
        settings.langsmith_project,
        settings.langsmith_endpoint,
    )


def tracing_active() -> bool:
    """True when LangSmith tracing is configured and active."""
    return _active


# ---------------------------------------------------------------------------
# Top-level turn trace (ask_turn, transcribe_turn)
# ---------------------------------------------------------------------------

@contextmanager
def turn_trace(
    name: str,
    inputs: dict[str, Any],
    metadata: Optional[dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """Context manager for a top-level turn trace.

    Yields the LangSmith RunTree or None when tracing is off.

    The RunTree object remains valid *after* the with-block exits (its id,
    trace_id and dotted_order stay in memory). Capture it before the block
    closes and pass it to background tasks so child spans can call
    parent_run.create_child() — which correctly inherits dotted_order::

        with turn_trace("ask_turn", {...}) as run:
            outcome = workflow.run(request)
            if run is not None:
                run.add_metadata({"route": ...})
            parent_run = run          # capture INSIDE the with-block
        # context exits → parent run is ended and patched
        background.add_task(render_turn, ..., parent_run)   # pass RunTree
    """
    if not _active:
        yield None
        return

    # Acquire the context manager first so a setup failure yields None cleanly.
    cm = None
    run = None
    try:
        from langsmith import trace as ls_trace  # type: ignore[import-untyped]

        cm = ls_trace(
            name,
            run_type="chain",
            inputs=inputs,
            metadata=metadata or {},
        )
        run = cm.__enter__()
    except Exception:
        logger.warning("LangSmith: turn_trace %r setup failed", name, exc_info=False)
        yield None
        return

    # Yield the RunTree to the caller. Do NOT catch caller exceptions.
    try:
        yield run
    finally:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                logger.warning(
                    "LangSmith: turn_trace %r close failed", name, exc_info=False
                )


# ---------------------------------------------------------------------------
# Synchronous child span (Kokoro TTS, Whisper STT)
# ---------------------------------------------------------------------------

@contextmanager
def child_span(
    parent_run: Optional[Any],
    name: str,
    run_type: str,
    inputs: dict[str, Any],
) -> Generator[Any, None, None]:
    """Synchronous child span using parent_run.create_child().

    parent_run is the RunTree yielded by turn_trace (may be None when tracing
    is off or after a LangSmith error). Using create_child() ensures the
    child's dotted_order is correctly prefixed with the parent's dotted_order,
    which is required by the LangSmith server for child-run validation.

    Yields the child RunTree so the caller can record final outputs::

        with child_span(parent_run, "kokoro_tts", "tool", {"text_chars": n}) as span:
            wav = synthesizer.synthesize(text)
            if span is not None:
                span.end(outputs={"audio_bytes": len(wav)})

    If the caller calls span.end(outputs=...) the context manager skips the
    auto-close; otherwise it closes with just ``duration_ms``.
    """
    if parent_run is None or not _active:
        yield None
        return

    child = None
    started = time.perf_counter()
    try:
        child = parent_run.create_child(
            name=name,
            run_type=run_type,
            inputs=inputs,
        )
        child.post()
    except Exception:
        logger.warning(
            "LangSmith: child_span %r open failed", name, exc_info=False
        )
        yield None
        return

    try:
        yield child
    finally:
        if child is not None:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            try:
                if child.end_time is None:
                    child.end(outputs={"duration_ms": elapsed_ms})
                child.patch()
            except Exception:
                logger.warning(
                    "LangSmith: child_span %r close failed", name, exc_info=False
                )


# ---------------------------------------------------------------------------
# Async-safe child span (Ditto avatar render — callback-based)
# ---------------------------------------------------------------------------

def open_child_span(
    parent_run: Optional[Any],
    name: str,
    run_type: str,
    inputs: dict[str, Any],
) -> Optional[Any]:
    """Open a child span immediately and return the RunTree.

    Uses parent_run.create_child() so the child's dotted_order is correctly
    prefixed with the parent's — required by the LangSmith server. The caller
    is responsible for closing it via close_child_span() from any thread
    (e.g. a Ditto on_done callback on the avatar-dispatch thread).

    Returns None when tracing is off or on any LangSmith error.
    """
    if parent_run is None or not _active:
        return None
    try:
        child = parent_run.create_child(
            name=name,
            run_type=run_type,
            inputs=inputs,
        )
        child.post()
        return child
    except Exception:
        logger.warning(
            "LangSmith: open_child_span %r failed", name, exc_info=False
        )
        return None


def close_child_span(
    child: Optional[Any],
    outputs: dict[str, Any],
    error: Optional[str] = None,
) -> None:
    """Close a span opened with open_child_span(). Thread-safe; no-op when child is None."""
    if child is None:
        return
    try:
        child.end(outputs=outputs, error=error)
        child.patch()
    except Exception:
        logger.warning("LangSmith: close_child_span failed", exc_info=False)
