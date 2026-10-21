"""AgentState: partial construction, expected keys, and LangGraph-style partial-update merging."""
from typing import get_type_hints

from app.agents.state import GUARDRAIL_ALLOW, GUARDRAIL_BLOCK, AgentState
from app.workflow.contracts import KioskRequest

EXPECTED_KEYS = {
    "request",
    "guardrail_status",
    "guardrail_reason",
    "decision",
    "passages",
    "rerank_applied",
    "tier",
    "generated",
    "cited",
    "validation",
    "result",
    "error",
    "timings",
}


def test_state_can_be_built_with_only_a_request():
    state: AgentState = {"request": KioskRequest(session_id="s1", text="hi")}
    assert set(state) == {"request"}


def test_state_declares_every_expected_key():
    assert set(get_type_hints(AgentState)) == EXPECTED_KEYS


def test_state_is_not_total():
    # total=False means no key is required (nodes fill the state in as the turn advances).
    assert AgentState.__required_keys__ == frozenset()


def test_node_style_partial_updates_merge_like_langgraph():
    state: AgentState = {"request": KioskRequest(session_id="s1", text="hi")}
    # LangGraph applies each node's returned dict on top of the running state.
    state.update({"guardrail_status": GUARDRAIL_ALLOW})
    state.update({"timings": {"guardrail": 0.4}})
    assert state["guardrail_status"] == "ALLOW"
    assert state["request"].text == "hi"  # earlier keys survive later partial updates
    assert GUARDRAIL_BLOCK == "BLOCK"
