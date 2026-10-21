"""Hybrid supervisor: rules first (no LLM), classifier only for the catch-all, plus greeting/deny nodes."""
import json

import pytest

from app.agents.supervisor import (
    CLASSIFIER_SYSTEM_PROMPT,
    IntentClassification,
    LLMIntentClassifier,
    make_deny_node,
    make_greeting_node,
    make_supervisor_node,
)
from app.workflow.contracts import KioskRequest, ResponseStatus, Route
from app.workflow.routing import ResponseBook, RouteDecision, Router
from tests.agents.helpers import POLICY, RESPONSES, FakeClassifier
from tests.fakes import ROOT

TRAPS = json.loads((ROOT / "data" / "unknown_traps.json").read_text(encoding="utf-8"))["items"]


def state_for(text: str) -> dict:
    """Initial graph state for one question."""
    return {"request": KioskRequest(session_id="s1", text=text)}


def supervisor(classifier):
    """Supervisor node using the real policy file and the given (fake) classifier."""
    return make_supervisor_node(Router.from_file(POLICY), classifier)


# ----------------------------------------------------------- Step 1: rules, no LLM
def test_greeting_is_decided_by_rules_without_calling_the_llm():
    clf = FakeClassifier()
    decision = supervisor(clf)(state_for("Hello"))["decision"]
    assert decision.route is Route.GREETING
    assert clf.calls == []


def test_flight_status_is_denied_by_rules_without_calling_the_llm():
    clf = FakeClassifier(label="AIRPORT_QUESTION")  # even a "wrong" classifier cannot override a rule
    decision = supervisor(clf)(state_for("What is the status of flight DM442?"))["decision"]
    assert decision.route is Route.OUT_OF_SCOPE
    assert decision.category == "flight_status"
    assert clf.calls == []


@pytest.mark.parametrize("trap", TRAPS, ids=[t["id"] for t in TRAPS])
def test_every_policy_trap_stays_deterministic(trap):
    clf = FakeClassifier(label="AIRPORT_QUESTION")
    decision = supervisor(clf)(state_for(trap["question_en"]))["decision"]
    assert decision.route is Route.OUT_OF_SCOPE
    assert clf.calls == []


# ----------------------------------------------------------- Step 2: LLM on the catch-all
def test_math_question_is_classified_out_of_scope_by_the_llm():
    clf = FakeClassifier(label="OUT_OF_SCOPE")
    decision = supervisor(clf)(state_for("what is 2+2"))["decision"]
    assert decision == RouteDecision(Route.OUT_OF_SCOPE, "out_of_scope")
    assert clf.calls == ["what is 2+2"]


def test_airport_question_is_classified_airport_by_the_llm():
    clf = FakeClassifier(label="AIRPORT_QUESTION")
    decision = supervisor(clf)(state_for("where is the prayer room"))["decision"]
    assert decision.route is Route.AIRPORT_QUESTION
    assert len(clf.calls) == 1


def test_classifier_failure_fails_safe_to_airport_question():
    clf = FakeClassifier(raises=TimeoutError("openai down"))
    decision = supervisor(clf)(state_for("where is the prayer room"))["decision"]
    # Fail-safe: evidence + validation gates downstream still protect the passenger.
    assert decision.route is Route.AIRPORT_QUESTION


def test_no_classifier_means_rules_only():
    decision = supervisor(None)(state_for("what is 2+2"))["decision"]
    assert decision.route is Route.AIRPORT_QUESTION


# ----------------------------------------------------------- greeting / deny nodes
@pytest.fixture(scope="module")
def book() -> ResponseBook:
    """The real approved-wording file."""
    return ResponseBook.from_file(RESPONSES)


def test_greeting_node_is_spoken_and_uses_approved_text(book):
    state = state_for("Hello") | {"decision": RouteDecision(Route.GREETING, "greeting")}
    result = make_greeting_node(book)(state)["result"]
    assert result.status is ResponseStatus.GREETING
    assert result.spoken is True
    assert result.text == book.greeting("greeting")


def test_thanks_category_gets_thanks_wording(book):
    state = state_for("thanks") | {"decision": RouteDecision(Route.GREETING, "thanks")}
    assert make_greeting_node(book)(state)["result"].text == book.greeting("thanks")


def test_deny_node_returns_the_policy_refusal(book):
    state = state_for("flight status") | {"decision": RouteDecision(Route.OUT_OF_SCOPE, "flight_status")}
    result = make_deny_node(book)(state)["result"]
    assert result.status is ResponseStatus.REFUSED
    assert result.spoken is True
    assert result.text == book.refusal("flight_status")


def test_deny_node_has_wording_for_llm_classified_refusals(book):
    state = state_for("what is 2+2") | {"decision": RouteDecision(Route.OUT_OF_SCOPE, "out_of_scope")}
    result = make_deny_node(book)(state)["result"]
    assert result.status is ResponseStatus.REFUSED
    assert "Dam International Airport" in result.text


# ----------------------------------------------------------- the real classifier class
class _FakeChain:
    """Stands in for ``ChatOpenAI(...).with_structured_output(...)``."""

    def __init__(self, reply):
        """``reply`` is what ``invoke`` will return."""
        self.reply = reply
        self.messages = None

    def invoke(self, messages):
        """Remember the messages and return the preset reply."""
        self.messages = messages
        return self.reply


def test_llm_classifier_accepts_a_pydantic_reply_and_sends_the_system_prompt():
    chain = _FakeChain(IntentClassification(intent="OUT_OF_SCOPE"))
    assert LLMIntentClassifier("k", "gpt-4o-mini", client=chain).classify("tell me a joke") == "OUT_OF_SCOPE"
    system, human = chain.messages
    assert system.content == CLASSIFIER_SYSTEM_PROMPT
    assert human.content == "tell me a joke"  # the question travels as a separate, untrusted human message


def test_llm_classifier_accepts_a_dict_reply():
    chain = _FakeChain({"intent": "AIRPORT_QUESTION"})
    assert LLMIntentClassifier("k", "m", client=chain).classify("gate?") == "AIRPORT_QUESTION"


def test_llm_classifier_rejects_unknown_labels():
    with pytest.raises(ValueError):
        LLMIntentClassifier("k", "m", client=_FakeChain({"intent": "GREETING"})).classify("hi")


def test_intent_schema_only_allows_the_two_labels():
    with pytest.raises(Exception):
        IntentClassification(intent="GREETING")
