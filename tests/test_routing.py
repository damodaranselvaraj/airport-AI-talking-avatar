"""Task 2: deterministic routing + approved wording."""
import json
from pathlib import Path

import pytest

from app.workflow.contracts import Route
from app.workflow.routing import ResponseBook, Router, normalize

ROOT = Path(__file__).resolve().parents[1]
TRAPS = json.loads((ROOT / "data" / "unknown_traps.json").read_text(encoding="utf-8"))["items"]

# trap id -> denial category the policy must assign
TRAP_CATEGORY = {
    "trap-flight-status": "flight_status",
    "trap-live-gate": "live_gate",
    "trap-visa": "visa",
    "trap-fit-to-fly": "fit_to_fly",
    "trap-fare": "fare",
    "trap-rebook": "rebook",
    "trap-other-airport": "other_airport",
    "trap-medical-advice": "medical",
    "trap-bag-on-plane": "lost_aircraft",
    "trap-weather-delay": "weather",
}


@pytest.fixture(scope="module")
def router() -> Router:
    return Router.from_file(ROOT / "config" / "policy.json")


@pytest.fixture(scope="module")
def book() -> ResponseBook:
    return ResponseBook.from_file(ROOT / "config" / "responses.json")


def test_every_trap_is_mapped():
    assert {t["id"] for t in TRAPS} == set(TRAP_CATEGORY)


@pytest.mark.parametrize("trap", TRAPS, ids=[t["id"] for t in TRAPS])
def test_trap_is_denied_with_expected_category(router, trap):
    decision = router.route(trap["question_en"])
    assert decision.route is Route.OUT_OF_SCOPE
    assert decision.category == TRAP_CATEGORY[trap["id"]]


@pytest.mark.parametrize("trap", TRAPS, ids=[t["id"] for t in TRAPS])
def test_refusal_wording_never_violates_trap_rules(router, book, trap):
    decision = router.route(trap["question_en"])
    text = book.refusal(decision.category).lower()
    for banned in trap["must_not_contain"]:
        assert banned.lower() not in text, f"{decision.category!r} refusal contains {banned!r}"


def test_all_wording_is_free_of_any_trap_banned_phrase(book):
    banned = {b.lower() for t in TRAPS for b in t["must_not_contain"]}
    # The generic fallback/greeting strings may legitimately mention a gate; only the
    # refusal and fallback texts (the ones a trap can reach) are held to the rules.
    for text in [book.fallback(), *[book.refusal(c) for c in TRAP_CATEGORY.values()]]:
        for phrase in banned - {"gate", "boarding"}:
            assert phrase not in text.lower()


@pytest.mark.parametrize(
    "text",
    [
        "Hello",
        "hi there!",
        "Good morning",
        "Namaste",
        "How are you?",
        "thanks",
        "Thank you very much.",
        "bye",
    ],
)
def test_small_talk_routes_to_greeting(router, text):
    assert router.route(text).route is Route.GREETING


@pytest.mark.parametrize(
    "text",
    [
        "Where is gate C22?",
        "Hello, where is gate C22?",
        "How long does it take to walk to gate C22 from security?",
        "Where is the prayer room?",
        "Where is lost and found?",
        "How do I get to the city?",
        "Is there a family security lane?",
        "Where can I get coffee near C22?",
        "Where is the medical clinic?",
        "What time does the information desk open?",
        "I am connecting. Do I go through security again?",
        "Can I take water through security?",
    ],
)
def test_wayfinding_questions_are_airport_questions(router, text):
    decision = router.route(text)
    assert decision.route is Route.AIRPORT_QUESTION, decision


def test_greeting_must_match_whole_utterance(router):
    assert router.route("hello where is the prayer room").route is Route.AIRPORT_QUESTION


def test_normalize_handles_punctuation_and_apostrophes():
    assert normalize("  What's   UP?! ") == "whats up"
    assert normalize("fit-to-fly") == "fit to fly"


def test_unknown_language_falls_back_to_default(router, book):
    assert router.route("hello", language="xx").route is Route.GREETING
    assert book.greeting("greeting", language="xx")
