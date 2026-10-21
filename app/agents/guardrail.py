"""Input Guardrail agent: the security boundary every passenger question crosses first.

Where it sits:   START -> input_guardrail -> (ALLOW) supervisor | (BLOCK) guardrail_result -> END
LLM call:        NONE. Pure Python (length checks and compiled regexes), so it is fast and
                 deterministic.
Reads:           ``state["request"].text``
Writes:          ``guardrail_status`` ("ALLOW" | "BLOCK") and, when blocked, ``guardrail_reason``.
                 ``guardrail_result_node`` then writes the ``result`` (an ``AnswerResult``).

Four checks run in this order; the first failure blocks the turn:

    1. EMPTY_INPUT       empty or whitespace-only text
    2. INPUT_TOO_LONG    more than ``max_chars`` characters (``settings.max_question_chars``)
    3. PROMPT_INJECTION  tries to override or reveal the kiosk's instructions
    4. JAILBREAK         tries to make the kiosk adopt another persona or drop its rules

A blocked turn returns ``ResponseStatus.REFUSED`` with ``spoken=True``: the approved wording is
shown on screen and spoken. With the avatar on, the text is one of the fixed phrases that is
pre-rendered with the default concierge face (never a traveler's photo).

NOT in scope for MVP1: PII detection / masking (no Presidio). The question is never logged or
echoed back here (existing privacy rule).

Limits: regexes catch the common, well-known phrasings. They are a first line of defence, not a
guarantee. The answer stage independently refuses to use anything outside the retrieved passages.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Callable, Optional

from app.agents.state import GUARDRAIL_ALLOW, GUARDRAIL_BLOCK, AgentState
from app.workflow.contracts import AnswerResult, ResponseStatus

# --- Reason codes (stored in ``guardrail_reason`` and used as keys in responses.json) ---
EMPTY_INPUT = "EMPTY_INPUT"
INPUT_TOO_LONG = "INPUT_TOO_LONG"
PROMPT_INJECTION = "PROMPT_INJECTION"
JAILBREAK = "JAILBREAK"

# Used when config/responses.json has no ``guardrail`` block. Same wording as the shipped JSON.
_DEFAULT_RESPONSES = {
    EMPTY_INPUT: "Please ask me an airport-related question.",
    INPUT_TOO_LONG: "Please shorten your question and try again.",
    PROMPT_INJECTION: "I can help with airport information, facilities, and passenger assistance.",
    JAILBREAK: "I can help with airport information, facilities, and passenger assistance.",
}

_SPACE_RE = re.compile(r"\s+")
# Zero-width / formatting characters are a common way to split a keyword so a regex misses it.
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")


def _flags(*patterns: str) -> tuple["re.Pattern[str]", ...]:
    """Compile each pattern case-insensitively. Inputs are normalized to lower case first anyway."""
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


# Prompt injection: text that tries to override, replace or reveal the kiosk's instructions.
# Each pattern is anchored on a verb AND an instruction-like noun so that ordinary airport
# questions ("can I override the gate sign?" is not an instruction attack) do not match.
INJECTION_PATTERNS = _flags(
    # "ignore / forget / disregard / override ... previous / all / your ... instructions / rules".
    # The middle word list (previous, your, all ...) is required so "disregard the policy sign"
    # style airport talk does not match; "the" is deliberately NOT in that list.
    r"\b(ignore|forget|disregard|override)\b.{0,25}\b(previous|prior|above|earlier|your|all|any|these|those|system|safety)\b.{0,20}\b(instructions?|rules|prompts?|guidelines|directions|polic(y|ies)|programming)\b",
    # asking for the hidden prompt ("your" only: "show me the rules for liquids" is a normal question)
    r"\bsystem prompt\b",
    r"\b(reveal|show|print|repeat|display|tell me|leak)\b.{0,20}\byour\b.{0,15}\b(prompt|instructions|rules|guidelines)\b",
    r"\b(hidden|secret|initial|original) (prompt|instructions)\b",
    # supplying replacement instructions
    r"\bnew instructions?\b",
    r"\bdo not follow\b.{0,20}\b(instructions?|rules|guidelines)\b",
    # privileged-mode tricks
    r"\b(developer|admin|administrator|debug|sudo|root|god) mode\b",
    # fake chat-role markers smuggled into the text
    r"<\s*/?\s*(system|assistant|instructions?)\s*>",
    r"\[\s*/?\s*(system|inst)\s*\]",
)

# Jailbreak: text that tries to make the kiosk become something else or abandon its limits.
# "act as" alone is NOT blocked (a passenger can ask whether a lane "acts as" a family lane);
# it must be followed by a persona word such as "different", "unrestricted" or "evil".
JAILBREAK_PATTERNS = _flags(
    r"\byou( are|re) now (a|an|my|free|dan|unrestricted|no longer|going to)\b",
    r"\bact (as|like) (a |an |my )?(different|new|another|unrestricted|uncensored|evil|jailbroken|hacker|dan)\b",
    r"\bact as if you (have no|had no|were not|are not|werent|arent)\b",
    r"\bpretend (that )?(you are|youre|to be|you have no|you dont have)\b",
    r"\brole ?play(ing)?\b.{0,20}\b(as|you are|youre)\b",
    r"\bdo anything now\b",
    r"\bdan mode\b",
    r"\b(you are|youre|be|as) dan\b",
    r"\bjailbreak(ing|ed|s)?\b",
    r"\bbypass\b.{0,15}\b(rules|restrictions|filters|safety|guardrails?|polic(y|ies)|limitations)\b",
    r"\bwithout (any )?(rules|restrictions|filters|limits|guardrails)\b.{0,20}\b(answer|respond|reply|talk)\b",
)


def normalize_for_matching(text: str) -> str:
    """Lower-case, fold compatibility Unicode, drop invisible characters and apostrophes, collapse spaces.

    This stops trivial evasions such as "IGNORE   previous", fullwidth letters (NFKC folds them),
    zero-width splits inside a keyword, and "you're" vs "youre". It does not map look-alike
    letters from other scripts (e.g. a Cyrillic "o"); that is a known limit of a regex guardrail.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = _INVISIBLE_RE.sub("", folded).lower()
    folded = folded.replace("\u2019", "'").replace("'", "")
    return _SPACE_RE.sub(" ", folded).strip()


def check_input(text: Optional[str], max_chars: int) -> Optional[str]:
    """Run the four checks in order; return the first failing reason code, or None when clean."""
    # 1. Empty / whitespace only.
    if not text or not text.strip():
        return EMPTY_INPUT
    # 2. Too long. Measured on the raw text so padding with spaces cannot sneak past it.
    if len(text) > max_chars:
        return INPUT_TOO_LONG
    normalized = normalize_for_matching(text)
    # 3. Prompt injection.
    if any(p.search(normalized) for p in INJECTION_PATTERNS):
        return PROMPT_INJECTION
    # 4. Jailbreak.
    if any(p.search(normalized) for p in JAILBREAK_PATTERNS):
        return JAILBREAK
    return None


def make_input_guardrail_node(max_chars: int) -> Callable[[AgentState], dict]:
    """Build the guardrail node. ``max_chars`` comes from ``settings.max_question_chars`` (never hardcoded)."""

    def input_guardrail_node(state: AgentState) -> dict:
        """Graph node: ALLOW the turn or BLOCK it with a reason code. Does not touch the question text."""
        reason = check_input(state["request"].text, max_chars)
        if reason is None:
            return {"guardrail_status": GUARDRAIL_ALLOW}
        return {"guardrail_status": GUARDRAIL_BLOCK, "guardrail_reason": reason}

    return input_guardrail_node


class GuardrailResponses:
    """Approved wording for blocked turns, per language, with a default-language fallback.

    Reads the ``guardrail`` block of ``config/responses.json``. Kept separate from
    ``ResponseBook`` so ``app/workflow`` stays untouched. Missing keys fall back to the default
    language and then to the built-in English wording above.
    """

    def __init__(self, responses: Optional[dict] = None, default_language: str = "en") -> None:
        """``responses`` is the parsed responses.json (or None to use only the built-in wording)."""
        self._default = default_language
        langs = (responses or {}).get("languages", {})
        # {language: {reason_code: text}}
        self._blocks: dict[str, dict[str, str]] = {
            lang: dict(block.get("guardrail", {})) for lang, block in langs.items()
        }

    @classmethod
    def from_file(cls, path: str | Path, default_language: str = "en") -> "GuardrailResponses":
        """Load from ``config/responses.json``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data, default_language)

    def text(self, reason: str, language: str = "en") -> str:
        """Wording for ``reason`` in ``language``; falls back to default language, then built-ins."""
        for lang in (language, self._default):
            found = self._blocks.get(lang, {}).get(reason)
            if found:
                return found
        return _DEFAULT_RESPONSES.get(reason, _DEFAULT_RESPONSES[PROMPT_INJECTION])


def make_guardrail_result_node(responses: GuardrailResponses) -> Callable[[AgentState], dict]:
    """Build the node that turns a BLOCK into the approved text, shown on screen and spoken."""

    def guardrail_result_node(state: AgentState) -> dict:
        """Graph node: produce a REFUSED, ``spoken=True`` result for a blocked turn."""
        req = state["request"]
        reason = state.get("guardrail_reason") or PROMPT_INJECTION
        return {
            "result": AnswerResult(
                session_id=req.session_id,
                status=ResponseStatus.REFUSED,
                text=responses.text(reason, req.language),
                language=req.language,
                spoken=True,  # spoken like any other fixed phrase (pre-rendered concierge clip)
            )
        }

    return guardrail_result_node
