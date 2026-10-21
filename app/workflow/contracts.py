"""Component data contracts for the kiosk workflow.

These are the stable interfaces between components (UI, STT, retrieval,
answer, TTS, avatar, orchestrator). Defining them once, up front, lets each
component be built and tested independently before it is wired together.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class InputType(str, Enum):
    """How the traveler's question arrived — keyboard entry or microphone capture."""

    TYPED = "typed"
    VOICE = "voice"


class Route(str, Enum):
    """Classifier output that determines which workflow branch handles the turn."""

    GREETING = "greeting"
    OUT_OF_SCOPE = "out_of_scope"
    AIRPORT_QUESTION = "airport_question"


class EvidenceTier(str, Enum):
    """Confidence level of the retrieved evidence backing an answer.

    STRONG  — multiple high-score passages fully cover the question.
    MEDIUM  — partial coverage; answer is hedged but still deliverable.
    WEAK    — insufficient grounding; workflow falls back to a safe response.
    """

    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


class ResponseStatus(str, Enum):
    """Terminal disposition of a turn, written to the trace and returned to the UI."""

    ANSWERED = "answered"   # grounded answer, validated against passages
    GREETING = "greeting"   # predefined response, no RAG / no LLM
    REFUSED = "refused"     # deterministic policy denial (out of scope)
    FALLBACK = "fallback"   # insufficient evidence or failed validation
    ERROR = "error"         # a stage failed; safe text-only response


class KioskRequest(BaseModel):
    """A unified question entering the workflow (typed or transcribed)."""

    session_id: str
    text: str
    input_type: InputType = InputType.TYPED
    language: str = "en"


class RetrievedPassage(BaseModel):
    """One retrieved document chunk with provenance for citation."""

    chunk_id: str
    text: str
    source: str = Field(..., description="Source document filename for the citation")
    score: float = Field(..., description="Similarity score from the vector store")


class Citation(BaseModel):
    """A minimal reference attached to an answer, pointing back to its source passage."""

    source: str
    chunk_id: str


class AnswerResult(BaseModel):
    """The approved response handed to speech + avatar.

    `text` is spoken verbatim by TTS, so it must already be the final,
    validated, passenger-friendly wording.
    """

    session_id: str
    status: ResponseStatus
    text: str = Field(..., description="Exact approved text; TTS speaks this verbatim")
    citations: list[Citation] = Field(default_factory=list)
    evidence_tier: Optional[EvidenceTier] = None
    language: str = "en"
    spoken: bool = Field(True, description="Whether this text should be spoken aloud")


class ErrorState(BaseModel):
    """Standard error envelope so every stage fails the same, visible way."""

    stage: str
    message: str
    recoverable: bool = True


class Transcript(BaseModel):
    """Speech-to-text output. Shown to the traveler for correction before any answer."""

    text: str
    duration_s: float
    confidence: Optional[float] = Field(
        None, description="0..1 heuristic from Whisper log-probs; None when unavailable"
    )
    language: str = "en"


class TurnTrace(BaseModel):
    """Per-turn observability record. Carries no raw audio or transcript text."""

    session_id: str
    route: Optional[Route] = None
    evidence_tier: Optional[EvidenceTier] = None
    status: Optional[ResponseStatus] = None
    latency_ms: Optional[float] = None
    jev_available: Optional[bool] = None
    validator: Optional[str] = None
    error_stage: Optional[str] = None
    stage_ms: dict[str, float] = Field(default_factory=dict)
    # Set only by the multi-agent workflow when the input guardrail blocked the turn (a reason
    # code such as "PROMPT_INJECTION", never the question text). None everywhere else.
    guardrail_reason: Optional[str] = None


class TurnOutcome(BaseModel):
    """What the workflow hands back for one turn: the approved answer plus its trace."""

    result: AnswerResult
    trace: TurnTrace
