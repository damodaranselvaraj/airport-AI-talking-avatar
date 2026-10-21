"""Test doubles and builders shared by the multi-agent tests (no network, no models)."""
from __future__ import annotations

from typing import Optional

from app.agents.graph import MultiAgentKioskWorkflow
from app.agents.guardrail import GuardrailResponses
from app.retrieval.search import AirportSearch
from app.workflow.contracts import InputType, KioskRequest
from app.workflow.evidence import EvidenceAssessor, EvidenceThresholds
from app.workflow.routing import ResponseBook, Router
from app.workflow.validation import CitationOverlapValidator
from tests.fakes import ROOT, FakeGenerator, FakeRetriever

POLICY = ROOT / "config" / "policy.json"
RESPONSES = ROOT / "config" / "responses.json"


class FakeClassifier:
    """Stands in for the gpt-4o-mini intent classifier; records every call."""

    def __init__(self, label: str = "AIRPORT_QUESTION", raises: Optional[Exception] = None) -> None:
        """``label`` is returned for every question; ``raises`` simulates an API failure."""
        self.label = label
        self.raises = raises
        self.calls: list[str] = []

    def classify(self, question: str) -> str:
        """Record the call and return the preset label (or raise)."""
        self.calls.append(question)
        if self.raises:
            raise self.raises
        return self.label


def make_request(text: str, **kw) -> KioskRequest:
    """A typed English request with a fixed session id."""
    return KioskRequest(session_id="agent-test-1", text=text, input_type=kw.pop("input_type", InputType.TYPED), **kw)


def make_agent_workflow(
    retriever: Optional[FakeRetriever] = None,
    generator: Optional[FakeGenerator] = None,
    classifier: Optional[FakeClassifier] = None,
    strong: float = 0.5,
    medium: float = 0.2,
    timeout: float = 5.0,
    max_chars: int = 500,
    use_classifier: bool = True,
) -> tuple[MultiAgentKioskWorkflow, FakeRetriever, FakeGenerator, Optional[FakeClassifier]]:
    """Build the full multi-agent workflow from fakes. Returns (workflow, retriever, generator, classifier)."""
    retriever = retriever or FakeRetriever()
    generator = generator or FakeGenerator()
    if use_classifier:
        classifier = classifier or FakeClassifier()
    else:
        classifier = None
    wf = MultiAgentKioskWorkflow(
        router=Router.from_file(POLICY),
        responses=ResponseBook.from_file(RESPONSES),
        search=AirportSearch(retriever, top_k=5),
        assessor=EvidenceAssessor(EvidenceThresholds(strong, medium, calibrated=True)),
        generator=generator,
        validator=CitationOverlapValidator(),
        node_timeout_seconds=timeout,
        classifier=classifier,
        guardrail_responses=GuardrailResponses.from_file(RESPONSES),
        max_question_chars=max_chars,
    )
    return wf, retriever, generator, classifier
