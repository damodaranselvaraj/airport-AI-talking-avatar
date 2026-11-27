"""FEAT-005: ask.py --lang / auto threads the language into retrieve_trace."""
from __future__ import annotations

import ask


class FakeTrace:
    def __init__(self) -> None:
        self.dense = []
        self.sparse = []
        self.fused = []
        self.reranked = []
        self.sources = {}
        self.texts = {}
        self.errors = {}
        self.rerank_fallback = False


class SpyRetriever:
    """Records the language passed into retrieve_trace."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def retrieve_trace(self, query, top_k, lang):
        self.calls.append((query, top_k, lang))
        return FakeTrace()


def test_explicit_lang_is_threaded_through(capsys):
    retriever = SpyRetriever()
    ask._ask(retriever, "where is gate C22", top_k=5, stage_k=5, full=False,
             llm_model=None, settings=None, lang="hi")
    assert retriever.calls[0][2] == "hi"


def test_auto_lang_detects_from_question(monkeypatch, capsys):
    retriever = SpyRetriever()

    class FakeDet:
        def detect(self, text):
            from app.language.detector import DetectionResult

            return DetectionResult("es", True)

    # Patch the detector constructor used inside _resolve_lang's lazy import.
    import app.language.detector as detector_mod

    monkeypatch.setattr(detector_mod, "LinguaLanguageDetector", lambda *a, **k: FakeDet())
    ask._ask(retriever, "donde esta la puerta", top_k=5, stage_k=5, full=False,
             llm_model=None, settings=None, lang="auto")
    assert retriever.calls[0][2] == "es"


def test_auto_lang_falls_back_to_default_when_unsupported(monkeypatch, capsys):
    retriever = SpyRetriever()

    class FakeDet:
        def detect(self, text):
            from app.language.detector import DetectionResult

            return DetectionResult("en", False)

    import app.language.detector as detector_mod

    monkeypatch.setattr(detector_mod, "LinguaLanguageDetector", lambda *a, **k: FakeDet())
    ask._ask(retriever, "???", top_k=5, stage_k=5, full=False,
             llm_model=None, settings=None, lang="auto")
    assert retriever.calls[0][2] == "en"
