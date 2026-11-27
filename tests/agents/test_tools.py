"""search_airport @tool: plain-dict output, the rerank flag, and fail-loudly when unconfigured."""
import pytest

from app.agents import tools
from app.agents.tools import configure_search, make_search_tool, search_airport
from app.retrieval.search import AirportSearch
from app.workflow.contracts import ErrorState
from tests.fakes import GATE_C22, PRAYER, FakeRetriever


@pytest.fixture(autouse=True)
def _reset_module_tool():
    """Always leave the module-level tool unconfigured so tests cannot affect each other."""
    configure_search(None)
    yield
    configure_search(None)


def test_unconfigured_tool_raises_runtime_error():
    with pytest.raises(RuntimeError, match="not configured"):
        search_airport.invoke({"query": "where is gate C22"})


def test_module_tool_returns_plain_dicts_after_configure():
    configure_search(AirportSearch(FakeRetriever([GATE_C22, PRAYER]), top_k=5))
    out = search_airport.invoke({"query": "where is gate C22"})
    assert out["rerank_applied"] is True
    assert [p["chunk_id"] for p in out["passages"]] == ["gates-c22", "faq-prayer"]
    assert set(out["passages"][0]) == {"chunk_id", "text", "source", "score"}
    assert out["passages"][0]["source"] == "data/gates.md"
    assert isinstance(out["passages"][0], dict)


def test_rerank_failure_is_reported_in_the_flag():
    err = ErrorState(stage="rerank", message="cohere down")
    tool = make_search_tool(AirportSearch(FakeRetriever([GATE_C22], error=err), top_k=5))
    assert tool.invoke({"query": "gate"})["rerank_applied"] is False


def test_bound_tool_does_not_touch_module_state():
    tool = make_search_tool(AirportSearch(FakeRetriever([PRAYER]), top_k=5))
    assert tool.invoke({"query": "prayer"})["passages"][0]["chunk_id"] == "faq-prayer"
    assert tools._search is None  # the module-level tool is still unconfigured


def test_tool_has_name_and_description_for_llm_binding():
    assert search_airport.name == "search_airport"
    assert "airport" in search_airport.description.lower()


def test_bound_tool_threads_language_to_the_retriever():
    retriever = FakeRetriever([GATE_C22])
    tool = make_search_tool(AirportSearch(retriever, top_k=5))
    tool.invoke({"query": "गेट सी22 कहाँ है?", "language": "hi"})
    assert retriever.languages == ["hi"]


def test_module_tool_threads_language_to_the_retriever():
    retriever = FakeRetriever([GATE_C22])
    configure_search(AirportSearch(retriever, top_k=5))
    search_airport.invoke({"query": "¿dónde está la puerta?", "language": "es"})
    assert retriever.languages == ["es"]


def test_tool_language_defaults_to_en():
    retriever = FakeRetriever([GATE_C22])
    tool = make_search_tool(AirportSearch(retriever, top_k=5))
    tool.invoke({"query": "where is gate C22"})
    assert retriever.languages == ["en"]
