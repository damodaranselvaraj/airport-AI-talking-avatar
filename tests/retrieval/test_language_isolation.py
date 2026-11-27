"""FEAT-002: language-isolated retrieval.

Covers the per-language BM25 path mapping and registry, the ChromaStore
``where`` filter, an end-to-end hi/en isolation round-trip, and per-language
ingestion parity. All fixtures are tiny and built in ``tmp_path``; no network,
no real embeddings, no models.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.retrieval.chunking import Chunker
from app.retrieval.ingestion import run_ingestion
from app.retrieval.manifest import ManifestError, ManifestValidator
from app.retrieval.parsing import ParsedDocument
from app.retrieval.retrievers import HybridRetriever
from app.retrieval.sparse import Bm25Index, Bm25Registry, bm25_path_for
from app.retrieval.store import ChromaStore
from config.settings import Settings

SECRETS = {
    "openai_api_key": "sk-test-OPENAI-secret-0123456789",
    "typesafe_api_key": "tsk-test-TYPESAFE-secret-0123456789",
    "cohere_api_key": "co-test-COHERE-secret-0123456789",
}

_DIMS = 8


def _vector(text: str) -> list[float]:
    """Deterministic, text-dependent unit-ish vector (no model needed)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i] / 255.0 for i in range(_DIMS)]


class FakeEmbedder:
    """Stands in for DenseEmbedder: deterministic vectors, no OpenAI, no config check."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _vector(text)


class PassthroughReranker:
    """Keeps fused order and scores so the isolation assertions are deterministic."""

    def rerank(self, query, candidates, top_k):
        return [(cid, 1.0) for cid, _text in candidates][:top_k]


# --------------------------------------------------------------------------- #
# bm25_path_for
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "base,lang,expected",
    [
        (".bm25/knowledge_base.pkl", "hi", ".bm25/knowledge_base.hi.pkl"),
        (".bm25/knowledge_base.pkl", "en", ".bm25/knowledge_base.en.pkl"),
        ("index.pkl", "es", "index.es.pkl"),
    ],
)
def test_bm25_path_for(base, lang, expected):
    assert Path(bm25_path_for(base, lang)) == Path(expected)


# --------------------------------------------------------------------------- #
# Bm25Registry round-trip
# --------------------------------------------------------------------------- #
def test_bm25_registry_load_get_roundtrip(tmp_path):
    base = str(tmp_path / "kb.pkl")
    en = Bm25Index()
    en.build(["en-1", "en-2"], ["gate information", "security line"])
    en.persist(bm25_path_for(base, "en"))
    hi = Bm25Index()
    hi.build(["hi-1"], ["गेट सूचना"])
    hi.persist(bm25_path_for(base, "hi"))

    registry = Bm25Registry.load(base, ["en", "hi"])

    assert registry.get("en").chunk_ids == frozenset({"en-1", "en-2"})
    assert registry.get("hi").chunk_ids == frozenset({"hi-1"})
    assert registry.chunk_ids("en") == frozenset({"en-1", "en-2"})


def test_bm25_registry_missing_file_raises(tmp_path):
    base = str(tmp_path / "kb.pkl")
    Bm25Index().persist(bm25_path_for(base, "en"))  # only en exists
    with pytest.raises(FileNotFoundError):
        Bm25Registry.load(base, ["en", "hi"])


# --------------------------------------------------------------------------- #
# ChromaStore where-filter
# --------------------------------------------------------------------------- #
def test_chroma_query_where_filters_by_lang(tmp_path):
    store = ChromaStore(str(tmp_path / "chroma"), "knowledge_base")
    store.upsert(
        ids=["en-1", "hi-1"],
        texts=["english text", "हिन्दी पाठ"],
        embeddings=[_vector("english text"), _vector("हिन्दी पाठ")],
        metadatas=[{"lang": "en", "source": "en.md"}, {"lang": "hi", "source": "hi.md"}],
    )

    # Unfiltered: both rows visible, behaves as before.
    assert store.ids() == {"en-1", "hi-1"}
    assert store.count() == 2

    # Filtered reads return only the matching language.
    assert store.ids(where={"lang": "hi"}) == {"hi-1"}
    assert store.count(where={"lang": "hi"}) == 1

    ranked = store.query(_vector("anything"), top_k=10, where={"lang": "hi"})
    assert [cid for cid, _ in ranked] == ["hi-1"]


# --------------------------------------------------------------------------- #
# Manifest entries() + chunk lang invariant
# --------------------------------------------------------------------------- #
def _write_two_lang_tree(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    (data / "en").mkdir(parents=True)
    (data / "hi").mkdir(parents=True)
    (data / "en" / "gates.md").write_text(
        "---\nlang: en\n---\n# Gates\nGate A1 near the English desk.\n", encoding="utf-8"
    )
    (data / "hi" / "gates-hi.md").write_text(
        "---\nlang: hi\n---\n# गेट\nगेट बी2 हिन्दी केंद्र के पास है।\n", encoding="utf-8"
    )
    return data


def test_manifest_entries_returns_path_and_lang(tmp_path):
    data = _write_two_lang_tree(tmp_path)
    manifest = data / "manifest.json"
    manifest.write_text(
        '{"documents": ['
        '{"id": "gates", "lang": "en", "source": "data/en/gates.md"},'
        '{"id": "gates-hi", "lang": "hi", "source": "data/hi/gates-hi.md"}'
        '], "do_not_ingest": ["manifest.json"]}',
        encoding="utf-8",
    )
    validator = ManifestValidator(str(manifest), str(data))
    entries = validator.entries()
    langs = {lang for _p, lang in entries}
    assert langs == {"en", "hi"}
    assert all(p.is_file() for p, _lang in entries)


def test_manifest_entries_rejects_unsupported_lang(tmp_path):
    data = _write_two_lang_tree(tmp_path)
    manifest = data / "manifest.json"
    manifest.write_text(
        '{"documents": ['
        '{"id": "gates", "lang": "xx", "source": "data/en/gates.md"}'
        '], "do_not_ingest": ["manifest.json"]}',
        encoding="utf-8",
    )
    validator = ManifestValidator(str(manifest), str(data))
    with pytest.raises(ManifestError):
        validator.entries()


def test_chunker_rejects_missing_or_unsupported_lang():
    chunker = Chunker()
    bad = ParsedDocument(
        source="bad.md", frontmatter={"lang": "xx"}, body="# H\nsome body text.\n"
    )
    with pytest.raises(ValueError):
        chunker.split(bad)
    missing = ParsedDocument(source="nolang.md", frontmatter={}, body="# H\nbody.\n")
    with pytest.raises(ValueError):
        chunker.split(missing)


def test_chunker_stamps_supported_lang():
    chunker = Chunker()
    doc = ParsedDocument(
        source="gates-hi.md", frontmatter={"lang": "hi"}, body="# गेट\nगेट बी2।\n"
    )
    chunks = chunker.split(doc)
    assert chunks
    assert all(c.metadata["lang"] == "hi" for c in chunks)


# --------------------------------------------------------------------------- #
# Ingestion fixture (tiny 2-language tree) + end-to-end isolation
# --------------------------------------------------------------------------- #
def _write_fixture(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    (data / "en").mkdir(parents=True)
    (data / "hi").mkdir(parents=True)
    (data / "en" / "gates.md").write_text(
        "---\nlang: en\n---\n# Gates\nGate A1 is near the English help desk.\n",
        encoding="utf-8",
    )
    (data / "hi" / "gates-hi.md").write_text(
        "---\nlang: hi\n---\n# गेट\nगेट बी2 हिन्दी सहायता केंद्र के पास है।\n",
        encoding="utf-8",
    )
    manifest = data / "manifest.json"
    manifest.write_text(
        """{
  "airport": "Test", "iata": "TST", "closed_world": true,
  "documents": [
    {"id": "gates", "title": "Gates", "lang": "en", "source": "data/en/gates.md"},
    {"id": "gates-hi", "title": "गेट", "lang": "hi", "source": "data/hi/gates-hi.md"}
  ],
  "do_not_ingest": ["manifest.json"]
}
""",
        encoding="utf-8",
    )
    return Settings(
        _env_file=None,
        manifest_path=str(manifest),
        document_root=str(data),
        chroma_dir=str(tmp_path / "chroma"),
        chroma_collection="knowledge_base",
        bm25_index_path=str(tmp_path / ".bm25" / "knowledge_base.pkl"),
        supported_languages=["en", "hi"],
        openai_embedding_dims=_DIMS,
        **SECRETS,
    )


def test_ingestion_writes_per_language_bm25_and_parity(tmp_path):
    settings = _write_fixture(tmp_path)
    report = run_ingestion(settings, embedder=FakeEmbedder())

    # Two per-language BM25 files written (en + hi).
    assert Path(bm25_path_for(settings.bm25_index_path, "en")).is_file()
    assert Path(bm25_path_for(settings.bm25_index_path, "hi")).is_file()

    assert report.parity_by_lang == {"en": True, "hi": True}
    assert report.parity_ok is True
    assert report.written == 2


def test_hi_and_en_queries_are_language_isolated(tmp_path):
    settings = _write_fixture(tmp_path)
    run_ingestion(settings, embedder=FakeEmbedder())

    chroma = ChromaStore(settings.chroma_dir, settings.chroma_collection)
    registry = Bm25Registry.load(settings.bm25_index_path, settings.supported_languages)
    retriever = HybridRetriever(
        settings, FakeEmbedder(), chroma, registry, PassthroughReranker()
    )

    hi_passages, _ = retriever.retrieve_with_status("गेट कहाँ है", top_k=5, lang="hi")
    assert hi_passages, "expected hi evidence"
    assert all(p.source.endswith("gates-hi.md") for p in hi_passages)

    en_passages, _ = retriever.retrieve_with_status("where is the gate", top_k=5, lang="en")
    assert en_passages, "expected en evidence"
    assert all(p.source.endswith("gates.md") and "hi" not in p.source for p in en_passages)
