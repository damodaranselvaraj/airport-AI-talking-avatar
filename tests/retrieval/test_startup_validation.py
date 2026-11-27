"""FEAT-005: per-language startup validation.

A tmp Chroma holding two languages + two per-language BM25 files passes when
every partition is non-empty and per-language BM25<->Chroma parity holds; it
fails loudly when a partition is empty, a BM25 file is missing, or parity breaks.
The Cohere probe is stubbed out; only the Chroma/BM25 checks are exercised.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.retrieval.errors import StartupValidationError
from app.retrieval.sparse import Bm25Index, bm25_path_for
from app.retrieval.startup import validate_startup
from app.retrieval.store import ChromaStore
from config.settings import Settings

SECRETS = {
    "openai_api_key": "sk-test-OPENAI-secret-0123456789",
    "typesafe_api_key": "tsk-test-TYPESAFE-secret-0123456789",
    "cohere_api_key": "co-test-COHERE-secret-0123456789",
}

_DIMS = 8


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i] / 255.0 for i in range(_DIMS)]


class _NoopReranker:
    def validate(self) -> None:  # the startup Cohere probe is not what we test here
        return None


def _build(tmp_path: Path, *, chroma_langs, bm25_langs, break_parity=False) -> Settings:
    """Create a tmp Chroma with ``chroma_langs`` chunks and BM25 files for ``bm25_langs``."""
    chroma_dir = tmp_path / "chroma"
    store = ChromaStore(str(chroma_dir), "knowledge_base")
    ids = {"en": "en-1", "hi": "hi-1"}
    texts = {"en": "english gate text", "hi": "हिन्दी गेट पाठ"}
    for lang in chroma_langs:
        store.upsert(
            ids=[ids[lang]],
            texts=[texts[lang]],
            embeddings=[_vector(texts[lang])],
            metadatas=[{"lang": lang, "source": f"{lang}.md"}],
        )

    base = str(tmp_path / ".bm25" / "knowledge_base.pkl")
    for lang in bm25_langs:
        index = Bm25Index()
        chunk_id = "mismatch-id" if break_parity else ids[lang]
        index.build([chunk_id], [texts[lang]])
        index.persist(bm25_path_for(base, lang))

    return Settings(
        _env_file=None,
        chroma_dir=str(chroma_dir),
        chroma_collection="knowledge_base",
        bm25_index_path=base,
        supported_languages=["en", "hi"],
        openai_embedding_dims=_DIMS,
        **SECRETS,
    )


def test_passes_when_partitions_nonempty_and_parity_holds(tmp_path):
    settings = _build(tmp_path, chroma_langs=["en", "hi"], bm25_langs=["en", "hi"])
    validate_startup(settings, reranker=_NoopReranker())  # no raise


def test_fails_when_a_partition_is_empty(tmp_path):
    # Chroma has only en; the hi partition is empty -> fatal.
    settings = _build(tmp_path, chroma_langs=["en"], bm25_langs=["en", "hi"])
    with pytest.raises(StartupValidationError) as exc:
        validate_startup(settings, reranker=_NoopReranker())
    assert "hi" in str(exc.value)


def test_fails_when_a_bm25_file_is_missing(tmp_path):
    settings = _build(tmp_path, chroma_langs=["en", "hi"], bm25_langs=["en"])
    with pytest.raises(StartupValidationError) as exc:
        validate_startup(settings, reranker=_NoopReranker())
    assert "hi" in str(exc.value)


def test_fails_when_per_language_parity_breaks(tmp_path):
    settings = _build(tmp_path, chroma_langs=["en", "hi"], bm25_langs=["en", "hi"], break_parity=True)
    with pytest.raises(StartupValidationError) as exc:
        validate_startup(settings, reranker=_NoopReranker())
    assert "differs from Chroma" in str(exc.value)
