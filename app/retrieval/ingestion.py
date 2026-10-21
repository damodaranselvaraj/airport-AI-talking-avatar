"""Ingestion orchestration: manifest -> parse -> chunk -> embed -> Chroma + BM25.

Both indexes are written from one canonical ``Chunk`` list: chunk text, chunk IDs
and source metadata are authored once by the ``Chunker`` and passed unchanged to
``ChromaStore.upsert`` and ``Bm25Index.build``.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from pydantic import BaseModel

from app.retrieval.chunking import Chunk, Chunker
from app.retrieval.embedding import DenseEmbedder
from app.retrieval.manifest import ManifestValidator
from app.retrieval.parsing import MarkdownParser
from app.retrieval.sparse import Bm25Index
from app.retrieval.store import ChromaStore
from config.settings import Settings

logger = logging.getLogger(__name__)


class ChunkIdCollisionError(ValueError):
    """Two chunks in one run share a chunk ID; the run is rejected before any write."""


class IngestionAbortedError(RuntimeError):
    """A store or persist step failed; both indexes were restored to pre-run state."""


class IngestionRollbackError(RuntimeError):
    """A store failed AND restoring the pre-run Chroma state also failed."""


class IngestionReport(BaseModel):
    planned: int  # chunks produced from eligible documents
    written: int  # chunks upserted into both indexes
    documents: int = 0  # eligible documents parsed
    chroma_count: int = 0  # chunk IDs in Chroma on completion
    bm25_count: int = 0  # chunk IDs in the persisted BM25 index on completion
    parity_ok: bool = True  # Chroma ID set == BM25 ID set
    chroma_only: int = 0  # IDs present only in Chroma
    bm25_only: int = 0  # IDs present only in BM25
    pruned: int = 0  # stale Chroma IDs (not in the current manifest scope) deleted
    parity_error: Optional[str] = None  # set when the post-commit parity check itself failed
    dry_run: bool = False  # plan only: written == 0 and no index was touched
    excluded_sources: list[str] = []  # dry-run: "<path>: <reason>" per unreadable source


def assert_unique_chunk_ids(chunks: list[Chunk]) -> None:
    """Reject the run if any chunk ID occurs more than once."""
    counts = Counter(c.chunk_id for c in chunks)
    duplicates = sorted(cid for cid, n in counts.items() if n > 1)
    if duplicates:
        raise ChunkIdCollisionError(
            f"{len(duplicates)} chunk ID collision(s); run rejected, indexes unchanged: "
            + ", ".join(duplicates[:5])
        )


def _plan_chunks(
    settings: Settings, excluded_sources: Optional[list[str]] = None
) -> tuple[list[Chunk], int]:
    """Manifest -> parser -> chunker. Returns (canonical chunks, document count).

    By default an unreadable/unparsable source raises. When ``excluded_sources`` is
    given (dry-run planning), such a source is recorded there as ``"<path>: <reason>"``,
    excluded from the plan, and planning continues.
    """
    validator = ManifestValidator(settings.manifest_path, settings.document_root)
    parser = MarkdownParser()
    chunker = Chunker(
        settings.chunk_size, settings.chunk_overlap, settings.primary_header_level
    )
    chunks: list[Chunk] = []
    documents = 0
    for path in validator.eligible_files():  # do_not_ingest excluded here
        assert not validator.is_excluded(path)  # invariant: never ingest excluded
        if excluded_sources is None:
            doc_chunks = chunker.split(parser.parse(path))
        else:
            try:
                doc_chunks = chunker.split(parser.parse(path))
            except Exception as exc:
                excluded_sources.append(f"{path}: {exc}")
                logger.error("Dry-run: excluded unreadable source %s: %s", path, exc)
                continue
        documents += 1
        chunks.extend(doc_chunks)
    return chunks, documents


def _restore_chroma(
    store: ChromaStore, before_ids: set[str], previous: dict[str, dict], ids: list[str]
) -> None:
    """Return Chroma to its pre-run state for the IDs this run touched."""
    new_ids = [cid for cid in ids if cid not in before_ids]
    store.delete(new_ids)
    if previous:
        old_ids = list(previous)
        store.upsert(
            ids=old_ids,
            texts=[previous[i]["text"] for i in old_ids],
            embeddings=[previous[i]["embedding"] for i in old_ids],
            metadatas=[previous[i]["metadata"] for i in old_ids],
        )


def run_ingestion(
    settings: Settings, embedder: Optional[DenseEmbedder] = None
) -> IngestionReport:
    """Ingest eligible documents into Chroma and BM25 as one consistent unit.

    ``embedder`` may be injected (tests); by default one is built from settings.

    A full run also prunes: Chroma IDs not produced by the current manifest scope are
    deleted and BM25 is rebuilt from the same chunk set, so both indexes hold exactly
    the current canonical chunks. Any failure restores both indexes to pre-run state.
    """
    if settings.ingest_dry_run:
        # Plan only: no embedder, no ChromaStore (even opening one creates files) and no
        # BM25 access, so both indexes stay byte-for-byte unmodified.
        excluded_sources: list[str] = []
        chunks, documents = _plan_chunks(settings, excluded_sources)
        assert_unique_chunk_ids(chunks)
        logger.info(
            "Dry-run plan: %d document(s), %d chunk(s), %d excluded source(s); "
            "nothing written",
            documents,
            len(chunks),
            len(excluded_sources),
        )
        return IngestionReport(
            planned=len(chunks),
            written=0,
            documents=documents,
            dry_run=True,
            excluded_sources=excluded_sources,
        )

    chunks, documents = _plan_chunks(settings)
    assert_unique_chunk_ids(chunks)  # collision -> reject before touching any store

    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    if embedder is None:
        embedder = DenseEmbedder(
            settings.openai_embedding_model,
            settings.openai_embedding_dims,
            api_key=settings.openai_api_key,
        )
    vectors = embedder.embed_documents(texts)  # failure here: nothing written yet
    if len(vectors) != len(chunks):
        raise IngestionAbortedError(
            f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
        )

    store = ChromaStore(settings.chroma_dir, settings.chroma_collection)
    before_ids = store.ids()
    # Stale = in Chroma but no longer produced by the current manifest scope. They are
    # snapshotted with the overwritten IDs so a failure can put them back.
    stale_ids = sorted(before_ids - set(ids))
    previous = store.get_records(
        [i for i in ids if i in before_ids] + stale_ids
    )

    # Chroma first; BM25 is persisted (temp file + atomic replace) only after
    # Chroma succeeded, so a BM25 failure never leaves a half-written index file.
    try:
        store.upsert(ids=ids, texts=texts, embeddings=vectors, metadatas=metadatas)
        store.delete(stale_ids)  # prune: Chroma ID set becomes exactly the current chunk set
        bm25 = Bm25Index()
        bm25.build(ids, texts)
        bm25.persist(settings.bm25_index_path)
    except Exception as exc:
        try:
            _restore_chroma(store, before_ids, previous, ids)
        except Exception as rollback_exc:
            raise IngestionRollbackError(
                f"Ingestion failed ({exc!r}) and Chroma rollback failed ({rollback_exc!r}); "
                "re-run ingestion to rebuild both indexes"
            ) from exc
        raise IngestionAbortedError(
            f"Ingestion aborted; both indexes left at pre-run state: {exc!r}"
        ) from exc

    # Both indexes are committed here. A failure of the verification reads below must not
    # surface as a raw exception: report parity as failed with the reason instead.
    try:
        chroma_ids = store.ids()
        bm25_ids = Bm25Index.load(settings.bm25_index_path).chunk_ids
    except Exception as exc:
        return IngestionReport(
            planned=len(chunks),
            written=len(chunks),
            documents=documents,
            pruned=len(stale_ids),
            parity_ok=False,
            parity_error=f"indexes committed but parity check failed: {exc!r}",
        )
    return IngestionReport(
        planned=len(chunks),
        written=len(chunks),
        documents=documents,
        chroma_count=len(chroma_ids),
        bm25_count=len(bm25_ids),
        parity_ok=chroma_ids == bm25_ids,
        chroma_only=len(chroma_ids - bm25_ids),
        bm25_only=len(bm25_ids - chroma_ids),
        pruned=len(stale_ids),
    )
