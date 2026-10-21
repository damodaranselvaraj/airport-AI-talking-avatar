"""Staged ingestion runner with a console message after each stage.

Usage (from the repo root):
    python scripts/run_ingestion_pipeline.py             # full run
    python scripts/run_ingestion_pipeline.py --dry-run   # stages 1-3 only, nothing written

Stages:
    1. Manifest validation / document discovery
    2. Parsing + chunking
    3. Chunk-ID uniqueness check
    4. Embedding (OpenAI)
    5. ChromaDB insertion (+ pruning of stale IDs)
    6. BM25 build + persist
    7. Chroma / BM25 parity verification

It uses the same components as ``app.retrieval.ingestion.run_ingestion`` (including
rollback of both indexes on failure), only split so each stage can report progress.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a plain script: put the repo root on sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.chunking import Chunker  # noqa: E402
from app.retrieval.embedding import DenseEmbedder  # noqa: E402
from app.retrieval.ingestion import (  # noqa: E402
    IngestionAbortedError,
    IngestionRollbackError,
    _restore_chroma,
    assert_unique_chunk_ids,
)
from app.retrieval.manifest import ManifestValidator  # noqa: E402
from app.retrieval.parsing import MarkdownParser  # noqa: E402
from app.retrieval.sparse import Bm25Index  # noqa: E402
from app.retrieval.store import ChromaStore  # noqa: E402
from config.settings import get_settings  # noqa: E402

TOTAL_STAGES = 7


def stage_done(number: int, message: str, started: float) -> None:
    print(f"[OK] Stage {number}/{TOTAL_STAGES} - {message} ({time.perf_counter() - started:.2f}s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ingestion into ChromaDB and BM25.")
    parser.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
    args = parser.parse_args(argv)

    settings = get_settings()
    # The settings default ("manifest.json") is not where the manifest lives and
    # .env may not set MANIFEST_PATH; fall back to the documented data/manifest.json.
    if not Path(settings.manifest_path).exists():
        fallback = REPO_ROOT / "data" / "manifest.json"
        if fallback.exists():
            settings = settings.model_copy(update={"manifest_path": str(fallback)})
    dry_run = args.dry_run or settings.ingest_dry_run
    print(f"Starting ingestion pipeline ({'DRY RUN' if dry_run else 'full run'})")

    try:
        # Stage 1: manifest validation and document discovery
        t = time.perf_counter()
        validator = ManifestValidator(settings.manifest_path, settings.document_root)
        files = list(validator.eligible_files())  # do_not_ingest excluded here
        stage_done(1, f"Manifest validated, {len(files)} eligible document(s) found", t)

        # Stage 2: parsing + chunking
        t = time.perf_counter()
        md_parser = MarkdownParser()
        chunker = Chunker(
            settings.chunk_size, settings.chunk_overlap, settings.primary_header_level
        )
        chunks = []
        for path in files:
            assert not validator.is_excluded(path)  # invariant: never ingest excluded
            chunks.extend(chunker.split(md_parser.parse(path)))
        stage_done(2, f"Parsed {len(files)} document(s) into {len(chunks)} chunk(s)", t)

        # Stage 3: chunk-ID uniqueness
        t = time.perf_counter()
        assert_unique_chunk_ids(chunks)
        stage_done(3, "Chunk IDs are unique", t)

        if dry_run:
            print("Dry run: skipping embedding and index writes. Nothing was written.")
            return 0

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Stage 4: embedding
        t = time.perf_counter()
        embedder = DenseEmbedder(
            settings.openai_embedding_model,
            settings.openai_embedding_dims,
            api_key=settings.openai_api_key,
        )
        vectors = embedder.embed_documents(texts)
        if len(vectors) != len(chunks):
            raise IngestionAbortedError(
                f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        stage_done(4, f"Embedded {len(vectors)} chunk(s) with {settings.openai_embedding_model}", t)

        # Snapshot Chroma so a failure in stage 5 or 6 can be rolled back.
        store = ChromaStore(settings.chroma_dir, settings.chroma_collection)
        before_ids = store.ids()
        stale_ids = sorted(before_ids - set(ids))
        previous = store.get_records([i for i in ids if i in before_ids] + stale_ids)

        try:
            # Stage 5: ChromaDB insertion
            t = time.perf_counter()
            store.upsert(ids=ids, texts=texts, embeddings=vectors, metadatas=metadatas)
            store.delete(stale_ids)
            stage_done(
                5,
                f"Inserted {len(ids)} chunk(s) into ChromaDB collection "
                f"'{settings.chroma_collection}' ({len(stale_ids)} stale pruned)",
                t,
            )

            # Stage 6: BM25 build + persist
            t = time.perf_counter()
            bm25 = Bm25Index()
            bm25.build(ids, texts)
            bm25.persist(settings.bm25_index_path)
            stage_done(6, f"BM25 index built and saved to {settings.bm25_index_path}", t)
        except Exception as exc:
            try:
                _restore_chroma(store, before_ids, previous, ids)
            except Exception as rollback_exc:
                raise IngestionRollbackError(
                    f"Ingestion failed ({exc!r}) and Chroma rollback failed "
                    f"({rollback_exc!r}); re-run ingestion to rebuild both indexes"
                ) from exc
            raise IngestionAbortedError(
                f"Ingestion aborted; both indexes left at pre-run state: {exc!r}"
            ) from exc

        # Stage 7: parity verification
        t = time.perf_counter()
        chroma_ids = store.ids()
        bm25_ids = Bm25Index.load(settings.bm25_index_path).chunk_ids
        if chroma_ids != bm25_ids:
            print(
                f"[FAIL] Stage 7/{TOTAL_STAGES} - Parity mismatch: "
                f"{len(chroma_ids - bm25_ids)} only in Chroma, "
                f"{len(bm25_ids - chroma_ids)} only in BM25",
                file=sys.stderr,
            )
            return 1
        stage_done(7, f"Parity verified: ChromaDB and BM25 both hold {len(chroma_ids)} chunk ID(s)", t)

    except Exception as exc:
        print(f"[FAIL] Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print("Ingestion pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
