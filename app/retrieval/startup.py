"""Fail-loudly startup validation: the service must not serve if any check fails."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from app.retrieval.errors import StartupValidationError
from app.retrieval.retrievers import CohereReranker
from app.retrieval.sparse import Bm25Index
from app.retrieval.store import ChromaStore

if TYPE_CHECKING:
    from config.settings import Settings

__all__ = ["StartupValidationError", "validate_startup"]

COHERE_PROBE_TIMEOUT_SECONDS = 10.0

CHECK_COHERE = "cohere_probe"
CHECK_CHROMA = "chroma_collection"
CHECK_DIMENSIONS = "embedding_dimensions"
CHECK_BM25 = "bm25_index"


def _fail(check: str, detail: str) -> StartupValidationError:
    return StartupValidationError(f"Startup check '{check}' failed: {detail}")


def _check_cohere(settings: "Settings", reranker: Optional[CohereReranker]) -> None:
    reranker = reranker or CohereReranker(
        settings.cohere_api_key,
        settings.reranker_model,
        timeout=COHERE_PROBE_TIMEOUT_SECONDS,
    )
    # The SDK timeout covers the real client; the hard cap also covers any client that hangs.
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(reranker.validate)
    try:
        future.result(timeout=COHERE_PROBE_TIMEOUT_SECONDS)
    except TimeoutError as exc:  # also what a mocked client timing out raises
        raise _fail(CHECK_COHERE, f"probe timed out after {COHERE_PROBE_TIMEOUT_SECONDS:g}s") from exc
    except Exception as exc:
        raise _fail(CHECK_COHERE, str(exc) or repr(exc)) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def validate_startup(settings: "Settings", reranker: Optional[CohereReranker] = None) -> None:
    """Run every startup check in order; raise ``StartupValidationError`` on the first failure.

    Checks: Cohere probe (10s timeout), Chroma collection present and non-empty,
    stored dimensions == ``settings.openai_embedding_dims``, BM25 ID set == Chroma ID set.
    """
    _check_cohere(settings, reranker)

    # Check presence without creating anything (ChromaStore would create a missing collection).
    chroma_dir = Path(settings.chroma_dir)
    if not chroma_dir.is_dir():
        raise _fail(CHECK_CHROMA, f"Chroma directory '{settings.chroma_dir}' does not exist")
    try:
        import chromadb

        names = {
            getattr(c, "name", c)
            for c in chromadb.PersistentClient(path=str(chroma_dir)).list_collections()
        }
        if settings.chroma_collection not in names:
            raise _fail(CHECK_CHROMA, f"collection '{settings.chroma_collection}' does not exist")
        store = ChromaStore(settings.chroma_dir, settings.chroma_collection)
        if store.count() <= 0:
            raise _fail(CHECK_CHROMA, f"collection '{settings.chroma_collection}' is empty")
        dims = store.collection_dimensions()
        chroma_ids = store.ids()
    except StartupValidationError:
        raise
    except Exception as exc:
        raise _fail(CHECK_CHROMA, f"could not read collection: {exc!r}") from exc

    expected_dims = settings.openai_embedding_dims
    if dims != expected_dims:
        raise _fail(CHECK_DIMENSIONS, f"stored vectors have {dims} dims; expected {expected_dims}")

    if not Path(settings.bm25_index_path).is_file():
        raise _fail(CHECK_BM25, f"index file '{settings.bm25_index_path}' does not exist")
    try:
        bm25_ids = set(Bm25Index.load(settings.bm25_index_path).chunk_ids)
    except Exception as exc:
        raise _fail(CHECK_BM25, f"could not load index: {exc!r}") from exc
    if bm25_ids != chroma_ids:
        raise _fail(
            CHECK_BM25,
            f"ID set differs from Chroma (bm25_only={len(bm25_ids - chroma_ids)}, "
            f"chroma_only={len(chroma_ids - bm25_ids)})",
        )
