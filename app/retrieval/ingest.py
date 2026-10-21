"""Ingestion CLI: ``python -m app.retrieval.ingest [--dry-run]``.

Importing this module needs no API keys and makes no network calls; ``Settings`` is
only loaded when ``main`` runs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from app.retrieval.embedding import DenseEmbedder
from app.retrieval.ingestion import IngestionReport, run_ingestion
from config.settings import Settings, get_settings


def format_report(report: IngestionReport) -> str:
    lines = [
        "Ingestion dry-run (nothing written)" if report.dry_run else "Ingestion complete",
        f"  documents: {report.documents}",
        f"  planned chunks: {report.planned}",
        f"  written chunks: {report.written}",
    ]
    if not report.dry_run:
        lines += [
            f"  chroma ids: {report.chroma_count}",
            f"  bm25 ids: {report.bm25_count}",
            f"  pruned: {report.pruned}",
            f"  parity_ok: {report.parity_ok}",
        ]
        if report.chroma_only or report.bm25_only:
            lines.append(
                f"  parity drift: {report.chroma_only} only in Chroma, "
                f"{report.bm25_only} only in BM25"
            )
        if report.parity_error:
            lines.append(f"  parity_error: {report.parity_error}")
    for entry in report.excluded_sources:
        lines.append(f"  ERROR excluded source: {entry}")
    return "\n".join(lines)


def main(
    argv: Optional[list[str]] = None,
    settings: Optional[Settings] = None,
    embedder: Optional[DenseEmbedder] = None,
) -> int:
    """Run ingestion and print the report. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m app.retrieval.ingest",
        description="Ingest manifest documents into ChromaDB and the BM25 index.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan only (overrides INGEST_DRY_RUN=false): report counts, write nothing",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        settings = settings or get_settings()
        if args.dry_run:
            settings = settings.model_copy(update={"ingest_dry_run": True})
        report = run_ingestion(settings, embedder=embedder)
    except Exception as exc:  # surface a clear message and a non-zero exit code
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(format_report(report))
    return 0 if report.parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
