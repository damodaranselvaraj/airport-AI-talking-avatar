"""Public API of the hybrid RAG ingestion and retrieval package.

Importing this package builds no clients and reads no settings; all
construction happens when ``run_ingestion``, ``HybridRetriever.from_settings``
or ``validate_startup`` is called.
"""
from app.retrieval.ingestion import run_ingestion
from app.retrieval.retrievers import HybridRetriever
from app.retrieval.startup import validate_startup

__all__ = ["run_ingestion", "HybridRetriever", "validate_startup"]
