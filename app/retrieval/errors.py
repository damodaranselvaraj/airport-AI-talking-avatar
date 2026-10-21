"""Shared retrieval errors, kept dependency-free so any module can import them."""
from __future__ import annotations


class StartupValidationError(RuntimeError):
    """A critical dependency failed validation; the service must not start.

    Raised by ``CohereReranker.validate`` and by ``validate_startup`` (re-exported from
    ``app.retrieval.startup``). There is no local fallback.
    """
