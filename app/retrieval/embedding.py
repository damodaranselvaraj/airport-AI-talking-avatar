"""Shared dense encoder used by both ingestion and query time.

One class, one configuration: ingestion calls ``embed_documents`` and retrieval
calls ``embed_query`` so both sides use the same model and dimensions.
"""
from __future__ import annotations

from typing import Any, Optional

from config.settings import get_settings

_BATCH_SIZE = 100


class EmbeddingConfigurationMismatchError(ValueError):
    """Query embedder model/dimensions differ from the ingestion configuration."""


class EmbeddingDimensionMismatchError(ValueError):
    """A produced vector does not have the expected dimensionality."""


class DenseEmbedder:
    def __init__(
        self,
        model: str,
        dimensions: int,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._client = client
        self._api_key = api_key

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            api_key = self._api_key
            if api_key is None:
                api_key = get_settings().openai_api_key
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            response = client.embeddings.create(
                model=self._model, input=batch, dimensions=self._dimensions
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in ordered])
        expected_dims = get_settings().openai_embedding_dims
        for vector in vectors:
            if len(vector) != expected_dims:
                raise EmbeddingDimensionMismatchError(
                    f"Embedding has {len(vector)} dimensions; "
                    f"expected {expected_dims}"
                )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        # Reject before any API call so no vector is produced on a mismatch.
        settings = get_settings()
        ingestion_model = settings.openai_embedding_model
        ingestion_dims = settings.openai_embedding_dims
        if self._model != ingestion_model or self._dimensions != ingestion_dims:
            raise EmbeddingConfigurationMismatchError(
                f"Query embedder config ({self._model}, {self._dimensions}) differs "
                f"from ingestion config ({ingestion_model}, {ingestion_dims})"
            )
        return self._embed([text])[0]
