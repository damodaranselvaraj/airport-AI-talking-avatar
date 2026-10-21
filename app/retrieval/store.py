"""Dense index: a local persistent ChromaDB collection keyed by chunk_id."""
from __future__ import annotations

import chromadb


class ChromaStore:
    def __init__(self, chroma_dir: str, collection: str) -> None:
        self._client = chromadb.PersistentClient(path=chroma_dir)
        # Cosine space so similarity = 1 - distance, higher is better.
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        try:
            self._batch_size = max(1, int(self._client.get_max_batch_size()))
        except Exception:  # older/limited clients: fall back to a conservative slice
            self._batch_size = 5000

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        # A single call is bounded by Chroma's max batch size, so write in slices.
        for start in range(0, len(ids), self._batch_size):
            end = start + self._batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

    def ids(self) -> set[str]:
        """All stored chunk IDs."""
        return set(self._collection.get(include=[])["ids"])

    def get_records(self, ids: list[str]) -> dict[str, dict]:
        """Stored text, embedding and metadata for each ID that exists (used for rollback)."""
        if not ids:
            return {}
        records: dict[str, dict] = {}
        for start in range(0, len(ids), self._batch_size):
            result = self._collection.get(
                ids=ids[start : start + self._batch_size],
                include=["documents", "embeddings", "metadatas"],
            )
            for i, chunk_id in enumerate(result["ids"]):
                records[chunk_id] = {
                    "text": result["documents"][i],
                    "embedding": [float(x) for x in result["embeddings"][i]],
                    "metadata": dict(result["metadatas"][i]),
                }
        return records

    def delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), self._batch_size):
            self._collection.delete(ids=ids[start : start + self._batch_size])

    def query(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, similarity)] ranked descending."""
        n_results = min(top_k, self.count())
        if n_results <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding], n_results=n_results, include=["distances"]
        )
        ranked = [
            (chunk_id, 1.0 - float(distance))
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0])
        ]
        return sorted(ranked, key=lambda item: -item[1])

    def get_texts(self, ids: list[str]) -> dict[str, str]:
        if not ids:
            return {}
        result = self._collection.get(ids=ids, include=["documents"])
        return dict(zip(result["ids"], result["documents"]))

    def metadata(self, chunk_id: str) -> dict:
        """Stored metadata for one chunk; raises KeyError if the ID is not indexed."""
        result = self._collection.get(ids=[chunk_id], include=["metadatas"])
        if not result["ids"]:
            raise KeyError(chunk_id)
        return dict(result["metadatas"][0])

    def collection_dimensions(self) -> int:
        """Dimensionality of stored vectors; 0 when the collection is empty."""
        result = self._collection.get(limit=1, include=["embeddings"])
        embeddings = result["embeddings"]
        if embeddings is None or len(embeddings) == 0:
            return 0
        return len(embeddings[0])

    def count(self) -> int:
        return self._collection.count()
