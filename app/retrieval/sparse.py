"""Sparse index: BM25 over the same canonical chunk records Chroma receives."""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """The single tokenizer used by both ``Bm25Index.build`` and ``Bm25Index.query``."""
    return _TOKEN_RE.findall(text.lower())


class Bm25Index:
    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have the same length")
        self._chunk_ids = list(chunk_ids)
        self._tokenized = [tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    @property
    def chunk_ids(self) -> frozenset[str]:
        """The set of indexed chunk IDs."""
        return frozenset(self._chunk_ids)

    def persist(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        try:
            with tmp.open("wb") as fh:
                pickle.dump(
                    {"chunk_ids": self._chunk_ids, "tokenized": self._tokenized}, fh
                )
            os.replace(tmp, target)  # atomic: a failed write never leaves a partial index
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str) -> "Bm25Index":
        with Path(path).open("rb") as fh:
            data = pickle.load(fh)  # locally produced by persist(); not untrusted input
        index = cls()
        index._chunk_ids = list(data["chunk_ids"])
        index._tokenized = [list(tokens) for tokens in data["tokenized"]]
        index._bm25 = BM25Okapi(index._tokenized) if index._tokenized else None
        return index

    def query(self, text: str, top_k: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, bm25_score)] ranked descending."""
        tokens = tokenize(text)
        if self._bm25 is None or not tokens or top_k <= 0:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        return [(self._chunk_ids[i], float(scores[i])) for i in order]
