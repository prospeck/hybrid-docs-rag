from __future__ import annotations

import json
import threading
from pathlib import Path

from rank_bm25 import BM25Okapi

from hybrid_rag.models import Chunk, SearchHit
from hybrid_rag.text import tokenize


class BM25Index:
    """Thread-safe BM25 index persisted as portable chunk JSON, never an unsafe pickle."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._index: BM25Okapi | None = None
        self.load()

    def rebuild(self, chunks: list[Chunk]) -> None:
        with self._lock:
            self._chunks = list(chunks)
            corpus = [tokenize(chunk.text) for chunk in chunks]
            self._index = BM25Okapi(corpus) if corpus else None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps([chunk.model_dump(mode="json") for chunk in chunks], ensure_ascii=False),
                encoding="utf-8",
            )
            temp_path.replace(self.path)

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._chunks = [Chunk.model_validate(item) for item in payload]
            corpus = [tokenize(chunk.text) for chunk in self._chunks]
            self._index = BM25Okapi(corpus) if corpus else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self._chunks, self._index = [], None

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        with self._lock:
            if self._index is None:
                return []
            scores = self._index.get_scores(tokenize(query))
            ranked = sorted(enumerate(scores), key=lambda pair: float(pair[1]), reverse=True)
            maximum = max((float(score) for _, score in ranked), default=0.0)
            output: list[SearchHit] = []
            for index, score in ranked[:top_k]:
                normalized = float(score) / maximum if maximum > 0 else 0.0
                output.append(
                    SearchHit(
                        chunk=self._chunks[index],
                        score=normalized,
                        sparse_score=normalized,
                    )
                )
            return output

    @property
    def size(self) -> int:
        return len(self._chunks)
