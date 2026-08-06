from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_rag.config import Settings
from hybrid_rag.embeddings import EmbeddingProvider, cosine_similarity
from hybrid_rag.models import Chunk, Document
from hybrid_rag.text import normalize_text, stable_hash

PAGE_PATTERN = re.compile(r"\[\[PAGE (\d+)\]\]")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class Chunker(Protocol):
    strategy: str

    def split(self, document: Document) -> list[Chunk]: ...


class BaseChunker:
    strategy: str

    def _chunks(self, document: Document, texts: Sequence[str]) -> list[Chunk]:
        output: list[Chunk] = []
        cursor = 0
        for index, raw_text in enumerate(texts):
            text = normalize_text(raw_text)
            if not text:
                continue
            anchor = text[: min(100, len(text))]
            position = document.text.find(anchor, cursor)
            if position < 0:
                position = document.text.find(anchor)
            if position < 0:
                position = cursor
            context = document.text[: position + len(text)]
            heading_matches = list(HEADING_PATTERN.finditer(context))
            page_matches = list(PAGE_PATTERN.finditer(context))
            heading = heading_matches[-1].group(2).strip() if heading_matches else None
            page = int(page_matches[-1].group(1)) if page_matches else None
            cursor = max(cursor, position + len(anchor))
            content_hash = stable_hash(text, length=64)
            chunk_id = stable_hash(f"{document.id}:{self.strategy}:{index}:{content_hash}")
            output.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    source=document.source,
                    text=text,
                    chunk_index=index,
                    strategy=self.strategy,  # type: ignore[arg-type]
                    section_heading=heading,
                    page_number=page,
                    character_count=len(text),
                    content_hash=content_hash,
                    metadata={"media_type": document.media_type},
                )
            )
        return output


class FixedChunker(BaseChunker):
    strategy = "fixed"

    def __init__(self, chunk_size: int, overlap: int) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        step = self.chunk_size - self.overlap
        texts = [
            document.text[start : start + self.chunk_size]
            for start in range(0, len(document.text), step)
        ]
        return self._chunks(document, texts)


class RecursiveChunker(BaseChunker):
    strategy = "recursive"

    def __init__(self, chunk_size: int, overlap: int) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def split(self, document: Document) -> list[Chunk]:
        return self._chunks(document, self.splitter.split_text(document.text))


class SemanticChunker(BaseChunker):
    strategy = "semantic"

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        target_size: int,
        breakpoint_percentile: float,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.target_size = target_size
        self.breakpoint_percentile = breakpoint_percentile

    def split(self, document: Document) -> list[Chunk]:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|\n{2,}", document.text)
            if part.strip()
        ]
        if len(sentences) < 2:
            return self._chunks(document, sentences)
        embeddings = self.embedding_provider.embed(sentences)
        similarities = [
            cosine_similarity(embeddings[index], embeddings[index + 1])
            for index in range(len(embeddings) - 1)
        ]
        threshold = float(np.percentile(similarities, 100 - self.breakpoint_percentile))
        groups: list[str] = []
        current: list[str] = []
        current_length = 0
        for index, sentence in enumerate(sentences):
            current.append(sentence)
            current_length += len(sentence)
            boundary = index < len(similarities) and similarities[index] <= threshold
            size_limit = current_length >= self.target_size
            if boundary or size_limit:
                groups.append(" ".join(current))
                current, current_length = [], 0
        if current:
            groups.append(" ".join(current))
        return self._chunks(document, groups)


def build_chunker(settings: Settings, embeddings: EmbeddingProvider) -> Chunker:
    if settings.chunk_strategy == "fixed":
        return FixedChunker(settings.chunk_size, settings.chunk_overlap)
    if settings.chunk_strategy == "semantic":
        return SemanticChunker(
            embeddings,
            target_size=settings.chunk_size,
            breakpoint_percentile=settings.semantic_breakpoint_percentile,
        )
    return RecursiveChunker(settings.chunk_size, settings.chunk_overlap)
