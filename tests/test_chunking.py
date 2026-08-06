from __future__ import annotations

from hybrid_rag.chunking.strategies import FixedChunker, RecursiveChunker, SemanticChunker
from hybrid_rag.embeddings import HashEmbeddingProvider
from hybrid_rag.models import Document


def document(text: str) -> Document:
    return Document(
        id="doc", source="guide.md", checksum="abc", text=text, media_type="text/markdown"
    )


def test_fixed_chunks_overlap_and_track_strategy() -> None:
    chunks = FixedChunker(chunk_size=20, overlap=5).split(document("alpha " * 20))
    assert len(chunks) > 1
    assert all(chunk.strategy == "fixed" for chunk in chunks)
    assert all(chunk.character_count == len(chunk.text) for chunk in chunks)


def test_recursive_prefers_headings() -> None:
    chunks = RecursiveChunker(chunk_size=70, overlap=10).split(
        document("# First\n\n" + "alpha " * 20 + "\n## Second\n\n" + "beta " * 20)
    )
    assert len(chunks) >= 2
    assert chunks[0].strategy == "recursive"
    assert any(chunk.section_heading == "Second" for chunk in chunks)


def test_semantic_chunker_tracks_topic_boundaries() -> None:
    chunks = SemanticChunker(HashEmbeddingProvider(), 90, 80).split(
        document(
            "Deployments use canaries. Rollbacks stop bad releases. "
            "Employee learning budgets fund conferences. Managers approve travel."
        )
    )
    assert chunks
    assert all(chunk.strategy == "semantic" for chunk in chunks)
