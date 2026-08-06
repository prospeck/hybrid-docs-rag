from __future__ import annotations

from hybrid_rag.models import Chunk, SearchHit
from hybrid_rag.retrieval import reciprocal_rank_fusion


def hit(identifier: str, score: float = 1.0) -> SearchHit:
    chunk = Chunk(
        id=identifier,
        document_id="doc",
        source="source.md",
        text=identifier,
        chunk_index=0,
        strategy="fixed",
        character_count=len(identifier),
        content_hash=identifier,
    )
    return SearchHit(chunk=chunk, score=score, dense_score=score)


def test_rrf_rewards_results_present_in_both_rankings() -> None:
    dense = [hit("a"), hit("b")]
    sparse = [hit("b"), hit("c")]
    fused = reciprocal_rank_fusion(
        dense, sparse, dense_weight=0.7, sparse_weight=0.3, rrf_k=60, limit=3
    )
    assert fused[0].chunk.id == "b"
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1
    assert fused[0].fusion_score == 1.0


def test_rrf_weighting_can_prefer_dense_first_result() -> None:
    fused = reciprocal_rank_fusion(
        [hit("dense")],
        [hit("sparse")],
        dense_weight=0.9,
        sparse_weight=0.1,
        rrf_k=60,
        limit=2,
    )
    assert fused[0].chunk.id == "dense"
