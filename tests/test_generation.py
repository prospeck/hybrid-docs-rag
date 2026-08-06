from __future__ import annotations

from hybrid_rag.generation import (
    citation_support_score,
    context_prompt,
    parse_claims,
    repair_missing_citations,
)
from hybrid_rag.models import Chunk, SearchHit


def search_hit() -> SearchHit:
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        source="runbook.md",
        text="Access expires after eight hours.",
        chunk_index=0,
        strategy="recursive",
        section_heading="Production access",
        page_number=2,
        character_count=33,
        content_hash="hash",
    )
    return SearchHit(chunk=chunk, score=0.9)


def test_claim_parser_keeps_trailing_references_with_sentence() -> None:
    claims = parse_claims("Access expires after eight hours. [1] It is recorded. [2]")
    assert claims == [
        ("Access expires after eight hours.", [1]),
        ("It is recorded.", [2]),
    ]


def test_context_prompt_includes_source_location() -> None:
    prompt = context_prompt("How long?", [search_hit()])
    assert "[1] SOURCE: runbook.md — Production access (page 2)" in prompt
    assert "Access expires after eight hours" in prompt


def test_citation_support_score_is_bounded_and_rewards_entailment() -> None:
    strong = citation_support_score(
        "Access expires after eight hours", "Production access expires after eight hours"
    )
    weak = citation_support_score("Access expires after eight hours", "Deployments use canaries")
    assert 0 <= weak < strong <= 1


def test_missing_citations_are_repaired_only_with_strong_evidence() -> None:
    supported = repair_missing_citations("Access expires after eight hours.", [search_hit()])
    unsupported = repair_missing_citations("Deployments use canaries.", [search_hit()])
    existing = repair_missing_citations("Access expires after eight hours. [1]", [search_hit()])

    assert supported == "Access expires after eight hours. [1]"
    assert unsupported == "Deployments use canaries."
    assert existing == "Access expires after eight hours. [1]"
