from __future__ import annotations

from pathlib import Path

from hybrid_rag.container import Components


class RejectingJudgeGenerator:
    name = "rejecting-judge"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        del system, user
        self.calls += 1
        return "API access tokens expire after 60 minutes. [1]"


def test_ingestion_keeps_dense_and_sparse_indexes_in_sync(
    components: Components, tmp_path: Path
) -> None:
    path = tmp_path / "runbook.md"
    path.write_text(
        "# Metrics\n\nOBS-1042 means the service token expired. Rotate METRICS_PUSH_TOKEN.",
        encoding="utf-8",
    )
    result = components.ingestion.ingest(path)
    chunks, _ = components.vector_store.all()
    assert result.chunks_created == len(chunks)
    assert components.sparse_index.size == len(chunks)


def test_reingestion_is_idempotent_and_near_duplicates_are_skipped(
    components: Components, tmp_path: Path
) -> None:
    path = tmp_path / "policy.md"
    path.write_text("# Access\n\nProduction access expires after eight hours.", encoding="utf-8")
    first = components.ingestion.ingest(path)
    second = components.ingestion.ingest(path)
    assert first.chunks_created == 1
    assert second.chunks_created == 0
    assert second.duplicates_skipped == 1


def test_changed_source_replaces_previous_document(components: Components, tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text("Production access expires after eight hours.", encoding="utf-8")
    first = components.ingestion.ingest(path)
    path.write_text("Production access expires after six hours.", encoding="utf-8")
    second = components.ingestion.ingest(path)
    chunks, _ = components.vector_store.all()
    assert second.document_id != first.document_id
    assert {chunk.document_id for chunk in chunks} == {second.document_id}
    assert "six hours" in chunks[0].text


def test_answer_contains_verified_source_and_confidence(
    components: Components, tmp_path: Path
) -> None:
    path = tmp_path / "api.txt"
    path.write_text(
        "AUTHENTICATION\nAPI access tokens expire after 60 minutes. "
        "AUTH-4017 means wrong audience.",
        encoding="utf-8",
    )
    components.ingestion.ingest(path)
    result = components.service.ask("How long do API access tokens last?")
    assert result.status == "answered"
    assert "60 minutes" in result.answer
    assert result.citations
    assert result.citations[0].source == "api.txt"
    assert 0 <= result.confidence.composite <= 1


def test_direct_evidence_is_verified_without_an_llm_judge_call(
    components: Components, tmp_path: Path
) -> None:
    path = tmp_path / "api.txt"
    path.write_text("API access tokens expire after 60 minutes.", encoding="utf-8")
    components.ingestion.ingest(path)
    generator = RejectingJudgeGenerator()
    components.service.generator = generator

    result = components.service.ask("How long do API access tokens last?")

    assert generator.calls == 1
    assert result.citations[0].verified is True


def test_low_retrieval_confidence_returns_structured_unknown(
    components: Components, tmp_path: Path
) -> None:
    path = tmp_path / "api.txt"
    path.write_text("API access tokens expire after 60 minutes.", encoding="utf-8")
    components.ingestion.ingest(path)
    components.settings.minimum_retrieval_confidence = 0.99
    result = components.service.ask("What is the parental leave policy?")
    assert result.status == "insufficient_context"
    assert "couldn't find" in result.answer
    assert result.citations == []
