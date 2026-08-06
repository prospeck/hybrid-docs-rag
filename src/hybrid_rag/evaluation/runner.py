from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from hybrid_rag.service import RAGService
from hybrid_rag.text import tokenize


class GoldenCase(BaseModel):
    id: str
    question: str
    expected_answer: str
    expected_sources: list[str] = Field(default_factory=list)
    answerable: bool = True
    category: str


@dataclass
class CaseMetrics:
    case_id: str
    answer_correctness: float
    faithfulness: float
    retrieval_relevance: float
    citation_accuracy: float
    latency_ms: float


class EvaluationRunner:
    def __init__(self, service: RAGService) -> None:
        self.service = service

    @staticmethod
    def load_dataset(path: Path) -> list[GoldenCase]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [GoldenCase.model_validate(item) for item in payload]

    def evaluate_case(self, case: GoldenCase, *, mode: str = "hybrid") -> CaseMetrics:
        result = self.service.ask(case.question, mode=mode)
        expected_tokens = set(tokenize(case.expected_answer))
        answer_tokens = set(tokenize(result.answer))
        correctness = len(expected_tokens & answer_tokens) / max(1, len(expected_tokens))
        if not case.answerable:
            correctness = 1.0 if result.status == "insufficient_context" else 0.0
        ranked_sources = [hit.chunk.source for hit in result.retrieved_chunks]
        expected_sources = list(dict.fromkeys(case.expected_sources))
        relevance = (
            mean(
                1.0 / (ranked_sources.index(source) + 1) if source in ranked_sources else 0.0
                for source in expected_sources
            )
            if expected_sources
            else float(result.status == "insufficient_context")
        )
        verified = [citation.verified for citation in result.citations]
        citation_accuracy = (
            sum(verified) / len(verified) if verified else float(not case.answerable)
        )
        return CaseMetrics(
            case_id=case.id,
            answer_correctness=correctness,
            faithfulness=result.confidence.citation_coverage,
            retrieval_relevance=relevance,
            citation_accuracy=citation_accuracy,
            latency_ms=result.latency_ms,
        )

    def run(self, dataset: list[GoldenCase], *, mode: str = "hybrid") -> dict[str, Any]:
        cases = [self.evaluate_case(case, mode=mode) for case in dataset]
        keys = ("answer_correctness", "faithfulness", "retrieval_relevance", "citation_accuracy")
        summary = {key: mean(getattr(case, key) for case in cases) for key in keys}
        summary["mean_latency_ms"] = mean(case.latency_ms for case in cases)
        return {
            "mode": mode,
            "case_count": len(cases),
            "summary": summary,
            "cases": [asdict(case) for case in cases],
        }
