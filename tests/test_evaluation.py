from __future__ import annotations

import json
from pathlib import Path

from hybrid_rag.container import Components
from hybrid_rag.evaluation.runner import EvaluationRunner


def test_evaluation_runs_answerable_and_unknown_cases(
    components: Components, tmp_path: Path
) -> None:
    source = tmp_path / "policy.md"
    source.write_text("# Access\n\nStandard access expires after eight hours.", encoding="utf-8")
    components.ingestion.ingest(source)
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "known",
                    "question": "When does standard access expire?",
                    "expected_answer": "after eight hours",
                    "expected_sources": ["policy.md"],
                    "answerable": True,
                    "category": "lookup",
                }
            ]
        ),
        encoding="utf-8",
    )
    runner = EvaluationRunner(components.service)
    report = runner.run(runner.load_dataset(dataset_path))
    assert report["case_count"] == 1
    assert report["summary"]["retrieval_relevance"] == 1.0
    assert report["cases"][0]["case_id"] == "known"
