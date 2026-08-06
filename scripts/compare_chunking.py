from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from hybrid_rag.config import Settings
from hybrid_rag.container import build_components
from hybrid_rag.evaluation import EvaluationRunner

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("fixed", "recursive", "semantic")
MODES = ("hybrid", "dense")


def run_experiment() -> dict[str, Any]:
    dataset_path = ROOT / "eval_dataset.json"
    corpus = ROOT / "sample_corpus"
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hybrid-rag-eval-") as temporary:
        for strategy in STRATEGIES:
            settings = Settings(
                data_dir=Path(temporary) / strategy,
                chunk_strategy=strategy,
                embedding_provider="hash",
                vector_backend="memory",
                generation_provider="extractive",
                reranker_enabled=False,
            )
            components = build_components(settings)
            for path in sorted(corpus.iterdir()):
                if path.suffix.lower() in settings.allowed_extensions:
                    components.ingestion.ingest(path)
            runner = EvaluationRunner(components.service)
            dataset = runner.load_dataset(dataset_path)
            for mode in MODES:
                report = runner.run(dataset, mode=mode)
                results.append(
                    {
                        "strategy": strategy,
                        "mode": mode,
                        "case_count": report["case_count"],
                        **report["summary"],
                    }
                )
    return {
        "configuration": {
            "embedding_provider": "hash",
            "generation_provider": "extractive",
            "reranker": "lexical",
            "purpose": "reproducible no-key regression baseline",
        },
        "results": results,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Chunking and retrieval comparison",
        "",
        "Reproducible no-key regression baseline. These figures measure the deterministic "
        "test stack, not the quality ceiling of the default neural stack.",
        "",
        "| Strategy | Retrieval | Cases | Correctness | Faithfulness | Retrieval relevance "
        "| Citation accuracy | Mean latency |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["results"]:
        lines.append(
            f"| {item['strategy']} | {item['mode']} | {item['case_count']} | "
            f"{item['answer_correctness']:.1%} | {item['faithfulness']:.1%} | "
            f"{item['retrieval_relevance']:.1%} | {item['citation_accuracy']:.1%} | "
            f"{item['mean_latency_ms']:.1f} ms |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    output = ROOT / "eval-results"
    output.mkdir(exist_ok=True)
    report = run_experiment()
    (output / "chunking-comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "chunking-comparison.md").write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))


if __name__ == "__main__":
    main()
