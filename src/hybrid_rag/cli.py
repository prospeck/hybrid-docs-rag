from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from hybrid_rag.container import build_components
from hybrid_rag.evaluation import EvaluationRunner


def seed(corpus: Path) -> None:
    components = build_components()
    supported = set(components.settings.allowed_extensions)
    paths = [
        path for path in corpus.rglob("*") if path.is_file() and path.suffix.lower() in supported
    ]
    for path in paths:
        result = components.ingestion.ingest(path)
        print(
            f"{result.source}: {result.chunks_created} chunks, "
            f"{result.duplicates_skipped} duplicates"
        )


def evaluate(dataset: Path, output: Path, mode: str) -> None:
    components = build_components()
    runner = EvaluationRunner(components.service)
    report = runner.run(runner.load_dataset(dataset), mode=mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="hybrid-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)
    seed_parser = subparsers.add_parser("seed")
    seed_parser.add_argument("corpus", type=Path, default=Path("sample_corpus"), nargs="?")
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("dataset", type=Path, default=Path("eval_dataset.json"), nargs="?")
    eval_parser.add_argument("--output", type=Path, default=Path("eval-results/report.json"))
    eval_parser.add_argument("--mode", choices=["hybrid", "dense"], default="hybrid")
    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run("hybrid_rag.api:app", host=args.host, port=args.port)
    elif args.command == "seed":
        seed(args.corpus)
    else:
        evaluate(args.dataset, args.output, args.mode)


if __name__ == "__main__":
    main()
