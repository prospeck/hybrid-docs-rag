.PHONY: install test lint typecheck run seed eval up

install:
	uv sync --extra dev --extra dashboard

test:
	uv run pytest --cov=hybrid_rag --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src

run:
	uv run uvicorn hybrid_rag.api:app --reload --port 8080

seed:
	uv run hybrid-rag seed sample_corpus

eval:
	uv run hybrid-rag evaluate eval_dataset.json

up:
	docker compose up --build
