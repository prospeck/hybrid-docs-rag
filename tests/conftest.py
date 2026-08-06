from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_rag.config import Settings
from hybrid_rag.container import Components, build_components


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        embedding_provider="hash",
        vector_backend="memory",
        generation_provider="extractive",
        reranker_enabled=False,
        chunk_size=300,
        chunk_overlap=40,
        minimum_retrieval_confidence=0.0,
    )


@pytest.fixture
def components(settings: Settings) -> Components:
    return build_components(settings)
