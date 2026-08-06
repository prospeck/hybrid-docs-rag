from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_rag.config import Settings
from hybrid_rag.loaders import DocumentLoader, UnsupportedDocumentError


def test_loads_markdown_and_persists_raw_and_processed(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nHello   world.\n\n\nNext.", encoding="utf-8")
    document = DocumentLoader(settings).load(path)
    assert document.text == "# Guide\n\nHello world.\n\nNext."
    assert document.metadata["headings"] == ["Guide"]
    assert list(settings.raw_dir.glob(f"{document.id}-*"))
    assert (settings.processed_dir / f"{document.id}.json").exists()
    assert DocumentLoader(settings).list_documents()[0].id == document.id


def test_loads_html_without_script_content(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "guide.html"
    path.write_text(
        "<h1>Runbook</h1><p>Rotate token.</p><script>secret()</script>", encoding="utf-8"
    )
    document = DocumentLoader(settings).load(path)
    assert "# Runbook" in document.text
    assert "Rotate token" in document.text
    assert "secret" not in document.text


def test_rejects_unsupported_type(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(UnsupportedDocumentError):
        DocumentLoader(settings).load(path)
