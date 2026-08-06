from __future__ import annotations

import json
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader

from hybrid_rag.config import Settings
from hybrid_rag.models import Document
from hybrid_rag.text import file_checksum, normalize_text, stable_hash

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class UnsupportedDocumentError(ValueError):
    pass


class DocumentLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()

    def load(self, path: Path) -> Document:
        path = path.resolve()
        suffix = path.suffix.lower()
        if suffix not in self.settings.allowed_extensions:
            raise UnsupportedDocumentError(
                f"Unsupported extension {suffix!r}; allowed: {self.settings.allowed_extensions}"
            )
        checksum = file_checksum(path)
        document_id = stable_hash(f"{path.name}:{checksum}")
        text, metadata = self._extract(path, suffix)
        document = Document(
            id=document_id,
            source=path.name,
            checksum=checksum,
            text=normalize_text(text),
            media_type=mimetypes.guess_type(path.name)[0] or "text/plain",
            metadata=metadata,
        )
        self._persist(path, document)
        return document

    def _extract(self, path: Path, suffix: str) -> tuple[str, dict[str, Any]]:
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            headings = [match.group(2).strip() for match in MARKDOWN_HEADING.finditer(text)]
            return text, {"headings": headings}
        if suffix in {".html", ".htm"}:
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
            for node in soup(["script", "style", "nav", "noscript"]):
                node.decompose()
            headings = [node.get_text(" ", strip=True) for node in soup.select("h1,h2,h3,h4,h5,h6")]
            blocks: list[str] = []
            for node in soup.select("h1,h2,h3,h4,h5,h6,p,li,pre,code,table"):
                value = node.get_text(" ", strip=True)
                if value and node.name and node.name.startswith("h"):
                    value = f"{'#' * int(node.name[1])} {value}"
                if value and (not blocks or blocks[-1] != value):
                    blocks.append(value)
            return "\n\n".join(blocks), {"headings": headings}
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            pages = []
            for index, page in enumerate(reader.pages, start=1):
                page_text = normalize_text(page.extract_text() or "")
                if page_text:
                    pages.append(f"[[PAGE {index}]]\n{page_text}")
            return "\n\n".join(pages), {"page_count": len(reader.pages)}
        raise UnsupportedDocumentError(f"No loader for {suffix}")

    def _persist(self, original: Path, document: Document) -> None:
        raw_target = self.settings.raw_dir / f"{document.id}-{original.name}"
        if not raw_target.exists():
            shutil.copy2(original, raw_target)
        processed_target = self.settings.processed_dir / f"{document.id}.json"
        processed_target.write_text(
            json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_documents(self) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(self.settings.processed_dir.glob("*.json")):
            try:
                documents.append(Document.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return documents
