"""Local-files connector — the offline path.

Reads a directory (recursively) of ``.txt``, ``.jsonl``, ``.jsonl.gz``, ``.md``,
``.html``, or ``.pdf`` files. Community-contributed texts for revived/endangered
languages often live only on someone's disk; this is how they enter the pipeline
(with permission — see DATA_ETHICS.md).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

from ...corpus import Document
from ...registry import SourceHint
from ..base import BaseConnector, require

_TEXT_EXT = {".txt", ".md"}
_JSONL_EXT = {".jsonl"}


class LocalConnector(BaseConnector):
    name = "local"
    kind = "mono"

    def iter_documents(
        self, hint: SourceHint, *, max_docs: int | None = None, token: str | None = None
    ) -> Iterator[Document]:
        raw_path = hint.params.get("path")
        if not raw_path:
            raise ValueError("local source needs params.path (a file or directory).")
        root = Path(raw_path)
        if not root.exists():
            raise FileNotFoundError(f"local path does not exist: {root}")
        text_field = hint.params.get("text_field", "text")

        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        n = 0
        for f in files:
            for text in self._read_file(f, text_field):
                if not text.strip():
                    continue
                yield Document(text=text, source=self.name, meta={"path": str(f)})
                n += 1
                if max_docs is not None and n >= max_docs:
                    return

    def _read_file(self, f: Path, text_field: str) -> Iterator[str]:
        suffixes = f.suffixes
        if f.suffix == ".gz" and ".jsonl" in suffixes:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line).get(text_field, "")
        elif f.suffix in _JSONL_EXT:
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line).get(text_field, "")
        elif f.suffix in _TEXT_EXT:
            yield f.read_text(encoding="utf-8", errors="replace")
        elif f.suffix in {".html", ".htm"}:
            yield self._extract_html(f.read_text(encoding="utf-8", errors="replace"))
        elif f.suffix == ".pdf":
            yield from self._extract_pdf(f)
        # silently skip unknown extensions

    def _extract_html(self, html: str) -> str:
        try:
            trafilatura = require("trafilatura")
            return trafilatura.extract(html) or ""
        except Exception:
            return html

    def _extract_pdf(self, f: Path) -> Iterator[str]:
        fitz = require("pymupdf", extra="data")
        with fitz.open(f) as doc:
            for page in doc:
                yield page.get_text()
