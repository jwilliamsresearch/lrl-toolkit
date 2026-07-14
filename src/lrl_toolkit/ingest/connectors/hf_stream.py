"""Monolingual connectors backed by streaming Hugging Face datasets.

Covers Wikipedia, Glot500, CulturaX (mC4+OSCAR), and OSCAR-2301. All stream in
constant memory via ``datasets.load_dataset(..., streaming=True)`` so a run
never needs to download a whole language split.
"""

from __future__ import annotations

from collections.abc import Iterator

from ...corpus import Document
from ...registry import SourceHint
from ..base import BaseConnector, require


class _HFStreamConnector(BaseConnector):
    """Shared streaming logic; subclasses supply the dataset coordinates."""

    kind = "mono"
    text_field = "text"

    def _load_kwargs(self, hint: SourceHint) -> dict:
        raise NotImplementedError

    def _extract_text(self, row: dict) -> str:
        return row.get(self.text_field) or ""

    def iter_documents(
        self, hint: SourceHint, *, max_docs: int | None = None, token: str | None = None
    ) -> Iterator[Document]:
        datasets = require("datasets")
        kwargs = self._load_kwargs(hint)
        try:
            ds = datasets.load_dataset(split="train", streaming=True, token=token, **kwargs)
        except Exception as exc:  # gated / not-found -> actionable message
            msg = str(exc).lower()
            gate_markers = ("gated", "401", "403", "awaiting", "access", "authenticated")
            if any(t in msg for t in gate_markers):
                raise PermissionError(
                    f"'{self.name}' is gated on the Hub. Accept the dataset's license and set "
                    f"HF_TOKEN (or `huggingface-cli login`). Coordinates: {kwargs}"
                ) from exc
            raise
        n = 0
        for row in ds:
            text = self._extract_text(row)
            if not text:
                continue
            meta = {k: row[k] for k in ("url", "title", "source") if k in row}
            yield Document(text=text, source=self.name, meta=meta)
            n += 1
            if max_docs is not None and n >= max_docs:
                break


class WikipediaConnector(_HFStreamConnector):
    name = "wikipedia"

    def _load_kwargs(self, hint: SourceHint) -> dict:
        wiki = hint.params.get("wiki")
        if not wiki:
            raise ValueError("wikipedia source needs params.wiki (e.g. 'cy').")
        snapshot = hint.params.get("snapshot", "20231101")
        return {"path": "wikimedia/wikipedia", "name": f"{snapshot}.{wiki}"}


class Glot500Connector(_HFStreamConnector):
    name = "glot500"

    def _load_kwargs(self, hint: SourceHint) -> dict:
        code = hint.params.get("glot") or hint.params.get("lang_script")
        if not code:
            raise ValueError("glot500 source needs params.glot (e.g. 'cym_Latn').")
        return {"path": "cis-lmu/Glot500", "name": code}


class CulturaXConnector(_HFStreamConnector):
    name = "culturax"
    gated = True  # requires accepting terms; usually auto-granted with a token

    def _load_kwargs(self, hint: SourceHint) -> dict:
        lang = hint.params.get("lang")
        if not lang:
            raise ValueError("culturax source needs params.lang (e.g. 'cy').")
        return {"path": "uonlp/CulturaX", "name": lang}


class OscarConnector(_HFStreamConnector):
    name = "oscar"
    gated = True

    def _load_kwargs(self, hint: SourceHint) -> dict:
        lang = hint.params.get("lang")
        if not lang:
            raise ValueError("oscar source needs params.lang (e.g. 'cy').")
        # OSCAR-2301 selects the split via `language=`, not `name=`.
        return {"path": "oscar-corpus/OSCAR-2301", "language": lang}
