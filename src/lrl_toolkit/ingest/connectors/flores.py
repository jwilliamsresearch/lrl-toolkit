"""FLORES-200 connector (openlanguagedata/flores_plus).

FLORES+ is a high-quality, human-translated evaluation set aligned across 200+
languages (the successor to Meta's FLORES-200, which also underpins NLLB-Seed).
It is small and gated: accept the license on the Hub and set HF_TOKEN.

Primarily an evaluation resource, but the sentences are clean parallel data
useful for seeding conversational/translation data too. Yields the target
language's sentences as monolingual documents.
"""

from __future__ import annotations

from collections.abc import Iterator

from ...corpus import Document
from ...registry import SourceHint
from ..base import BaseConnector, require


class FloresConnector(BaseConnector):
    name = "flores"
    kind = "parallel"
    gated = True

    def iter_documents(
        self, hint: SourceHint, *, max_docs: int | None = None, token: str | None = None
    ) -> Iterator[Document]:
        datasets = require("datasets")
        code = hint.params.get("flores") or hint.params.get("lang_script")
        if not code:
            raise ValueError("flores source needs params.flores (e.g. 'cym_Latn').")
        split = hint.params.get("split", "dev")
        try:
            ds = datasets.load_dataset(
                "openlanguagedata/flores_plus", code, split=split, streaming=True, token=token
            )
        except Exception as exc:
            msg = str(exc).lower()
            if any(t in msg for t in ("gated", "401", "403", "access", "awaiting")):
                raise PermissionError(
                    "FLORES+ is gated. Accept the license at "
                    "https://huggingface.co/datasets/openlanguagedata/flores_plus and set HF_TOKEN."
                ) from exc
            raise
        n = 0
        for row in ds:
            text = row.get("text") or ""
            if not text:
                continue
            yield Document(text=text, source=self.name, meta={"flores": code, "split": split})
            n += 1
            if max_docs is not None and n >= max_docs:
                return
