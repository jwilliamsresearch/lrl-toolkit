"""MADLAD-400 connector via direct raw-file access.

``allenai/madlad-400`` ships as a dataset *script*, which current ``datasets``
versions no longer execute. We instead read its per-language gzipped JSONL files
straight from the Hub with ``HfFileSystem`` — robust and streaming-friendly.
Layout: ``datasets/allenai/madlad-400/data/<lang>/<lang>_clean_XXXX.jsonl.gz``.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterator

from ...corpus import Document
from ...registry import SourceHint
from ..base import BaseConnector, require


class MadladConnector(BaseConnector):
    name = "madlad400"
    kind = "mono"

    def iter_documents(
        self, hint: SourceHint, *, max_docs: int | None = None, token: str | None = None
    ) -> Iterator[Document]:
        hub = require("huggingface_hub")
        lang = hint.params.get("lang")
        if not lang:
            raise ValueError("madlad400 source needs params.lang (e.g. 'cy').")
        variant = hint.params.get("variant", "clean")  # 'clean' or 'noisy'
        fs = hub.HfFileSystem(token=token)
        base = f"datasets/allenai/madlad-400/data/{lang}"
        try:
            files = [f for f in fs.ls(base, detail=False) if f.endswith(".jsonl.gz")]
        except FileNotFoundError as exc:
            raise ValueError(f"MADLAD-400 has no data for language '{lang}'.") from exc
        files = sorted(f for f in files if variant in f) or sorted(files)

        n = 0
        for path in files:
            with fs.open(path, "rb") as raw:
                with gzip.GzipFile(fileobj=io.BytesIO(raw.read())) as gz:
                    for line in gz:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        text = rec.get("text") or ""
                        if not text:
                            continue
                        yield Document(text=text, source=self.name, meta={"variant": variant})
                        n += 1
                        if max_docs is not None and n >= max_docs:
                            return
