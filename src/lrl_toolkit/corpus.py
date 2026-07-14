"""Corpus data model and sharded JSONL(.gz) storage.

Connectors emit :class:`Document`s; the ingest stage streams them through a
:class:`ShardWriter` into gzipped JSONL shards. The same format is read back by
the clean stage and rewritten (cleaned) using the same writer.

Documents are plain dataclasses (not Pydantic) because we handle millions of
them and per-object validation would dominate runtime.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Document:
    """One unit of text in a corpus.

    For monolingual sources ``text`` is the document body. For parallel sources
    ``text`` holds the target-language side (so it can also feed monolingual
    training) and ``meta['translation']`` carries the aligned pair.
    """

    text: str
    source: str
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict) -> Document:
        return Document(text=d.get("text", ""), source=d.get("source", ""), meta=d.get("meta", {}))


class ShardWriter:
    """Writes documents into gzipped JSONL shards of a fixed size."""

    def __init__(
        self,
        out_dir: str | Path,
        prefix: str = "part",
        docs_per_shard: int = 50_000,
        compress: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.docs_per_shard = docs_per_shard
        self.compress = compress
        self._shard_idx = 0
        self._in_shard = 0
        self._fh = None
        self.n_docs = 0
        self.n_bytes = 0  # uncompressed text bytes written
        self.shards: list[str] = []

    def _ext(self) -> str:
        return ".jsonl.gz" if self.compress else ".jsonl"

    def _open_new_shard(self) -> None:
        self._close_fh()
        name = f"{self.prefix}-{self._shard_idx:05d}{self._ext()}"
        path = self.out_dir / name
        self._fh = gzip.open(path, "wt", encoding="utf-8") if self.compress else open(
            path, "w", encoding="utf-8"
        )
        self.shards.append(str(path))
        self._in_shard = 0
        self._shard_idx += 1

    def _close_fh(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, doc: Document) -> None:
        if self._fh is None or self._in_shard >= self.docs_per_shard:
            self._open_new_shard()
        line = doc.to_json()
        self._fh.write(line + "\n")
        self._in_shard += 1
        self.n_docs += 1
        self.n_bytes += len(doc.text.encode("utf-8"))

    def close(self) -> None:
        self._close_fh()


def iter_documents(path: str | Path) -> Iterator[Document]:
    """Iterate documents from a shard file or a directory of shards."""
    path = Path(path)
    files: Iterable[Path]
    if path.is_dir():
        files = sorted(list(path.glob("*.jsonl")) + list(path.glob("*.jsonl.gz")))
    else:
        files = [path]
    for f in files:
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield Document.from_dict(json.loads(line))
