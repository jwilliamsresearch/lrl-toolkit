"""Deduplication: exact (hashing) and near-duplicate (MinHash/LSH).

Exact dedup is pure-Python and always available. Near-dup uses ``datasketch``
(from the ``[data]`` extra); if it is not installed, the deduper degrades to
exact-only with a warning rather than failing.
"""

from __future__ import annotations

import hashlib
import re

from ..utils import get_logger

log = get_logger("lrl.clean.dedup")

_WS_RE = re.compile(r"\s+")


def _canonical(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


class Deduper:
    """Stateful deduplicator. Call :meth:`is_duplicate` per document in order."""

    def __init__(self, method: str = "minhash", *, threshold: float = 0.85, num_perm: int = 128):
        self.method = method
        self.threshold = threshold
        self.num_perm = num_perm
        self._exact_seen: set[str] = set()
        self._lsh = None
        self._minhash_cls = None
        self._idx = 0
        if method == "minhash":
            self._init_minhash()

    def _init_minhash(self) -> None:
        try:
            from datasketch import MinHash, MinHashLSH

            self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            self._minhash_cls = MinHash
        except ImportError:
            log.warning(
                "datasketch not installed; falling back to exact dedup. "
                "Install with: pip install 'lrl-toolkit[data]'"
            )
            self.method = "exact"

    def _minhash(self, text: str):
        m = self._minhash_cls(num_perm=self.num_perm)
        for token in set(_canonical(text).split()):
            m.update(token.encode("utf-8"))
        return m

    def is_duplicate(self, text: str) -> bool:
        if self.method == "none":
            return False

        digest = hashlib.sha1(_canonical(text).encode("utf-8")).hexdigest()
        if digest in self._exact_seen:
            return True
        self._exact_seen.add(digest)

        if self.method == "minhash" and self._lsh is not None:
            m = self._minhash(text)
            if self._lsh.query(m):
                return True
            self._lsh.insert(f"d{self._idx}", m)
            self._idx += 1
        return False
