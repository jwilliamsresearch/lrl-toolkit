"""Connector interface shared by every ingest source.

A connector knows how to turn a :class:`SourceHint` (from a language profile)
into a stream of :class:`Document`s, plus a base :class:`ProvenanceRecord`
describing where the data came from and under what license. Heavy / network
dependencies are imported lazily inside :meth:`iter_documents` so the core
package stays importable without the ``[data]`` extra installed.
"""

from __future__ import annotations

import abc
import importlib
from collections.abc import Iterator

from ..corpus import Document
from ..manifest import ProvenanceRecord
from ..registry import SourceHint


class MissingDependencyError(RuntimeError):
    """Raised when a connector's optional dependency is not installed."""


def require(module: str, *, extra: str = "data") -> object:
    """Import an optional dependency or raise a helpful install hint."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only when dep missing
        raise MissingDependencyError(
            f"This connector needs '{module}'. Install it with: "
            f"pip install 'lrl-toolkit[{extra}]'"
        ) from exc


class BaseConnector(abc.ABC):
    """Base class for corpus source connectors."""

    #: Connector slug used in language-profile source catalogs.
    name: str
    #: "mono" (monolingual corpus) or "parallel" (aligned pairs).
    kind: str = "mono"
    #: Whether the source is gated and typically needs an auth token.
    gated: bool = False

    def provenance(self, hint: SourceHint) -> ProvenanceRecord:
        """Base provenance for this source (n_docs filled in by the stage)."""
        return ProvenanceRecord(
            source=self.name,
            url=hint.params.get("url"),
            license=hint.params.get("license"),
            notes=hint.notes,
        )

    @abc.abstractmethod
    def iter_documents(
        self, hint: SourceHint, *, max_docs: int | None = None, token: str | None = None
    ) -> Iterator[Document]:
        """Yield documents for one source. Implementations import heavy deps lazily."""
        raise NotImplementedError
