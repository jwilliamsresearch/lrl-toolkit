"""Connector registry: maps source names to connector instances."""

from __future__ import annotations

from .base import BaseConnector
from .connectors.commoncrawl import CommonCrawlConnector
from .connectors.flores import FloresConnector
from .connectors.hf_stream import (
    CulturaXConnector,
    Glot500Connector,
    OscarConnector,
    WikipediaConnector,
)
from .connectors.local import LocalConnector
from .connectors.madlad import MadladConnector
from .connectors.opus import OpusConnector
from .connectors.smol import SmolConnector

_CONNECTORS: dict[str, BaseConnector] = {
    c.name: c
    for c in [
        WikipediaConnector(),
        Glot500Connector(),
        CulturaXConnector(),
        OscarConnector(),
        MadladConnector(),
        CommonCrawlConnector(),
        LocalConnector(),
        OpusConnector(),
        SmolConnector(),
        FloresConnector(),
    ]
}


def get_connector(name: str) -> BaseConnector:
    try:
        return _CONNECTORS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown connector '{name}'. Known: {sorted(_CONNECTORS)}"
        ) from exc


def available_connectors() -> list[str]:
    return sorted(_CONNECTORS)
