"""Ingest connector registry and the offline LocalConnector."""

import gzip
import json

import pytest

from lrl_toolkit.ingest import available_connectors, get_connector
from lrl_toolkit.ingest.connectors.local import LocalConnector
from lrl_toolkit.registry import SourceHint

EXPECTED = {
    "wikipedia", "glot500", "culturax", "madlad400", "oscar",
    "commoncrawl", "local", "opus", "smol", "flores",
}


def test_all_connectors_registered():
    assert EXPECTED.issubset(set(available_connectors()))


def test_get_unknown_connector_raises():
    with pytest.raises(ValueError):
        get_connector("nope")


def test_local_connector_reads_txt_and_jsonl(tmp_path):
    (tmp_path / "a.txt").write_text("hello local corpus", encoding="utf-8")
    with gzip.open(tmp_path / "b.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "from jsonl gz"}) + "\n")

    conn = LocalConnector()
    hint = SourceHint(connector="local", params={"path": str(tmp_path)})
    texts = sorted(d.text for d in conn.iter_documents(hint))
    assert texts == ["from jsonl gz", "hello local corpus"]


def test_local_connector_respects_max_docs(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text(f"document number {i}", encoding="utf-8")
    conn = LocalConnector()
    hint = SourceHint(connector="local", params={"path": str(tmp_path)})
    docs = list(conn.iter_documents(hint, max_docs=2))
    assert len(docs) == 2


def test_local_connector_missing_path_raises():
    conn = LocalConnector()
    with pytest.raises(ValueError):
        list(conn.iter_documents(SourceHint(connector="local", params={})))
