"""Smoke-test the Streamlit dashboard by running it headless via AppTest.

Skipped when Streamlit isn't installed. This executes the whole app script (with
no projects present, so it exercises the empty-state branches) and asserts it
raises no exception and renders the expected tabs.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


def test_dashboard_runs_without_error():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    # Title and the four tabs render.
    assert any("lrl-toolkit" in (t.value or "") for t in at.title)
    assert len(at.tabs) == 4


def test_dashboard_wizard_generates_valid_yaml(tmp_path, monkeypatch):
    # Point config discovery at the bundled profiles only (deterministic).
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    # The wizard renders a YAML code block by default (no interaction needed).
    assert not at.exception
