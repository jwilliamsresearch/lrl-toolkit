"""Project config loading, validation, and profile resolution."""

import pytest
import yaml
from pydantic import ValidationError

from lrl_toolkit.config import STAGE_ORDER, ProjectConfig, load_project


def _write(tmp_path, data):
    p = tmp_path / "project.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_load_smoke_example():
    proj = load_project("examples/smoke.yaml")
    assert proj.name == "smoke"
    assert proj.language_profile.iso639_3 == "cym"
    assert proj.model_profile.hf_id.endswith("SmolLM2-135M")
    assert proj.compute_profile.quantization.value == "4bit"


def test_selected_stages_default_is_full_order():
    cfg = ProjectConfig(
        name="x", language="welsh", base_model="qwen2.5-1.5b", compute="consumer_gpu"
    )
    assert cfg.selected_stages() == list(STAGE_ORDER)


def test_stage_subset_is_canonically_ordered():
    cfg = ProjectConfig(
        name="x",
        language="welsh",
        base_model="qwen2.5-1.5b",
        compute="consumer_gpu",
        stages=["export", "ingest", "clean"],
    )
    assert cfg.selected_stages() == ["ingest", "clean", "export"]


def test_unknown_stage_rejected():
    with pytest.raises(ValueError):
        ProjectConfig(
            name="x", language="welsh", base_model="qwen2.5-1.5b", compute="consumer_gpu",
            stages=["not-a-stage"],
        )


def test_extra_keys_rejected(tmp_path):
    p = _write(tmp_path, {
        "name": "x", "language": "welsh", "base_model": "qwen2.5-1.5b",
        "compute": "consumer_gpu", "bogus_key": 1,
    })
    with pytest.raises(ValidationError):
        load_project(p)


def test_workdir_override(tmp_path):
    p = _write(tmp_path, {
        "name": "x", "language": "welsh", "base_model": "qwen2.5-1.5b",
        "compute": "consumer_gpu", "workdir": str(tmp_path / "wd"),
    })
    proj = load_project(p)
    assert proj.workdir == tmp_path / "wd"
