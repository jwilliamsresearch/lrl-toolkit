"""M2 (offline): tokenizer extension and continued pretraining on a tiny model.

Skipped automatically when the training deps (torch/transformers) are absent.
"""

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("transformers")

from lrl_toolkit.config import load_project  # noqa: E402
from lrl_toolkit.pipeline import run_pipeline, run_single_stage  # noqa: E402
from lrl_toolkit.utils import read_json  # noqa: E402


def _project(tmp_path, tiny_model, **overrides):
    data = {
        "name": "m2",
        "language": "testlang",
        "base_model": tiny_model["model"],
        "compute": "consumer_gpu",
        "workdir": str(tmp_path / "wd"),
        "ingest": {"sources": ["local"]},
        "clean": {"lang_id": "none", "dedup": "exact", "min_quality": 0.3},
        "tokenizer": {"strategy": "extend", "added_tokens": 50},
        "pretrain": {"method": "qlora", "max_steps": 2, "seq_len": 32},
    }
    data.update(overrides)
    p = tmp_path / "m2.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_project(p)


def test_tokenizer_extends_and_reports_fertility(tmp_path, tiny_model):
    proj = _project(tmp_path, tiny_model)
    run_pipeline(proj, stages=["ingest", "clean", "tokenizer"])
    card = read_json(proj.stage_dir("tokenizer") / "tokenizer_card.json")
    assert card["tokens_added"] >= 1
    assert card["final_vocab_size"] > card["base_vocab_size"]
    # Fertility is measured and reported for both (direction depends on the base;
    # the real improvement is asserted in the live SmolLM2 verification).
    assert card["fertility"]["base"] > 0
    assert card["fertility"]["extended"] > 0
    assert (proj.stage_dir("tokenizer") / "tokenizer").is_dir()


def test_pretrain_falls_back_to_lora_on_cpu_and_trains(tmp_path, tiny_model):
    proj = _project(tmp_path, tiny_model)
    run_pipeline(proj, stages=["ingest", "clean", "tokenizer"])
    outcome = run_single_stage(proj, "pretrain")
    assert outcome.status == "ran"

    card = read_json(proj.stage_dir("pretrain") / "pretrain_card.json")
    # QLoRA requested, but on CPU it must fall back to LoRA and still train.
    assert card["method_requested"] == "qlora"
    assert card["method_used"] == "lora"
    assert card["steps"] == 2
    assert card["train_loss"] > 0
    assert (proj.stage_dir("pretrain") / "adapter").is_dir()


def test_full_eight_stage_pipeline_with_tiny_model(tmp_path, tiny_model):
    proj = _project(tmp_path, tiny_model)
    outcomes = run_pipeline(proj)
    assert [o.stage for o in outcomes] == list(proj.config.selected_stages())
    assert all(o.status == "ran" for o in outcomes)
    # Real adapter from pretrain + auto-generated model card from export.
    assert (proj.stage_dir("pretrain") / "adapter").is_dir()
    assert (proj.stage_dir("export") / "MODEL_CARD.md").is_file()
