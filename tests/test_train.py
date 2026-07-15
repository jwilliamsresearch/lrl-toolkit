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
        "convdata": {
            "provider": "mock",
            "translate": [],
            "synth": {"provider": "mock", "n": 8},
            "review": False,
        },
        "finetune": {"method": "qlora", "max_steps": 2, "max_seq_len": 64},
        "evaluate": {"benchmarks": ["perplexity"]},
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


def test_pretrain_falls_back_to_lora_without_4bit_and_trains(tmp_path, tiny_model, monkeypatch):
    # Force the no-4bit path so the fallback is exercised deterministically,
    # regardless of whether the test host has a CUDA GPU + bitsandbytes.
    monkeypatch.setattr("lrl_toolkit.pretrain.train._can_use_4bit", lambda: False)
    proj = _project(tmp_path, tiny_model)
    run_pipeline(proj, stages=["ingest", "clean", "tokenizer"])
    outcome = run_single_stage(proj, "pretrain")
    assert outcome.status == "ran"

    card = read_json(proj.stage_dir("pretrain") / "pretrain_card.json")
    # QLoRA requested, but without 4-bit it must fall back to LoRA and still train.
    assert card["method_requested"] == "qlora"
    assert card["method_used"] == "lora"
    assert card["steps"] == 2
    assert card["train_loss"] > 0
    assert (proj.stage_dir("pretrain") / "adapter").is_dir()


def test_sft_finetune_trains_on_mock_pairs(tmp_path, tiny_model, monkeypatch):
    # `finetune.train` imports `_can_use_4bit` into its own namespace, so patch it there.
    monkeypatch.setattr("lrl_toolkit.finetune.train._can_use_4bit", lambda: False)
    proj = _project(tmp_path, tiny_model)
    run_pipeline(proj, stages=["ingest", "clean", "tokenizer", "pretrain", "convdata"])
    outcome = run_single_stage(proj, "finetune")
    assert outcome.status == "ran"
    card = read_json(proj.stage_dir("finetune") / "finetune_card.json")
    assert card["method_used"] == "lora"  # QLoRA -> LoRA without 4-bit
    assert card["steps"] == 2
    assert card["n_examples"] == 8
    assert (proj.stage_dir("finetune") / "adapter").is_dir()


def test_evaluate_computes_perplexity(tmp_path, tiny_model):
    proj = _project(tmp_path, tiny_model)
    run_pipeline(proj, stages=["ingest", "clean", "tokenizer", "pretrain"])
    run_single_stage(proj, "evaluate")
    report = read_json(proj.stage_dir("evaluate") / "report_card.json")
    ppl = report["results"]["perplexity"]
    assert isinstance(ppl["value"], float) and ppl["value"] > 0


def test_export_merges_adapter_and_writes_card(tmp_path, tiny_model):
    proj = _project(tmp_path, tiny_model)
    run_pipeline(
        proj, stages=["ingest", "clean", "tokenizer", "pretrain", "convdata", "finetune"]
    )
    run_single_stage(proj, "export")
    assert (proj.stage_dir("export") / "merged").is_dir()
    assert (proj.stage_dir("export") / "MODEL_CARD.md").is_file()
    assert (proj.stage_dir("export") / "Modelfile").is_file()
    export_card = read_json(proj.stage_dir("export") / "export_card.json")
    # GGUF conversion is best-effort; without llama.cpp it is reported as skipped.
    assert export_card["packaging"]["gguf"]["status"] in {"skipped", "available", "not_requested"}


def test_full_eight_stage_pipeline_with_tiny_model(tmp_path, tiny_model):
    proj = _project(tmp_path, tiny_model)
    outcomes = run_pipeline(proj)
    assert [o.stage for o in outcomes] == list(proj.config.selected_stages())
    assert all(o.status == "ran" for o in outcomes)
    # Real adapters from pretrain + SFT, and a merged export.
    assert (proj.stage_dir("pretrain") / "adapter").is_dir()
    assert (proj.stage_dir("finetune") / "adapter").is_dir()
    assert (proj.stage_dir("export") / "merged").is_dir()
    assert (proj.stage_dir("export") / "MODEL_CARD.md").is_file()


# --------------------------------------------------------------------------- #
# Smart embedding init + Unsloth backend
# --------------------------------------------------------------------------- #
def test_smart_embed_init_sets_mean_of_base_subwords(tiny_model):
    """A new token is re-initialized to the mean of its base-tokenizer pieces."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lrl_toolkit.pretrain.embed_init import smart_init_new_embeddings

    path = str(tiny_model["model_path"])
    base_tok = AutoTokenizer.from_pretrained(path)
    ext_tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    old_num = model.get_input_embeddings().weight.shape[0]

    assert ext_tok.add_tokens(["abcde"]) == 1
    model.resize_token_embeddings(len(ext_tok))
    new_id = ext_tok.convert_tokens_to_ids("abcde")

    # Reproduce the function's own decomposition to know the expected mean.
    surface = ext_tok.convert_tokens_to_string([ext_tok.convert_ids_to_tokens(new_id)])
    skip = {i for i in base_tok.all_special_ids if i is not None}
    pieces = [
        i
        for i in base_tok(surface, add_special_tokens=False).input_ids
        if i < old_num and i not in skip
    ]
    if not pieces:
        pytest.skip("surface did not decompose into known base pieces on this tiny tokenizer")

    # Corrupt the new row so we can prove it was overwritten.
    with torch.no_grad():
        model.get_input_embeddings().weight[new_id] = 123.456

    n = smart_init_new_embeddings(model, base_tok, ext_tok, old_num)
    assert n == 1
    emb = model.get_input_embeddings().weight
    expected = emb[torch.tensor(pieces)].mean(dim=0)
    assert torch.allclose(emb[new_id], expected, atol=1e-5)


def test_pretrain_records_smart_embed_init(tmp_path, tiny_model, monkeypatch):
    monkeypatch.setattr("lrl_toolkit.pretrain.train._can_use_4bit", lambda: False)
    proj = _project(tmp_path, tiny_model)  # tokenizer stage adds 50 tokens
    run_pipeline(proj, stages=["ingest", "clean", "tokenizer"])
    run_single_stage(proj, "pretrain")
    card = read_json(proj.stage_dir("pretrain") / "pretrain_card.json")
    assert card["backend"] == "hf"
    assert card["embed_init"] == "subword_mean"
    assert card["embed_init_tokens"] >= 1


def test_pretrain_default_embed_init_skips_smart_init(tmp_path, tiny_model, monkeypatch):
    monkeypatch.setattr("lrl_toolkit.pretrain.train._can_use_4bit", lambda: False)
    proj = _project(
        tmp_path,
        tiny_model,
        pretrain={"method": "qlora", "max_steps": 2, "seq_len": 32, "embed_init": "default"},
    )
    run_pipeline(proj, stages=["ingest", "clean", "tokenizer"])
    run_single_stage(proj, "pretrain")
    card = read_json(proj.stage_dir("pretrain") / "pretrain_card.json")
    assert card["embed_init"] == "default"
    assert card["embed_init_tokens"] == 0


def test_pretrain_unsloth_failure_falls_back_to_hf(tmp_path, tiny_model, monkeypatch):
    """When Unsloth is requested but its build fails, the run must degrade to HF."""
    monkeypatch.setattr("lrl_toolkit.pretrain.train._can_use_4bit", lambda: False)
    monkeypatch.setattr("lrl_toolkit.pretrain.train._unsloth_available", lambda: True)

    def _boom(**_kwargs):
        raise RuntimeError("unsloth not really installed")

    monkeypatch.setattr("lrl_toolkit.pretrain.train._build_unsloth", _boom)

    proj = _project(tmp_path, tiny_model)
    proj.compute_profile.use_unsloth = True  # trigger the Unsloth branch
    run_pipeline(proj, stages=["ingest", "clean", "tokenizer"])
    outcome = run_single_stage(proj, "pretrain")
    assert outcome.status == "ran"
    card = read_json(proj.stage_dir("pretrain") / "pretrain_card.json")
    assert card["backend"] == "hf"  # fell back cleanly
    assert card["embed_init_tokens"] >= 1  # HF path still smart-inits
