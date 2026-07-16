"""Conversational-data generation (offline, mock teacher — no torch needed)."""

import json

import yaml

from lrl_toolkit.config import load_project
from lrl_toolkit.convdata import review as review_mod
from lrl_toolkit.convdata.schema import instruction_of, read_jsonl, to_messages
from lrl_toolkit.convdata.teacher import get_teacher
from lrl_toolkit.convdata.translators import available_backends, get_translator
from lrl_toolkit.pipeline import run_pipeline
from lrl_toolkit.registry import load_language


def test_mock_teacher_generate_and_translate():
    t = get_teacher("mock")
    pairs = t.generate_pairs(5, "Welsh")
    assert len(pairs) == 5
    assert all(p["instruction"] and p["response"] for p in pairs)
    assert t.translate("hello world", "Welsh").startswith("[Welsh]")


def test_proprietary_providers_are_rejected():
    import pytest

    for provider in ("claude", "openai", "gemini"):
        with pytest.raises(ValueError):
            get_teacher(provider)


def test_translator_backends_registered_and_mock_works():
    assert set(available_backends()) >= {"nllb", "m2m100", "opusmt", "madlad", "teacher", "mock"}
    welsh = load_language("welsh")
    assert welsh.nllb_code == "cym_Latn"
    out = get_translator("mock").translate("hello world", welsh)
    assert out.startswith("[Welsh]")


def test_unknown_translator_backend_raises():
    import pytest

    with pytest.raises(ValueError):
        get_translator("googletranslate")


def test_schema_and_review_flow():
    msgs = to_messages("q", "a", system="s")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]

    pairs = [{"messages": to_messages("q1", "a1"), "source": "synth", "meta": {}}]
    queue = review_mod.build_queue(pairs)
    assert queue[0]["status"] == "pending"
    # Review disabled -> everything accepted; enabled -> only accepted items.
    assert len(review_mod.accepted_pairs(queue, review_enabled=False)) == 1
    assert len(review_mod.accepted_pairs(queue, review_enabled=True)) == 0
    queue[0]["status"] = "accepted"
    assert len(review_mod.accepted_pairs(queue, review_enabled=True)) == 1


def _project(tmp_path, offline_configs, convdata):
    data = {
        "name": "cd",
        "language": "testlang",
        "base_model": "smollm2-135m",
        "compute": "consumer_gpu",
        "workdir": str(tmp_path / "wd"),
        "ingest": {"sources": ["local"]},
        "clean": {"lang_id": "none", "dedup": "exact"},
        "convdata": convdata,
    }
    p = tmp_path / "cd.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_project(p)


def test_convdata_synth_mock_offline(tmp_path, offline_configs):
    proj = _project(
        tmp_path,
        offline_configs,
        {
            "provider": "mock",
            "translate": [],
            "synth": {"provider": "mock", "n": 6},
            "review": False,
        },
    )
    run_pipeline(proj, stages=["ingest", "clean", "convdata"])
    accepted = read_jsonl(proj.stage_dir("convdata") / "accepted.jsonl")
    assert len(accepted) == 6
    assert all(instruction_of(p) for p in accepted)


def test_convdata_translate_local_jsonl(tmp_path, offline_configs):
    seed = tmp_path / "seed.jsonl"
    with seed.open("w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps({"instruction": f"Q{i}", "response": f"A{i}"}) + "\n")

    proj = _project(
        tmp_path,
        offline_configs,
        {
            "translate": [str(seed)],
            "translate_limit": 10,
            "translate_backend": "mock",
            "review": False,
        },
    )
    run_pipeline(proj, stages=["ingest", "clean", "convdata"])
    accepted = read_jsonl(proj.stage_dir("convdata") / "accepted.jsonl")
    assert len(accepted) == 3
    # Mock translation tags the target language.
    assert instruction_of(accepted[0]).startswith("[Test Language]")


def test_convdata_native_set_used_as_is_and_excludes_flores(tmp_path, offline_configs):
    # Native target-language instruction data with a `dataset` provenance column.
    native = tmp_path / "native.jsonl"
    with native.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"inputs": "Pirs 1", "targets": "Bersiv 1", "dataset": "wikiqa"}) + "\n")
        fh.write(json.dumps({"inputs": "Pirs 2", "targets": "Bersiv 2", "dataset": "wikiqa"}) + "\n")
        # A FLORES-derived row that must be filtered out of training.
        fh.write(json.dumps({"inputs": "MT src", "targets": "MT tgt", "dataset": "flores200"}) + "\n")

    proj = _project(
        tmp_path,
        offline_configs,
        {
            "translate": [],
            "native_sets": [
                {"repo": str(native), "exclude": ["flores"], "limit": 10}
            ],
            "review": False,
        },
    )
    run_pipeline(proj, stages=["ingest", "clean", "convdata"])
    accepted = read_jsonl(proj.stage_dir("convdata") / "accepted.jsonl")
    # Two wikiqa rows kept, flores row dropped.
    assert len(accepted) == 2
    instructions = {instruction_of(p) for p in accepted}
    assert instructions == {"Pirs 1", "Pirs 2"}
    # Used as-is: NOT machine-translated (no "[Test Language]" prefix).
    assert all(not i.startswith("[") for i in instructions)
