"""Conversational-data generation (offline, mock teacher — no torch needed)."""

import json
from pathlib import Path

import yaml

from lrl_toolkit.config import load_project
from lrl_toolkit.convdata import review as review_mod
from lrl_toolkit.convdata.schema import instruction_of, read_jsonl, to_messages
from lrl_toolkit.convdata.teacher import get_teacher
from lrl_toolkit.convdata.translators import available_backends, get_translator
from lrl_toolkit.pipeline import run_pipeline
from lrl_toolkit.registry import load_language


def test_prompt_teacher_batches_large_requests(monkeypatch):
    """A single completion can't hold e.g. 40 full pairs before truncating; the
    teacher must split large n into multiple batched calls to actually reach n."""
    from lrl_toolkit.convdata.teacher import _GEN_BATCH_SIZE, OllamaTeacher

    teacher = OllamaTeacher(model="fake")
    calls: list[str] = []

    def fake_chat(self, system, user, max_tokens=2048):
        calls.append(user)
        # Simulate a real backend: only ever returns up to _GEN_BATCH_SIZE pairs
        # per completion, regardless of how many were asked for.
        n_in_request = min(_GEN_BATCH_SIZE, 999)
        pairs = [{"instruction": f"q{i}", "response": f"a{i}"} for i in range(n_in_request)]
        import json

        return json.dumps(pairs)

    monkeypatch.setattr(OllamaTeacher, "_chat", fake_chat)

    n = _GEN_BATCH_SIZE * 3  # requires multiple batches to satisfy
    pairs = teacher.generate_pairs(n, "Welsh")
    assert len(pairs) == n
    assert len(calls) >= 3  # had to make multiple requests, not one


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


def test_degenerate_repetition_is_detected():
    from lrl_toolkit.convdata.schema import is_degenerate, pair_is_degenerate

    # The real failure observed in the Kurmanji SFT set.
    assert is_degenerate("Şemzînan zûtirîn zûtirîn zûtirîn zûtirîn zûtirîn zûtirîn")
    assert not is_degenerate("Ez kurdî me û ez li Kurdistanê dijîm.")

    degenerate_pair = {
        "messages": to_messages("Çima?", "zûtirîn zûtirîn zûtirîn zûtirîn zûtirîn zûtirîn")
    }
    normal_pair = {"messages": to_messages("Çima?", "Ji ber ku ew rast e.")}
    assert pair_is_degenerate(degenerate_pair)
    assert not pair_is_degenerate(normal_pair)


def test_convdata_drops_degenerate_pairs(tmp_path, offline_configs):
    proj = _project(
        tmp_path,
        offline_configs,
        {
            "translate": [],
            "native_sets": [
                {
                    "repo": str(_write_jsonl(tmp_path, [
                        {"inputs": "Q1", "targets": "A normal answer."},
                        {"inputs": "Q2", "targets": "bad bad bad bad bad bad bad"},
                    ])),
                    "limit": 10,
                }
            ],
            "review": False,
        },
    )
    run_pipeline(proj, stages=["ingest", "clean", "convdata"])
    accepted = read_jsonl(proj.stage_dir("convdata") / "accepted.jsonl")
    assert len(accepted) == 1
    assert instruction_of(accepted[0]) == "Q1"
    card = read_jsonl(proj.stage_dir("convdata") / "pairs.jsonl")
    assert len(card) == 1  # the degenerate pair never made it past dedup/filtering


def _write_jsonl(tmp_path, rows) -> Path:
    import json as _json

    p = tmp_path / f"native_{len(rows)}_{id(rows)}.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(_json.dumps(r) + "\n")
    return p


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
