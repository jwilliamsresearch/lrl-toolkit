"""Core pipeline behavior (offline, no training deps): manifest + license gate.

These tests avoid the model stages (tokenizer/pretrain/finetune) so they run
without torch; the model path is covered in test_train.py with a tiny model.
"""

import yaml

from lrl_toolkit.config import load_project
from lrl_toolkit.corpus import iter_documents
from lrl_toolkit.export import LicenseGateError
from lrl_toolkit.manifest import Manifest
from lrl_toolkit.pipeline import run_pipeline

# Light stages that need no ML dependencies.
LIGHT = ["ingest", "clean", "evaluate", "export"]


def _project(tmp_path, offline_configs, *, language="testlang", **overrides):
    data = {
        "name": "t",
        "language": language,
        "base_model": "smollm2-135m",
        "compute": "consumer_gpu",
        "workdir": str(tmp_path / "wd"),
        "ingest": {"sources": ["local"]},
        "clean": {"lang_id": "none", "dedup": "exact"},
        # Intrinsic-only so the light pipeline stays offline (no benchmark-dataset
        # config lookups over the network).
        "evaluate": {"benchmarks": ["native_cloze", "perplexity"]},
    }
    data.update(overrides)
    p = tmp_path / "project.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_project(p)


def test_light_pipeline_runs_and_produces_artifacts(tmp_path, offline_configs):
    proj = _project(tmp_path, offline_configs)
    outcomes = run_pipeline(proj, stages=LIGHT)
    assert [o.stage for o in outcomes] == LIGHT
    assert all(o.status == "ran" for o in outcomes)
    assert (proj.stage_dir("clean") / "data_card.json").is_file()
    assert (proj.stage_dir("export") / "MODEL_CARD.md").is_file()

    manifest = Manifest.load(proj.workdir, proj.name)
    assert set(manifest.stages) == set(LIGHT)


def test_ingest_and_clean_counts(tmp_path, offline_configs):
    proj = _project(tmp_path, offline_configs)
    run_pipeline(proj, stages=["ingest", "clean"])

    ingested = list(iter_documents(proj.stage_dir("ingest") / "corpus"))
    assert len(ingested) == 5  # 3 good + 1 dup + 1 junk

    cleaned = list(iter_documents(proj.stage_dir("clean") / "corpus"))
    assert len(cleaned) == offline_configs["n_good"]  # dup deduped, junk dropped


def test_rerun_is_all_manifest_hits(tmp_path, offline_configs):
    proj = _project(tmp_path, offline_configs)
    run_pipeline(proj, stages=LIGHT)
    second = run_pipeline(proj, stages=LIGHT)
    assert all(o.status == "skipped" for o in second)


def test_config_change_invalidates_downstream(tmp_path, offline_configs):
    proj = _project(tmp_path, offline_configs)
    run_pipeline(proj, stages=LIGHT)

    proj2 = _project(tmp_path, offline_configs, clean={"lang_id": "none", "min_quality": 0.9})
    outcomes = {o.stage: o.status for o in run_pipeline(proj2, stages=LIGHT)}
    assert outcomes["ingest"] == "skipped"
    assert outcomes["clean"] == "ran"
    assert outcomes["evaluate"] == "ran"
    assert outcomes["export"] == "ran"


def test_export_license_gate_blocks_unresolved(tmp_path, offline_configs):
    proj = _project(tmp_path, offline_configs, language="testlang_unlicensed")
    try:
        run_pipeline(proj, stages=["ingest", "export"])
        raised = False
    except LicenseGateError:
        raised = True
    assert raised, "export should refuse when a source license is unresolved"


def test_export_license_gate_passes_when_resolved(tmp_path, offline_configs):
    proj = _project(tmp_path, offline_configs)
    outcomes = run_pipeline(proj, stages=["ingest", "export"])
    assert outcomes[-1].stage == "export"
    assert outcomes[-1].status == "ran"
