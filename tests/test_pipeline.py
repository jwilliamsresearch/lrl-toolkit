"""End-to-end pipeline behavior: artifacts, manifest hits, and the license gate."""

import yaml

from lrl_toolkit.config import load_project
from lrl_toolkit.export import LicenseGateError
from lrl_toolkit.manifest import Manifest
from lrl_toolkit.pipeline import run_pipeline


def _project(tmp_path, **overrides):
    data = {
        "name": "t",
        "language": "welsh",
        "base_model": "smollm2-135m",
        "compute": "consumer_gpu",
        "workdir": str(tmp_path / "wd"),
    }
    data.update(overrides)
    p = tmp_path / "project.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_project(p)


def test_full_pipeline_runs_and_produces_artifacts(tmp_path):
    proj = _project(tmp_path, ingest={"sources": ["wikipedia"]})
    outcomes = run_pipeline(proj)
    assert [o.stage for o in outcomes] == list(proj.config.selected_stages())
    assert all(o.status == "ran" for o in outcomes)

    # Every stage wrote at least one artifact under its stage dir.
    for stage in proj.config.selected_stages():
        assert proj.stage_dir(stage).exists()
    assert (proj.stage_dir("clean") / "data_card.json").is_file()
    assert (proj.stage_dir("export") / "MODEL_CARD.md").is_file()

    # Manifest persisted with a record per stage.
    manifest = Manifest.load(proj.workdir, proj.name)
    assert set(manifest.stages) == set(proj.config.selected_stages())


def test_rerun_is_all_manifest_hits(tmp_path):
    proj = _project(tmp_path, ingest={"sources": ["wikipedia"]})
    run_pipeline(proj)
    second = run_pipeline(proj)
    assert all(o.status == "skipped" for o in second)


def test_config_change_invalidates_downstream(tmp_path):
    proj = _project(tmp_path, ingest={"sources": ["wikipedia"]})
    run_pipeline(proj)

    # Change a mid-pipeline stage's config; it and everything after must re-run,
    # earlier stages stay cached.
    proj2 = _project(
        tmp_path, ingest={"sources": ["wikipedia"]}, tokenizer={"added_tokens": 4321}
    )
    outcomes = {o.stage: o.status for o in run_pipeline(proj2)}
    assert outcomes["ingest"] == "skipped"
    assert outcomes["clean"] == "skipped"
    assert outcomes["tokenizer"] == "ran"
    assert outcomes["pretrain"] == "ran"
    assert outcomes["export"] == "ran"


def test_export_license_gate_blocks_unresolved(tmp_path):
    # Cornish's 'local' source carries an unknown license -> export must refuse.
    proj = _project(tmp_path, language="cornish", ingest={"sources": ["local"]})
    try:
        run_pipeline(proj)
        raised = False
    except LicenseGateError:
        raised = True
    assert raised, "export should refuse when a source license is unresolved"


def test_export_license_gate_passes_when_resolved(tmp_path):
    proj = _project(tmp_path, language="welsh", ingest={"sources": ["wikipedia"]})
    outcomes = run_pipeline(proj)
    assert outcomes[-1].stage == "export"
    assert outcomes[-1].status == "ran"
