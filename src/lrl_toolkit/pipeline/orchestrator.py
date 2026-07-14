"""The pipeline orchestrator: chains stages with manifest-based skip/resume.

Each stage's fingerprint incorporates the *previous* executed stage's
fingerprint, so a change anywhere upstream naturally invalidates everything
downstream (their fingerprints change and they recompute). Unchanged stages are
skipped as "manifest hits".
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import __version__
from ..config import STAGE_ORDER, ResolvedProject
from ..manifest import Manifest, stable_fingerprint
from ..utils import get_logger
from .base import Stage, StageContext

log = get_logger("lrl.pipeline")


def get_stage(name: str) -> Stage:
    """Instantiate a stage by name (lazy import to keep startup light)."""
    if name == "ingest":
        from ..ingest import IngestStage

        return IngestStage()
    if name == "clean":
        from ..clean import CleanStage

        return CleanStage()
    if name == "tokenizer":
        from ..tokenizer import TokenizerStage

        return TokenizerStage()
    if name == "pretrain":
        from ..pretrain import PretrainStage

        return PretrainStage()
    if name == "convdata":
        from ..convdata import ConvDataStage

        return ConvDataStage()
    if name == "finetune":
        from ..finetune import FinetuneStage

        return FinetuneStage()
    if name == "evaluate":
        from ..evaluate import EvaluateStage

        return EvaluateStage()
    if name == "export":
        from ..export import ExportStage

        return ExportStage()
    raise ValueError(f"Unknown stage: {name}")


@dataclass
class StageOutcome:
    stage: str
    status: str  # "ran" | "skipped"
    fingerprint: str
    metrics: dict


def _fingerprint(stage: Stage, project: ResolvedProject, upstream_fp: str | None) -> str:
    payload = stage.fingerprint_payload(project)
    payload["upstream"] = upstream_fp
    payload["toolkit_version"] = __version__
    return stable_fingerprint(payload)


def run_pipeline(
    project: ResolvedProject,
    stages: list[str] | None = None,
    *,
    force: bool = False,
    force_from: str | None = None,
) -> list[StageOutcome]:
    """Run the selected stages in canonical order.

    Args:
        stages: subset to run; defaults to the project's selected stages.
        force: re-run every selected stage even on a manifest hit.
        force_from: re-run from this stage onward (invalidates downstream too).
    """
    project.workdir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(project.workdir, project.name)

    selected = stages if stages is not None else project.config.selected_stages()
    selected = [s for s in STAGE_ORDER if s in selected]  # enforce canonical order

    if force_from:
        manifest.invalidate_from(force_from, STAGE_ORDER)
    force_idx = STAGE_ORDER.index(force_from) if force_from else None

    outcomes: list[StageOutcome] = []
    upstream_fp: str | None = None

    for stage_name in selected:
        stage = get_stage(stage_name)
        fp = _fingerprint(stage, project, upstream_fp)

        forced = force or (force_idx is not None and STAGE_ORDER.index(stage_name) >= force_idx)
        if not forced and manifest.is_current(stage_name, fp):
            log.info("[skip] %s (manifest hit %s)", stage_name, fp)
            metrics = manifest.stages[stage_name].metrics
            outcomes.append(StageOutcome(stage_name, "skipped", fp, metrics))
            upstream_fp = fp
            continue

        log.info("[run ] %s", stage_name)
        ctx = StageContext(project=project, manifest=manifest)
        result = stage.run(ctx)
        for pr in result.provenance:
            manifest.add_provenance(pr)
        manifest.record_stage(stage_name, fp, outputs=result.outputs, metrics=result.metrics)
        manifest.save(project.workdir)
        outcomes.append(StageOutcome(stage_name, "ran", fp, result.metrics))
        upstream_fp = fp

    return outcomes


def run_single_stage(
    project: ResolvedProject, stage_name: str, *, force: bool = False
) -> StageOutcome:
    """Run one stage on its own (used by ``lrl <stage> -c ...``).

    The upstream fingerprint is reconstructed from prior manifest records so a
    standalone run stays consistent with a full-pipeline run.
    """
    manifest = Manifest.load(project.workdir, project.name)
    idx = STAGE_ORDER.index(stage_name)
    upstream_fp = None
    if idx > 0:
        prev = STAGE_ORDER[idx - 1]
        rec = manifest.stages.get(prev)
        upstream_fp = rec.fingerprint if rec else None

    stage = get_stage(stage_name)
    fp = _fingerprint(stage, project, upstream_fp)
    if not force and manifest.is_current(stage_name, fp):
        log.info("[skip] %s (manifest hit %s)", stage_name, fp)
        return StageOutcome(stage_name, "skipped", fp, manifest.stages[stage_name].metrics)

    log.info("[run ] %s", stage_name)
    ctx = StageContext(project=project, manifest=manifest)
    result = stage.run(ctx)
    for pr in result.provenance:
        manifest.add_provenance(pr)
    manifest.record_stage(stage_name, fp, outputs=result.outputs, metrics=result.metrics)
    manifest.save(project.workdir)
    return StageOutcome(stage_name, "ran", fp, result.metrics)
