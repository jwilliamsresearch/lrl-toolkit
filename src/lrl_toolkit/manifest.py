"""Project manifest: artifact tracking, provenance, and resume/skip logic.

The manifest is a small JSON file under ``<workdir>/.lrl/manifest.json`` that
records, for each completed stage, a *fingerprint* of the inputs that produced
it. On re-run, a stage whose fingerprint is unchanged is skipped (a "manifest
hit"), giving Make/DVC-like incremental behavior without a heavy dependency.

It also aggregates :class:`ProvenanceRecord` entries emitted by ingest
connectors so that ``export`` can enforce the license-resolution gate described
in DATA_ETHICS.md.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_fingerprint(payload: dict) -> str:
    """Deterministic short hash of a JSON-serializable payload."""
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class ProvenanceRecord(BaseModel):
    """Where a slice of corpus data came from and under what license."""

    source: str = Field(..., description="Connector / dataset name.")
    url: str | None = None
    retrieved_at: str = Field(default_factory=_now)
    license: str | None = Field(default=None, description="SPDX id or free text; None = unknown.")
    attribution: str | None = None
    n_docs: int | None = None
    notes: str | None = None

    model_config = {"extra": "forbid"}

    @property
    def license_resolved(self) -> bool:
        return bool(self.license) and self.license.lower() not in {"unknown", "unresolved"}


class StageRecord(BaseModel):
    """Bookkeeping for one completed stage."""

    stage: str
    fingerprint: str
    completed_at: str = Field(default_factory=_now)
    outputs: list[str] = Field(default_factory=list, description="Artifact paths (relative).")
    metrics: dict = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class Manifest(BaseModel):
    """The persisted state of a project's pipeline run."""

    project: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    # ---- persistence ---------------------------------------------------- #
    @staticmethod
    def path_for(workdir: Path) -> Path:
        return workdir / ".lrl" / "manifest.json"

    @classmethod
    def load(cls, workdir: Path, project: str) -> Manifest:
        p = cls.path_for(workdir)
        if p.is_file():
            with p.open("r", encoding="utf-8") as fh:
                return cls.model_validate_json(fh.read())
        return cls(project=project)

    def save(self, workdir: Path) -> None:
        p = self.path_for(workdir)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        with p.open("w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))

    # ---- resume / skip -------------------------------------------------- #
    def is_current(self, stage: str, fingerprint: str) -> bool:
        rec = self.stages.get(stage)
        return rec is not None and rec.fingerprint == fingerprint

    def record_stage(
        self,
        stage: str,
        fingerprint: str,
        outputs: list[str] | None = None,
        metrics: dict | None = None,
    ) -> StageRecord:
        rec = StageRecord(
            stage=stage,
            fingerprint=fingerprint,
            outputs=outputs or [],
            metrics=metrics or {},
        )
        self.stages[stage] = rec
        return rec

    def invalidate_from(self, stage: str, stage_order: tuple[str, ...]) -> None:
        """Drop records for ``stage`` and every stage after it (a re-run forces
        downstream stages to recompute)."""
        if stage not in stage_order:
            return
        idx = stage_order.index(stage)
        for downstream in stage_order[idx:]:
            self.stages.pop(downstream, None)

    # ---- provenance / license gate ------------------------------------- #
    def add_provenance(self, record: ProvenanceRecord) -> None:
        self.provenance.append(record)

    def unresolved_licenses(self) -> list[ProvenanceRecord]:
        return [r for r in self.provenance if not r.license_resolved]
