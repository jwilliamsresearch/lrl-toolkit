"""Stage abstraction shared by every pipeline step.

A :class:`Stage` declares its name, computes a *fingerprint* of the inputs that
determine its output (used by the manifest for skip/resume), and implements
:meth:`run`. Stages communicate only through artifacts on disk plus the
manifest, which keeps them independently runnable and resumable.
"""

from __future__ import annotations

import abc
from pathlib import Path

from pydantic import BaseModel, Field

from ..config import ResolvedProject
from ..manifest import Manifest, ProvenanceRecord


class StageContext(BaseModel):
    """Everything a stage needs at run time."""

    project: ResolvedProject
    manifest: Manifest

    model_config = {"arbitrary_types_allowed": True}

    def stage_dir(self, stage: str) -> Path:
        d = self.project.stage_dir(stage)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def relpath(self, path: Path) -> str:
        """Path relative to the project workdir, for manifest storage."""
        try:
            return str(Path(path).resolve().relative_to(self.project.workdir.resolve()))
        except ValueError:
            return str(path)


class StageResult(BaseModel):
    """What a stage produces: artifact paths, metrics, and any provenance."""

    outputs: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class Stage(abc.ABC):
    """Base class for a pipeline stage."""

    #: One of :data:`lrl_toolkit.config.STAGE_ORDER`.
    name: str

    def stage_config(self, project: ResolvedProject) -> BaseModel:
        return project.config.stage_config(self.name)

    def fingerprint_payload(self, project: ResolvedProject) -> dict:
        """Inputs that determine this stage's output.

        Included in the manifest fingerprint so the stage re-runs when any of
        these change. Subclasses may extend this (e.g. to include a hash of a
        local input file), but should call ``super()`` and merge.
        """
        return {
            "stage": self.name,
            "config": self.stage_config(project).model_dump(mode="json"),
            "language": project.language_profile.model_dump(mode="json"),
            "base_model": project.model_profile.model_dump(mode="json"),
            "compute": project.compute_profile.model_dump(mode="json"),
            "seed": project.config.seed,
        }

    @abc.abstractmethod
    def run(self, ctx: StageContext) -> StageResult:
        """Execute the stage and return its result."""
        raise NotImplementedError
