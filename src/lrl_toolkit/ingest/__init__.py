"""Ingest stage: fetch raw corpora from configured sources.

M0 status: scaffolding. This stage currently emits provenance records and a
placeholder corpus manifest derived from the language profile's source catalog,
which exercises the pipeline and the license-tracking machinery end to end.

M1 fills in real connectors (Wikipedia, OSCAR/HF, OPUS, Leipzig, local files)
behind the ``BaseConnector`` interface below.
"""

from __future__ import annotations

import abc

from ..manifest import ProvenanceRecord
from ..pipeline.base import Stage, StageContext, StageResult
from ..registry import SourceHint
from ..utils import get_logger, write_json

log = get_logger("lrl.ingest")


class BaseConnector(abc.ABC):
    """Interface every source connector implements (filled in from M1)."""

    name: str

    @abc.abstractmethod
    def fetch(self, hint: SourceHint, out_dir, max_docs: int | None) -> ProvenanceRecord:
        """Fetch data for one source and return its provenance record."""
        raise NotImplementedError


class IngestStage(Stage):
    name = "ingest"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.ingest
        lang = ctx.project.language_profile
        out_dir = ctx.stage_dir(self.name)

        # Decide which sources to pull: explicit config subset, else the whole
        # language source catalog.
        catalog: list[SourceHint] = lang.sources
        if cfg.sources:
            catalog = [s for s in catalog if s.connector in cfg.sources]

        provenance: list[ProvenanceRecord] = []
        for hint in catalog:
            # M1: dispatch to the real connector here. For now, record intent +
            # provenance with an as-yet-unresolved license so the export gate
            # is meaningfully exercised.
            license_ = hint.params.get("license")
            provenance.append(
                ProvenanceRecord(
                    source=hint.connector,
                    url=hint.params.get("url"),
                    license=license_,
                    notes=hint.notes or "M0 placeholder ingest (no data fetched yet).",
                )
            )
            log.info("[ingest] planned source: %s (license=%s)", hint.connector, license_)

        card = {
            "language": lang.name,
            "script": lang.resolved_script(),
            "planned_sources": [h.connector for h in catalog],
            "max_gb": cfg.max_gb,
            "status": "placeholder",
        }
        card_path = write_json(out_dir / "ingest_card.json", card)

        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={"n_sources": len(catalog)},
            provenance=provenance,
        )
