"""Ingest stage: fetch raw corpora from configured sources.

Dispatches each source in the language profile's catalog to its connector,
streams the returned documents into gzipped JSONL shards (monolingual text under
``ingest/corpus/``, aligned pairs under ``ingest/parallel/``), enforces per-source
and total size limits, and records a :class:`ProvenanceRecord` per source so the
export license gate has something to check.

A failing or gated source is logged and recorded but does not abort the run —
one dead source should not sink a multi-source ingest.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..corpus import ShardWriter
from ..manifest import ProvenanceRecord
from ..pipeline.base import Stage, StageContext, StageResult
from ..registry import SourceHint
from ..utils import get_logger, write_json
from .base import BaseConnector, MissingDependencyError
from .registry import available_connectors, get_connector

log = get_logger("lrl.ingest")

__all__ = ["IngestStage", "BaseConnector", "available_connectors", "get_connector"]


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class IngestStage(Stage):
    name = "ingest"

    def _catalog(self, ctx: StageContext) -> list[SourceHint]:
        cfg = ctx.project.config.ingest
        catalog = ctx.project.language_profile.sources
        if cfg.sources:
            catalog = [s for s in catalog if s.connector in cfg.sources]
        return catalog

    def fingerprint_payload(self, project) -> dict:
        payload = super().fingerprint_payload(project)
        # Make local-file sources reproducible: include file signatures.
        sigs = []
        for hint in project.language_profile.sources:
            if hint.connector == "local" and hint.params.get("path"):
                root = Path(hint.params["path"])
                if root.exists():
                    for p in sorted(root.rglob("*") if root.is_dir() else [root]):
                        if p.is_file():
                            st = p.stat()
                            sigs.append([str(p), st.st_size, int(st.st_mtime)])
        payload["local_signatures"] = sigs
        return payload

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.ingest
        out_dir = ctx.stage_dir(self.name)
        corpus_dir = out_dir / "corpus"
        parallel_dir = out_dir / "parallel"
        token = _hf_token()

        max_bytes = int(cfg.max_gb * 1e9) if cfg.max_gb else None
        catalog = self._catalog(ctx)

        mono_writer = ShardWriter(corpus_dir, prefix="mono")
        par_writer = ShardWriter(parallel_dir, prefix="parallel")
        provenance: list[ProvenanceRecord] = []
        per_source: dict[str, int] = {}
        budget_hit = False

        for hint in catalog:
            try:
                connector = get_connector(hint.connector)
            except ValueError as exc:
                log.warning("[ingest] %s", exc)
                continue

            writer = par_writer if connector.kind == "parallel" else mono_writer
            record = connector.provenance(hint)
            count = 0
            try:
                for doc in connector.iter_documents(
                    hint, max_docs=cfg.max_docs_per_source, token=token
                ):
                    writer.write(doc)
                    count += 1
                    if max_bytes is not None and mono_writer.n_bytes >= max_bytes:
                        budget_hit = True
                        break
                log.info("[ingest] %s: %d docs (%s)", hint.connector, count, connector.kind)
            except PermissionError as exc:
                record.notes = f"GATED — skipped: {exc}"
                log.warning("[ingest] %s gated/skipped: %s", hint.connector, exc)
            except MissingDependencyError as exc:
                record.notes = f"dependency missing — skipped: {exc}"
                log.warning("[ingest] %s skipped: %s", hint.connector, exc)
            except Exception as exc:  # keep the run alive; surface in provenance
                record.notes = f"ERROR — skipped: {type(exc).__name__}: {exc}"
                log.warning("[ingest] %s failed: %s", hint.connector, exc)

            record.n_docs = count
            provenance.append(record)
            per_source[hint.connector] = count
            if budget_hit:
                log.info("[ingest] max_gb budget reached; stopping.")
                break

        mono_writer.close()
        par_writer.close()

        card = {
            "language": ctx.project.language_profile.name,
            "sources": per_source,
            "mono_docs": mono_writer.n_docs,
            "mono_bytes": mono_writer.n_bytes,
            "parallel_docs": par_writer.n_docs,
            "budget_hit": budget_hit,
            "used_token": token is not None,
        }
        card_path = write_json(out_dir / "ingest_card.json", card)

        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={
                "mono_docs": mono_writer.n_docs,
                "parallel_docs": par_writer.n_docs,
                "sources": per_source,
            },
            provenance=provenance,
        )
