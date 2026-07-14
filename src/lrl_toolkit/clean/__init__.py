"""Clean stage: normalize, language-filter, dedup, quality-filter, and scrub PII.

Reads the monolingual corpus produced by ``ingest``, runs each document through
the filter chain, writes the surviving cleaned documents as new shards, and emits
a real data card (source mix, token counts, and per-reason drop counts) — a
first-class deliverable per DATA_ETHICS.md.
"""

from __future__ import annotations

from ..corpus import ShardWriter, iter_documents
from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json
from . import filters, pii
from .dedup import Deduper
from .langid import LanguageIdentifier
from .normalize import normalize_text

log = get_logger("lrl.clean")

__all__ = ["CleanStage"]


class CleanStage(Stage):
    name = "clean"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.clean
        lang = ctx.project.language_profile
        out_dir = ctx.stage_dir(self.name)

        ingest_corpus = ctx.project.stage_dir("ingest") / "corpus"
        writer = ShardWriter(out_dir / "corpus", prefix="clean")

        identifier = LanguageIdentifier(cfg.lang_id.value)
        deduper = Deduper(cfg.dedup.value)

        stats = {
            "docs_in": 0,
            "docs_out": 0,
            "tokens_out": 0,
            "chars_out": 0,
            "dropped": {},
            "pii_redactions": 0,
            "per_source": {},
        }

        def drop(reason: str) -> None:
            stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1

        if ingest_corpus.exists():
            for doc in iter_documents(ingest_corpus):
                stats["docs_in"] += 1
                text = normalize_text(doc.text, lang.normalization)
                if not text:
                    drop("empty_after_normalize")
                    continue

                # Language filter: drop only when we're confident it's another lang.
                if identifier.available:
                    iso, prob = identifier.predict(text)
                    if iso and iso != lang.iso639_3 and prob >= cfg.min_doc_lang_prob:
                        drop("wrong_language")
                        continue

                q = filters.assess(text, min_quality=cfg.min_quality)
                if not q.keep:
                    drop(q.reason or "low_quality")
                    continue

                if deduper.is_duplicate(text):
                    drop("duplicate")
                    continue

                if cfg.scrub_pii:
                    text, n = pii.scrub(text)
                    stats["pii_redactions"] += n

                doc.text = text
                writer.write(doc)
                stats["docs_out"] += 1
                stats["chars_out"] += len(text)
                stats["tokens_out"] += len(text.split())
                src = doc.source
                stats["per_source"][src] = stats["per_source"].get(src, 0) + 1
        else:
            log.warning("[clean] no ingest corpus found at %s", ingest_corpus)

        writer.close()

        data_card = {
            "language": lang.name,
            "iso639_3": lang.iso639_3,
            "scripts": lang.scripts,
            "normalization": lang.normalization.model_dump(mode="json"),
            "plan": {
                "lang_id": cfg.lang_id.value,
                "lang_id_available": identifier.available,
                "min_doc_lang_prob": cfg.min_doc_lang_prob,
                "dedup": cfg.dedup.value,
                "min_quality": cfg.min_quality,
                "scrub_pii": cfg.scrub_pii,
            },
            "stats": stats,
        }
        card_path = write_json(out_dir / "data_card.json", data_card)
        log.info(
            "[clean] in=%d out=%d dropped=%s",
            stats["docs_in"],
            stats["docs_out"],
            stats["dropped"],
        )

        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={
                "docs_in": stats["docs_in"],
                "docs_out": stats["docs_out"],
                "tokens_out": stats["tokens_out"],
            },
        )
