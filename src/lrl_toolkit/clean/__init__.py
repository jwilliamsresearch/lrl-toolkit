"""Clean stage: boilerplate strip, language ID, normalization, dedup, PII scrub.

M0 status: scaffolding — emits a data card describing the configured cleaning
plan. M1 implements the real filters (GlotLID/fastText language ID, MinHash
dedup, script normalization from the language profile, PII scrubbing).
"""

from __future__ import annotations

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json

log = get_logger("lrl.clean")


class CleanStage(Stage):
    name = "clean"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.clean
        lang = ctx.project.language_profile
        out_dir = ctx.stage_dir(self.name)

        # The data card is a first-class deliverable (see DATA_ETHICS.md): it
        # reports the source mix and filter behavior for every model built.
        data_card = {
            "language": lang.name,
            "scripts": lang.scripts,
            "normalization": lang.normalization.model_dump(mode="json"),
            "plan": {
                "lang_id": cfg.lang_id.value,
                "min_doc_lang_prob": cfg.min_doc_lang_prob,
                "dedup": cfg.dedup.value,
                "min_quality": cfg.min_quality,
                "strip_boilerplate": cfg.strip_boilerplate,
                "scrub_pii": cfg.scrub_pii,
            },
            "stats": {
                # M1 fills these with real counts.
                "docs_in": None,
                "docs_out": None,
                "tokens_out": None,
                "dropped": {},
            },
            "status": "placeholder",
        }
        card_path = write_json(out_dir / "data_card.json", data_card)
        log.info("[clean] wrote data card: %s", card_path)

        return StageResult(outputs=[ctx.relpath(card_path)], metrics={"status": "placeholder"})
