"""Convdata stage: build conversational/instruction pairs.

Three configurable generators, all emitting a unified ``messages`` schema:
  * translate open instruction datasets into the target language,
  * synthesize native pairs via a configurable teacher LLM (default Claude),
  * route everything through human-in-the-loop review before training.

M0 status: scaffolding — records the plan and expected counts. M3 implements the
generators, the back-translation/chrF quality filter, and the review queue
(surfaced in the dashboard).
"""

from __future__ import annotations

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json

log = get_logger("lrl.convdata")


class ConvDataStage(Stage):
    name = "convdata"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.convdata
        out_dir = ctx.stage_dir(self.name)

        plan = {
            "translate_datasets": cfg.translate,
            "synth": cfg.synth.model_dump(mode="json") if cfg.synth else None,
            "review": cfg.review,
            "schema": "messages",  # ChatML/ShareGPT-compatible
            "counts": {"translated": None, "synthetic": None, "accepted": None},
            "status": "placeholder",
        }
        card_path = write_json(out_dir / "convdata_card.json", plan)
        synth_n = cfg.synth.n if cfg.synth else 0
        log.info(
            "[convdata] translate=%s synth_n=%s review=%s",
            cfg.translate,
            synth_n,
            cfg.review,
        )

        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={"translate": len(cfg.translate), "synth_n": synth_n, "review": cfg.review},
        )
