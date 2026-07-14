"""Evaluate stage: build a model report card.

M0 status: scaffolding — records the requested benchmarks and an empty report
card. M4 wires the real metrics: held-out perplexity, chrF/BLEU on FLORES-200,
Belebele and translated-MMLU via lm-eval-harness, and LLM-as-judge chat scoring.
"""

from __future__ import annotations

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json

log = get_logger("lrl.evaluate")


class EvaluateStage(Stage):
    name = "evaluate"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.evaluate
        lang = ctx.project.language_profile
        out_dir = ctx.stage_dir(self.name)

        report = {
            "language": lang.name,
            "benchmarks": cfg.benchmarks,
            "results": {b: None for b in cfg.benchmarks},  # M4 fills these in.
            "status": "placeholder",
        }
        card_path = write_json(out_dir / "report_card.json", report)
        log.info("[evaluate] benchmarks=%s", cfg.benchmarks)

        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={"n_benchmarks": len(cfg.benchmarks)},
        )
