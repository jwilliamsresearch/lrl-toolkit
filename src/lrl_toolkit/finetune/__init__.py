"""Finetune stage: instruction fine-tuning (SFT) with optional preference tuning.

M0 status: scaffolding — records the plan. M3 implements SFT via TRL's
``SFTTrainer`` (LoRA/QLoRA) with the base model's chat template, plus optional
DPO/ORPO on reviewed preference pairs.
"""

from __future__ import annotations

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json

log = get_logger("lrl.finetune")


class FinetuneStage(Stage):
    name = "finetune"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.finetune
        model = ctx.project.model_profile
        out_dir = ctx.stage_dir(self.name)

        plan = {
            "base_model": model.hf_id,
            "method": cfg.method.value,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "dpo": cfg.dpo,
            "status": "placeholder",
        }
        card_path = write_json(out_dir / "finetune_card.json", plan)
        log.info("[finetune] %s via %s (dpo=%s)", model.hf_id, cfg.method.value, cfg.dpo)

        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={"method": cfg.method.value, "dpo": cfg.dpo},
        )
