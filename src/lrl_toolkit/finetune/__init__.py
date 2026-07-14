"""Finetune stage: instruction fine-tuning (SFT) on accepted conversational pairs.

Delegates to :mod:`lrl_toolkit.finetune.train`. Writes a plan-only card if there
is no accepted data yet (e.g. review is enabled but nothing has been accepted).
"""

from __future__ import annotations

import os

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json

log = get_logger("lrl.finetune")

__all__ = ["FinetuneStage"]


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class FinetuneStage(Stage):
    name = "finetune"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.finetune
        compute = ctx.project.compute_profile
        model = ctx.project.model_profile
        out_dir = ctx.stage_dir(self.name)

        pairs_path = ctx.project.stage_dir("convdata") / "accepted.jsonl"
        if not pairs_path.exists() or pairs_path.stat().st_size == 0:
            log.warning("[finetune] no accepted pairs at %s; writing plan only.", pairs_path)
            card_path = write_json(
                out_dir / "finetune_card.json",
                {"base_model": model.hf_id, "method": cfg.method.value, "status": "no_data"},
            )
            return StageResult(outputs=[ctx.relpath(card_path)], metrics={"status": "no_data"})

        tok_dir = ctx.project.stage_dir("tokenizer") / "tokenizer"
        pretrain_adapter = ctx.project.stage_dir("pretrain") / "adapter"

        from .train import run_sft

        report = run_sft(
            base_id=model.hf_id,
            tokenizer_dir=tok_dir,
            pretrain_adapter_dir=pretrain_adapter,
            pairs_path=pairs_path,
            out_dir=out_dir,
            method=cfg.method.value,
            compute=compute,
            max_seq_len=cfg.max_seq_len,
            epochs=cfg.epochs,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            seed=ctx.project.config.seed,
            token=_hf_token(),
        )
        card_path = write_json(out_dir / "finetune_card.json", report)
        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={
                "method_used": report["method_used"],
                "steps": report["steps"],
                "train_loss": report["train_loss"],
            },
        )
