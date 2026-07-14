"""Pretrain stage: continued causal-LM pretraining on the clean corpus.

Delegates the heavy lifting to :mod:`lrl_toolkit.pretrain.train`. If there is no
clean corpus yet, writes a plan-only card rather than failing, keeping the stage
safe to run in a dry pipeline.
"""

from __future__ import annotations

import os

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json

log = get_logger("lrl.pretrain")

__all__ = ["PretrainStage"]


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class PretrainStage(Stage):
    name = "pretrain"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.pretrain
        compute = ctx.project.compute_profile
        model = ctx.project.model_profile
        out_dir = ctx.stage_dir(self.name)
        corpus_dir = ctx.project.stage_dir("clean") / "corpus"

        if not corpus_dir.exists() or not any(corpus_dir.iterdir()):
            log.warning("[pretrain] no clean corpus at %s; writing plan only.", corpus_dir)
            card_path = write_json(
                out_dir / "pretrain_card.json",
                {"base_model": model.hf_id, "method": cfg.method.value, "status": "no_corpus"},
            )
            return StageResult(outputs=[ctx.relpath(card_path)], metrics={"status": "no_corpus"})

        # Prefer the extended tokenizer from the tokenizer stage if present.
        tok_dir = ctx.project.stage_dir("tokenizer") / "tokenizer"
        seq_len = cfg.seq_len
        if compute.max_seq_len_cap:
            seq_len = min(seq_len, compute.max_seq_len_cap)

        from .train import run_pretraining

        report = run_pretraining(
            base_id=model.hf_id,
            tokenizer_dir=tok_dir,
            corpus_dir=corpus_dir,
            out_dir=out_dir,
            method=cfg.method.value,
            compute=compute,
            seq_len=seq_len,
            epochs=cfg.epochs,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            seed=ctx.project.config.seed,
            token=_hf_token(),
        )
        card_path = write_json(out_dir / "pretrain_card.json", report)
        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={
                "method_used": report["method_used"],
                "steps": report["steps"],
                "train_loss": report["train_loss"],
            },
        )
