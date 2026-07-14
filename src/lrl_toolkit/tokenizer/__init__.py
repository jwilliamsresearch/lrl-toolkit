"""Tokenizer stage: extend the base tokenizer for the target script.

Trains a tokenizer on the cleaned corpus, merges novel pieces into the base
tokenizer (the ``extend`` strategy), saves the result, and reports fertility
before/after. Falls back gracefully to a card-only run if no clean corpus exists
yet (e.g. a dry run), so the stage never hard-fails the pipeline.
"""

from __future__ import annotations

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json

log = get_logger("lrl.tokenizer")

__all__ = ["TokenizerStage"]


class TokenizerStage(Stage):
    name = "tokenizer"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.tokenizer
        model = ctx.project.model_profile
        out_dir = ctx.stage_dir(self.name)
        corpus_dir = ctx.project.stage_dir("clean") / "corpus"
        token = _hf_token()

        if not corpus_dir.exists() or not any(corpus_dir.iterdir()):
            log.warning("[tokenizer] no clean corpus at %s; writing plan only.", corpus_dir)
            report = {
                "strategy": cfg.strategy.value,
                "base_tokenizer": model.hf_id,
                "status": "no_corpus",
            }
            card_path = write_json(out_dir / "tokenizer_card.json", report)
            return StageResult(outputs=[ctx.relpath(card_path)], metrics={"status": "no_corpus"})

        from .extend import build_extended_tokenizer

        report = build_extended_tokenizer(
            model.hf_id,
            corpus_dir,
            strategy=cfg.strategy.value,
            added_tokens=cfg.added_tokens,
            out_dir=out_dir / "tokenizer",
            token=token,
        )
        card_path = write_json(out_dir / "tokenizer_card.json", report)
        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={
                "tokens_added": report.get("tokens_added"),
                "fertility": report.get("fertility"),
            },
        )


def _hf_token() -> str | None:
    import os

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
