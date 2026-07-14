"""Tokenizer stage: extend the base tokenizer for the target script.

M0 status: scaffolding — records the tokenizer plan and a fertility placeholder.
M2 implements SentencePiece/BPE training on the clean corpus, merging new pieces
into the base tokenizer, resizing model embeddings (mean-init), and reporting
tokenizer fertility before/after — the key LRL efficiency win.
"""

from __future__ import annotations

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json

log = get_logger("lrl.tokenizer")


class TokenizerStage(Stage):
    name = "tokenizer"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.tokenizer
        model = ctx.project.model_profile
        out_dir = ctx.stage_dir(self.name)

        card = {
            "strategy": cfg.strategy.value,
            "added_tokens": cfg.added_tokens,
            "base_tokenizer": model.hf_id,
            "tokenizer_type": model.tokenizer_type,
            "fertility": {"base": None, "extended": None},  # M2 fills these in.
            "status": "placeholder",
        }
        card_path = write_json(out_dir / "tokenizer_card.json", card)
        log.info("[tokenizer] strategy=%s added_tokens=%s", cfg.strategy.value, cfg.added_tokens)

        return StageResult(
            outputs=[ctx.relpath(card_path)], metrics={"strategy": cfg.strategy.value}
        )
