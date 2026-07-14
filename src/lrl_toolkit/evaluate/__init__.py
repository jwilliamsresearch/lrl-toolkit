"""Evaluate stage: build a model report card.

Computes real held-out perplexity on the trained model. Other benchmarks
(FLORES chrF, Belebele, LLM-as-judge) are declared and reported as ``pending``
until wired up, so the card never overstates what was measured.
"""

from __future__ import annotations

import os

from ..corpus import iter_documents
from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json

log = get_logger("lrl.evaluate")

__all__ = ["EvaluateStage"]

# Benchmarks not yet implemented get an honest 'pending' rather than a fake score.
_PENDING = {"flores", "belebele", "judge", "mmlu"}


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class EvaluateStage(Stage):
    name = "evaluate"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.evaluate
        lang = ctx.project.language_profile
        out_dir = ctx.stage_dir(self.name)
        results: dict[str, object] = {}

        if "perplexity" in cfg.benchmarks:
            results["perplexity"] = self._perplexity(ctx)

        for bench in cfg.benchmarks:
            if bench in _PENDING:
                results[bench] = {"status": "pending", "note": "planned; not yet implemented"}

        report = {
            "language": lang.name,
            "benchmarks": cfg.benchmarks,
            "results": results,
        }
        card_path = write_json(out_dir / "report_card.json", report)
        log.info("[evaluate] results=%s", {k: v for k, v in results.items()})
        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={"perplexity": results.get("perplexity")},
        )

    def _perplexity(self, ctx: StageContext):
        clean_dir = ctx.project.stage_dir("clean") / "corpus"
        if not clean_dir.exists():
            return {"status": "skipped", "note": "no clean corpus"}
        texts = [d.text for d in iter_documents(clean_dir)][:200]
        if not texts:
            return {"status": "skipped", "note": "empty corpus"}

        from ..modeling import load_model_and_tokenizer, resolve_artifacts
        from .metrics import perplexity

        tok_dir, adapter, kind = resolve_artifacts(ctx.project)
        model, tokenizer = load_model_and_tokenizer(
            ctx.project.model_profile.hf_id, tok_dir, adapter, token=_hf_token()
        )
        ppl = perplexity(model, tokenizer, texts, seq_len=256, max_blocks=20)
        return {"value": ppl, "model": kind}
