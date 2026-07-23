"""Evaluate stage: a coverage-aware, base-vs-adapted benchmark report.

Rather than run a fixed list of benchmarks and hope they apply, this stage asks
each benchmark whether it *covers* the target language, runs the covered ones on
both the untouched base model and the adapted model, and reports the **delta** as
the headline (an absolute score conflates what the pipeline added with what the
base already knew). It emits a **coverage matrix** so the (common) case of a
language with no external benchmarks is documented rather than silently skipped —
the corpus-derived intrinsic metrics (native_cloze, perplexity) are the honest
fallback there. See :mod:`lrl_toolkit.evaluate.benchmarks.base` for the rationale.
"""

from __future__ import annotations

import os

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json
from .benchmarks import BenchmarkResult, ModelBundle, get_benchmark
from .benchmarks.base import MetricDirection

log = get_logger("lrl.evaluate")

__all__ = ["EvaluateStage"]


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class EvaluateStage(Stage):
    name = "evaluate"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.evaluate
        lang = ctx.project.language_profile
        out_dir = ctx.stage_dir(self.name)
        limit = getattr(cfg, "limit", 100)
        has_token = _hf_token() is not None

        # Instantiate benchmarks and give each the context it needs.
        prepared = [(name, self._prepare(get_benchmark(name), ctx)) for name in cfg.benchmarks]

        # --- coverage first (cheap; no model load) ------------------------ #
        coverage = {}
        for name, bench in prepared:
            try:
                cov = bench.coverage(lang, has_token=has_token)
            except Exception as exc:  # never let coverage detection sink the stage
                from .benchmarks.base import Coverage

                cov = Coverage(False, f"coverage check failed: {type(exc).__name__}: {exc}")
            coverage[name] = cov
            log.info("[evaluate] coverage %s: available=%s (%s)", name, cov.available, cov.reason)

        any_covered = any(c.available for c in coverage.values())
        need_base = any(
            coverage[name].available and bench.compare_to_base for name, bench in prepared
        )

        # --- load models once, reuse across benchmarks -------------------- #
        adapted_bundle = base_bundle = None
        if any_covered:
            adapted_bundle, base_bundle = self._load_bundles(ctx, need_base)

        # --- score covered benchmarks on both bundles --------------------- #
        results: dict[str, dict] = {}
        summary: dict[str, str] = {}
        for name, bench in prepared:
            cov = coverage[name]
            result = BenchmarkResult(
                benchmark=name, direction=bench.direction, coverage=cov,
                caveats=list(bench.caveats),
            )
            if cov.available and adapted_bundle is not None:
                result.adapted = self._safe_score(bench, adapted_bundle, lang, limit)
                if bench.compare_to_base and base_bundle is not None:
                    result.base = self._safe_score(bench, base_bundle, lang, limit)
            results[name] = result.to_dict()
            summary[name] = self._summarize(result)

        # --- optional: native-speaker output-review queue ---------------- #
        human_review = None
        if getattr(cfg, "human_review", False) and adapted_bundle is not None:
            human_review = self._emit_output_review(ctx, adapted_bundle, cfg.human_review_n)

        report = {
            "language": lang.name,
            "benchmarks_requested": list(cfg.benchmarks),
            "human_review": human_review,
            "coverage_matrix": {
                name: {
                    "available": c.available,
                    "reason": c.reason,
                    "resolved_code": c.resolved_code,
                    "gated": c.gated,
                }
                for name, c in coverage.items()
            },
            "results": results,
            "summary": summary,
        }
        card_path = write_json(out_dir / "report_card.json", report)
        log.info("[evaluate] %s", summary)
        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={"summary": summary},
        )

    # ------------------------------------------------------------------ #
    def _prepare(self, bench, ctx: StageContext):
        """Inject the context a benchmark needs (corpus dir, judge prompts/teacher)."""
        clean_dir = ctx.project.stage_dir("clean") / "corpus"
        bench._clean_dir = clean_dir  # used by native_cloze / perplexity

        if bench.name == "judge":
            bench._prompts = self._judge_prompts(ctx)
            bench._teacher = self._teacher(ctx)
        return bench

    @staticmethod
    def _judge_prompts(ctx: StageContext) -> list[str]:
        from ..convdata.schema import instruction_of, read_jsonl

        accepted = ctx.project.stage_dir("convdata") / "accepted.jsonl"
        if not accepted.exists():
            return []
        pairs = read_jsonl(accepted)
        return [instruction_of(p) for p in pairs if instruction_of(p)]

    @staticmethod
    def _teacher(ctx: StageContext):
        cd = ctx.project.config.convdata
        try:
            from ..convdata.teacher import get_teacher

            return get_teacher(cd.provider, cd.model)
        except Exception as exc:
            log.warning("[evaluate] judge teacher unavailable: %s", exc)
            return None

    def _emit_output_review(self, ctx: StageContext, bundle, n: int):
        """Generate model responses to held-out instructions and write a native-
        speaker review queue (see human_review.py). Returns a summary of it."""
        import torch

        from . import human_review as hr

        prompts = self._judge_prompts(ctx)[:n]
        if not prompts:
            return {"status": "skipped", "note": "no held-out instructions (run convdata)"}
        model, tok = bundle.model, bundle.tokenizer
        device = next(model.parameters()).device
        items = []
        for instruction in prompts:
            ids = tok(instruction, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=128, do_sample=False)
            resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            items.append({"instruction": instruction, "response": resp})

        out_dir = ctx.stage_dir(self.name)
        queue_path = out_dir / "human_review_queue.jsonl"
        queue = hr.merge_reviews(hr.build_output_queue(items), queue_path)
        hr.save_queue(queue, queue_path)
        log.info("[evaluate] wrote %d model responses for native-speaker review", len(queue))
        return hr.summarize(queue)

    def _load_bundles(self, ctx: StageContext, need_base: bool):
        from ..modeling import load_model_and_tokenizer, resolve_artifacts

        base_id = ctx.project.model_profile.hf_id
        tok_dir, adapter, kind = resolve_artifacts(ctx.project)
        adapted_model, adapted_tok = load_model_and_tokenizer(
            base_id, tok_dir, adapter, token=_hf_token()
        )
        adapted = ModelBundle(adapted_model, adapted_tok, f"adapted:{kind}")
        base = None
        if need_base:
            # untouched base: original tokenizer, no adapter -> a fair reference point
            base_model, base_tok = load_model_and_tokenizer(base_id, None, None, token=_hf_token())
            base = ModelBundle(base_model, base_tok, "base")
        return adapted, base

    @staticmethod
    def _safe_score(bench, bundle, lang, limit):
        from .benchmarks.base import Score

        try:
            return bench.score(bundle, lang, limit=limit)
        except Exception as exc:  # one benchmark's failure shouldn't sink the report
            log.warning("[evaluate] %s failed on %s: %s", bench.name, bundle.label, exc)
            return Score(value=None, note=f"error: {type(exc).__name__}: {exc}")

    @staticmethod
    def _summarize(result: BenchmarkResult) -> str:
        if not result.coverage.available:
            return f"not covered — {result.coverage.reason}"
        a = result.adapted.value if result.adapted else None
        if a is None:
            note = result.adapted.note if result.adapted else "not run"
            return f"not measured ({note})"
        unit = "↓" if result.direction is MetricDirection.lower_better else "↑"
        s = f"adapted={a:.3f}{unit}"
        if result.adapted and result.adapted.ci95:
            lo, hi = result.adapted.ci95
            s += f" (95% CI {lo:.3f}–{hi:.3f})"
        d = result.delta
        if d is not None:
            s += f", Δ vs base {d:+.3f}"
        return s
