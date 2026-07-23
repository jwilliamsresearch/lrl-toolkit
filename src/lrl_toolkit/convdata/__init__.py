"""Convdata stage: build conversational/instruction pairs for fine-tuning.

Combines three generators — translating open instruction datasets, synthesizing
native pairs with a teacher LLM (grounded in the corpus), and a human review
queue — into a single accepted set in the unified ``messages`` schema.
"""

from __future__ import annotations

from pathlib import Path

from ..corpus import iter_documents
from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json
from . import review as review_mod
from .instruction_sets import load_instruction_set, load_native_set
from .schema import chat_pair, instruction_of, pair_is_degenerate, write_jsonl
from .teacher import get_teacher
from .translators import get_translator

log = get_logger("lrl.convdata")

__all__ = ["ConvDataStage"]


def _corpus_contexts(clean_dir: Path, k: int = 20) -> list[str]:
    contexts: list[str] = []
    if not clean_dir.exists():
        return contexts
    for doc in iter_documents(clean_dir):
        contexts.append(doc.text)
        if len(contexts) >= k:
            break
    return contexts


class ConvDataStage(Stage):
    name = "convdata"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.convdata
        lang = ctx.project.language_profile
        lang_name = lang.display_name
        out_dir = ctx.stage_dir(self.name)
        clean_dir = ctx.project.stage_dir("clean") / "corpus"

        pairs: list[dict] = []

        # --- translate open instruction datasets ------------------------- #
        if cfg.translate:
            translator = get_translator(
                cfg.translate_backend,
                cfg.translate_model,
                teacher_provider=cfg.provider,
                teacher_model=cfg.model,
            )
            log.info("[convdata] translating with backend=%s", cfg.translate_backend)
            for ds_name in cfg.translate:
                count = 0
                for ex in load_instruction_set(ds_name, limit=cfg.translate_limit):
                    instr = translator.translate(ex["instruction"], lang)
                    resp = translator.translate(ex["response"], lang)
                    pairs.append(chat_pair(instr, resp, source=f"translate:{ds_name}"))
                    count += 1
                log.info("[convdata] translated %d from %s", count, ds_name)

        # --- native target-language instruction sets (used as-is, no MT) - #
        # Profile-level sources (e.g. Aya for languages Aya covers) plus any
        # declared on the project config.
        native_sources = [*lang.instruction_sources, *cfg.native_sets]
        for ns in native_sources:
            count = 0
            for ex in load_native_set(ns):
                pairs.append(
                    chat_pair(ex["instruction"], ex["response"], source=f"native:{ns.repo}")
                )
                count += 1
            log.info("[convdata] loaded %d native pairs from %s", count, ns.repo)

        # --- synthesize native pairs via a teacher LLM ------------------- #
        if cfg.synth and cfg.synth.n > 0:
            teacher = get_teacher(cfg.synth.provider, cfg.synth.model)
            contexts = _corpus_contexts(clean_dir) if cfg.synth.ground_in_corpus else None
            synth_pairs = teacher.generate_pairs(cfg.synth.n, lang_name, contexts)
            for p in synth_pairs:
                pairs.append(chat_pair(p["instruction"], p["response"], source="synth"))
            log.info("[convdata] synthesized %d pairs", len(synth_pairs))

        # --- dedup by instruction + drop repetition-degenerate pairs ----- #
        seen: set[str] = set()
        deduped: list[dict] = []
        n_degenerate = 0
        for p in pairs:
            key = instruction_of(p).strip().lower()
            if not key or key in seen:
                continue
            if pair_is_degenerate(p):
                n_degenerate += 1
                continue
            seen.add(key)
            deduped.append(p)

        # --- review queue ----------------------------------------------- #
        queue = review_mod.build_queue(deduped)
        queue_path = out_dir / "review_queue.jsonl"
        queue = review_mod.merge_reviews(queue, queue_path)
        review_mod.save_queue(queue, queue_path)

        accepted = review_mod.accepted_pairs(queue, review_enabled=cfg.review)
        write_jsonl(out_dir / "pairs.jsonl", deduped)
        write_jsonl(out_dir / "accepted.jsonl", accepted)

        card = {
            "language": lang_name,
            "translate_datasets": cfg.translate,
            "translate_backend": cfg.translate_backend if cfg.translate else None,
            "native_sets": [ns.model_dump(mode="json") for ns in native_sources],
            "provider": cfg.provider,
            "synth": cfg.synth.model_dump(mode="json") if cfg.synth else None,
            "review": cfg.review,
            "counts": {
                "generated": len(pairs),
                "degenerate_dropped": n_degenerate,
                "deduped": len(deduped),
                "accepted": len(accepted),
            },
        }
        card_path = write_json(out_dir / "convdata_card.json", card)
        log.info(
            "[convdata] generated=%d deduped=%d accepted=%d",
            len(pairs),
            len(deduped),
            len(accepted),
        )

        return StageResult(
            outputs=[ctx.relpath(card_path)],
            metrics={"accepted": len(accepted), "deduped": len(deduped)},
        )
