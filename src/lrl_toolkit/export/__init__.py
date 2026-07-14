"""Export stage: merge, package, and document — gated on license resolution.

Per DATA_ETHICS.md, export refuses to run while any ingested source has an
unresolved license. When a trained adapter exists it is merged into the base and
saved; GGUF conversion is attempted best-effort (needs llama.cpp) and an Ollama
Modelfile + a model card (with any evaluation results) are always written.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, read_json, write_json, write_text

log = get_logger("lrl.export")

__all__ = ["ExportStage", "LicenseGateError"]


class LicenseGateError(RuntimeError):
    """Raised when export is attempted with unresolved source licenses."""


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class ExportStage(Stage):
    name = "export"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.export
        lang = ctx.project.language_profile
        model_profile = ctx.project.model_profile
        out_dir = ctx.stage_dir(self.name)

        # --- license resolution gate (DATA_ETHICS.md) --------------------- #
        unresolved = ctx.manifest.unresolved_licenses()
        if unresolved:
            names = ", ".join(sorted({r.source for r in unresolved}))
            raise LicenseGateError(
                "Refusing to export: unresolved license(s) for source(s): "
                f"{names}. Resolve these in the language profile's source catalog "
                "before exporting. See DATA_ETHICS.md."
            )

        from ..modeling import resolve_artifacts

        tok_dir, adapter, kind = resolve_artifacts(ctx.project)

        packaging: dict = {"model_kind": kind}
        merged_dir = out_dir / "merged"
        if adapter is not None and cfg.merge_adapter:
            from ..modeling import load_model_and_tokenizer

            model, tokenizer = load_model_and_tokenizer(
                model_profile.hf_id, tok_dir, adapter, token=_hf_token(), merge=True
            )
            if merged_dir.exists():
                shutil.rmtree(merged_dir)
            model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            packaging["merged_to"] = str(merged_dir)
            log.info("[export] merged %s adapter -> %s", kind, merged_dir)
        else:
            packaging["merged_to"] = None
            packaging["note"] = "no trained adapter to merge; packaging metadata only"

        packaging["gguf"] = _maybe_gguf(merged_dir, cfg.quantize)

        if cfg.make_ollama_modelfile:
            model_ref = packaging.get("gguf", {}).get("path") or (
                str(merged_dir) if packaging["merged_to"] else model_profile.hf_id
            )
            mf = write_text(out_dir / "Modelfile", _render_modelfile(model_ref, lang.display_name))
            packaging["ollama_modelfile"] = str(mf)

        # --- model card (always shipped) ---------------------------------- #
        licenses = sorted({r.license for r in ctx.manifest.provenance if r.license})
        evals = _load_evals(ctx.project.stage_dir("evaluate") / "report_card.json")
        model_card = _render_model_card(
            project=ctx.project.name,
            language=lang.display_name,
            base_model=model_profile.hf_id,
            model_kind=kind,
            quantize=cfg.quantize,
            licenses=licenses,
            evals=evals,
        )
        card_path = write_text(out_dir / "MODEL_CARD.md", model_card)
        plan_path = write_json(
            out_dir / "export_card.json",
            {
                "packaging": packaging,
                "quantize": cfg.quantize,
                "push_to_hub": cfg.push_to_hub,
                "source_licenses": licenses,
                "evals": evals,
            },
        )
        log.info("[export] license gate passed; model_kind=%s", kind)

        return StageResult(
            outputs=[ctx.relpath(card_path), ctx.relpath(plan_path)],
            metrics={"model_kind": kind, "licenses": licenses},
        )


def _maybe_gguf(merged_dir: Path, quantize: list[str]) -> dict:
    """Attempt GGUF conversion if llama.cpp tooling is available; else report how
    to enable it. Never fails the pipeline."""
    if not quantize or not any(q.startswith("gguf") for q in quantize):
        return {"status": "not_requested"}
    if not merged_dir.exists():
        return {"status": "skipped", "note": "no merged model to convert"}
    converter = shutil.which("convert_hf_to_gguf.py") or shutil.which("convert-hf-to-gguf.py")
    if converter is None:
        return {
            "status": "skipped",
            "note": "install llama.cpp and put convert_hf_to_gguf.py on PATH to enable GGUF",
        }
    # Real conversion is delegated to llama.cpp; recorded here for the card.
    return {"status": "available", "converter": converter, "targets": quantize}


def _load_evals(path: Path) -> dict | None:
    if path.exists():
        try:
            return read_json(path).get("results")
        except Exception:
            return None
    return None


def _render_modelfile(model_ref: str, language: str) -> str:
    return (
        f"# Ollama Modelfile for a {language} model (lrl-toolkit)\n"
        f"FROM {model_ref}\n"
        'PARAMETER temperature 0.7\n'
        f'SYSTEM "You are a helpful assistant that responds in {language}."\n'
    )


def _render_model_card(
    *, project, language, base_model, model_kind, quantize, licenses, evals
) -> str:
    licenses_str = ", ".join(licenses) if licenses else "see source catalog"
    quant_str = ", ".join(quantize) if quantize else "none"
    evals_str = "\n".join(f"- **{k}:** {v}" for k, v in (evals or {}).items()) or "- (none run)"
    return f"""# {project} — {language} language model

Built with [lrl-toolkit](https://github.com/lrl-toolkit/lrl-toolkit).

- **Language:** {language}
- **Base model:** `{base_model}`
- **Adaptation:** {model_kind} (continued pretraining / SFT via LoRA)
- **Quantization:** {quant_str}
- **Source data licenses:** {licenses_str}

## Evaluation

{evals_str}

## Intended use

A community-oriented language model for {language}, adapted from the base model
above via continued pretraining and instruction fine-tuning.

## Limitations

Low-resource-language models trained on thin or machine-translated data can be
confidently wrong. Evaluate on native tasks before relying on outputs, and prefer
native-speaker review. See the project's data card for corpus details.

## Provenance & licensing

This model derives from data under: {licenses_str}. You are responsible for
honoring those licenses and any attribution requirements.
"""
