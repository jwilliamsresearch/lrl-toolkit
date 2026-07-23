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

from ..licensing import LicenseConflictError, resolve_release_license
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

        try:
            resolved_license = resolve_release_license(ctx.manifest.provenance)
        except LicenseConflictError as exc:
            raise LicenseGateError(f"Refusing to export: {exc}") from exc

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
        evals = _load_evals(ctx.project.stage_dir("evaluate") / "report_card.json")
        model_card = _render_model_card(
            project=ctx.project.name,
            language=lang.display_name,
            base_model=model_profile.hf_id,
            model_kind=kind,
            quantize=cfg.quantize,
            resolved_license=resolved_license,
            evals=evals,
        )
        card_path = write_text(out_dir / "MODEL_CARD.md", model_card)
        plan_path = write_json(
            out_dir / "export_card.json",
            {
                "packaging": packaging,
                "quantize": cfg.quantize,
                "push_to_hub": cfg.push_to_hub,
                "model_license": resolved_license.license,
                "model_license_rationale": resolved_license.rationale,
                "attributions": resolved_license.attributions,
                "evals": evals,
            },
        )
        log.info(
            "[export] license gate passed; model_kind=%s license=%s",
            kind,
            resolved_license.license,
        )

        return StageResult(
            outputs=[ctx.relpath(card_path), ctx.relpath(plan_path)],
            metrics={"model_kind": kind, "license": resolved_license.license},
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
    """Load the evaluate stage's human-readable per-benchmark summary for the model
    card. Prefers the coverage-aware 'summary' (name -> readable string); falls back
    to 'results' for older report cards."""
    if path.exists():
        try:
            report = read_json(path)
            return report.get("summary") or report.get("results")
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
    *, project, language, base_model, model_kind, quantize, resolved_license, evals
) -> str:
    quant_str = ", ".join(quantize) if quantize else "none"
    evals_str = "\n".join(f"- **{k}:** {v}" for k, v in (evals or {}).items()) or "- (none run)"
    attributions_str = (
        "\n".join(f"- {a}" for a in resolved_license.attributions) or "- (no sources recorded)"
    )
    return f"""# {project} — {language} language model

Built with [lrl-toolkit](https://github.com/lrl-toolkit/lrl-toolkit).

- **Language:** {language}
- **Base model:** `{base_model}`
- **Adaptation:** {model_kind} (continued pretraining / SFT via LoRA)
- **Quantization:** {quant_str}
- **License:** {resolved_license.license}

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

**This model is released under {resolved_license.license}.**

{resolved_license.rationale}

### Source attribution

{attributions_str}
"""
