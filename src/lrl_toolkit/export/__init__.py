"""Export stage: merge, quantize, and publish — gated on license resolution.

Per DATA_ETHICS.md, a project cannot export while any ingested source has an
unresolved license. This stage enforces that gate before doing any packaging.

M0 status: the license gate is real; packaging is scaffolding. M4 implements
LoRA merge, GGUF/AWQ/GPTQ quantization, the Ollama Modelfile, and Hub upload.
"""

from __future__ import annotations

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json, write_text

log = get_logger("lrl.export")


class LicenseGateError(RuntimeError):
    """Raised when export is attempted with unresolved source licenses."""


class ExportStage(Stage):
    name = "export"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.export
        lang = ctx.project.language_profile
        model = ctx.project.model_profile
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

        # --- model card (always shipped) ---------------------------------- #
        licenses = sorted({r.license for r in ctx.manifest.provenance if r.license})
        model_card = _render_model_card(
            project=ctx.project.name,
            language=lang.display_name,
            base_model=model.hf_id,
            quantize=cfg.quantize,
            licenses=licenses,
        )
        card_path = write_text(out_dir / "MODEL_CARD.md", model_card)

        plan = {
            "quantize": cfg.quantize,
            "push_to_hub": cfg.push_to_hub,
            "hub_repo": cfg.hub_repo,
            "make_ollama_modelfile": cfg.make_ollama_modelfile,
            "source_licenses": licenses,
            "status": "placeholder",
        }
        plan_path = write_json(out_dir / "export_card.json", plan)
        log.info("[export] license gate passed; quantize=%s", cfg.quantize)

        return StageResult(
            outputs=[ctx.relpath(card_path), ctx.relpath(plan_path)],
            metrics={"quantize": cfg.quantize, "licenses": licenses},
        )


def _render_model_card(
    *, project: str, language: str, base_model: str, quantize: list[str], licenses: list[str]
) -> str:
    licenses_str = ", ".join(licenses) if licenses else "see source catalog"
    quant_str = ", ".join(quantize) if quantize else "none"
    return f"""# {project} — {language} language model

Built with [lrl-toolkit](https://github.com/lrl-toolkit/lrl-toolkit).

- **Language:** {language}
- **Base model:** `{base_model}`
- **Quantization:** {quant_str}
- **Source data licenses:** {licenses_str}

## Intended use

A community-oriented language model for {language}. Adapted from the base model
above via continued pretraining and instruction fine-tuning.

## Limitations

Low-resource-language models trained on thin or machine-translated data can be
confidently wrong. Evaluate on native tasks before relying on outputs, and
prefer native-speaker review. See the project's data card for corpus details.

## Provenance & licensing

This model derives from data under: {licenses_str}. You are responsible for
honoring those licenses and any attribution requirements.

*(This card is auto-generated; M4 will populate evaluation results.)*
"""
