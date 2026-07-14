# Architecture

`lrl-toolkit` is a **config-driven pipeline**. A *project* is one YAML that names
three reusable profiles (`language`, `base_model`, `compute`) and carries
per-stage settings. Every stage reads the same validated config object and
communicates only through **artifacts on disk plus a manifest**, which keeps
stages independently runnable and resumable.

```
project.yaml ──load_project──▶ ResolvedProject ──▶ orchestrator ──▶ stages
                                   │
        ┌──────────────────────────┼──────────────────────────┐
   LanguageProfile           ModelProfile               ComputeProfile
   (configs/languages)       (configs/models)           (configs/compute)
```

## Key modules (`src/lrl_toolkit/`)

| Module | Responsibility |
|--------|----------------|
| `config.py` | Pydantic schemas for every stage + `ProjectConfig` / `ResolvedProject`; `load_project()`. |
| `registry/` | `LanguageProfile`, `ModelProfile`, `ComputeProfile` + a search-path loader (env → `./configs` → bundled). |
| `manifest.py` | `Manifest` (stage fingerprints for skip/resume) + `ProvenanceRecord` (license tracking). |
| `pipeline/` | `Stage` base class, `StageContext`/`StageResult`, and the orchestrator. |
| `ingest/ clean/ tokenizer/ pretrain/ convdata/ finetune/ evaluate/ export/` | One package per pipeline stage, each exposing a `Stage` subclass. |
| `cli.py` | Typer app: `init`, `sources`, `run`, per-stage commands, `dashboard`. |

## Stage contract

A stage subclasses `pipeline.base.Stage`, sets `name`, and implements
`run(ctx) -> StageResult`. It may override `fingerprint_payload()` to declare the
inputs that determine its output. The orchestrator:

1. computes each stage's fingerprint, **chaining in the previous stage's
   fingerprint** so any upstream change invalidates everything downstream;
2. **skips** a stage whose fingerprint matches the manifest (a "manifest hit");
3. otherwise runs it, records outputs/metrics/provenance, and saves the manifest.

## Invariants worth preserving

- **Provenance in, license gate out.** Ingest emits `ProvenanceRecord`s; `export`
  refuses to run while any source license is unresolved (see `DATA_ETHICS.md`).
- **Config is the source of truth.** Prefer new capability via config + a small
  module over hard-coding. `ProjectConfig` forbids unknown keys.
- **Canonical stage order.** `config.STAGE_ORDER` defines execution order; subsets
  are always run in that order.

## Where each milestone lands

Implemented and verified on live data: **ingest** (10 connectors) + **clean**
(M1); **tokenizer** extension + **pretrain** LoRA/QLoRA (M2); **convdata**
(translate via NLLB/M2M-100/OPUS-MT/MADLAD/teacher + synth via local/Ollama +
review) + **finetune** SFT via TRL (M3); **evaluate** (perplexity) + **export**
(LoRA merge, Ollama Modelfile, model card, best-effort GGUF) (M4); the Streamlit
**dashboard** (M5, `dashboard/app.py`). 33 languages + 19 base models ship as
config. The license gate in `export` is enforced. Remaining (M6): broader eval
benchmarks (FLORES chrF, Belebele, judge), docs, CI, release.
