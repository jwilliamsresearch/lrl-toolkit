# lrl-toolkit

**A config-driven, open-source pipeline for building custom language models for low-resource languages (LRLs).**

Take a language from *"here is an ISO code"* to *"here is a fine-tuned, evaluated, publishable chat model"* with a single managed workflow — ingestion, cleaning, tokenizer adaptation, continued pretraining, conversational-data generation, instruction fine-tuning, evaluation, and export.

Built for the messy reality of LRLs — Kurmanji/Sorani Kurdish, Welsh, Cornish, Farsi, and beyond — where corpora are scattered, scripts and dialects vary, and instruction data barely exists.

> **Status:** Alpha (M0 scaffolding). The pipeline skeleton, config system, registry, and CLI run today; individual stages are being filled in. See [the roadmap](#roadmap).

---

## Why

Building an LLM for a low-resource language means re-solving the same plumbing every time: finding corpora across a dozen incompatible sources, cleaning them, extending a tokenizer that barely covers your script, and manufacturing chat data that doesn't exist. `lrl-toolkit` makes that a **managed, reproducible, one-config workflow** with sensible defaults that run on a **single consumer GPU**.

## Design principles

- **Adapt, don't pretrain.** Continued pretraining + tokenizer extension + instruction tuning on a strong multilingual base (Llama / Qwen / Gemma / XLM-R) — best quality per dollar. No from-scratch pretraining in v1.
- **Consumer-GPU first.** QLoRA / 4-bit defaults; runnable on Colab/Kaggle. A100 and multi-GPU are alternate profiles.
- **Config over code.** One YAML per language project; every stage reads the same validated config. Fully resumable.
- **Provenance & ethics are first-class.** Every datum carries a license/source record; export is gated on license resolution. See [DATA_ETHICS.md](DATA_ETHICS.md).

## The pipeline

```
registry ─▶ ingest ─▶ clean ─▶ tokenizer ─▶ pretrain ─┐
                                                       ├─▶ finetune ─▶ evaluate ─▶ export
                              convdata (translate/synth/review) ─────┘
```

| Stage | Produces |
|-------|----------|
| **ingest** | raw sharded corpus + provenance records |
| **clean** | deduped, language-filtered, normalized text + data card |
| **tokenizer** | base tokenizer extended for the target script |
| **pretrain** | continued-pretrained base (LoRA/QLoRA adapter) |
| **convdata** | reviewed chat pairs (translate + synthesize + human review) |
| **finetune** | instruction-tuned chat model (SFT ± DPO) |
| **evaluate** | model report card (perplexity, chrF/FLORES, Belebele, judge) |
| **export** | merged/quantized weights (GGUF/Ollama) + HF model card |

## Install

```bash
pip install -e .              # core: config, CLI, orchestration
pip install -e ".[data]"      # + ingestion / cleaning
pip install -e ".[train]"     # + training (torch, transformers, peft, trl, ...)
pip install -e ".[all]"       # everything
```

## Quickstart

```bash
lrl init welsh                     # scaffold a project config
lrl sources list welsh             # see which corpora are available
lrl run -c projects/welsh.yaml     # run the full pipeline (resumable)
lrl dashboard                      # or drive it from the web UI
```

Run a single stage:

```bash
lrl ingest   -c projects/welsh.yaml
lrl clean    -c projects/welsh.yaml
lrl pretrain -c projects/welsh.yaml
```

## Roadmap

- **M0 — Scaffolding** *(current)*: package, config schemas, manifest, CLI, registry, governance docs.
- **M1 — Data path**: ingest connectors (Wikipedia, OSCAR/HF, OPUS, Leipzig, local) + full clean stage.
- **M2 — Model path**: tokenizer extension + QLoRA continued pretraining.
- **M3 — Conversational + SFT**: translate + synth + review + SFT → a working Welsh chat model.
- **M4 — Evaluate + export**: report card, GGUF/Ollama, HF model card.
- **M5 — Dashboard**: wizard, run monitor, review queue, chat.
- **M6 — Launch**: docs, tutorials for the seed languages, CI, first release.

## Contributing

Contributions — especially **new language profiles, source connectors, and native-speaker review** — are very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

[Apache-2.0](LICENSE). Note that **models and datasets you produce are governed by the licenses of their source data** — the toolkit tracks this for you, but you are responsible for honoring it.
