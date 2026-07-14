# lrl-toolkit

**A config-driven, open-source pipeline for building custom language models for low-resource languages (LRLs).**

Take a language from *"here is an ISO code"* to *"here is a fine-tuned, evaluated, publishable chat model"* with a single managed workflow — ingestion, cleaning, tokenizer adaptation, continued pretraining, conversational-data generation, instruction fine-tuning, evaluation, and export.

Built for the messy reality of LRLs — Kurmanji/Sorani Kurdish, Welsh, Cornish, Farsi, and beyond — where corpora are scattered, scripts and dialects vary, and instruction data barely exists.

> **Status:** Alpha (M4). **The full pipeline runs end-to-end today** — ingest → clean → tokenizer → pretrain → convdata → finetune → evaluate → export — verified on live data (Welsh, SmolLM2, Ollama teacher). The web dashboard (M5) is next. See [the roadmap](#roadmap).

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

## Data sources

The ingest stage ships connectors for the major LRL corpora. Each is a small
class implementing `BaseConnector`; sources for a language are declared in its
profile's catalog (`lrl sources list <lang>`).

| Connector | Source | Type | Notes |
|-----------|--------|------|-------|
| `wikipedia` | `wikimedia/wikipedia` | mono | Per-language Wikipedia dumps |
| `glot500` | `cis-lmu/Glot500` | mono | 500+ under-resourced languages |
| `culturax` | `uonlp/CulturaX` | mono | Cleaned mC4 + OSCAR (may need HF token) |
| `madlad400` | `allenai/madlad-400` | mono | Google's audited 419-language set |
| `oscar` | `oscar-corpus/OSCAR-2301` | mono | **Gated** — accept license + set `HF_TOKEN` |
| `commoncrawl` | raw CDX + WARC | mono | Target specific domains; language-filtered in `clean` |
| `opus` | `opus.nlpl.eu` API | parallel | Aggregates **NLLB**, OpenSubtitles, Tatoeba, bible… |
| `smol` | `google/smol` | parallel | GATITOS/SmolSent/SmolDoc for 100+ LRLs |
| `flores` | `openlanguagedata/flores_plus` | parallel | **Gated** — FLORES-200 eval/seed data |
| `local` | your filesystem | mono | `.txt/.jsonl(.gz)/.md/.html/.pdf`; the offline path |

Gated sources need a Hugging Face token: `export HF_TOKEN=hf_...` (or
`huggingface-cli login`). Every fetched source records a `ProvenanceRecord` with
its license; **export is blocked until all licenses resolve** (see
[DATA_ETHICS.md](DATA_ETHICS.md)).

### Conversational data (teacher LLMs)

The `convdata` stage builds instruction/chat pairs by translating open
instruction sets and synthesizing native pairs with a **teacher LLM**. Only
**local / open-weight teachers** are supported — proprietary hosted APIs (Claude,
OpenAI, Gemini) prohibit using their outputs to train other models:

| Provider | Backend | Notes |
|----------|---------|-------|
| `ollama` | local Ollama server | **Recommended.** Free, no key. `ollama pull qwen2.5:7b` |
| `local` | transformers model | Runs an open HF model in-process |
| `mock` | deterministic | Offline/CI; no model |

Generated pairs pass through an optional **human review** queue (native speakers
accept/edit/reject) before fine-tuning — see [DATA_ETHICS.md](DATA_ETHICS.md).

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

- **M0 — Scaffolding** ✅: package, config schemas, manifest, CLI, registry, governance docs.
- **M1 — Data path** ✅: 10 ingest connectors (Wikipedia, Glot500, CulturaX, MADLAD-400, OSCAR, Common Crawl, OPUS/NLLB, SMOL, FLORES, local) + full clean stage (normalize, language-ID, dedup, quality filters, PII).
- **M2 — Model path** ✅: tokenizer extension (fertility-reported) + LoRA/QLoRA continued pretraining, with automatic QLoRA→LoRA fallback when there's no GPU.
- **M3 — Conversational + SFT** ✅: translate + synth (local/Ollama teacher) + review queue + SFT via TRL.
- **M4 — Evaluate + export** ✅: held-out perplexity report card, LoRA merge, Ollama Modelfile, HF model card (best-effort GGUF via llama.cpp).
- **M5 — Dashboard** *(next)*: wizard, run monitor, review queue, chat.
- **M6 — Launch**: docs, tutorials for the seed languages, CI, first release.

## Contributing

Contributions — especially **new language profiles, source connectors, and native-speaker review** — are very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

[Apache-2.0](LICENSE). Note that **models and datasets you produce are governed by the licenses of their source data** — the toolkit tracks this for you, but you are responsible for honoring it.
