# lrl-toolkit

**A config-driven, open-source pipeline for building custom language models for low-resource languages (LRLs).**

Take a language from *"here is an ISO code"* to *"here is a fine-tuned, evaluated, publishable chat model"* with a single managed workflow — ingestion, cleaning, tokenizer adaptation, continued pretraining, conversational-data generation, instruction fine-tuning, evaluation, and export.

Built for the messy reality of LRLs — Kurmanji/Sorani Kurdish, Welsh, Cornish, Farsi, and beyond — where corpora are scattered, scripts and dialects vary, and instruction data barely exists.

> **Status:** Alpha (M5). **The full pipeline runs end-to-end** — ingest → clean → tokenizer → pretrain → convdata → finetune → evaluate → export — plus a **Streamlit dashboard**, **33 built-in languages**, and **19 base models**. Verified on live data (Welsh, SmolLM2, Ollama/NLLB). See [the roadmap](#roadmap).

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

### Conversational data

The `convdata` stage builds instruction/chat pairs three ways, mixed per project:
**translate** open instruction sets, **synthesize** native pairs with a teacher
LLM, and **review** them. Everything runs on **local / open-weight models** —
proprietary hosted APIs (Claude, OpenAI, Gemini) prohibit using their outputs to
train other models, so they are not offered.

**Translation backends** (`translate_backend`) — pick per language:

| Backend | Model | Notes |
|---------|-------|-------|
| `nllb` | NLLB-200 (600M default) | **Recommended for LRLs.** 200+ languages; best quality/size |
| `m2m100` | M2M-100 (418M) | 100 languages; small/fast |
| `opusmt` | Helsinki-NLP OPUS-MT | One tiny model per language pair (not all pairs exist) |
| `madlad` | MADLAD-400 (3B) | 400+ languages; large |
| `teacher` | the synth teacher LLM | Reuse Ollama/local; convenient but weaker on LRLs |
| `mock` | — | Offline/CI |

**Synth teacher** (`provider`) — local LLM that generates native pairs:

| Provider | Backend | Notes |
|----------|---------|-------|
| `ollama` | local Ollama server | **Recommended.** Free, no key. `ollama pull qwen2.5:7b` |
| `local` | transformers model | Runs an open HF model in-process |
| `mock` | deterministic | Offline/CI |

Generated pairs pass through an optional **human review** queue (native speakers
accept/edit/reject) before fine-tuning — see [DATA_ETHICS.md](DATA_ETHICS.md).

## Install

```bash
pip install -e .              # core: config, CLI, orchestration
pip install -e ".[data]"      # + ingestion / cleaning
pip install -e ".[train]"     # + training (torch, transformers, peft, trl, ...)
pip install -e ".[all]"       # everything
```

### GPU / platform notes

- **CUDA:** `pip install torch` pulls the **CPU-only** build. For GPU training, install a
  CUDA build first, then the extras — e.g. for CUDA 12.8:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install -e ".[train]"
  ```
  Verify with `python -c "import torch; print(torch.cuda.is_available())"`.
- **Language ID (fastText):** the `clean` stage's language filter uses fastText, which has no
  wheels for Python ≥ 3.12. `[data]` therefore skips it there and the filter degrades gracefully
  (all docs kept). On Python ≤ 3.11 install it explicitly with `pip install -e ".[langid]"`.
- **Windows:** use a venv at a short path (e.g. `C:\venv`) — the deep MS Store Python path plus
  torch's long internal paths can trip the 260-char limit during install. Prefer running the CLI
  as `python -m lrl_toolkit.cli ...` (or `python -m streamlit run dashboard/app.py`) if the
  `Scripts` dir isn't on `PATH`.
- **Recommended local base model:** on ~8&nbsp;GB VRAM, `qwen2.5-1.5b`/`qwen2.5-3b` (Apache-2.0)
  train comfortably under QLoRA; the Aya models are stronger for LRLs but want more VRAM.

## Quickstart

```bash
lrl init welsh                     # scaffold a project config
lrl sources list welsh             # see which corpora are available
lrl run -c projects/welsh.yaml     # run the full pipeline (resumable)
lrl dashboard                      # or drive it from the web UI
```

### Try it on Colab — 0 → hero

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jwilliamsresearch/lrl-toolkit/blob/main/notebooks/lrl_colab.ipynb)

[`notebooks/lrl_colab.ipynb`](notebooks/lrl_colab.ipynb) trains a **genuine** Welsh
chat model end-to-end on a free Colab T4, lets you chat with it, and publishes it to
the Hub — using [`projects/welsh-colab.yaml`](projects/welsh-colab.yaml):
**Qwen3-1.7B-Base** under QLoRA, with real instruction data (Dolly + OASST1
machine-translated to Welsh — no Ollama, no API, no mock). ~1–3 h, fully resumable.
The config is **release-clean** (Apache-2.0 base + m2m100/MIT translator → the model
publishes under `cc-by-sa-4.0`); swap `translate_backend: nllb` for higher quality at
the cost of a non-commercial licence. Scale up to `qwen3-4b` (also Apache-2.0) on
Colab Pro. (For a 30-minute offline smoke test of the plumbing, use
[`projects/quick-30min.yaml`](projects/quick-30min.yaml) instead.)

Run a single stage:

```bash
lrl ingest   -c projects/welsh.yaml
lrl clean    -c projects/welsh.yaml
lrl pretrain -c projects/welsh.yaml
```

### Built-in languages & models

Run `lrl languages` to list what ships. Currently **33 languages** across Celtic
(Welsh, Irish, Scottish Gaelic, Breton, Cornish…), African (Swahili, Yoruba,
Hausa, Amharic, Zulu…), Asian (Uyghur, Tibetan, Pashto, Nepali, Khmer…), Kurdish
(Kurmanji, Sorani), and more — each with script/dialect metadata, NLLB codes, and
a source catalog. And **19 base models**: Qwen2.5 (0.5B–7B), Llama 3.1/3.2,
Gemma 2, Mistral, Phi-3.5, **Aya-23 / Aya-Expanse** and **BLOOMZ** (multilingual,
strong for LRLs), SmolLM2, TinyLlama, and XLM-R. Add your own by dropping a YAML
in `configs/languages/` or `configs/models/` (or a dir on `$LRL_CONFIG_PATH`).

## Roadmap

- **M0 — Scaffolding** ✅: package, config schemas, manifest, CLI, registry, governance docs.
- **M1 — Data path** ✅: 10 ingest connectors (Wikipedia, Glot500, CulturaX, MADLAD-400, OSCAR, Common Crawl, OPUS/NLLB, SMOL, FLORES, local) + full clean stage (normalize, language-ID, dedup, quality filters, PII).
- **M2 — Model path** ✅: tokenizer extension (fertility-reported) + LoRA/QLoRA continued pretraining, with automatic QLoRA→LoRA fallback when there's no GPU.
- **M3 — Conversational + SFT** ✅: translate (NLLB/M2M-100/OPUS-MT/MADLAD/teacher) + synth (Ollama/local) + review queue + SFT via TRL.
- **M4 — Evaluate + export** ✅: held-out perplexity report card, LoRA merge, Ollama Modelfile, HF model card (best-effort GGUF via llama.cpp).
- **M5 — Dashboard** ✅: Streamlit app — wizard (generate a project), run monitor (run stages, view cards), review queue, chat with the trained model.
- **M6 — Launch** *(next)*: docs, tutorials, CI, first release. Broader eval benchmarks (FLORES chrF, Belebele).

## Contributing

Contributions — especially **new language profiles, source connectors, and native-speaker review** — are very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

[Apache-2.0](LICENSE). Note that **models and datasets you produce are governed by the licenses of their source data** — the toolkit tracks this for you, but you are responsible for honoring it.
