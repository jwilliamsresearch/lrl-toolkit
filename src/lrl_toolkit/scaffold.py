"""Generate a project YAML from a few choices. Shared by the CLI and dashboard."""

from __future__ import annotations


def scaffold_yaml(
    name: str,
    language: str,
    base_model: str = "qwen2.5-1.5b",
    compute: str = "consumer_gpu",
    *,
    translate: list[str] | None = None,
    translate_backend: str = "nllb",
    synth_provider: str = "ollama",
    synth_n: int = 2000,
    review: bool = True,
) -> str:
    """Return a ready-to-run project YAML string."""
    translate = translate if translate is not None else ["dolly"]
    translate_list = ", ".join(translate)
    return f"""# lrl-toolkit project: {name}
name: {name}
language: {language}
base_model: {base_model}
compute: {compute}
seed: 42

ingest:
  sources: []          # empty = use every source in the language profile
  max_gb: 5

clean:
  lang_id: glotlid
  dedup: minhash
  min_quality: 0.6

tokenizer:
  strategy: extend
  added_tokens: 8000

pretrain:
  method: qlora
  epochs: 1
  seq_len: 2048

convdata:
  translate: [{translate_list}]      # instruction sets: dolly/alpaca/oasst1 or a local path
  translate_limit: 500
  translate_backend: {translate_backend}   # nllb/m2m100/opusmt/madlad/teacher/mock
  provider: {synth_provider}           # local teacher for synth (ollama/local/mock)
  synth:
    provider: {synth_provider}
    n: {synth_n}
  review: {str(review).lower()}

finetune:
  method: qlora
  dpo: false

evaluate:
  benchmarks: [perplexity, flores]

export:
  quantize: [gguf_q4_k_m]
  push_to_hub: false
"""
