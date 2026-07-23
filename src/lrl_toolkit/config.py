"""Project configuration: the single YAML that drives a language project.

A project references three profiles by slug (``language``, ``base_model``,
``compute``) and carries per-stage settings. :func:`load_project` validates the
YAML and resolves the referenced profiles into a :class:`ResolvedProject` that
every stage consumes.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from .registry import (
    ComputeProfile,
    LanguageProfile,
    ModelProfile,
    load_compute,
    load_language,
    load_model,
)

# The canonical pipeline order. `lrl run` executes this sequence; individual
# stages may be run on their own via the CLI.
STAGE_ORDER: tuple[str, ...] = (
    "ingest",
    "clean",
    "tokenizer",
    "pretrain",
    "convdata",
    "finetune",
    "evaluate",
    "export",
)


# --------------------------------------------------------------------------- #
# Per-stage configuration blocks
# --------------------------------------------------------------------------- #
class LangID(str, Enum):
    glotlid = "glotlid"
    fasttext = "fasttext"
    none = "none"


class Dedup(str, Enum):
    none = "none"
    exact = "exact"
    minhash = "minhash"


class IngestConfig(BaseModel):
    # Connector names to run; if empty, use every source in the language profile.
    sources: list[str] = Field(default_factory=list)
    max_gb: float | None = Field(default=None, description="Soft cap on total raw text.")
    max_docs_per_source: int | None = None
    model_config = {"extra": "forbid"}


class CleanConfig(BaseModel):
    lang_id: LangID = LangID.glotlid
    min_doc_lang_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    dedup: Dedup = Dedup.minhash
    min_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    strip_boilerplate: bool = True
    normalize: bool = True
    scrub_pii: bool = True
    model_config = {"extra": "forbid"}


class TokenizerStrategy(str, Enum):
    extend = "extend"  # add pieces to the base tokenizer (default)
    train = "train"  # train a fresh tokenizer (encoder / from-scratch experiments)
    none = "none"  # keep the base tokenizer unchanged


class TokenizerConfig(BaseModel):
    strategy: TokenizerStrategy = TokenizerStrategy.extend
    added_tokens: int = Field(default=8000, ge=0)
    model_config = {"extra": "forbid"}


class TuneMethod(str, Enum):
    qlora = "qlora"
    lora = "lora"
    full = "full"


class PretrainConfig(BaseModel):
    method: TuneMethod = TuneMethod.qlora
    epochs: float = 1.0
    seq_len: int = 2048
    learning_rate: float = 2e-4
    max_steps: int | None = Field(default=None, description="Overrides epochs when set.")
    lora_r: int = 32
    lora_alpha: int = 64
    embed_init: str = Field(
        default="subword_mean",
        description=(
            "How to initialize embeddings for newly added tokens: 'subword_mean' "
            "(FOCUS-style mean of base-tokenizer pieces) or 'default' (random)."
        ),
    )
    model_config = {"extra": "forbid"}


class SynthConfig(BaseModel):
    provider: str = Field(default="ollama", description="Local teacher LLM: ollama/local/mock.")
    model: str | None = Field(default=None, description="Override the provider's default model.")
    n: int = Field(default=5000, ge=0, description="Number of synthetic pairs to generate.")
    ground_in_corpus: bool = True
    model_config = {"extra": "forbid"}


class NativeSetConfig(BaseModel):
    """A target-language instruction dataset used *as-is* — no machine translation.

    For datasets that already contain the target language (e.g. xP3x's per-language
    splits). Rows are mapped to ``{instruction, response}`` and added directly to the
    SFT set. ``exclude`` drops rows whose ``source_field`` contains any listed
    substring — use it to keep eval-only corpora (e.g. FLORES) out of training.
    """

    repo: str = Field(..., description="HF dataset id or local .jsonl path.")
    name: str | None = Field(
        default=None, description="HF config name (e.g. an xP3x language like 'kmr_Latn')."
    )
    split: str = "train"
    instruction_field: str = Field(default="inputs", description="Column holding the prompt.")
    response_field: str = Field(default="targets", description="Column holding the response.")
    source_field: str = Field(
        default="dataset", description="Column naming the constituent source, for exclusion."
    )
    exclude: list[str] = Field(
        default_factory=list,
        description="Drop rows whose source_field contains any of these substrings, e.g. 'flores'.",
    )
    select_field: str | None = Field(
        default=None, description="Column to filter on (e.g. 'language_code')."
    )
    select_value: str | None = Field(
        default=None, description="Keep only rows whose select_field equals this."
    )
    limit: int | None = Field(default=5000, description="Max rows to take.")
    model_config = {"extra": "forbid"}


class ConvDataConfig(BaseModel):
    # Names of open instruction datasets (HF ids or local paths) to translate.
    translate: list[str] = Field(default_factory=list)
    # Target-language instruction datasets used as-is (no translation).
    native_sets: list[NativeSetConfig] = Field(default_factory=list)
    translate_limit: int | None = Field(
        default=500, description="Max examples to translate per dataset."
    )
    translate_backend: str = Field(
        default="nllb", description="MT backend: nllb/m2m100/opusmt/madlad/teacher/mock."
    )
    translate_model: str | None = Field(default=None, description="Override the MT model.")
    provider: str = Field(default="ollama", description="Local teacher: ollama/local/mock.")
    model: str | None = Field(default=None, description="Override the provider's default model.")
    synth: SynthConfig | None = None
    review: bool = Field(default=True, description="Route pairs through human review before SFT.")
    model_config = {"extra": "forbid"}


class FinetuneConfig(BaseModel):
    method: TuneMethod = TuneMethod.qlora
    epochs: float = 3.0
    max_steps: int | None = Field(default=None, description="Overrides epochs when set.")
    max_seq_len: int = 1024
    learning_rate: float = 2e-4
    dpo: bool = False
    model_config = {"extra": "forbid"}


class EvaluateConfig(BaseModel):
    # Benchmark ids from the benchmark registry: 'native_cloze', 'perplexity',
    # 'belebele', 'global_mmlu', 'flores', 'judge'. Each self-reports whether it
    # covers the target language; uncovered ones are recorded in the coverage
    # matrix rather than failing.
    benchmarks: list[str] = Field(
        default_factory=lambda: [
            "native_cloze",
            "perplexity",
            "belebele",
            "global_mmlu",
            "flores",
        ]
    )
    limit: int = Field(default=100, ge=1, description="Max items scored per benchmark.")
    human_review: bool = Field(
        default=False,
        description="Emit a native-speaker output-review queue (model responses to rate).",
    )
    human_review_n: int = Field(default=50, ge=1, description="Responses to queue for review.")
    model_config = {"extra": "forbid"}


class ExportConfig(BaseModel):
    merge_adapter: bool = Field(default=True, description="Merge the LoRA adapter into the base.")
    # Quantization targets, e.g. 'gguf_q4_k_m', 'awq', 'gptq'.
    quantize: list[str] = Field(default_factory=lambda: ["gguf_q4_k_m"])
    push_to_hub: bool = False
    hub_repo: str | None = None
    make_ollama_modelfile: bool = True
    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------- #
# Top-level project config
# --------------------------------------------------------------------------- #
class ProjectConfig(BaseModel):
    """The validated contents of a project YAML (profiles still unresolved)."""

    name: str = Field(..., description="Project slug; used for the working directory name.")
    language: str = Field(..., description="Language profile slug.")
    base_model: str = Field(..., description="Model profile slug.")
    compute: str = Field(..., description="Compute profile slug.")
    seed: int = 42
    workdir: str | None = Field(
        default=None, description="Where artifacts live; defaults to ./projects/<name>."
    )
    # Subset of stages to run with `lrl run`; empty means the full STAGE_ORDER.
    stages: list[str] = Field(default_factory=list)

    ingest: IngestConfig = Field(default_factory=IngestConfig)
    clean: CleanConfig = Field(default_factory=CleanConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    pretrain: PretrainConfig = Field(default_factory=PretrainConfig)
    convdata: ConvDataConfig = Field(default_factory=ConvDataConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_stages(self) -> ProjectConfig:
        unknown = [s for s in self.stages if s not in STAGE_ORDER]
        if unknown:
            raise ValueError(
                f"Unknown stage(s) {unknown}; valid stages are {list(STAGE_ORDER)}."
            )
        return self

    def selected_stages(self) -> list[str]:
        """Stages to execute, in canonical order."""
        if not self.stages:
            return list(STAGE_ORDER)
        return [s for s in STAGE_ORDER if s in self.stages]

    def stage_config(self, stage: str) -> BaseModel:
        return getattr(self, stage)


class ResolvedProject(BaseModel):
    """A project with its profiles resolved and a concrete working directory."""

    config: ProjectConfig
    language_profile: LanguageProfile
    model_profile: ModelProfile
    compute_profile: ComputeProfile
    workdir: Path
    source_path: Path | None = None

    # `protected_namespaces=()` silences pydantic's warning about the
    # `model_profile` field colliding with its reserved `model_` namespace.
    model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}

    @property
    def name(self) -> str:
        return self.config.name

    def artifacts_dir(self) -> Path:
        return self.workdir / "artifacts"

    def stage_dir(self, stage: str) -> Path:
        return self.artifacts_dir() / stage


def load_project(path: str | Path) -> ResolvedProject:
    """Load, validate, and resolve a project YAML into a :class:`ResolvedProject`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Project config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    config = ProjectConfig.model_validate(raw)

    workdir = Path(config.workdir) if config.workdir else Path.cwd() / "projects" / config.name

    return ResolvedProject(
        config=config,
        language_profile=load_language(config.language),
        model_profile=load_model(config.base_model),
        compute_profile=load_compute(config.compute),
        workdir=workdir,
        source_path=path.resolve(),
    )
