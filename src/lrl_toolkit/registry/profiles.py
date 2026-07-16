"""Profile schemas: languages, base models, and compute environments.

A *language project* is described by one YAML that references three reusable
profiles by name (``language``, ``base_model``, ``compute``). These Pydantic
models define the shape of those profiles. The concrete YAML files live under
``configs/{languages,models,compute}`` (bundled with the package and
discoverable in a user's working directory) and are loaded by
:mod:`lrl_toolkit.registry.loader`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

# Human script name -> ISO 15924 tag, used to derive dataset codes like
# ``kmr_Latn`` / ``ckb_Arab`` from a profile's iso639_3 + script.
_SCRIPT_TAG = {
    "Latin": "Latn",
    "Arabic": "Arab",
    "Cyrillic": "Cyrl",
    "Ethiopic": "Ethi",
    "Devanagari": "Deva",
    "Sinhala": "Sinh",
    "Tibetan": "Tibt",
    "Khmer": "Khmr",
    "Hebrew": "Hebr",
    "Greek": "Grek",
    "Armenian": "Armn",
    "Georgian": "Geor",
    "Hangul": "Hang",
    "Thai": "Thai",
    "Bengali": "Beng",
    "Tamil": "Taml",
}

# Languages whose FineWeb-2 config code differs from the derived <iso639_3>_<Script>
# (usually a macrolanguage iso vs FineWeb-2's specific code). Verified against the
# live FineWeb-2 config list. Add a line here if a new language needs an override.
_FINEWEB2_OVERRIDE = {
    "grn": "gug_Latn",  # Guarani
    "mon": "khk_Cyrl",  # Mongolian -> Halh
    "nep": "npi_Deva",  # Nepali (individual)
    "pus": "pbt_Arab",  # Pashto -> Southern
    "que": "quy_Latn",  # Quechua -> Ayacucho
    "swa": "swh_Latn",  # Swahili (individual)
}


class TextDirection(str, Enum):
    ltr = "ltr"
    rtl = "rtl"


class SourceHint(BaseModel):
    """A pointer telling an ingest connector how to fetch data for a language.

    ``connector`` names a module under :mod:`lrl_toolkit.ingest`; ``params``
    carries connector-specific hints (e.g. the Wikipedia language code, or the
    list of OPUS corpora to pull).
    """

    connector: str = Field(..., description="Ingest connector name, e.g. 'wikipedia', 'opus'.")
    params: dict = Field(default_factory=dict, description="Connector-specific hints.")
    notes: str | None = Field(default=None, description="Human note about this source.")

    model_config = {"extra": "forbid"}


class InstructionSource(BaseModel):
    """A target-language instruction dataset used as-is (no MT) in the convdata stage.

    Mirrors the shape ``convdata.native_sets`` accepts, but lives on the language
    profile so it applies to every project for that language (e.g. Aya for the
    languages Aya covers). Loaded by :func:`lrl_toolkit.convdata.load_native_set`.
    """

    repo: str = Field(..., description="HF dataset id or local .jsonl path.")
    name: str | None = Field(default=None, description="HF config name, if the dataset uses one.")
    split: str = "train"
    instruction_field: str = "inputs"
    response_field: str = "targets"
    source_field: str = Field(
        default="dataset", description="Column naming the constituent source, for exclusion."
    )
    exclude: list[str] = Field(
        default_factory=list, description="Drop rows whose source_field contains any substring."
    )
    select_field: str | None = Field(
        default=None, description="Column to filter on (e.g. 'language_code' for Aya)."
    )
    select_value: str | None = Field(
        default=None, description="Keep only rows whose select_field equals this."
    )
    limit: int | None = 5000
    notes: str | None = None

    model_config = {"extra": "forbid"}


class NormalizationRules(BaseModel):
    """Per-language text normalization applied in the clean stage."""

    unicode_form: str = Field(default="NFC", description="Unicode normalization form.")
    normalize_digits: bool = Field(default=False, description="Map non-ASCII digits to ASCII.")
    # Named, language-specific rule sets implemented in lrl_toolkit.clean.normalize,
    # e.g. "arabic_presentation_forms", "kurdish_latin", "persian_yeh_kaf".
    rules: list[str] = Field(default_factory=list, description="Named normalization rule sets.")

    model_config = {"extra": "forbid"}


class LanguageProfile(BaseModel):
    """Everything the pipeline needs to know about a target language."""

    name: str = Field(..., description="Profile slug, e.g. 'kurmanji'.")
    display_name: str = Field(..., description="Human-readable name, e.g. 'Kurmanji Kurdish'.")
    iso639_3: str = Field(..., description="ISO 639-3 code, e.g. 'kmr'.")
    iso639_1: str | None = Field(default=None, description="ISO 639-1 code if one exists.")
    nllb_code: str | None = Field(
        default=None, description="NLLB-200 FLORES code, e.g. 'cym_Latn'. None if unsupported."
    )
    scripts: list[str] = Field(..., min_length=1, description="Scripts, e.g. ['Latin'].")
    default_script: str | None = Field(default=None, description="Primary script if multiple.")
    direction: TextDirection = TextDirection.ltr
    family: str | None = Field(default=None, description="Language family, informational.")
    dialects: list[str] = Field(default_factory=list, description="Known dialects/varieties.")
    sources: list[SourceHint] = Field(default_factory=list, description="Corpus source catalog.")
    instruction_sources: list[InstructionSource] = Field(
        default_factory=list, description="Native-language instruction datasets for convdata."
    )
    normalization: NormalizationRules = Field(default_factory=NormalizationRules)
    notes: str | None = None

    model_config = {"extra": "forbid"}

    def resolved_script(self) -> str:
        return self.default_script or self.scripts[0]

    def lang_script_code(self) -> str:
        """Derive the standard ``<iso639_3>_<Script>`` code (e.g. 'kmr_Latn').

        Used to address per-language configs of multilingual datasets like
        FineWeb-2. Kurmanji -> ``kmr_Latn``, Sorani -> ``ckb_Arab``.
        """
        script = self.resolved_script()
        tag = _SCRIPT_TAG.get(script, script[:4].title())
        return f"{self.iso639_3}_{tag}"

    @model_validator(mode="after")
    def _add_standard_sources(self) -> LanguageProfile:
        """Auto-add FineWeb-2 for any language that doesn't already list it.

        Keeps standard web-text coverage DRY: a new language profile only needs
        iso639_3 + script and gets FineWeb-2 for free, with the correct per-language
        code. Languages whose FineWeb-2 code differs from the derived one (e.g. macro
        codes) override by listing ``fineweb2`` explicitly. A language not in
        FineWeb-2 simply logs and skips that source at ingest.
        """
        if not any(s.connector == "fineweb2" for s in self.sources):
            code = _FINEWEB2_OVERRIDE.get(self.iso639_3, self.lang_script_code())
            self.sources.append(
                SourceHint(
                    connector="fineweb2",
                    params={"fineweb": code, "license": "ODC-BY-1.0"},
                    notes="Auto-added: FineWeb-2 web text (ODC-BY-1.0).",
                )
            )
        return self


class ModelArch(str, Enum):
    decoder = "decoder"  # causal LM (Llama, Qwen, Gemma)
    encoder = "encoder"  # masked LM (XLM-R, mBERT)


class ModelProfile(BaseModel):
    """A base model that can be adapted to a target language."""

    name: str = Field(..., description="Profile slug, e.g. 'qwen2.5-1.5b'.")
    hf_id: str = Field(..., description="Hugging Face repo id.")
    family: str = Field(..., description="Model family, e.g. 'qwen', 'llama', 'gemma', 'xlmr'.")
    arch: ModelArch = ModelArch.decoder
    tokenizer_type: str = Field(default="bpe", description="'bpe' or 'sentencepiece'.")
    context_length: int = Field(default=2048)
    chat_template: str | None = Field(
        default=None, description="Override chat template; else use the model's own."
    )
    good_for_lrl: bool = Field(
        default=True, description="Whether this base is a sensible LRL default."
    )
    notes: str | None = None

    model_config = {"extra": "forbid"}


class Quantization(str, Enum):
    none = "none"
    int8 = "8bit"
    int4 = "4bit"


class Distributed(str, Enum):
    none = "none"
    ddp = "ddp"
    deepspeed = "deepspeed"
    fsdp = "fsdp"


class ComputeProfile(BaseModel):
    """A hardware/training environment profile.

    Stage defaults (batch size, quantization, etc.) are drawn from here so the
    same project YAML runs on a laptop GPU or a cluster by swapping one name.
    """

    name: str = Field(..., description="Profile slug, e.g. 'consumer_gpu'.")
    device: str = Field(default="cuda", description="'cuda', 'cpu', or 'mps'.")
    precision: str = Field(default="bf16", description="'bf16', 'fp16', or 'fp32'.")
    quantization: Quantization = Quantization.int4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    use_unsloth: bool = Field(default=False, description="Use Unsloth backend if available.")
    distributed: Distributed = Distributed.none
    max_seq_len_cap: int | None = Field(
        default=None, description="Hard cap on sequence length regardless of stage config."
    )
    notes: str | None = None

    model_config = {"extra": "forbid"}
