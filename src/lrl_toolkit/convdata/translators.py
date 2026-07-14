"""Translation backends for the ``translate`` conversational-data path.

Separate from the teacher LLM: for LRLs a dedicated multilingual MT model usually
beats a general chat model. All backends are local / open-weight:

* ``nllb``    — Meta NLLB-200 (default; 200+ languages incl. many LRLs).
* ``m2m100``  — Meta M2M-100 (100 languages, small/fast).
* ``opusmt``  — Helsinki-NLP OPUS-MT, one model per language pair (tiny/fast).
* ``madlad``  — Google MADLAD-400 MT (400+ languages; large 3B model).
* ``teacher`` — reuse the configured teacher LLM (Ollama/local) to translate.
* ``mock``    — deterministic, offline; for tests.

Each translator maps English instruction data into the target language described
by a :class:`LanguageProfile` (using its ``nllb_code`` / ``iso639_1``).
"""

from __future__ import annotations

import abc

from ..registry import LanguageProfile
from ..utils import get_logger

log = get_logger("lrl.convdata.translate")

_MAX_LEN = 512


class BaseTranslator(abc.ABC):
    name: str

    @abc.abstractmethod
    def translate(self, text: str, lang: LanguageProfile) -> str:
        ...


class MockTranslator(BaseTranslator):
    name = "mock"

    def translate(self, text: str, lang: LanguageProfile) -> str:
        return f"[{lang.display_name}] " + " ".join(text.split())


class TeacherTranslator(BaseTranslator):
    name = "teacher"

    def __init__(self, provider: str, model: str | None):
        from .teacher import get_teacher

        self._teacher = get_teacher(provider, model)

    def translate(self, text: str, lang: LanguageProfile) -> str:
        return self._teacher.translate(text, lang.display_name)


class _HFTranslator(BaseTranslator):
    """Lazily-loaded seq2seq MT model (tokenizer + model), driven via ``generate``.

    We call the model directly rather than the ``pipeline('translation')`` task,
    which is not stable across transformers versions.
    """

    default_model: str = ""

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model
        self._tok = None
        self._net = None

    def _load(self, name: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(name)
        self._net = AutoModelForSeq2SeqLM.from_pretrained(name)

    def _generate(self, text: str, **gen_kwargs) -> str:
        inputs = self._tok(text, return_tensors="pt", truncation=True, max_length=_MAX_LEN)
        out = self._net.generate(**inputs, max_length=_MAX_LEN, **gen_kwargs)
        return self._tok.batch_decode(out, skip_special_tokens=True)[0]

    def translate(self, text: str, lang: LanguageProfile) -> str:
        if self._net is None:
            self._build(lang)
        return self._run(text, lang)

    @abc.abstractmethod
    def _build(self, lang: LanguageProfile) -> None:
        ...

    @abc.abstractmethod
    def _run(self, text: str, lang: LanguageProfile) -> str:
        ...


class NLLBTranslator(_HFTranslator):
    name = "nllb"
    default_model = "facebook/nllb-200-distilled-600M"

    def _build(self, lang: LanguageProfile) -> None:
        if not lang.nllb_code:
            raise ValueError(
                f"{lang.display_name} has no nllb_code (not in NLLB-200). "
                "Use translate_backend: teacher (or madlad)."
            )
        log.info("[translate] loading NLLB %s", self.model)
        self._load(self.model)

    def _run(self, text: str, lang: LanguageProfile) -> str:
        self._tok.src_lang = "eng_Latn"
        bos = self._tok.convert_tokens_to_ids(lang.nllb_code)
        return self._generate(text, forced_bos_token_id=bos)


class M2M100Translator(_HFTranslator):
    name = "m2m100"
    default_model = "facebook/m2m100_418M"

    def _build(self, lang: LanguageProfile) -> None:
        if not lang.iso639_1:
            raise ValueError(f"{lang.display_name} has no iso639_1 code for M2M-100.")
        log.info("[translate] loading M2M-100 %s", self.model)
        self._load(self.model)

    def _run(self, text: str, lang: LanguageProfile) -> str:
        self._tok.src_lang = "en"
        return self._generate(text, forced_bos_token_id=self._tok.get_lang_id(lang.iso639_1))


class OpusMTTranslator(_HFTranslator):
    name = "opusmt"

    def _build(self, lang: LanguageProfile) -> None:
        if not (self.model or lang.iso639_1):
            raise ValueError(f"{lang.display_name} has no iso639_1 code for OPUS-MT.")
        name = self.model or f"Helsinki-NLP/opus-mt-en-{lang.iso639_1}"
        try:
            log.info("[translate] loading OPUS-MT %s", name)
            self._load(name)
        except Exception as exc:
            raise ValueError(
                f"OPUS-MT model '{name}' unavailable for en->{lang.iso639_1}. "
                "Not every pair has a model; try nllb or teacher."
            ) from exc

    def _run(self, text: str, lang: LanguageProfile) -> str:
        return self._generate(text)


class MadladTranslator(_HFTranslator):
    name = "madlad"
    default_model = "google/madlad400-3b-mt"

    def _build(self, lang: LanguageProfile) -> None:
        log.info("[translate] loading MADLAD-400 %s (large ~3B)", self.model)
        self._load(self.model)

    def _run(self, text: str, lang: LanguageProfile) -> str:
        code = lang.iso639_1 or lang.iso639_3
        return self._generate(f"<2{code}> {text}")


_BACKENDS = {
    "nllb": NLLBTranslator,
    "m2m100": M2M100Translator,
    "opusmt": OpusMTTranslator,
    "madlad": MadladTranslator,
}


def get_translator(
    backend: str,
    model: str | None = None,
    *,
    teacher_provider: str = "ollama",
    teacher_model: str | None = None,
) -> BaseTranslator:
    b = backend.lower()
    if b == "mock":
        return MockTranslator()
    if b == "teacher":
        return TeacherTranslator(teacher_provider, teacher_model)
    if b in _BACKENDS:
        return _BACKENDS[b](model)
    raise ValueError(
        f"Unknown translate backend '{backend}'. Use: nllb/m2m100/opusmt/madlad/teacher/mock."
    )


def available_backends() -> list[str]:
    return ["nllb", "m2m100", "opusmt", "madlad", "teacher", "mock"]
