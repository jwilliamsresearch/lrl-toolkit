"""Language identification for LRLs.

Wraps GlotLID (``cis-lmu/glotlid``), a fastText model covering 2000+ languages —
far better LRL coverage than the classic lid.176. The model is downloaded from
the Hub on first use. If fastText/the model is unavailable the identifier reports
``available=False`` and the clean stage skips language filtering (rather than
silently dropping everything).
"""

from __future__ import annotations

from ..utils import get_logger

log = get_logger("lrl.clean.langid")


class LanguageIdentifier:
    """Lazy GlotLID / fastText language identifier."""

    def __init__(self, backend: str = "glotlid"):
        self.backend = backend
        self._model = None
        self.available = False
        if backend != "none":
            self._try_load()

    def _try_load(self) -> None:
        try:
            import fasttext
            from huggingface_hub import hf_hub_download

            if self.backend == "glotlid":
                path = hf_hub_download("cis-lmu/glotlid", "model.bin")
            else:  # generic fastText lid.176
                path = hf_hub_download("facebook/fasttext-language-identification", "model.bin")
            self._model = fasttext.load_model(path)
            self.available = True
        except Exception as exc:  # missing dep or offline -> disable gracefully
            log.warning("Language ID unavailable (%s); skipping language filter.", exc)
            self.available = False

    def predict(self, text: str) -> tuple[str, float]:
        """Return (iso639_3_label, probability). Empty label if unavailable."""
        if not self.available or self._model is None:
            return "", 0.0
        # fastText wants single-line input.
        labels, probs = self._model.predict(text.replace("\n", " "), k=1)
        if not labels:
            return "", 0.0
        # GlotLID labels look like "__label__cym_Latn"; keep the ISO 639-3 part.
        raw = labels[0].replace("__label__", "")
        iso = raw.split("_")[0]
        return iso, float(probs[0])
