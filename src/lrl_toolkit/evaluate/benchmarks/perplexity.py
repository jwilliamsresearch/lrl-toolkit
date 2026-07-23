"""Held-out perplexity — intrinsic, corpus-derived, universal.

Kept as a benchmark for continuity, but note: unlike the other metrics it does
**not** produce a base-vs-adapted delta, because the pipeline extends the
tokenizer and token-level perplexity is not comparable across different
vocabularies (the adapted model tokenizes the same text into fewer/more tokens).
For a fair "did adaptation help the LM" signal, native_cloze (word-level) is the
one to read. Perplexity is reported for the adapted model only.
"""

from __future__ import annotations

from ...registry import LanguageProfile
from .base import Benchmark, Coverage, MetricDirection, ModelBundle, Score


class PerplexityBenchmark(Benchmark):
    name = "perplexity"
    direction = MetricDirection.lower_better
    compare_to_base = False  # token-level PPL is not comparable across tokenizers
    caveats = (
        "Adapted-model only: token-level perplexity is not comparable across the "
        "base and extended tokenizers, so no base delta is reported. See native_cloze "
        "for a tokenizer-fair intrinsic comparison.",
    )

    def coverage(self, lang: LanguageProfile, *, has_token: bool) -> Coverage:
        return Coverage(available=True, reason="Corpus-derived; applies to every language.")

    def score(self, bundle: ModelBundle, lang: LanguageProfile, *, limit: int) -> Score:
        clean_dir = getattr(self, "_clean_dir", None)
        if clean_dir is None or not clean_dir.exists():
            return Score(value=None, note="no clean corpus available")
        from ...corpus import iter_documents
        from ..metrics import perplexity

        texts = [d.text for d in iter_documents(clean_dir)][:limit]
        if not texts:
            return Score(value=None, note="empty corpus")
        ppl = perplexity(bundle.model, bundle.tokenizer, texts, seq_len=256, max_blocks=20)
        return Score(value=ppl, n=len(texts), note="token-level perplexity")
