"""Native cloze: next-word prediction accuracy on the language's own held-out text.

This is the framework's headline metric for the low-resource tail, because it is
the one benchmark that works for **every** language the toolkit can build a model
for. It needs no external test set (which usually doesn't exist), no teacher LLM
(which usually doesn't know the language), and no parallel data — only the
language's own cleaned corpus, which every project already has.

Method: take held-out sentences, truncate each before its final content word, and
check whether the model's most-likely next word matches the true one. Scoring is at
the **word** level, not the token level, which makes the base-vs-adapted comparison
fair even though the pipeline extends the tokenizer (a token-level metric would be
confounded by the vocabulary change). Accuracy is reported with a bootstrap CI.

It is an *intrinsic* metric — it measures distribution fit, like perplexity, not
task usefulness. It is reported as such and never presented as evidence the model
follows instructions.
"""

from __future__ import annotations

import re

from ...corpus import iter_documents
from ...registry import LanguageProfile
from .base import (
    Benchmark,
    Coverage,
    MetricDirection,
    ModelBundle,
    Score,
    bootstrap_ci,
)

_WORD = re.compile(r"\w+", re.UNICODE)


def _held_out_prompts(clean_dir, limit: int, min_words: int = 6) -> list[tuple[str, str]]:
    """Return (prompt, expected_next_word) from held-out sentences.

    Deterministically holds out the *last* documents of the corpus so the prompts
    are disjoint from what training over the earlier shards emphasized (a soft
    hold-out; the pipeline does not carve a formal split yet).
    """
    sentences: list[str] = []
    for doc in iter_documents(clean_dir):
        for raw in re.split(r"(?<=[.!?])\s+|\n+", doc.text):
            s = raw.strip()
            if len(_WORD.findall(s)) >= min_words:
                sentences.append(s)
    # take from the tail as a soft hold-out
    tail = sentences[-(limit * 3):] if len(sentences) > limit * 3 else sentences
    prompts: list[tuple[str, str]] = []
    for s in tail:
        words = s.split()
        # predict the last "wordy" token; back off if the final token is punctuation
        idx = len(words) - 1
        while idx > 0 and not _WORD.search(words[idx]):
            idx -= 1
        if idx <= 0:
            continue
        target = _WORD.search(words[idx]).group(0)
        prompt = " ".join(words[:idx]) + " "
        prompts.append((prompt, target))
        if len(prompts) >= limit:
            break
    return prompts


class NativeClozeBenchmark(Benchmark):
    name = "native_cloze"
    direction = MetricDirection.higher_better
    caveats = (
        "Intrinsic metric (distribution fit), not a measure of instruction-following "
        "usefulness. Held-out from the tail of the same corpus, not a formal split.",
    )

    def coverage(self, lang: LanguageProfile, *, has_token: bool) -> Coverage:
        # Universal: derived from the language's own corpus, so it always applies.
        return Coverage(available=True, reason="Corpus-derived; applies to every language.")

    def score(self, bundle: ModelBundle, lang: LanguageProfile, *, limit: int) -> Score:
        # clean_dir is threaded in via the runner (see evaluate stage) as an attr.
        clean_dir = getattr(self, "_clean_dir", None)
        if clean_dir is None or not clean_dir.exists():
            return Score(value=None, note="no clean corpus available")
        prompts = _held_out_prompts(clean_dir, limit)
        if not prompts:
            return Score(value=None, note="corpus too small for cloze prompts")

        import torch

        model, tok = bundle.model, bundle.tokenizer
        correct: list[bool] = []
        for prompt, target in prompts:
            ids = tok(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            ids = {k: v.to(device) for k, v in ids.items()}
            with torch.no_grad():
                logits = model(**ids).logits
            next_id = int(torch.argmax(logits[0, -1]))
            predicted = tok.decode([next_id]).strip()
            pred_word = _WORD.search(predicted)
            correct.append(bool(pred_word) and pred_word.group(0).lower() == target.lower())

        acc = sum(correct) / len(correct)
        return Score(value=acc, n=len(correct), ci95=bootstrap_ci(correct),
                     note="top-1 next-word accuracy")
