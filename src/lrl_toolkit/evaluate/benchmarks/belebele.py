"""Belebele: multiple-choice machine reading comprehension (facebook/belebele).

122 language variants — but *not* all languages, and the split is by specific
code: Sorani Kurdish is present as `ckb_Arab`, Kurmanji (`kmr_Latn`) is not. So
coverage is resolved from the language's own `<iso639_3>_<Script>` code and honestly
reports "not covered" for the majority of the toolkit's languages.

Scored by comparing the length-normalized log-probability the model assigns to each
of the four answer choices (standard log-likelihood MC scoring), so it needs no
generation and works on a base LM. Accuracy is reported with a bootstrap CI.

Caveat surfaced in every result: Belebele's passages are derived from FLORES, and a
strong base model may have seen both — so the *delta* over base is the trustworthy
signal, not the absolute score.
"""

from __future__ import annotations

from ...registry import LanguageProfile
from .base import (
    Benchmark,
    Coverage,
    MetricDirection,
    ModelBundle,
    Score,
    bootstrap_ci,
    choose_mc,
    dataset_configs,
)

_REPO = "facebook/belebele"


class BelebeleBenchmark(Benchmark):
    name = "belebele"
    direction = MetricDirection.higher_better
    caveats = (
        "Belebele passages derive from FLORES; a strong base model may have seen "
        "them. Read the base-vs-adapted delta, not the absolute score.",
    )

    def _code(self, lang: LanguageProfile) -> str:
        return lang.lang_script_code()

    def coverage(self, lang: LanguageProfile, *, has_token: bool) -> Coverage:
        code = self._code(lang)
        configs = dataset_configs(_REPO)
        if configs is None:
            return Coverage(False, "Could not list Belebele configs (offline).", code)
        if code in configs:
            return Coverage(True, f"Belebele covers {code}.", code)
        return Coverage(False, f"{lang.display_name} ({code}) is not in Belebele.", code)

    def score(self, bundle: ModelBundle, lang: LanguageProfile, *, limit: int) -> Score:
        from datasets import load_dataset

        code = self._code(lang)
        ds = load_dataset(_REPO, code, split="test", streaming=True)
        correct: list[bool] = []
        for row in ds:
            prompt = (
                f"{row['flores_passage']}\n"
                f"Question: {row['question']}\n"
                f"Answer:"
            )
            options = [row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]]
            pred = choose_mc(bundle.model, bundle.tokenizer, prompt, options)
            gold = int(row["correct_answer_num"]) - 1
            correct.append(pred == gold)
            if len(correct) >= limit:
                break
        if not correct:
            return Score(value=None, note="no items scored")
        acc = sum(correct) / len(correct)
        return Score(value=acc, n=len(correct), ci95=bootstrap_ci(correct),
                     note="4-way MC accuracy (log-likelihood scoring)")
