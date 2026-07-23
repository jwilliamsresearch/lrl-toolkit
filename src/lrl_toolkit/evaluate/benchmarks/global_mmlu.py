"""Global-MMLU: multilingual knowledge multiple-choice (CohereForAI/Global-MMLU).

42 languages, keyed by ISO 639-1 (e.g. `fa` Farsi, `am` Amharic). Kurdish is not
covered at all — so for Kurmanji this benchmark honestly reports "not covered,"
which is exactly the coverage-gap story the framework is meant to make visible.

Same log-likelihood MC scoring as Belebele. Because MMLU probes world knowledge the
base model largely already has, the base-vs-adapted delta here is expected to be
small — and that's a *finding*, not a failure: continued pretraining on a small LRL
corpus improves language modeling, not general knowledge. Reporting the delta keeps
that honest.
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

_REPO = "CohereForAI/Global-MMLU"


class GlobalMMLUBenchmark(Benchmark):
    name = "global_mmlu"
    direction = MetricDirection.higher_better
    caveats = (
        "MMLU probes world knowledge already largely present in the base model; a "
        "near-zero delta is expected and is not a defect of language adaptation.",
    )

    def _code(self, lang: LanguageProfile) -> str | None:
        return lang.iso639_1

    def coverage(self, lang: LanguageProfile, *, has_token: bool) -> Coverage:
        code = self._code(lang)
        if not code:
            return Coverage(False, f"{lang.display_name} has no ISO 639-1 code for Global-MMLU.")
        configs = dataset_configs(_REPO)
        if configs is None:
            return Coverage(False, "Could not list Global-MMLU configs (offline).", code)
        if code in configs:
            return Coverage(True, f"Global-MMLU covers '{code}'.", code)
        return Coverage(False, f"{lang.display_name} ('{code}') is not in Global-MMLU.", code)

    def score(self, bundle: ModelBundle, lang: LanguageProfile, *, limit: int) -> Score:
        from datasets import load_dataset

        code = self._code(lang)
        ds = load_dataset(_REPO, code, split="test", streaming=True)
        correct: list[bool] = []
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        for row in ds:
            prompt = f"{row['question']}\nAnswer:"
            options = [row["option_a"], row["option_b"], row["option_c"], row["option_d"]]
            pred = choose_mc(bundle.model, bundle.tokenizer, prompt, options)
            gold = letter_to_idx.get(str(row["answer"]).strip().upper())
            if gold is None:
                continue
            correct.append(pred == gold)
            if len(correct) >= limit:
                break
        if not correct:
            return Score(value=None, note="no items scored")
        acc = sum(correct) / len(correct)
        return Score(value=acc, n=len(correct), ci95=bootstrap_ci(correct),
                     note="4-way MC accuracy (log-likelihood scoring)")
