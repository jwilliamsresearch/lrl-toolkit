"""Benchmark framework: coverage-aware, base-vs-adapted, honest-by-construction.

The evaluation problem for low-resource languages is *coverage*, not computation.
Most external benchmarks (Belebele, Global-MMLU, FLORES) exist for only a fraction
of the world's languages — Kurmanji, for instance, has none of them. So a benchmark
here is not just "a thing that produces a score"; it is a thing that first reports
**whether it applies at all** to a given language and *why*, and only then runs.

Two design commitments make the numbers trustworthy rather than decorative:

1. **Base-vs-adapted deltas.** A raw score on an adapted model conflates what the
   pipeline added with what the base model already knew (Qwen3 has seen a lot).
   Every benchmark is run on *both* the untouched base model and the adapted model,
   and the delta is the headline. A metric that only moves because the base was
   already good is not a result about the toolkit.

2. **Coverage as a first-class artifact.** The evaluate stage emits a coverage
   matrix (which benchmarks apply, which don't, and the reason) so a language with
   no external benchmarks is documented as such rather than silently scored on
   nothing. For that long tail, the corpus-derived intrinsic metrics
   (native_cloze, perplexity) are the honest fallback.
"""

from __future__ import annotations

import abc
import functools
from dataclasses import dataclass, field
from enum import Enum

from ...registry import LanguageProfile


@functools.lru_cache(maxsize=32)
def dataset_configs(repo: str, token: str | None = None) -> frozenset[str] | None:
    """Config names available for an HF dataset, or None if they can't be listed
    (offline, or gated without access). Cached so coverage detection is cheap.

    Isolated behind one function so tests can monkeypatch it and run fully offline.
    """
    try:
        from datasets import get_dataset_config_names

        return frozenset(get_dataset_config_names(repo, token=token))
    except Exception:
        return None


class MetricDirection(str, Enum):
    higher_better = "higher_better"  # accuracy, chrF, judge score
    lower_better = "lower_better"  # perplexity


@dataclass
class Coverage:
    """Whether a benchmark can run for a language, and why / why not."""

    available: bool
    reason: str
    resolved_code: str | None = None  # e.g. the Belebele config 'ckb_Arab'
    gated: bool = False  # available in principle but needs auth (e.g. HF_TOKEN)


@dataclass
class Score:
    """A single measurement with enough context to be honest about it."""

    value: float | None
    n: int = 0  # number of items scored (0 => not run)
    ci95: tuple[float, float] | None = None  # bootstrap 95% CI where meaningful
    note: str | None = None


@dataclass
class BenchmarkResult:
    """Base vs. adapted, plus the delta and any caveats a reader needs."""

    benchmark: str
    direction: MetricDirection
    coverage: Coverage
    base: Score | None = None
    adapted: Score | None = None
    caveats: list[str] = field(default_factory=list)

    @property
    def delta(self) -> float | None:
        """Improvement of adapted over base, signed so positive is always better."""
        if (
            self.base is None
            or self.adapted is None
            or self.base.value is None
            or self.adapted.value is None
        ):
            return None
        raw = self.adapted.value - self.base.value
        return raw if self.direction is MetricDirection.higher_better else -raw

    def to_dict(self) -> dict:
        def _score(s: Score | None) -> dict | None:
            if s is None:
                return None
            return {"value": s.value, "n": s.n, "ci95": list(s.ci95) if s.ci95 else None,
                    "note": s.note}

        return {
            "benchmark": self.benchmark,
            "direction": self.direction.value,
            "coverage": {
                "available": self.coverage.available,
                "reason": self.coverage.reason,
                "resolved_code": self.coverage.resolved_code,
                "gated": self.coverage.gated,
            },
            "base": _score(self.base),
            "adapted": _score(self.adapted),
            "delta": self.delta,
            "caveats": self.caveats,
        }


@dataclass
class ModelBundle:
    """A model + its tokenizer + a label ('base' | 'adapted')."""

    model: object
    tokenizer: object
    label: str


class Benchmark(abc.ABC):
    """One evaluation. Subclasses declare coverage and how to score a bundle."""

    #: Registry slug, e.g. 'belebele'.
    name: str
    #: Which direction is "better" for this metric.
    direction: MetricDirection = MetricDirection.higher_better
    #: Fixed caveats always attached to this benchmark's result (e.g. contamination).
    caveats: tuple[str, ...] = ()
    #: Whether this benchmark needs the base model too (intrinsic ones still do,
    #: for the delta). Set False only for metrics where base comparison is
    #: methodologically invalid (documented on the subclass).
    compare_to_base: bool = True

    @abc.abstractmethod
    def coverage(self, lang: LanguageProfile, *, has_token: bool) -> Coverage:
        """Can this benchmark run for `lang`? Report the reason either way."""

    @abc.abstractmethod
    def score(self, bundle: ModelBundle, lang: LanguageProfile, *, limit: int) -> Score:
        """Score one model bundle. Only called when coverage.available is True."""


# --------------------------------------------------------------------------- #
# Shared scoring helpers
# --------------------------------------------------------------------------- #
def continuation_logprob(model, tokenizer, prompt: str, continuation: str) -> tuple[float, int]:
    """Sum log-probability the model assigns to `continuation` following `prompt`.

    Returns (total_logprob, n_continuation_tokens). Used for multiple-choice
    benchmarks (score each option, pick the most likely) and for cloze.
    """
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if not cont_ids:
        return float("-inf"), 0
    input_ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    start = len(prompt_ids)
    for i in range(start, start + len(cont_ids)):
        tok = input_ids[0, i]
        total += float(log_probs[0, i - 1, tok])
    return total, len(cont_ids)


def choose_mc(model, tokenizer, prompt: str, options: list[str]) -> int:
    """Return the index of the highest length-normalized log-prob option."""
    scored = []
    for opt in options:
        lp, n = continuation_logprob(model, tokenizer, prompt, " " + opt.strip())
        scored.append(lp / max(n, 1))  # length-normalize so long answers aren't penalized
    return max(range(len(scored)), key=lambda i: scored[i])


def bootstrap_ci(correct: list[bool], *, iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for a proportion (accuracy). Cheap, distribution-free."""
    import random

    if not correct:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(correct)
    means = []
    for _ in range(iters):
        s = sum(correct[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (lo, hi)
