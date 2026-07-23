"""LLM-as-judge: a local teacher LLM rates the model's responses (1-5).

The model is prompted with held-out instructions (drawn from the convdata review
queue), generates responses, and a local/open teacher scores each for helpfulness
and fluency in the target language. This is the only benchmark here that touches
what the model is actually *for* — instruction-following — but it comes with the
loudest caveat in the framework:

**For genuinely low-resource languages the judge is unreliable**, because the judge
model often does not know the language well either and ends up grading confident-
sounding fluency rather than correctness. It is reported, never used as a headline
for a low-resource language, and always carries that warning. It also respects the
toolkit's rule that only local/open teachers are used — never a proprietary API.
"""

from __future__ import annotations

import re

from ...registry import LanguageProfile
from .base import Benchmark, Coverage, MetricDirection, ModelBundle, Score, bootstrap_ci

_SCORE_RE = re.compile(r"[1-5]")


class JudgeBenchmark(Benchmark):
    name = "judge"
    direction = MetricDirection.higher_better
    compare_to_base = True
    caveats = (
        "UNRELIABLE for low-resource languages: the judge model may not know the "
        "language and can grade fluency-of-confidence, not correctness. Never treat "
        "as a headline metric for a low-resource language.",
    )

    def coverage(self, lang: LanguageProfile, *, has_token: bool) -> Coverage:
        # Needs held-out instructions to answer; those come from convdata.
        prompts = getattr(self, "_prompts", None)
        if not prompts:
            return Coverage(False, "No held-out instructions available (run convdata first).")
        return Coverage(True, "Local teacher will rate model responses (see caveat).")

    def score(self, bundle: ModelBundle, lang: LanguageProfile, *, limit: int) -> Score:
        import torch

        prompts = list(getattr(self, "_prompts", []))[:limit]
        if not prompts:
            return Score(value=None, note="no instructions to judge")
        teacher = getattr(self, "_teacher", None)
        if teacher is None:
            return Score(value=None, note="no teacher configured")

        model, tok = bundle.model, bundle.tokenizer
        device = next(model.parameters()).device
        ratings: list[float] = []
        for instruction in prompts:
            ids = tok(instruction, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=128, do_sample=False)
            response = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            rating = self._rate(teacher, lang.display_name, instruction, response)
            if rating is not None:
                ratings.append(rating)
        if not ratings:
            return Score(value=None, note="judge returned no parseable scores")
        mean = sum(ratings) / len(ratings)
        # normalize a 1-5 score to [0,1] for a comparable direction; keep raw in note
        correct_like = [(r - 1) / 4 for r in ratings]
        lo, hi = bootstrap_ci([bool(c > 0.5) for c in correct_like])  # coarse CI proxy
        return Score(value=mean, n=len(ratings), ci95=(lo * 4 + 1, hi * 4 + 1),
                     note="mean 1-5 rating by local judge (see caveat)")

    @staticmethod
    def _rate(teacher, lang_name: str, instruction: str, response: str) -> float | None:
        system = (
            f"You are grading an AI assistant's reply in {lang_name}. Rate it 1-5 for "
            "helpfulness and fluency. Reply with ONLY the digit."
        )
        prompt = f"Instruction: {instruction}\nReply: {response}\nScore (1-5):"
        try:
            raw = teacher._chat(system, prompt, max_tokens=8)  # noqa: SLF001
        except Exception:
            return None
        m = _SCORE_RE.search(raw or "")
        return float(m.group(0)) if m else None
