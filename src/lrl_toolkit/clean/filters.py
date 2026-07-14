"""Document quality filters.

Pure-Python heuristics (no heavy deps) that catch the common junk in scraped
LRL text: too-short fragments, symbol/markup soup, and heavy repetition. Returns
a quality score in [0, 1] plus a hard-drop reason for disqualifying documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"\w+", re.UNICODE)
# "Letter-ish" = any word char or whitespace; everything else counts as symbol.
_NON_TEXT_RE = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass
class QualityResult:
    keep: bool
    score: float
    reason: str | None = None


def _repetition_ratio(text: str) -> float:
    """Fraction of lines that are duplicates (0 = all unique, 1 = all dupes)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return 0.0
    unique = len(set(lines))
    return 1.0 - (unique / len(lines))


def assess(
    text: str,
    *,
    min_chars: int = 200,
    min_words: int = 20,
    max_symbol_ratio: float = 0.3,
    max_repetition: float = 0.5,
    min_quality: float = 0.5,
) -> QualityResult:
    """Score a document and decide whether to keep it."""
    n_chars = len(text)
    if n_chars < min_chars:
        return QualityResult(False, 0.0, "too_short_chars")

    words = _WORD_RE.findall(text)
    if len(words) < min_words:
        return QualityResult(False, 0.0, "too_few_words")

    symbols = len(_NON_TEXT_RE.findall(text))
    symbol_ratio = symbols / max(n_chars, 1)
    if symbol_ratio > max_symbol_ratio:
        return QualityResult(False, 0.0, "symbol_heavy")

    repetition = _repetition_ratio(text)
    if repetition > max_repetition:
        return QualityResult(False, 0.0, "repetitive")

    # Composite score: reward text density, penalize symbols and repetition.
    score = (
        (1.0 - symbol_ratio / max_symbol_ratio) * 0.5
        + (1.0 - repetition) * 0.5
    )
    score = max(0.0, min(1.0, score))
    if score < min_quality:
        return QualityResult(False, score, "low_quality")
    return QualityResult(True, score, None)
