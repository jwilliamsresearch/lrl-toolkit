"""Text normalization: Unicode form, digit folding, and named script rules.

Rule sets are referenced by name from a language profile's ``normalization.rules``
and applied in order after Unicode normalization. They target the specific
inconsistencies that hurt LRL tokenization — e.g. Arabic vs. Persian/Kurdish
letter variants, tatweel, and zero-width non-joiner handling.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from ..registry import NormalizationRules

# Arabic-Indic (U+0660..) and extended/Persian (U+06F0..) digits -> ASCII.
_ARABIC_INDIC = "".join(chr(0x0660 + i) for i in range(10))
_EXT_ARABIC_INDIC = "".join(chr(0x06F0 + i) for i in range(10))
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(_ARABIC_INDIC)}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(_EXT_ARABIC_INDIC)})

_TATWEEL = "ـ"  # Arabic tatweel / kashida
_ZWNJ = "‌"  # zero-width non-joiner
_ARABIC_YEH = "ي"  # ي
_PERSIAN_YEH = "ی"  # ی
_ARABIC_KAF = "ك"  # ك
_PERSIAN_KAF = "ک"  # ک

# Strip C0/C1 control chars except tab (\t) and newline (\n).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _strip_controls(text: str) -> str:
    return _CONTROL_RE.sub("", text)


def _persian_yeh_kaf(text: str) -> str:
    # Normalize Arabic yeh/kaf to Persian forms; drop tatweel; collapse ZWNJ runs.
    text = text.replace(_ARABIC_YEH, _PERSIAN_YEH)
    text = text.replace(_ARABIC_KAF, _PERSIAN_KAF)
    text = text.replace(_TATWEEL, "")
    text = re.sub(_ZWNJ + "+", _ZWNJ, text)
    return text


def _sorani_arabic(text: str) -> str:
    # Central Kurdish shares yeh/kaf normalization with Persian. We deliberately
    # avoid rewriting heh/ae (U+06D5) since that is orthography-dependent.
    return _persian_yeh_kaf(text)


def _arabic_presentation_forms(text: str) -> str:
    # Presentation-form ligatures/isolated glyphs -> canonical letters.
    return unicodedata.normalize("NFKC", text)


def _kurdish_latin(text: str) -> str:
    # Kurmanji uses î û ê ç ş; NFC (applied earlier) composes diacritics. Here we
    # only strip stray control chars.
    return _strip_controls(text)


def _welsh_digraphs(text: str) -> str:
    # Welsh digraphs (ch, dd, ff, ng, ll, ph, rh, th) are two code points and need
    # no character rewriting; digraph-awareness belongs to tokenization. No-op.
    return text


_RULES: dict[str, Callable[[str], str]] = {
    "persian_yeh_kaf": _persian_yeh_kaf,
    "sorani_arabic": _sorani_arabic,
    "arabic_presentation_forms": _arabic_presentation_forms,
    "kurdish_latin": _kurdish_latin,
    "welsh_digraphs": _welsh_digraphs,
}


def normalize_text(text: str, rules: NormalizationRules) -> str:
    """Apply Unicode normalization, optional digit folding, then named rules."""
    text = unicodedata.normalize(rules.unicode_form, text)
    text = _strip_controls(text)
    if rules.normalize_digits:
        text = text.translate(_DIGIT_MAP)
    for name in rules.rules:
        fn = _RULES.get(name)
        if fn is not None:
            text = fn(text)
    # Collapse excessive whitespace while preserving paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def known_rules() -> list[str]:
    return sorted(_RULES)
