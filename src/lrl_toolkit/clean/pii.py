"""Lightweight PII scrubbing.

Regex-based redaction of emails, URLs, and long digit runs (phone/card-like).
Deliberately conservative — it errs toward leaving text intact rather than
mangling it, and is not a substitute for a full PII pipeline where one is needed.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# 7+ digit runs (allowing spaces/dashes) catch most phone / card numbers.
_LONG_DIGITS_RE = re.compile(r"(?<!\w)(?:\d[\d\s-]{6,}\d)(?!\w)")


def scrub(text: str) -> tuple[str, int]:
    """Redact PII. Returns (scrubbed_text, n_redactions)."""
    count = 0

    def _sub(pattern: re.Pattern, token: str, s: str) -> str:
        nonlocal count
        new, n = pattern.subn(token, s)
        count += n
        return new

    text = _sub(_EMAIL_RE, "[EMAIL]", text)
    text = _sub(_URL_RE, "[URL]", text)
    text = _sub(_LONG_DIGITS_RE, "[NUM]", text)
    return text, count
