"""Unified chat-pair schema (ChatML/ShareGPT-compatible) and JSONL IO."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any


def to_messages(instruction: str, response: str, system: str | None = None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": response})
    return messages


def chat_pair(
    instruction: str,
    response: str,
    source: str,
    *,
    system: str | None = None,
    meta: dict | None = None,
) -> dict:
    return {
        "messages": to_messages(instruction, response, system),
        "source": source,
        "meta": meta or {},
    }


def instruction_of(pair: dict) -> str:
    for m in pair.get("messages", []):
        if m["role"] == "user":
            return m["content"]
    return ""


def response_of(pair: dict) -> str:
    for m in pair.get("messages", []):
        if m["role"] == "assistant":
            return m["content"]
    return ""


def is_degenerate(text: str, *, max_run: int = 6) -> bool:
    """Catch repetition-loop degeneration: the same word repeated many times in a
    row (e.g. "zûtirîn zûtirîn zûtirîn ..."). Undertrained models and machine
    translation both occasionally produce this; it's a different failure shape
    than the clean stage's line-level duplicate check, so it needs its own gate.
    """
    words = text.split()
    if not words:
        return False
    longest_run = max((len(list(g)) for _, g in itertools.groupby(words)), default=1)
    return longest_run >= max_run


def pair_is_degenerate(pair: dict, *, max_run: int = 6) -> bool:
    return is_degenerate(instruction_of(pair), max_run=max_run) or is_degenerate(
        response_of(pair), max_run=max_run
    )


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
