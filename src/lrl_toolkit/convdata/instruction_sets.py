"""Load open instruction datasets (to be translated into the target language).

Accepts a short alias (``dolly``, ``alpaca``, ``oasst1``), a raw Hugging Face
dataset id, or a local ``.jsonl`` path. Normalizes each example to
``{'instruction': str, 'response': str}``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import NativeSetConfig

_ALIASES = {
    "dolly": ("databricks/databricks-dolly-15k", "instruction", "response"),
    "alpaca": ("tatsu-lab/alpaca", "instruction", "output"),
    "oasst1": ("OpenAssistant/oasst1", None, None),  # handled specially below
}


def _iter_local_raw(path: Path) -> Iterator[dict]:
    """Yield raw JSON objects from a .jsonl file (fields untouched)."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_native_set(cfg: NativeSetConfig) -> Iterator[dict]:
    """Yield normalized ``{instruction, response}`` from a target-language dataset.

    Used *as-is* (no translation) for data already in the target language, e.g.
    xP3x's per-language splits. Rows whose ``source_field`` contains any substring
    in ``cfg.exclude`` are dropped — this is how eval-only corpora (FLORES) are kept
    out of training.
    """
    excl = [e.lower() for e in cfg.exclude]
    select_field = getattr(cfg, "select_field", None)
    select_value = getattr(cfg, "select_value", None)

    def _excluded(row: dict) -> bool:
        if excl:
            src = str(row.get(cfg.source_field, "")).lower()
            if any(e in src for e in excl):
                return True
        # Inclusion filter: keep only rows matching select_value (e.g. one
        # language's rows in a mixed dataset like Aya).
        if select_field is not None and str(row.get(select_field)) != str(select_value):
            return True
        return False

    p = Path(cfg.repo)
    if p.exists() and p.is_file():
        rows: Iterator[dict] = _iter_local_raw(p)
    else:
        from datasets import load_dataset

        rows = load_dataset(cfg.repo, cfg.name, split=cfg.split, streaming=True)

    n = 0
    for row in rows:
        if _excluded(row):
            continue
        instr = row.get(cfg.instruction_field)
        resp = row.get(cfg.response_field)
        if instr and resp:
            yield {"instruction": str(instr), "response": str(resp)}
            n += 1
            if cfg.limit is not None and n >= cfg.limit:
                return


def _iter_local(path: Path, limit: int | None) -> Iterator[dict]:
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            instr = rec.get("instruction") or rec.get("prompt") or rec.get("input")
            resp = rec.get("response") or rec.get("output") or rec.get("completion")
            if instr and resp:
                yield {"instruction": instr, "response": resp}
                n += 1
                if limit is not None and n >= limit:
                    return


def load_instruction_set(name: str, limit: int | None = None) -> Iterator[dict]:
    """Yield normalized {instruction, response} examples from a dataset."""
    # Local file path takes precedence.
    p = Path(name)
    if p.exists() and p.is_file():
        yield from _iter_local(p, limit)
        return

    from datasets import load_dataset

    repo, instr_field, resp_field = _ALIASES.get(name, (name, "instruction", "response"))

    if repo == "OpenAssistant/oasst1":
        yield from _iter_oasst(limit)
        return

    ds = load_dataset(repo, split="train", streaming=True)
    n = 0
    for row in ds:
        instr = row.get(instr_field)
        resp = row.get(resp_field)
        if instr and resp:
            yield {"instruction": instr, "response": resp}
            n += 1
            if limit is not None and n >= limit:
                return


def _iter_oasst(limit: int | None) -> Iterator[dict]:
    """Pair top-level prompter messages with their first assistant reply."""
    from datasets import load_dataset

    ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    prompts: dict[str, str] = {}
    n = 0
    for row in ds:
        if row.get("role") == "prompter" and row.get("parent_id") is None:
            prompts[row["message_id"]] = row["text"]
        elif row.get("role") == "assistant" and row.get("parent_id") in prompts:
            yield {"instruction": prompts[row["parent_id"]], "response": row["text"]}
            n += 1
            if limit is not None and n >= limit:
                return
