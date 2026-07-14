"""Load open instruction datasets (to be translated into the target language).

Accepts a short alias (``dolly``, ``alpaca``, ``oasst1``), a raw Hugging Face
dataset id, or a local ``.jsonl`` path. Normalizes each example to
``{'instruction': str, 'response': str}``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

_ALIASES = {
    "dolly": ("databricks/databricks-dolly-15k", "instruction", "response"),
    "alpaca": ("tatsu-lab/alpaca", "instruction", "output"),
    "oasst1": ("OpenAssistant/oasst1", None, None),  # handled specially below
}


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
