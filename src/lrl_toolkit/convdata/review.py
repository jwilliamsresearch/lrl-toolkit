"""Human-in-the-loop review queue for conversational pairs.

Pairs can be routed through a review queue where native speakers accept, edit, or
reject them before they train a model (see DATA_ETHICS.md). The queue is a plain
JSONL so reviewers can use the dashboard or any external tool. When review is
disabled, all pairs are auto-accepted.
"""

from __future__ import annotations

from pathlib import Path

from .schema import read_jsonl, write_jsonl


def build_queue(pairs: list[dict]) -> list[dict]:
    """Wrap pairs as review items with a pending status and stable ids."""
    return [
        {"id": i, "status": "pending", **pair}
        for i, pair in enumerate(pairs)
    ]


def merge_reviews(new_queue: list[dict], existing_path: Path) -> list[dict]:
    """Preserve prior review decisions (by id) when regenerating the queue."""
    if not existing_path.exists():
        return new_queue
    prior = {item["id"]: item.get("status", "pending") for item in read_jsonl(existing_path)}
    for item in new_queue:
        if item["id"] in prior:
            item["status"] = prior[item["id"]]
    return new_queue


def _strip(q: dict) -> dict:
    return {"messages": q["messages"], "source": q.get("source", ""), "meta": q.get("meta", {})}


def accepted_pairs(queue: list[dict], *, review_enabled: bool) -> list[dict]:
    """Return the pairs to train on."""
    if not review_enabled:
        return [_strip(q) for q in queue]
    return [_strip(q) for q in queue if q.get("status") == "accepted"]


def save_queue(queue: list[dict], path: str | Path) -> Path:
    return write_jsonl(path, queue)
