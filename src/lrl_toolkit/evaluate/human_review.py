"""Human evaluation of model outputs by native speakers.

Automated metrics are proxies; for the languages this toolkit serves, the people
best placed to judge a model's output are members of the language community — the
same principle (CARE: Authority to control, Responsibility) that puts humans in the
loop for *data* review in the convdata stage. This module extends that loop from
data to **model outputs**: it generates responses to held-out instructions and
writes a review queue in which a native speaker rates each for fluency and
correctness and can leave a correction.

It is deliberately a *queue producer + aggregator*, not an automated score: the
values only become meaningful once a human fills them in (via the dashboard or any
JSONL editor). Until then the report records it as awaiting review — honest by
construction, like the rest of the eval stage.
"""

from __future__ import annotations

from pathlib import Path

from ..convdata.schema import read_jsonl, write_jsonl

_RATING_FIELDS = ("fluency", "correctness")  # 1-5, filled by a human reviewer


def build_output_queue(items: list[dict]) -> list[dict]:
    """Wrap {instruction, response} items as pending review rows with rating slots."""
    return [
        {
            "id": i,
            "status": "pending",  # pending | reviewed | rejected
            "instruction": it["instruction"],
            "response": it["response"],
            "fluency": None,
            "correctness": None,
            "correction": None,
            "notes": None,
        }
        for i, it in enumerate(items)
    ]


def merge_reviews(new_queue: list[dict], existing_path: Path) -> list[dict]:
    """Preserve prior human ratings (by id) when the queue is regenerated."""
    if not existing_path.exists():
        return new_queue
    prior = {row["id"]: row for row in read_jsonl(existing_path)}
    for row in new_queue:
        old = prior.get(row["id"])
        if old and old.get("status") != "pending":
            row.update(
                {
                    "status": old.get("status", "pending"),
                    "fluency": old.get("fluency"),
                    "correctness": old.get("correctness"),
                    "correction": old.get("correction"),
                    "notes": old.get("notes"),
                }
            )
    return new_queue


def save_queue(queue: list[dict], path: str | Path) -> Path:
    return write_jsonl(path, queue)


def summarize(queue: list[dict]) -> dict:
    """Aggregate whatever a human has filled in so far (honest about coverage)."""
    reviewed = [r for r in queue if r.get("status") == "reviewed"]
    summary: dict[str, object] = {
        "total": len(queue),
        "reviewed": len(reviewed),
        "pending": len(queue) - len(reviewed),
    }
    for field in _RATING_FIELDS:
        vals = [r[field] for r in reviewed if isinstance(r.get(field), (int, float))]
        summary[f"mean_{field}"] = round(sum(vals) / len(vals), 3) if vals else None
    if not reviewed:
        summary["status"] = "awaiting native-speaker review"
    return summary
