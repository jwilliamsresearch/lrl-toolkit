"""Benchmark framework: coverage detection, deltas/CIs, and human review — all
offline (dataset config lookups are monkeypatched; no model or network needed)."""

import lrl_toolkit.evaluate.benchmarks.base as base_mod
from lrl_toolkit.evaluate import human_review as hr
from lrl_toolkit.evaluate.benchmarks import get_benchmark
from lrl_toolkit.evaluate.benchmarks.base import (
    BenchmarkResult,
    Coverage,
    MetricDirection,
    Score,
    bootstrap_ci,
)
from lrl_toolkit.evaluate.benchmarks.native_cloze import _held_out_prompts
from lrl_toolkit.registry import load_language


# ---- coverage: the core "does this benchmark apply to this language" logic ---
def test_belebele_coverage_uses_resolved_code(monkeypatch):
    # Sorani IS in Belebele (ckb_Arab); Kurmanji (kmr_Latn) is NOT.
    monkeypatch.setattr(base_mod, "dataset_configs",
                        lambda repo, token=None: frozenset({"ckb_Arab", "eng_Latn"}))
    bel = get_benchmark("belebele")
    assert bel.coverage(load_language("sorani"), has_token=False).available is True
    kmr = bel.coverage(load_language("kurmanji"), has_token=False)
    assert kmr.available is False
    assert kmr.resolved_code == "kmr_Latn"


def test_global_mmlu_coverage_needs_iso639_1(monkeypatch):
    monkeypatch.setattr(base_mod, "dataset_configs",
                        lambda repo, token=None: frozenset({"fa", "am", "en"}))
    mmlu = get_benchmark("global_mmlu")
    # Farsi has iso639_1 'fa' and is covered; Kurmanji ('ku') is not.
    assert mmlu.coverage(load_language("farsi"), has_token=False).available is True
    assert mmlu.coverage(load_language("kurmanji"), has_token=False).available is False


def test_flores_reports_gated_without_token():
    cov = get_benchmark("flores").coverage(load_language("kurmanji"), has_token=False)
    assert cov.available is False and cov.gated is True


def test_intrinsic_benchmarks_are_universal():
    for name in ("native_cloze", "perplexity"):
        cov = get_benchmark(name).coverage(load_language("kurmanji"), has_token=False)
        assert cov.available is True


# ---- deltas + CIs ------------------------------------------------------------
def test_delta_sign_respects_direction():
    up = BenchmarkResult("acc", MetricDirection.higher_better, Coverage(True, "ok"),
                         base=Score(0.40), adapted=Score(0.55))
    assert round(up.delta, 3) == 0.15  # improvement is positive
    down = BenchmarkResult("ppl", MetricDirection.lower_better, Coverage(True, "ok"),
                           base=Score(50.0), adapted=Score(30.0))
    assert down.delta == 20.0  # lower perplexity => positive (better) delta


def test_delta_none_when_missing_base():
    r = BenchmarkResult("ppl", MetricDirection.lower_better, Coverage(True, "ok"),
                        base=None, adapted=Score(30.0))
    assert r.delta is None


def test_bootstrap_ci_brackets_accuracy():
    lo, hi = bootstrap_ci([True] * 8 + [False] * 2, iters=500)
    assert 0.0 <= lo <= 0.8 <= hi <= 1.0


# ---- native cloze prompt construction --------------------------------------
def test_cloze_prompts_hold_out_final_word(monkeypatch, tmp_path):
    import lrl_toolkit.evaluate.benchmarks.native_cloze as nc

    class _Doc:
        def __init__(self, text):
            self.text = text

    docs = [_Doc(f"the quick brown fox jumps over the lazy dog{i}.") for i in range(20)]
    monkeypatch.setattr(nc, "iter_documents", lambda _dir: iter(docs))

    d = tmp_path / "corpus"
    d.mkdir()
    prompts = _held_out_prompts(d, limit=5)
    assert len(prompts) == 5
    for prompt, target in prompts:
        # The target is the sentence's final content word, held out of the prompt.
        assert target.startswith("dog")
        assert "dog" not in prompt


# ---- human output review (CARE-aligned scaffold) -----------------------------
def test_human_review_queue_and_summary(tmp_path):
    items = [{"instruction": "Q1", "response": "A1"}, {"instruction": "Q2", "response": "A2"}]
    queue = hr.build_output_queue(items)
    assert all(row["status"] == "pending" for row in queue)
    assert hr.summarize(queue)["status"] == "awaiting native-speaker review"

    # A reviewer fills one in; regenerating the queue must preserve that.
    path = tmp_path / "q.jsonl"
    queue[0].update({"status": "reviewed", "fluency": 4, "correctness": 5})
    hr.save_queue(queue, path)

    regenerated = hr.merge_reviews(hr.build_output_queue(items), path)
    assert regenerated[0]["status"] == "reviewed"
    assert regenerated[0]["fluency"] == 4
    summary = hr.summarize(regenerated)
    assert summary["reviewed"] == 1
    assert summary["mean_correctness"] == 5.0
