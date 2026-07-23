"""Benchmark registry: maps names to Benchmark instances (plugin pattern, like the
ingest connectors / teachers / translators)."""

from __future__ import annotations

from .base import Benchmark
from .belebele import BelebeleBenchmark
from .flores_chrf import FloresChrfBenchmark
from .global_mmlu import GlobalMMLUBenchmark
from .judge import JudgeBenchmark
from .native_cloze import NativeClozeBenchmark
from .perplexity import PerplexityBenchmark

_BENCHMARKS: dict[str, type[Benchmark]] = {
    b.name: b
    for b in (
        NativeClozeBenchmark,
        PerplexityBenchmark,
        BelebeleBenchmark,
        GlobalMMLUBenchmark,
        FloresChrfBenchmark,
        JudgeBenchmark,
    )
}


def get_benchmark(name: str) -> Benchmark:
    try:
        return _BENCHMARKS[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown benchmark '{name}'. Known: {sorted(_BENCHMARKS)}"
        ) from exc


def available_benchmarks() -> list[str]:
    return sorted(_BENCHMARKS)
