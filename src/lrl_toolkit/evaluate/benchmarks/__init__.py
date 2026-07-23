"""Coverage-aware, base-vs-adapted benchmark framework.

See :mod:`lrl_toolkit.evaluate.benchmarks.base` for the design rationale: the hard
problem for LRL evaluation is *coverage* (most languages have no external
benchmarks), so benchmarks self-report whether they apply, deltas over the base
model are the headline, and a coverage matrix is emitted as a first-class artifact.
"""

from __future__ import annotations

from .base import (
    Benchmark,
    BenchmarkResult,
    Coverage,
    MetricDirection,
    ModelBundle,
    Score,
)
from .registry import available_benchmarks, get_benchmark

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "Coverage",
    "MetricDirection",
    "ModelBundle",
    "Score",
    "available_benchmarks",
    "get_benchmark",
]
