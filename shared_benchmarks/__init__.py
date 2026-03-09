"""Shared benchmark library for standardised agent evaluation."""

from .schema import (
    AggregateScore,
    CoreMetrics,
    MetricsOutput,
    RunSpec,
    SuiteResult,
)
from .metrics import (
    BenchmarkMetrics,
    LatencyTracker,
    compute_accuracy,
    compute_exact_match,
    compute_hop_success_rate,
    compute_latency,
    compute_llm_match,
    compute_recall_at_k,
)
from .scoring import WeightedScoreAggregator
from .adapter import AgentAdapter

__all__ = [
    "AgentAdapter",
    "AggregateScore",
    "BenchmarkMetrics",
    "CoreMetrics",
    "LatencyTracker",
    "MetricsOutput",
    "RunSpec",
    "SuiteResult",
    "WeightedScoreAggregator",
    "compute_accuracy",
    "compute_exact_match",
    "compute_hop_success_rate",
    "compute_latency",
    "compute_llm_match",
    "compute_recall_at_k",
]
