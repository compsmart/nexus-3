"""Standardised benchmark data schemas (v2.0).

Both agents import these dataclasses so their JSON output is
structurally identical and directly comparable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "2.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunSpec:
    """Immutable run configuration written at benchmark start."""

    run_id: str
    name: str
    created_at: str
    agent: str = ""  # "nexus-1" or "nexus-2"
    schema_version: str = SCHEMA_VERSION
    profile: str = "standard"
    suites: List[str] = field(default_factory=list)
    baselines: List[str] = field(default_factory=list)
    primary_suite: str = "memory_recall"
    seed: int = 42
    model_name: str = ""
    use_4bit: bool = True
    max_new_tokens: int = 128
    suite_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunSpec:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class CoreMetrics:
    """Standard metrics every suite must report."""

    accuracy: float = 0.0
    exact_match: float = 0.0
    total_queries: int = 0
    correct: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteResult:
    """Result of running one suite across all baselines."""

    suite_id: str
    baseline_metrics: Dict[str, Dict[str, Any]]
    case_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "baseline_metrics": self.baseline_metrics,
            "case_results": self.case_results,
            "metadata": self.metadata,
        }


@dataclass
class AggregateScore:
    """Weighted aggregate score for one baseline across all suites."""

    baseline_id: str
    overall_score: float
    suite_scores: Dict[str, float] = field(default_factory=dict)
    metric_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricsOutput:
    """Top-level metrics.json schema with embedded suite weights."""

    run_id: str
    schema_version: str = SCHEMA_VERSION
    agent: str = ""
    suite_weights: Dict[str, float] = field(default_factory=dict)
    aggregate_scores: List[Dict[str, Any]] = field(default_factory=list)
    rank_estimates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankEstimate:
    """Position estimate against a public leaderboard."""

    baseline_id: str
    comparable: bool
    reference_source: str
    reference_date: str
    percentile: Optional[float] = None
    estimated_rank: Optional[int] = None
    total_models: Optional[int] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
