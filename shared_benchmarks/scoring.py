"""Weighted score aggregation — extracted from nexus-1 scoring_engine.py."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .schema import AggregateScore, SuiteResult

_CONFIG_DIR = Path(__file__).parent / "config"


def load_scoring_config(path: Path | None = None) -> dict:
    """Load scoring.yaml from *path* or the default shared config."""
    p = path or (_CONFIG_DIR / "scoring.yaml")
    with open(p) as f:
        return yaml.safe_load(f)


class WeightedScoreAggregator:
    """Compute weighted overall scores from suite results."""

    def __init__(self, scoring_cfg: dict | None = None) -> None:
        if scoring_cfg is None:
            scoring_cfg = load_scoring_config()
        self.suite_weights: Dict[str, float] = scoring_cfg.get("suite_weights", {})
        self.primary_metric: Dict[str, str] = scoring_cfg.get("primary_metric", {})
        self.fallback_metric_keys: List[str] = scoring_cfg.get(
            "fallback_metric_keys",
            ["accuracy", "exact_match", "composite_score"],
        )
        self.normalize: bool = bool(scoring_cfg.get("normalize_suite_weights", True))

    def _suite_metric_value(self, suite_result: SuiteResult, baseline_id: str) -> float:
        metrics = suite_result.baseline_metrics.get(baseline_id, {})
        metric_key = self.primary_metric.get(suite_result.suite_id)
        if metric_key and metric_key in metrics:
            return float(metrics[metric_key])
        for key in self.fallback_metric_keys:
            if key in metrics:
                return float(metrics[key])
        if metrics:
            numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
            return float(sum(numeric) / len(numeric)) if numeric else 0.0
        return 0.0

    def aggregate(self, suite_results: List[SuiteResult]) -> List[AggregateScore]:
        baseline_ids = sorted(
            {bid for sr in suite_results for bid in sr.baseline_metrics}
        )
        raw_weights = {
            sr.suite_id: float(self.suite_weights.get(sr.suite_id, 1.0))
            for sr in suite_results
        }
        if self.normalize:
            total = sum(raw_weights.values()) or 1.0
            weights = {k: v / total for k, v in raw_weights.items()}
        else:
            weights = raw_weights

        out: List[AggregateScore] = []
        for bid in baseline_ids:
            suite_scores: Dict[str, float] = {}
            weighted_sum = 0.0
            for sr in suite_results:
                value = self._suite_metric_value(sr, bid)
                suite_scores[sr.suite_id] = value
                weighted_sum += value * weights.get(sr.suite_id, 0.0)
            out.append(AggregateScore(
                baseline_id=bid,
                overall_score=weighted_sum,
                suite_scores=suite_scores,
                metric_breakdown={"weighted_sum": weighted_sum},
            ))
        return sorted(out, key=lambda x: x.overall_score, reverse=True)
