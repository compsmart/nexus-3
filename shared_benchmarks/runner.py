#!/usr/bin/env python3
"""Universal benchmark runner for nexus agents.

Runs canonical suites against an agent adapter and produces structured output
compatible with the UI benchmark dashboard.

Usage from an agent's run_benchmark.py:
    from benchmarks.adapter import Nexus2Adapter
    from shared_benchmarks.runner import run_benchmark
    run_benchmark(adapter_class=Nexus2Adapter)
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .metrics import BenchmarkMetrics
from .suites import (
    MemoryRecallSuite,
    MultihopChainSuite,
    ScalabilitySuite,
    LearningTransferSuite,
    CompositeSuite,
)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _load_profiles() -> dict:
    """Load profile configs from profiles.yaml."""
    cfg_path = Path(__file__).resolve().parent / "config" / "profiles.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
            return data.get("profiles", {})
    return {}


def _load_scoring_config() -> dict:
    """Load suite weights from scoring.yaml."""
    pkg_path = Path(__file__).resolve().parent / "config" / "scoring.yaml"
    if pkg_path.exists():
        with open(pkg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_DEFAULT_SUITE_WEIGHTS = {
    "memory_recall": 0.20,
    "multihop": 0.30,
    "scalability": 0.15,
    "learning_transfer": 0.15,
    "composite": 0.20,
}


class TrackedBenchmarkRunner:
    """Benchmark runner that emits events and writes structured output."""

    def __init__(self, run_dir: Path, suites: dict, adapter, run_spec: dict):
        self.run_dir = run_dir
        self.suites = suites
        self.adapter = adapter
        self.run_spec = run_spec
        self.events_path = run_dir / "events.jsonl"

    def emit(self, event_type: str, message: str, data: dict = None):
        """Append an event to events.jsonl."""
        event = {
            "event_type": event_type,
            "message": message,
            "timestamp": _now_iso(),
        }
        if data:
            event["data"] = data
        with open(self.events_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        print(f"  [{event_type}] {message}", flush=True)

    def update_status(self, status: str):
        """Update status.json."""
        status_data = {
            "run_id": self.run_spec["run_id"],
            "status": status,
            "updated_at": _now_iso(),
        }
        (self.run_dir / "status.json").write_text(json.dumps(status_data, indent=2))

    def _flush_partial_results(self, suite_results, aggregate_scores_raw, suite_weights):
        """Write intermediate results.json and metrics.json."""
        results_data = {
            "run_id": self.run_spec["run_id"],
            "created_at": self.run_spec["created_at"],
            "suite_results": suite_results,
        }
        (self.run_dir / "results.json").write_text(json.dumps(results_data, indent=2))

        baseline_id = self.adapter.agent_name
        agg = aggregate_scores_raw.get(baseline_id, {"total": 0, "count": 0, "scores": {}})
        overall = agg["total"] / agg["count"] if agg["count"] > 0 else 0.0
        aggregate_scores = [{
            "baseline_id": baseline_id,
            "overall_score": round(overall, 4),
            "suite_scores": {k: round(v, 4) for k, v in agg["scores"].items()},
        }]

        metrics_data = {
            "run_id": self.run_spec["run_id"],
            "schema_version": "2.0",
            "agent": self.adapter.agent_name,
            "suite_weights": suite_weights,
            "aggregate_scores": aggregate_scores,
        }
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2))

    def run(self) -> dict:
        """Run all suites against the adapter with event tracking."""
        baseline_id = self.adapter.agent_name
        self.emit("run_start", f"Starting benchmark run: {self.run_spec['name']}", {
            "suites": list(self.suites.keys()),
            "adapter": baseline_id,
        })
        self.update_status("running")

        all_suite_results = []
        all_aggregate_scores = {baseline_id: {"scores": {}, "total": 0.0, "count": 0}}

        scoring_cfg = _load_scoring_config()
        suite_weights = scoring_cfg.get("suite_weights", _DEFAULT_SUITE_WEIGHTS)

        total_suites = len(self.suites)
        completed = 0

        for suite_name, suite in self.suites.items():
            self.emit("suite_start", f"Running suite: {suite_name}")
            suite_result = {
                "suite_id": suite_name,
                "baseline_metrics": {},
                "weight": suite_weights.get(suite_name, 0.20),
            }

            self.emit("baseline_start", f"Running {suite_name} x {baseline_id}")
            t0 = time.time()

            try:
                metrics = suite.run(self.adapter)
                elapsed = time.time() - t0
                metrics_dict = asdict(metrics)
                suite_result["baseline_metrics"][baseline_id] = metrics_dict

                weight = suite_weights.get(suite_name, 0.20)
                score = metrics.exact_match
                all_aggregate_scores[baseline_id]["scores"][suite_name] = score
                all_aggregate_scores[baseline_id]["total"] += score * weight
                all_aggregate_scores[baseline_id]["count"] += weight

                self.emit("baseline_end", f"Completed {suite_name} x {baseline_id}", {
                    "accuracy": metrics.exact_match,
                    "exact_match": metrics.exact_match,
                    "latency_p50_ms": metrics.latency_p50_ms,
                    "total_queries": metrics.total_queries,
                    "elapsed_s": round(elapsed, 1),
                })
            except Exception as e:
                self.emit("baseline_error", f"Error in {suite_name} x {baseline_id}: {e}", {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                suite_result["baseline_metrics"][baseline_id] = asdict(BenchmarkMetrics())

            completed += 1
            self.emit("suite_end", f"Completed suite: {suite_name}", {
                "completed": completed,
                "total": total_suites,
            })
            all_suite_results.append(suite_result)
            self._flush_partial_results(all_suite_results, all_aggregate_scores, suite_weights)

        self._flush_partial_results(all_suite_results, all_aggregate_scores, suite_weights)
        self.emit("run_end", "Benchmark run completed", {
            "aggregate_scores": json.loads(
                (self.run_dir / "metrics.json").read_text()
            ).get("aggregate_scores", []),
        })
        self.update_status("completed")

        # Persist to DB
        try:
            metrics_data = json.loads((self.run_dir / "metrics.json").read_text())
            results_data = json.loads((self.run_dir / "results.json").read_text())
            asyncio.run(_persist_run_to_db(self.run_spec, metrics_data, results_data, "completed"))
            print("[DB] Benchmark run persisted to database.", flush=True)
        except Exception as e:
            print(f"[DB] Warning: could not persist benchmark run to DB: {e}", flush=True)

        return json.loads((self.run_dir / "results.json").read_text())


async def _persist_run_to_db(run_spec: dict, metrics: dict, results: dict, status: str):
    """Persist a completed benchmark run to PostgreSQL."""
    try:
        import asyncpg
        db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:Pa55w0rd123%21@localhost:5432/ai_lab",
        )
        ts_str = run_spec.get("created_at")
        try:
            created_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            created_ts = datetime.now(timezone.utc)

        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(
                """
                INSERT INTO benchmark_runs
                    (run_id, agent_name, name, status, profile, model_name,
                     suites, baselines, suite_weights, aggregate_scores, suite_results,
                     source_branch, is_master_baseline,
                     created_at, completed_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,NOW())
                ON CONFLICT (run_id) DO UPDATE SET
                    status           = EXCLUDED.status,
                    aggregate_scores = EXCLUDED.aggregate_scores,
                    suite_results    = EXCLUDED.suite_results,
                    completed_at     = EXCLUDED.completed_at
                """,
                run_spec["run_id"],
                run_spec.get("agent", ""),
                run_spec.get("name", ""),
                status,
                run_spec.get("profile", ""),
                run_spec.get("model_name", ""),
                json.dumps(run_spec.get("suites", [])),
                json.dumps(run_spec.get("baselines", [])),
                json.dumps(metrics.get("suite_weights", {})),
                json.dumps(metrics.get("aggregate_scores", [])),
                json.dumps(results.get("suite_results", [])),
                run_spec.get("source_branch", ""),
                run_spec.get("is_master_baseline", False),
                created_ts,
            )
        finally:
            await conn.close()
    except Exception as e:
        raise RuntimeError(f"DB persist failed: {e}") from e


def _build_suites(suite_names: list, profile_config: dict = None) -> dict:
    """Build suite instances, optionally configured by profile."""
    all_suite_classes = {
        "memory_recall": MemoryRecallSuite,
        "multihop": MultihopChainSuite,
        "scalability": ScalabilitySuite,
        "learning_transfer": LearningTransferSuite,
        "composite": CompositeSuite,
    }

    suites = {}
    for name in suite_names:
        if name not in all_suite_classes:
            continue
        kwargs = {}
        if profile_config and name in profile_config:
            kwargs = profile_config[name]
        suites[name] = all_suite_classes[name](**kwargs)
    return suites


def run_benchmark(adapter_class, default_device="cuda"):
    """Main entry point for running benchmarks with a given adapter.

    Called from each agent's run_benchmark.py:
        from benchmarks.adapter import Nexus2Adapter
        from shared_benchmarks.runner import run_benchmark
        run_benchmark(adapter_class=Nexus2Adapter)
    """
    parser = argparse.ArgumentParser(description="Nexus Benchmark Runner")
    parser.add_argument("--name", default=None, help="Run name")
    parser.add_argument("--suites", default="all", help="Comma-separated suite names or 'all'")
    parser.add_argument("--baselines", default="", help="Ignored (kept for CLI compat)")
    parser.add_argument("--device", default=default_device, help="Device: cpu or cuda")
    parser.add_argument("--profile", default="standard", help="Profile: smoke, standard, quality_first")
    args = parser.parse_args()

    # Create adapter
    adapter = adapter_class(device=args.device)
    agent_name = adapter.agent_name

    # Load profile config
    profiles = _load_profiles()
    profile_config = None
    if args.profile in profiles:
        profile_config = profiles[args.profile].get("suites", {})

    # Determine suites
    default_suites = ["memory_recall", "multihop", "scalability", "learning_transfer", "composite"]
    if args.suites == "all":
        suite_names = list(profile_config.keys()) if profile_config else default_suites
        # Ensure we use the standard suite list order
        suite_names = [s for s in default_suites if s in suite_names] or default_suites
    else:
        suite_names = [s.strip() for s in args.suites.split(",")]

    # Create run directory
    run_id = uuid.uuid4().hex[:12]
    runs_dir = Path.cwd() / "benchmarks" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.name or f"{agent_name}-benchmark"

    # Build run spec
    run_spec = {
        "run_id": run_id,
        "name": run_name,
        "created_at": _now_iso(),
        "agent": agent_name,
        "schema_version": "2.0",
        "profile": args.profile,
        "model_name": getattr(adapter, "model_name", ""),
        "suites": suite_names,
        "baselines": [agent_name],
        "source_branch": os.environ.get("NEXUS_BRANCH", ""),
        "is_master_baseline": os.environ.get("NEXUS_IS_MASTER_BASELINE", "").lower() == "true",
    }
    (run_dir / "run_spec.json").write_text(json.dumps(run_spec, indent=2))
    (run_dir / "status.json").write_text(json.dumps({
        "run_id": run_id, "status": "queued", "updated_at": _now_iso()
    }, indent=2))
    (run_dir / "events.jsonl").write_text("")

    try:
        import setproctitle
        setproctitle.setproctitle(f"{agent_name}-benchmark:{run_name}")
    except ImportError:
        pass

    print("=" * 60, flush=True)
    print(f"{agent_name.upper()} Benchmark -- Run {run_id}", flush=True)
    print("=" * 60, flush=True)
    print(f"  Name:      {run_name}", flush=True)
    print(f"  Profile:   {args.profile}", flush=True)
    print(f"  Suites:    {suite_names}", flush=True)
    print(f"  Device:    {args.device}", flush=True)
    print(f"  Output:    {run_dir}", flush=True)
    print(flush=True)

    # Build suites with profile config
    suites = _build_suites(suite_names, profile_config)

    # Run
    runner = TrackedBenchmarkRunner(run_dir, suites, adapter, run_spec)
    try:
        results = runner.run()
        print(f"\nRun complete: {run_dir}", flush=True)
    except Exception as e:
        runner.emit("run_error", f"Fatal error: {e}", {"traceback": traceback.format_exc()})
        runner.update_status("failed")
        print(f"\nRun failed: {e}", flush=True)
        sys.exit(1)
