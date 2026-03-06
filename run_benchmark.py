#!/usr/bin/env python3
"""Run NEXUS-3 benchmarks and output results compatible with the UI dashboard.

Imports benchmark suites from nexus-2 (the shared infrastructure) and runs
nexus-3, rag, and llm_only baselines, persisting results to the benchmark_runs
PostgreSQL table.

Usage:
    python run_benchmark.py                          # Full run
    python run_benchmark.py --suites memory_recall   # Single suite
    python run_benchmark.py --baselines nexus3        # Only nexus3 baseline
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

# ---------------------------------------------------------------------------
# Make nexus-2 benchmark suites importable
# ---------------------------------------------------------------------------
_POC_DIR = Path(__file__).resolve().parent.parent   # poc/
_NEXUS2_DIR = _POC_DIR / "nexus-2"
if str(_NEXUS2_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS2_DIR))

# Make nexus-3 itself importable for the baseline
_NEXUS3_DIR = Path(__file__).resolve().parent
if str(_NEXUS3_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS3_DIR))

from benchmarks.runner import BenchmarkRunner          # noqa: F401 (nexus-2)
from benchmarks.metrics import BenchmarkMetrics        # noqa: F401 (nexus-2)
from benchmarks.suites.memory_recall import MemoryRecallSuite
from benchmarks.suites.multihop_chain import MultihopChainSuite
from benchmarks.suites.scalability import ScalabilitySuite
from benchmarks.suites.learning_transfer import LearningTransferSuite
from benchmarks.suites.composite import CompositeSuite


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _load_scoring_config() -> dict:
    cfg_path = _NEXUS2_DIR / "benchmarks" / "config" / "scoring.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


class TrackedBenchmarkRunner:
    """Benchmark runner that emits events and writes structured output."""

    def __init__(self, run_dir: Path, suites: dict, baselines: dict, run_spec: dict):
        self.run_dir = run_dir
        self.suites = suites
        self.baselines = baselines
        self.run_spec = run_spec
        self.events_path = run_dir / "events.jsonl"

    def emit(self, event_type: str, message: str, data: dict = None):
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
        status_data = {
            "run_id": self.run_spec["run_id"],
            "status": status,
            "updated_at": _now_iso(),
        }
        (self.run_dir / "status.json").write_text(json.dumps(status_data, indent=2))

    def _flush_partial_results(self, suite_results, aggregate_scores_raw, suite_weights):
        results_data = {
            "run_id": self.run_spec["run_id"],
            "created_at": self.run_spec["created_at"],
            "suite_results": suite_results,
        }
        (self.run_dir / "results.json").write_text(json.dumps(results_data, indent=2))

        aggregate_scores = []
        for bl_name, agg in aggregate_scores_raw.items():
            overall = agg["total"] / agg["count"] if agg["count"] > 0 else 0.0
            aggregate_scores.append({
                "baseline_id": bl_name,
                "overall_score": round(overall, 4),
                "suite_scores": {k: round(v, 4) for k, v in agg["scores"].items()},
            })
        aggregate_scores.sort(key=lambda x: x["overall_score"], reverse=True)
        metrics_data = {
            "run_id": self.run_spec["run_id"],
            "schema_version": "2.0",
            "agent": "nexus-3",
            "suite_weights": suite_weights,
            "aggregate_scores": aggregate_scores,
        }
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2))

    def run(self) -> dict:
        self.emit("run_start", f"Starting benchmark run: {self.run_spec['name']}", {
            "suites": list(self.suites.keys()),
            "baselines": list(self.baselines.keys()),
        })
        self.update_status("running")

        all_suite_results = []
        all_aggregate_scores = {
            bl_name: {"scores": {}, "total": 0.0, "count": 0}
            for bl_name in self.baselines
        }

        scoring_cfg = _load_scoring_config()
        suite_weights = scoring_cfg.get("suite_weights", {
            "memory_recall": 0.20,
            "multihop": 0.30,
            "scalability": 0.15,
            "learning_transfer": 0.15,
            "composite": 0.20,
        })

        total_suites = len(self.suites)
        completed = 0

        for suite_name, suite in self.suites.items():
            self.emit("suite_start", f"Running suite: {suite_name}")
            suite_result = {
                "suite_id": suite_name,
                "baseline_metrics": {},
                "weight": suite_weights.get(suite_name, 0.20),
            }

            for bl_name, bl in self.baselines.items():
                self.emit("baseline_start", f"Running {suite_name} x {bl_name}")
                t0 = time.time()
                try:
                    metrics = suite.run(bl)
                    elapsed = time.time() - t0
                    metrics_dict = asdict(metrics)
                    suite_result["baseline_metrics"][bl_name] = metrics_dict
                    weight = suite_weights.get(suite_name, 0.20)
                    score = metrics.exact_match
                    all_aggregate_scores[bl_name]["scores"][suite_name] = score
                    all_aggregate_scores[bl_name]["total"] += score * weight
                    all_aggregate_scores[bl_name]["count"] += weight
                    self.emit("baseline_end", f"Completed {suite_name} x {bl_name}", {
                        "accuracy": metrics.exact_match,
                        "exact_match": metrics.exact_match,
                        "latency_p50_ms": metrics.latency_p50_ms,
                        "total_queries": metrics.total_queries,
                        "elapsed_s": round(elapsed, 1),
                    })
                except Exception as e:
                    self.emit("baseline_error", f"Error in {suite_name} x {bl_name}: {e}", {
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })
                    suite_result["baseline_metrics"][bl_name] = asdict(BenchmarkMetrics())

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

        try:
            metrics_data = json.loads((self.run_dir / "metrics.json").read_text())
            results_data = json.loads((self.run_dir / "results.json").read_text())
            asyncio.run(_persist_run_to_db(self.run_spec, metrics_data, results_data, "completed"))
            print("[DB] Benchmark run persisted to database.", flush=True)
        except Exception as e:
            print(f"[DB] Warning: could not persist benchmark run to DB: {e}", flush=True)

        return json.loads((self.run_dir / "results.json").read_text())


async def _persist_run_to_db(run_spec: dict, metrics: dict, results: dict, status: str):
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
                     created_at, completed_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,NOW())
                ON CONFLICT (run_id) DO UPDATE SET
                    status           = EXCLUDED.status,
                    aggregate_scores = EXCLUDED.aggregate_scores,
                    suite_results    = EXCLUDED.suite_results,
                    completed_at     = EXCLUDED.completed_at
                """,
                run_spec["run_id"],
                run_spec.get("agent", "nexus-3"),
                run_spec.get("name", ""),
                status,
                run_spec.get("profile", ""),
                run_spec.get("model_name", ""),
                json.dumps(run_spec.get("suites", [])),
                json.dumps(run_spec.get("baselines", [])),
                json.dumps(metrics.get("suite_weights", {})),
                json.dumps(metrics.get("aggregate_scores", [])),
                json.dumps(results.get("suite_results", [])),
                created_ts,
            )
        finally:
            await conn.close()
    except Exception as e:
        raise RuntimeError(f"DB persist failed: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="NEXUS-3 Tracked Benchmark Runner")
    parser.add_argument("--name", default="nexus3-benchmark", help="Run name")
    parser.add_argument("--suites", default="all", help="Comma-separated suite names or 'all'")
    parser.add_argument("--baselines", default="", help="Comma-separated baselines (default: all)")
    parser.add_argument("--device", default="cuda", help="Device: cpu or cuda")
    args = parser.parse_args()

    suite_names = None if args.suites == "all" else [s.strip() for s in args.suites.split(",")]
    baseline_names = [b.strip() for b in args.baselines.split(",") if b.strip()]

    run_id = uuid.uuid4().hex[:12]
    runs_dir = Path(__file__).resolve().parent / "benchmarks" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        import setproctitle
        setproctitle.setproctitle(f"nexus3-benchmark:{args.name}")
    except ImportError:
        pass

    default_suites = ["memory_recall", "multihop", "scalability", "learning_transfer", "composite"]
    default_baselines = ["nexus3", "rag", "llm_only"]
    run_spec = {
        "run_id": run_id,
        "name": args.name,
        "created_at": _now_iso(),
        "agent": "nexus-3",
        "schema_version": "2.0",
        "profile": "standard",
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "suites": suite_names or default_suites,
        "baselines": baseline_names or default_baselines,
    }
    (run_dir / "run_spec.json").write_text(json.dumps(run_spec, indent=2))
    (run_dir / "status.json").write_text(json.dumps({
        "run_id": run_id, "status": "queued", "updated_at": _now_iso()
    }, indent=2))
    (run_dir / "events.jsonl").write_text("")

    print("=" * 60, flush=True)
    print(f"NEXUS-3 Benchmark -- Run {run_id}", flush=True)
    print("=" * 60, flush=True)
    print(f"  Name:      {args.name}", flush=True)
    print(f"  Suites:    {run_spec['suites']}", flush=True)
    print(f"  Baselines: {run_spec['baselines']}", flush=True)
    print(f"  Device:    {args.device}", flush=True)
    print(f"  Output:    {run_dir}", flush=True)
    print(flush=True)

    # Build suites
    all_suites = {
        "memory_recall": MemoryRecallSuite(),
        "multihop": MultihopChainSuite(),
        "scalability": ScalabilitySuite(),
        "learning_transfer": LearningTransferSuite(),
        "composite": CompositeSuite(),
    }
    suites = {n: all_suites[n] for n in run_spec["suites"] if n in all_suites}

    # Build baselines
    baselines = {}
    for bl_name in run_spec["baselines"]:
        if bl_name == "nexus3":
            from benchmarks.baselines.nexus3_baseline import Nexus3Baseline
            baselines["nexus3"] = Nexus3Baseline(device=args.device)
        elif bl_name == "rag":
            from benchmarks.baselines.rag_baseline import RagBaseline
            baselines["rag"] = RagBaseline(device=args.device)
        elif bl_name in ("llm_only", "phi_only"):
            from benchmarks.baselines.phi_only_baseline import LLMOnlyBaseline
            baselines[bl_name] = LLMOnlyBaseline(device=args.device)

    runner = TrackedBenchmarkRunner(run_dir, suites, baselines, run_spec)
    try:
        runner.run()
        print(f"\nRun complete: {run_dir}", flush=True)
    except Exception as e:
        runner.emit("run_error", f"Fatal error: {e}", {"traceback": traceback.format_exc()})
        runner.update_status("failed")
        print(f"\nRun failed: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
