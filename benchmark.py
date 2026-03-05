#!/usr/bin/env python3
"""Nexus-3 Benchmark Runner.

Evaluates the agent on HotpotQA (distractor setting) using multiple conditions:
- oracle: Gold context only (upper bound)
- distractor: Gold + distractor paragraphs (realistic)
- bridge_guided: 2-call bridge-guided architecture
- memory_retrieval: Store context in memory, then retrieve

Usage:
    python benchmark.py                              # Full benchmark
    python benchmark.py --n 50 --conditions oracle   # Quick oracle test
    python benchmark.py --seeds 42 7 137             # Multi-seed run
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from config import Nexus3Config
from agent import Nexus3Agent
from data_loader import (
    load_hotpotqa,
    get_gold_context,
    get_distractor_context,
    compute_em,
    compute_f1,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def run_condition_oracle(agent: Nexus3Agent, examples: List[Dict]) -> List[Dict]:
    """Oracle condition: answer using only gold supporting paragraphs."""
    results = []
    for ex in tqdm(examples, desc="oracle"):
        gold_ctx = get_gold_context(ex)
        answer = agent.answer_question(
            question=ex["question"],
            context_paragraphs=gold_ctx,
            greedy=True,
        )
        em = compute_em(answer, ex["answer"])
        f1 = compute_f1(answer, ex["answer"])
        results.append({
            "id": ex["id"],
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "predicted": answer,
            "em": em,
            "f1": f1,
            "type": ex["type"],
            "level": ex["level"],
        })
    return results


def run_condition_distractor(agent: Nexus3Agent, examples: List[Dict]) -> List[Dict]:
    """Distractor condition: answer with gold + distractor paragraphs."""
    results = []
    for ex in tqdm(examples, desc="distractor"):
        ctx = get_distractor_context(ex, n_distractors=8)
        answer = agent.answer_question(
            question=ex["question"],
            context_paragraphs=ctx,
            greedy=True,
        )
        em = compute_em(answer, ex["answer"])
        f1 = compute_f1(answer, ex["answer"])
        results.append({
            "id": ex["id"],
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "predicted": answer,
            "em": em,
            "f1": f1,
            "type": ex["type"],
            "level": ex["level"],
        })
    return results


def run_condition_bridge_guided(agent: Nexus3Agent, examples: List[Dict]) -> List[Dict]:
    """Bridge-guided condition: 2-call architecture with bridge entity extraction."""
    results = []
    for ex in tqdm(examples, desc="bridge_guided"):
        ctx = get_distractor_context(ex, n_distractors=8)
        result = agent.answer_multihop(
            question=ex["question"],
            context_paragraphs=ctx,
            supporting_facts=ex.get("supporting_facts"),
            greedy=True,
        )
        answer = result["answer"]
        em = compute_em(answer, ex["answer"])
        f1 = compute_f1(answer, ex["answer"])
        results.append({
            "id": ex["id"],
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "predicted": answer,
            "bridge": result.get("bridge"),
            "em": em,
            "f1": f1,
            "type": ex["type"],
            "level": ex["level"],
        })
    return results


def run_condition_memory_retrieval(agent: Nexus3Agent, examples: List[Dict]) -> List[Dict]:
    """Memory retrieval condition: store all context in memory, then answer via retrieval."""
    results = []
    for ex in tqdm(examples, desc="memory_retrieval"):
        agent.reset()

        ctx = get_distractor_context(ex, n_distractors=8)
        facts = []
        for title, sentences in ctx.items():
            for sent in sentences:
                facts.append({
                    "text": sent,
                    "narrative": f"From the article '{title}': {sent}",
                    "type": "fact",
                })
        agent.store_knowledge(facts, source="hotpotqa_context")

        retrieval = agent.retriever.retrieve_bridge_guided(ex["question"])
        context_text = retrieval["full_context"]

        if context_text:
            answer = agent.answer_question(
                question=ex["question"],
                context_paragraphs=None,
                greedy=True,
            )
        else:
            answer = "Insufficient information"

        em = compute_em(answer, ex["answer"])
        f1 = compute_f1(answer, ex["answer"])
        results.append({
            "id": ex["id"],
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "predicted": answer,
            "em": em,
            "f1": f1,
            "type": ex["type"],
            "level": ex["level"],
            "bridge": retrieval.get("bridge_entity"),
        })
    return results


CONDITION_MAP = {
    "oracle": run_condition_oracle,
    "distractor": run_condition_distractor,
    "bridge_guided": run_condition_bridge_guided,
    "memory_retrieval": run_condition_memory_retrieval,
}


def summarize_results(results: List[Dict], condition: str) -> Dict:
    """Compute aggregate metrics from individual results."""
    if not results:
        return {"condition": condition, "n": 0, "em": 0.0, "f1": 0.0}

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]

    summary = {
        "condition": condition,
        "n": len(results),
        "em": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }

    for qtype in ("bridge", "comparison"):
        typed = [r for r in results if r.get("type") == qtype]
        if typed:
            summary[f"em_{qtype}"] = sum(r["em"] for r in typed) / len(typed)
            summary[f"f1_{qtype}"] = sum(r["f1"] for r in typed) / len(typed)
            summary[f"n_{qtype}"] = len(typed)

    for level in ("easy", "medium", "hard"):
        leveled = [r for r in results if r.get("level") == level]
        if leveled:
            summary[f"em_{level}"] = sum(r["em"] for r in leveled) / len(leveled)
            summary[f"n_{level}"] = len(leveled)

    return summary


def run_benchmark(
    n_samples: int = 200,
    seeds: List[int] = None,
    conditions: List[str] = None,
    output_dir: str = "data/benchmark_results",
    device: str = "auto",
):
    """Run the full benchmark suite."""
    seeds = seeds or [42]
    conditions = conditions or list(CONDITION_MAP.keys())

    os.makedirs(output_dir, exist_ok=True)

    config = Nexus3Config(device=device)
    agent = Nexus3Agent(config=config, load_llm=True)

    all_summaries = []

    for seed in seeds:
        log.info("=== Seed %d ===", seed)

        examples = load_hotpotqa(
            split="validation",
            n_samples=n_samples,
            seed=seed,
            cache_dir="data",
        )

        for condition_name in conditions:
            if condition_name not in CONDITION_MAP:
                log.warning("Unknown condition: %s", condition_name)
                continue

            log.info("Running condition: %s (seed=%d, n=%d)", condition_name, seed, len(examples))
            t0 = time.time()

            run_fn = CONDITION_MAP[condition_name]
            results = run_fn(agent, examples)
            elapsed = time.time() - t0

            summary = summarize_results(results, condition_name)
            summary["seed"] = seed
            summary["elapsed_seconds"] = round(elapsed, 1)
            all_summaries.append(summary)

            log.info(
                "  %s seed=%d: EM=%.3f F1=%.3f (n=%d, %.1fs)",
                condition_name, seed, summary["em"], summary["f1"],
                summary["n"], elapsed,
            )

            detail_path = os.path.join(
                output_dir, f"{condition_name}_seed{seed}_details.json"
            )
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": config.model_name,
            "n_samples": n_samples,
            "seeds": seeds,
            "conditions": conditions,
            "summaries": all_summaries,
        }, f, indent=2)

    log.info("Results saved to %s", results_path)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Condition':<20} {'Seed':<6} {'EM':>8} {'F1':>8} {'N':>6} {'Time':>8}")
    print("-" * 70)
    for s in all_summaries:
        print(f"{s['condition']:<20} {s['seed']:<6} {s['em']:>8.3f} {s['f1']:>8.3f} {s['n']:>6} {s['elapsed_seconds']:>7.1f}s")
    print("=" * 70)

    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Nexus-3 Benchmark Runner")
    parser.add_argument("--n", type=int, default=200, help="Number of samples per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds")
    parser.add_argument(
        "--conditions", nargs="+",
        default=list(CONDITION_MAP.keys()),
        choices=list(CONDITION_MAP.keys()),
        help="Conditions to run",
    )
    parser.add_argument("--output", default="data/benchmark_results", help="Output directory")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    run_benchmark(
        n_samples=args.n,
        seeds=args.seeds,
        conditions=args.conditions,
        output_dir=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
