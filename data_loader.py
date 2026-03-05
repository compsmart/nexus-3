"""HotpotQA data loader for Nexus-3 benchmarking.

Loads the HotpotQA dataset (distractor setting) from HuggingFace datasets.
Each example contains:
- question: The multi-hop question
- answer: Gold answer string
- type: 'bridge' or 'comparison'
- level: 'easy', 'medium', or 'hard'
- supporting_facts: {'title': [...], 'sent_id': [...]}
- context: {'title': [...], 'sentences': [[...], ...]}
"""

import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


def load_hotpotqa(
    split: str = "validation",
    n_samples: Optional[int] = None,
    seed: int = 42,
    question_type: Optional[str] = None,
    level: Optional[str] = None,
    cache_dir: str = "data",
) -> List[Dict]:
    """Load HotpotQA from HuggingFace datasets.

    Args:
        split: Dataset split ('train' or 'validation').
        n_samples: Number of samples to load (None = all).
        seed: Random seed for sampling.
        question_type: Filter by type ('bridge' or 'comparison').
        level: Filter by difficulty ('easy', 'medium', 'hard').
        cache_dir: Directory to cache the dataset.

    Returns:
        List of example dicts with standardized keys.
    """
    from datasets import load_dataset

    log.info("Loading HotpotQA split=%s ...", split)
    ds = load_dataset("hotpot_qa", "distractor", split=split, cache_dir=cache_dir)

    examples = []
    for ex in ds:
        if question_type and ex.get("type") != question_type:
            continue
        if level and ex.get("level") != level:
            continue

        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        paragraphs = {}
        for t, s in zip(titles, sentences):
            paragraphs[t] = s

        sf_titles = ex["supporting_facts"]["title"]
        sf_sent_ids = ex["supporting_facts"]["sent_id"]

        gold_titles = list(dict.fromkeys(sf_titles))

        examples.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex.get("type", "unknown"),
            "level": ex.get("level", "unknown"),
            "context": paragraphs,
            "supporting_facts": {
                "title": sf_titles,
                "sent_id": sf_sent_ids,
            },
            "gold_titles": gold_titles,
        })

    log.info("Loaded %d examples (filtered from %d)", len(examples), len(ds))

    if n_samples and n_samples < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, n_samples)
        log.info("Sampled %d examples (seed=%d)", n_samples, seed)

    return examples


def get_gold_context(example: Dict) -> Dict[str, List[str]]:
    """Extract only the gold (supporting) paragraphs for an example."""
    gold_titles = example["gold_titles"]
    return {t: example["context"][t] for t in gold_titles if t in example["context"]}


def get_distractor_context(example: Dict, n_distractors: int = 8) -> Dict[str, List[str]]:
    """Get gold paragraphs plus distractor paragraphs.

    This simulates the realistic retrieval scenario where the agent must
    identify relevant paragraphs among distractors.
    """
    gold_titles = set(example["gold_titles"])
    context = {}

    for t in example["gold_titles"]:
        if t in example["context"]:
            context[t] = example["context"][t]

    distractor_titles = [t for t in example["context"] if t not in gold_titles]
    for t in distractor_titles[:n_distractors]:
        context[t] = example["context"][t]

    return context


def compute_em(prediction: str, gold: str) -> float:
    """Compute Exact Match score (normalized)."""
    pred = _normalize(prediction)
    gold_norm = _normalize(gold)
    return 1.0 if pred == gold_norm else 0.0


def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> str:
    """Normalize text for evaluation: lowercase, strip articles/punctuation."""
    import re
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
