"""Nexus-3 configuration."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Nexus3Config:
    # --- LLM ---
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    use_4bit: bool = True
    device: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.15

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # --- Narrative Memory ---
    max_memory_slots: int = 10000
    memory_top_k: int = 5
    novelty_threshold: float = 0.5
    narrative_max_length: int = 512
    dedup_threshold: float = 0.95

    # --- Bridge-Guided Retrieval ---
    bridge_top_k: int = 3
    bridge_token_budget: int = 200
    hop1_top_k: int = 5
    hop2_top_k: int = 5

    # --- Confidence Gate ---
    confidence_high: float = 0.55
    confidence_low: float = 0.30
    confidence_reject: float = 0.15

    # --- Tools ---
    max_tool_calls: int = 3
    tool_call_pattern: str = r'\[\s*TOOL_CALL\s*:\s*(\w+)\s*\|\s*(.+?)\s*\]'

    # --- Persistence ---
    memory_path: str = "data/memory.json"
    memory_pt_path: str = "data/memory.pt"

    # --- Benchmark ---
    benchmark_n_samples: int = 200
    benchmark_seeds: List[int] = field(default_factory=lambda: [42, 7, 137])
