"""Nexus-3 baseline adapter for benchmarks.

Clean pass-through to the real Nexus3Agent -- no shortcuts, no regex,
no pattern matching. All queries go through the full agent pipeline
(embedding retrieval + LLM generation).
"""

import sys
import os


class Nexus3Baseline:
    """Wraps the real Nexus3Agent for benchmark evaluation.

    Lazy-loads the agent on first use. All teach/query calls go through
    the agent's real memory and interact pipeline.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._agent = None

    def _ensure_loaded(self):
        if self._agent is not None:
            return
        nexus3_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        if nexus3_dir not in sys.path:
            sys.path.insert(0, nexus3_dir)
        from config import Nexus3Config
        from agent import Nexus3Agent

        cfg = Nexus3Config(device=self.device)
        self._agent = Nexus3Agent(config=cfg, load_llm=True)

    def reset(self):
        """Clear all stored facts."""
        self._ensure_loaded()
        self._agent.reset()

    def teach(self, text: str):
        """Store a fact through the real agent memory pipeline."""
        self._ensure_loaded()
        self._agent.memory.store(text=text, narrative=text, mem_type="fact", source="benchmark")

    def query(self, text: str) -> str:
        """Answer a query through the real agent pipeline."""
        self._ensure_loaded()
        return self._agent.interact(text)
