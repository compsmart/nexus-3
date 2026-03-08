"""Clean adapter wrapping the actual Nexus-3 agent for benchmarking.

No shortcuts -- all queries go through the real agent pipeline.
"""

import sys
from pathlib import Path

# Ensure nexus-3 root is importable
_NEXUS3_DIR = Path(__file__).resolve().parent.parent
if str(_NEXUS3_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS3_DIR))


class Nexus3Adapter:
    """Wraps the actual Nexus3Agent for benchmark evaluation."""

    agent_name = "nexus-3"

    def __init__(self, device="cpu", **kwargs):
        self.device = device
        self._agent = None
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"

    def _ensure_loaded(self):
        if self._agent is not None:
            return
        from config import Nexus3Config
        from agent import Nexus3Agent

        cfg = Nexus3Config(device=self.device)
        self._agent = Nexus3Agent(config=cfg, load_llm=True)

    def reset(self):
        self._ensure_loaded()
        self._agent.reset()

    def teach(self, text: str):
        self._ensure_loaded()
        self._agent.memory.store(text=text, narrative=text, mem_type="fact", source="benchmark")

    def query(self, text: str) -> str:
        self._ensure_loaded()
        return self._agent.interact(text)
