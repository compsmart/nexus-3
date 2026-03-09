"""Agent adapter protocol for standardized benchmarking.

Every nexus agent must implement this protocol to be benchmarked
against the canonical suites. No shortcuts allowed -- the adapter
must route through the actual agent's inference pipeline.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentAdapter(Protocol):
    """Protocol that all agent adapters must implement.

    The benchmark suites call these methods to interact with the agent:
    - reset(): Clear all agent state for a fresh test
    - teach(text): Feed a fact into the agent's memory
    - query(text): Ask the agent a question, get a string answer
    """
    agent_name: str

    def reset(self) -> None: ...
    def teach(self, text: str) -> None: ...
    def query(self, text: str) -> str: ...
