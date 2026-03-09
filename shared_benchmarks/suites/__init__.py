"""Canonical benchmark suites for nexus agent evaluation."""

from .memory_recall import MemoryRecallSuite
from .multihop_chain import MultihopChainSuite
from .scalability import ScalabilitySuite
from .learning_transfer import LearningTransferSuite
from .composite import CompositeSuite

__all__ = [
    "MemoryRecallSuite",
    "MultihopChainSuite",
    "ScalabilitySuite",
    "LearningTransferSuite",
    "CompositeSuite",
]
