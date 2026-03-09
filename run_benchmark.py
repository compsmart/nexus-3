#!/usr/bin/env python3
"""Run NEXUS-3 benchmarks using the shared benchmark runner.

Usage:
    python run_benchmark.py                           # Standard run
    python run_benchmark.py --profile smoke           # Fast dev check
    python run_benchmark.py --profile quality_first   # Full validation
    python run_benchmark.py --suites memory_recall    # Single suite
"""

import sys
from pathlib import Path

# Ensure nexus-3 root is importable
_NEXUS3_DIR = Path(__file__).resolve().parent
if str(_NEXUS3_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS3_DIR))

from benchmarks.adapter import Nexus3Adapter

# Prefer the bundled copy of shared_benchmarks (avoids pip install dependency)
_LOCAL_SHARED = _NEXUS3_DIR / "shared_benchmarks"
if _LOCAL_SHARED.exists() and str(_NEXUS3_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS3_DIR))

try:
    from shared_benchmarks.runner import run_benchmark
except ImportError:
    # Fallback: shared_benchmarks may be at a sibling path
    _SHARED = _NEXUS3_DIR.parent / "shared_benchmarks"
    if _SHARED.exists() and str(_SHARED.parent) not in sys.path:
        sys.path.insert(0, str(_SHARED.parent))
    from shared_benchmarks.runner import run_benchmark


if __name__ == "__main__":
    run_benchmark(adapter_class=Nexus3Adapter)
