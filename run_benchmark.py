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

try:
    from shared_benchmarks.runner import run_benchmark
except ImportError:
    # Fallback: pip install from local nexus-benchmarks repo or GitHub
    import subprocess
    _LOCAL = _NEXUS3_DIR.parent / "nexus-benchmarks"
    _pkg = str(_LOCAL) if _LOCAL.exists() else "git+https://github.com/compsmart/nexus-benchmarks.git"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)
    from shared_benchmarks.runner import run_benchmark


if __name__ == "__main__":
    run_benchmark(adapter_class=Nexus3Adapter)
