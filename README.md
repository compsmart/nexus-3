# Nexus-3: Advanced AI Agent with Narrative Memory

Nexus-3 synthesizes key research findings from the AI Research Lab into a single agent:

- **Narrative Memory** (D-316, D-352): Stores reasoning chains as coherent stories instead of entity lists. Achieves 83-97% EM vs 0% for list-based memory at long reasoning chains.
- **Bridge-Guided Retrieval** (D-335, D-332): 2-call architecture that identifies bridge entities for multi-hop QA. +5pp EM and +14pp recall over standard iterative retrieval.
- **Confidence Gating** (D-222, D-227): Routes queries based on retrieval confidence to prevent hallucination.
- **Llama 3 8B** backbone with 4-bit quantization for efficient inference.

## Architecture

```
User Query
    |
    v
[Bridge-Guided Retriever]
    |-- Hop 1: Retrieve initial context
    |-- Bridge ID: Extract connecting entity (pattern or LLM)
    |-- Hop 2: Retrieve with bridge context
    |-- Confidence: Score and route
    v
[Narrative Memory]
    |-- Store as stories, not lists
    |-- Dedup by semantic similarity
    |-- Temporal decay + recency boost
    v
[Llama 3 8B Engine]
    |-- System prompt with retrieved memory
    |-- Greedy decoding for determinism
    |-- Tool call parsing and execution
    v
Response
```

## Setup

### Requirements

- Python 3.10+
- CUDA GPU with 8+ GB VRAM (for 4-bit Llama 3 8B)
- ~16 GB system RAM

### Install

```bash
cd poc/nexus-3
pip install -r requirements.txt
```

### HuggingFace Access

Llama 3 requires authentication. Set your token:

```bash
huggingface-cli login
# Or set: export HF_TOKEN=hf_...
```

## Usage

### Interactive Mode

```bash
python main.py                    # Full agent with LLM
python main.py --device cpu       # Force CPU (slow)
python main.py --no-llm           # Memory/retrieval only
python main.py --model path/to/model  # Custom model
```

### Benchmarking (HotpotQA)

```bash
# Full benchmark (4 conditions x 200 samples)
python benchmark.py

# Quick test
python benchmark.py --n 50 --conditions oracle

# Multi-seed for statistical significance
python benchmark.py --n 200 --seeds 42 7 137

# Specific conditions
python benchmark.py --conditions oracle bridge_guided --n 100
```

### Benchmark Conditions

| Condition | Description |
|-----------|-------------|
| `oracle` | Gold supporting paragraphs only (upper bound) |
| `distractor` | Gold + distractor paragraphs (realistic) |
| `bridge_guided` | 2-call architecture with bridge entity extraction |
| `memory_retrieval` | Store context in narrative memory, answer via retrieval |

## File Structure

```
poc/nexus-3/
├── main.py           # Interactive agent entry point
├── agent.py          # Main agent orchestrator
├── llm.py            # Llama 3 8B inference engine
├── memory.py         # Narrative Memory system
├── retriever.py      # Bridge-Guided Retriever
├── config.py         # Configuration dataclass
├── data_loader.py    # HotpotQA data loader + metrics
├── benchmark.py      # Benchmark runner
├── requirements.txt  # Python dependencies
├── data/             # Cached datasets and results
│   └── benchmark_results/
└── README.md         # This file
```

## Migration to GPU Server

1. Copy the entire `poc/nexus-3/` directory to the target server
2. Install dependencies: `pip install -r requirements.txt`
3. Authenticate with HuggingFace: `huggingface-cli login`
4. Run benchmark: `python benchmark.py --n 200 --seeds 42 7 137`
5. Results will be in `data/benchmark_results/benchmark_results.json`

## Key Research Findings Integrated

| Finding | Insight | Impact |
|---------|---------|--------|
| D-316 | Narrative chains >> entity lists at k=10 | +33pp EM |
| D-352 | Multi-seed confirms narrative (p=1.2e-9) | Statistical rigor |
| D-335 | Bridge-guided retrieval | +5pp EM, +14pp recall |
| D-332 | 2-call > single-call oracle | Architecture validation |
| L-323 | Call 2 needs full context | Critical constraint |
| D-346 | Greedy decoding = deterministic | Reproducibility |
| D-220 | Adversarial validation mandatory | Robustness |
