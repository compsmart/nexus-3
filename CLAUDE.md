# Nexus-3 Developer Agent Instructions

## Mission

Nexus-3 must work on REAL-WORLD text data — messy, ambiguous, unstructured natural language. Every change you make must be evaluated against that standard. If a technique only works on clean synthetic benchmark data, it is worthless. Do not ship it.

## Architecture

- **NarrativeMemory** (`memory.py`): Stores facts as narrative chains with embeddings and connections
- **BridgeGuidedRetriever** (`retriever.py`): Multi-hop retrieval via bridge entity identification
- **LlamaEngine** (`llm.py`): Qwen2.5-7B inference
- **Agent** (`agent.py`): Orchestrator with routing (INJECT/HEDGE/SKIP/REJECT)

## Non-Negotiable Engineering Standards

### 1. Real-World Robustness is Mandatory
- **NO hardcoded structural assumptions** about input text (e.g., `text.split()[0]` as entity extraction is unacceptable)
- Use proper NER, coreference resolution, or semantic similarity for entity extraction
- All retrieval must handle: ambiguous entities, pronouns, implicit relationships, noisy/incomplete text
- Test mentally: "Would this work on a Wikipedia paragraph? A customer support transcript? A legal document?" If no, don't do it.

### 2. Benchmarks are Diagnostics, NOT Targets
- Benchmark scores measure progress — they are not the goal
- NEVER tune logic to match synthetic benchmark data patterns
- NEVER add special-case handling that only helps benchmark-shaped inputs
- If a change improves benchmark scores but relies on synthetic data structure, REJECT it
- A smaller benchmark improvement that generalizes is worth more than a large one that doesn't

### 3. No Shortcuts
- No `text.split()[0]` for entity extraction — use embedding similarity or NER
- No assumptions about fact ordering in memory entries
- No prompt hacks to dodge scorer edge cases (e.g., rewording "I don't know" to avoid negation detection)
- No benchmark-specific context ordering that wouldn't work on heterogeneous real data

### 4. Implementation Quality
- Adapter integrity: < 60 lines, no `import re`, one call to real agent
- Changes go in agent code (agent.py, memory.py, retriever.py, config.py, llm.py) — NOT adapter.py
- Every retrieval improvement must degrade gracefully: if entity extraction fails, fall back to semantic search, not crash or return nothing
- Code must handle edge cases: empty queries, single-hop questions routed through multi-hop, very long documents

## What Good Changes Look Like

- Replacing string-matching bridge detection with embedding-based semantic bridge finding
- Adding proper entity extraction (spaCy, regex patterns for named entities, or LLM-based extraction)
- Improving context ranking by relevance score rather than assumed structural position
- Making multi-hop work when chain links use pronouns or paraphrases instead of exact entity names
- Robustness improvements: handling coreference, partial matches, ambiguous entities

## What Bad Changes Look Like

- Reordering context based on `text.split()[0]` matching a known chain
- Changing the "I don't know" phrasing to game the scorer
- Any logic that assumes facts start with the subject entity
- Any logic that assumes a clean Alpha→Bravo→Charlie chain structure
- Optimizing for the specific distractor/chain ratio in benchmarks

## Evaluation Criteria

When you benchmark a change, ask yourself:
1. Does this improvement come from better understanding of language, or from matching synthetic patterns?
2. Would this work if I fed in real conversation transcripts instead of synthetic facts?
3. Am I making the retriever smarter, or just making it better at this specific test?

If the answer to #1 is "synthetic patterns" or #2 is "no" or #3 is "this specific test" — roll it back and try again.
