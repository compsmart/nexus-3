"""Bridge-Guided Retriever for Nexus-3.

Key findings integrated:
- D-335: Bridge-guided retrieval improves EM +5pp and Hop2 recall +14pp
- D-332: 2-call architecture with correct context EXCEEDS single-call oracle
- L-323: Call 2 MUST receive hop1+hop2 context (not just bridge hint)
- D-304: Dynamic per-question entity lookup resolves coverage ceiling
- L-299: LLM multi-hop reasoning is the bottleneck after retrieval is solved
- D-432: Context enrichment HURTS hop-2 retrieval (-8pp); use bridge entity name only
- D-433: Entity name extraction achieves 83% hop-2 recall (vs 73% for descriptive phrases)

The bridge-guided strategy:
1. Given a question, retrieve initial context (Hop 1)
2. Identify the "bridge entity" connecting two pieces of evidence (use exact name, D-433)
3. Use ONLY the bridge entity name for hop-2 retrieval (no context enrichment, D-432)
4. Provide FULL context (hop1 + hop2) for final answer generation
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from memory import NarrativeMemory, MemoryEntry

log = logging.getLogger(__name__)


class BridgeGuidedRetriever:
    """Two-step retrieval that identifies bridge entities for multi-hop QA.

    The key architectural insight (L-323): the second retrieval call must
    receive the COMPLETE context from both hops, not just the bridge hint.
    Single-hop context with bridge hint degrades to 11% vs 37.5% oracle.
    """

    def __init__(
        self,
        memory: NarrativeMemory,
        llm=None,
        hop1_top_k: int = 5,
        hop2_top_k: int = 5,
        bridge_top_k: int = 3,
    ):
        self.memory = memory
        self.llm = llm
        self.hop1_top_k = hop1_top_k
        self.hop2_top_k = hop2_top_k
        self.bridge_top_k = bridge_top_k

    def retrieve_simple(self, query: str, top_k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        """Single-hop retrieval (baseline)."""
        return self.memory.retrieve(query, top_k=top_k)

    def retrieve_bridge_guided(
        self,
        question: str,
        entities: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, object]:
        """Full bridge-guided 2-hop retrieval.

        Args:
            question: The user's question.
            entities: Optional dict mapping entity titles to their facts/paragraphs.

        Returns:
            Dict with keys:
                hop1_entries: Retrieved memories from hop 1
                bridge_entity: The identified bridge entity/concept
                hop2_entries: Retrieved memories from hop 2
                full_context: Combined narrative context for answer generation
                hop1_scores: Similarity scores for hop 1
                hop2_scores: Similarity scores for hop 2
        """
        hop1_results = self.memory.retrieve(question, top_k=self.hop1_top_k)
        hop1_entries = [e for e, _ in hop1_results]
        hop1_scores = [s for _, s in hop1_results]

        if not hop1_entries:
            return {
                "hop1_entries": [],
                "bridge_entity": None,
                "hop2_entries": [],
                "full_context": "",
                "hop1_scores": [],
                "hop2_scores": [],
            }

        # D-432/D-433: Use bridge entity name only for hop retrieval (no context enrichment).
        # Iteratively follow the chain for up to max_hops to support N-hop reasoning.
        # Each hop extracts a new bridge entity from the previous hop's entries and
        # retrieves the next link — critical for 3/4/5-hop chain queries.
        seen_texts = {e.text for e in hop1_entries}
        all_entries = list(hop1_entries)
        all_scores = list(hop1_scores)

        first_bridge = None
        hop2_entries: List[MemoryEntry] = []
        hop2_scores: List[float] = []

        current_entries = hop1_entries
        current_query = question
        max_hops = 5

        for hop in range(max_hops - 1):
            bridge_entity = self._identify_bridge(current_query, current_entries)

            if hop == 0:
                first_bridge = bridge_entity

            hop_query = bridge_entity if bridge_entity else None
            if not hop_query:
                break

            hop_results = self.memory.retrieve(hop_query, top_k=self.hop2_top_k)
            new_entries = [e for e, _ in hop_results if e.text not in seen_texts]
            new_scores = [s for e, s in hop_results if e.text not in seen_texts]

            if not new_entries:
                break

            for e in new_entries:
                seen_texts.add(e.text)
            all_entries.extend(new_entries)
            all_scores.extend(new_scores)

            if hop == 0:
                hop2_entries = new_entries
                hop2_scores = new_scores

            # Advance: use bridge entity as next query for further hops
            current_entries = new_entries
            current_query = hop_query

        # Build ordered context: chain links first (best hop1 seed + bridge-hop entries),
        # then remaining hop1 entries. This ensures the LLM sees the chain path
        # in order rather than buried under distractors (Context Structure > Content).
        bridge_hop_entries = all_entries[len(hop1_entries):]
        if hop1_entries:
            ordered_entries = [hop1_entries[0]] + bridge_hop_entries + hop1_entries[1:]
        else:
            ordered_entries = bridge_hop_entries
        full_context = self.memory.build_narrative_context(ordered_entries)

        return {
            "hop1_entries": hop1_entries,
            "bridge_entity": first_bridge,
            "hop2_entries": hop2_entries,
            "full_context": full_context,
            "hop1_scores": hop1_scores,
            "hop2_scores": hop2_scores,
        }

    def _identify_bridge(
        self,
        question: str,
        hop1_entries: List[MemoryEntry],
    ) -> Optional[str]:
        """Identify the bridge entity connecting question to deeper context.

        Tries pattern-based extraction first (faster, deterministic for structured data),
        falls back to LLM for complex/ambiguous cases (D-264, L-255).
        """
        bridge = self._identify_bridge_pattern(question, hop1_entries)
        if bridge:
            return bridge
        if self.llm is not None:
            return self._identify_bridge_llm(question, hop1_entries)
        return None

    def _identify_bridge_pattern(
        self,
        question: str,
        hop1_entries: List[MemoryEntry],
    ) -> Optional[str]:
        """Extract bridge entity using pattern matching (no LLM needed)."""
        combined_text = " ".join(e.text for e in hop1_entries[:3])

        proper_nouns = set()
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', combined_text):
            candidate = match.group(1)
            if candidate.lower() not in question.lower() and len(candidate) > 2:
                proper_nouns.add(candidate)

        if proper_nouns:
            return sorted(proper_nouns, key=len, reverse=True)[0]
        return None

    def _identify_bridge_llm(
        self,
        question: str,
        hop1_entries: List[MemoryEntry],
    ) -> Optional[str]:
        """Use the LLM to identify the bridge entity."""
        context = "\n".join(f"- {e.text}" for e in hop1_entries[:3])

        # D-433: Exact entity names achieve 83% hop-2 recall vs 73% for descriptive phrases.
        # Force the LLM to return a short entity name, not a description.
        messages = [
            {"role": "system", "content": (
                "You are a bridge entity extractor. Given a question and initial context, "
                "identify the key NAMED ENTITY (person, place, organization, or thing) that "
                "connects the known information to the answer. "
                "Respond with ONLY the entity name (1-4 words max). No descriptions, no sentences."
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\nInitial context:\n{context}\n\n"
                "Bridge entity name:"
            )},
        ]

        try:
            response = self.llm.generate(messages, max_new_tokens=30, greedy=True)
            bridge = response.strip().strip('"').strip("'")
            if bridge and len(bridge) < 100:
                return bridge
        except Exception as e:
            log.warning("Bridge LLM extraction failed: %s", e)

        return self._identify_bridge_pattern(question, hop1_entries)

    def retrieve_with_confidence(
        self,
        question: str,
    ) -> Tuple[str, float, str]:
        """Retrieve context and compute a confidence score.

        Returns:
            (context_text, confidence_score, route_level)
            route_level is one of: INJECT_FULL, INJECT_TOP1, HEDGE, SKIP, REJECT
        """
        results = self.retrieve_bridge_guided(question)

        if not results["hop1_entries"]:
            return "", 0.0, "REJECT"

        all_scores = results["hop1_scores"] + results["hop2_scores"]
        if not all_scores:
            return "", 0.0, "REJECT"

        max_score = max(all_scores)
        margin = max_score - (sorted(all_scores, reverse=True)[1] if len(all_scores) > 1 else 0.0)

        if max_score >= 0.55:
            route = "INJECT_FULL"
        elif max_score >= 0.40:
            route = "INJECT_TOP1" if margin > 0.10 else "HEDGE"
        elif max_score >= 0.15:
            route = "SKIP"
        else:
            route = "REJECT"

        return results["full_context"], max_score, route
