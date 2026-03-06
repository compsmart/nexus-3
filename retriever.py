"""Bridge-Guided Retriever for Nexus-3.

Key findings integrated:
- D-335: Bridge-guided retrieval improves EM +5pp and Hop2 recall +14pp
- D-332: 2-call architecture with correct context EXCEEDS single-call oracle
- L-323: Call 2 MUST receive hop1+hop2 context (not just bridge hint)
- D-304: Dynamic per-question entity lookup resolves coverage ceiling
- L-299: LLM multi-hop reasoning is the bottleneck after retrieval is solved
- D-422: Medium-quality semantic descriptors beat exact entity names for hop-2 recall
  (57.4% vs 41.6%): multi-query retrieval with both exact name and semantic descriptor

The bridge-guided strategy:
1. Given a question, retrieve initial context (Hop 1)
2. Identify the "bridge entity" + semantic descriptor connecting two pieces of evidence
3. Use BOTH exact-name and descriptor queries to retrieve hop-2 context (D-422)
4. Merge hop-2 results with deduplication, sort by score
5. Provide FULL context (hop1 + hop2) for final answer generation
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

        # D-422: multi-query hop-2 retrieval with both exact entity name and
        # semantic descriptor -- medium-quality semantic queries outperform
        # exact entity names (57.4% vs 41.6% hop-2 recall).
        hop2_queries, bridge_entity = self._extract_bridge_queries(question, hop1_entries)

        seen_texts = {e.text for e in hop1_entries}
        hop2_seen: dict = {}  # text -> (entry, best_score)
        for q in hop2_queries:
            for entry, score in self.memory.retrieve(q, top_k=self.hop2_top_k):
                if entry.text not in seen_texts:
                    if entry.text not in hop2_seen or score > hop2_seen[entry.text][1]:
                        hop2_seen[entry.text] = (entry, score)

        hop2_sorted = sorted(hop2_seen.values(), key=lambda x: x[1], reverse=True)
        hop2_entries = [e for e, _ in hop2_sorted[:self.hop2_top_k]]
        hop2_scores = [s for _, s in hop2_sorted[:self.hop2_top_k]]

        all_entries = hop1_entries + hop2_entries
        full_context = self.memory.build_narrative_context(all_entries)

        return {
            "hop1_entries": hop1_entries,
            "bridge_entity": bridge_entity,
            "hop2_entries": hop2_entries,
            "full_context": full_context,
            "hop1_scores": hop1_scores,
            "hop2_scores": hop2_scores,
        }

    def _extract_bridge_queries(
        self,
        question: str,
        hop1_entries: List[MemoryEntry],
    ) -> Tuple[List[str], Optional[str]]:
        """Build multiple hop-2 retrieval queries using entity name + semantic descriptor.

        D-422: Embedding-based retrieval rewards semantic variation. An exact entity
        name may be too narrow -- a semantic descriptor (what this entity IS about)
        retrieves documents that exact-match queries miss (57.4% vs 41.6% hop-2 recall).

        Returns:
            (queries, bridge_entity) where queries is a list of hop-2 query strings
            and bridge_entity is the exact name (may be None).
        """
        base_ctx = hop1_entries[0].narrative if hop1_entries else ""
        queries: List[str] = []
        bridge_entity: Optional[str] = None

        if self.llm is not None:
            context = "\n".join(f"- {e.text}" for e in hop1_entries[:3])
            messages = [
                {"role": "system", "content": (
                    "You are a bridge entity extractor. Given a question and initial "
                    "context, identify the key entity or concept that connects the "
                    "information to the answer.\n"
                    "Respond in exactly this format (two lines):\n"
                    "ENTITY: <exact entity name>\n"
                    "DESCRIPTION: <brief semantic description in 5-10 words>\n"
                    "Example:\n"
                    "ENTITY: YG Entertainment\n"
                    "DESCRIPTION: South Korean K-pop music talent agency"
                )},
                {"role": "user", "content": (
                    f"Question: {question}\n\nInitial context:\n{context}\n\nBridge:"
                )},
            ]
            try:
                response = self.llm.generate(messages, max_new_tokens=60, greedy=True)
                descriptor: Optional[str] = None
                for line in response.strip().splitlines():
                    if line.startswith("ENTITY:"):
                        bridge_entity = line[7:].strip().strip('"').strip("'")
                    elif line.startswith("DESCRIPTION:"):
                        descriptor = line[12:].strip().strip('"').strip("'")

                if bridge_entity and len(bridge_entity) < 100:
                    # Query 1: exact entity name (traditional, high-quality)
                    queries.append(
                        f"{question} | Bridge: {bridge_entity} | Context: {base_ctx}"
                    )
                if descriptor and len(descriptor) < 200:
                    # Query 2: semantic descriptor (D-422: medium-quality beats exact)
                    queries.append(
                        f"{question} | {descriptor} | Context: {base_ctx}"
                    )
            except Exception as e:
                log.warning("Bridge extraction failed: %s", e)

        if not queries:
            # Fallback: pattern-based entity extraction
            bridge_entity = self._identify_bridge_pattern(question, hop1_entries)
            if bridge_entity:
                queries.append(
                    f"{question} | Bridge: {bridge_entity} | Context: {base_ctx}"
                )
            else:
                queries.append(f"{question} | Context: {base_ctx}")

        return queries, bridge_entity

    def _identify_bridge(
        self,
        question: str,
        hop1_entries: List[MemoryEntry],
    ) -> Optional[str]:
        """Identify the bridge entity connecting question to deeper context.

        Uses the LLM if available, otherwise falls back to pattern extraction.
        """
        if self.llm is not None:
            return self._identify_bridge_llm(question, hop1_entries)
        return self._identify_bridge_pattern(question, hop1_entries)

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

        messages = [
            {"role": "system", "content": (
                "You are a bridge entity extractor. Given a question and initial context, "
                "identify the key entity or concept that connects the known information "
                "to the answer. Respond with ONLY the bridge entity name, nothing else."
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\nInitial context:\n{context}\n\n"
                "Bridge entity:"
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
