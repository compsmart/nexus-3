"""Narrative Memory for Nexus-3.

Key findings integrated:
- D-316: Narrative chain accumulation achieves 83-97% EM vs 0% for entity lists at k=10
- D-352: Multi-seed validation confirms narrative superiority (p=1.2e-9)
- D-248: AMM handles inference-time fact updates with zero accuracy loss
- D-245: Zero catastrophic forgetting with cross-task fine-tuning

The core insight: store reasoning chains as coherent natural language stories,
not isolated key-value pairs. Each memory entry contains both the raw fact
AND the narrative context of how it connects to other knowledge.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory slot with narrative context."""
    text: str
    narrative: str
    embedding: Optional[np.ndarray] = None
    mem_type: str = "fact"
    timestamp: float = 0.0
    access_count: int = 0
    last_accessed: float = 0.0
    source: str = ""
    connections: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "narrative": self.narrative,
            "mem_type": self.mem_type,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "source": self.source,
            "connections": self.connections,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            text=d["text"],
            narrative=d.get("narrative", d["text"]),
            mem_type=d.get("mem_type", "fact"),
            timestamp=d.get("timestamp", 0.0),
            access_count=d.get("access_count", 0),
            last_accessed=d.get("last_accessed", 0.0),
            source=d.get("source", ""),
            connections=d.get("connections", []),
        )


class NarrativeMemory:
    """Memory system that stores and retrieves narrative chains.

    Instead of storing isolated facts, each entry maintains a 'narrative' field
    that describes how the fact connects to related knowledge. This dramatically
    improves multi-hop reasoning (D-316: +33pp over entity lists at k=10).
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_slots: int = 10000,
        top_k: int = 5,
        dedup_threshold: float = 0.95,
        device: str = "cpu",
    ):
        self.max_slots = max_slots
        self.top_k = top_k
        self.dedup_threshold = dedup_threshold
        self.device = device

        self._encoder = None
        self._encoder_model_name = embedding_model
        self.entries: List[MemoryEntry] = []
        self._embeddings: Optional[np.ndarray] = None

    def _load_encoder(self):
        if self._encoder is not None:
            return
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model %s ...", self._encoder_model_name)
        self._encoder = SentenceTransformer(self._encoder_model_name, device=self.device)
        log.info("Embedding model loaded.")

    def _encode(self, texts: List[str]) -> np.ndarray:
        self._load_encoder()
        return self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _rebuild_index(self):
        """Rebuild the embedding matrix from all entries."""
        if not self.entries:
            self._embeddings = None
            return
        texts = [e.narrative for e in self.entries]
        self._embeddings = self._encode(texts)

    def store(
        self,
        text: str,
        narrative: Optional[str] = None,
        mem_type: str = "fact",
        source: str = "",
        connections: Optional[List[str]] = None,
    ) -> int:
        """Store a fact with its narrative context.

        Args:
            text: The raw factual content.
            narrative: Natural language story connecting this fact to context.
                       If None, defaults to the text itself.
            mem_type: Type tag (fact, identity, correction, observation, etc.)
            source: Where this fact came from.
            connections: List of related memory texts.

        Returns:
            Index of the stored entry.
        """
        narrative = narrative or text
        now = time.time()

        if self.entries and self.dedup_threshold < 1.0:
            emb = self._encode([narrative])
            if self._embeddings is not None and len(self._embeddings) > 0:
                sims = emb @ self._embeddings.T
                max_sim = float(sims.max())
                if max_sim >= self.dedup_threshold:
                    idx = int(sims.argmax())
                    existing = self.entries[idx]
                    if mem_type == "correction" or len(narrative) > len(existing.narrative):
                        existing.text = text
                        existing.narrative = narrative
                        existing.mem_type = mem_type
                        existing.timestamp = now
                        existing.source = source
                        self._embeddings[idx] = emb[0]
                    existing.access_count += 1
                    existing.last_accessed = now
                    log.debug("Dedup: updated existing entry %d (sim=%.3f)", idx, max_sim)
                    return idx

        entry = MemoryEntry(
            text=text,
            narrative=narrative,
            mem_type=mem_type,
            timestamp=now,
            access_count=0,
            last_accessed=now,
            source=source,
            connections=connections or [],
        )

        emb = self._encode([narrative])
        entry.embedding = emb[0]

        if len(self.entries) >= self.max_slots:
            oldest_idx = min(range(len(self.entries)), key=lambda i: self.entries[i].last_accessed)
            self.entries[oldest_idx] = entry
            self._embeddings[oldest_idx] = emb[0]
            idx = oldest_idx
        else:
            self.entries.append(entry)
            if self._embeddings is None:
                self._embeddings = emb
            else:
                self._embeddings = np.vstack([self._embeddings, emb])
            idx = len(self.entries) - 1

        return idx

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        type_filter: Optional[set] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Retrieve the most relevant memories for a query.

        Args:
            query: The search query.
            top_k: Number of results (defaults to self.top_k).
            type_filter: Only return entries matching these types.
            min_score: Minimum cosine similarity threshold.

        Returns:
            List of (entry, score) tuples, sorted by score descending.
        """
        if not self.entries or self._embeddings is None:
            return []

        k = top_k or self.top_k
        q_emb = self._encode([query])
        sims = (q_emb @ self._embeddings.T)[0]

        now = time.time()
        scores = []
        for i, sim in enumerate(sims):
            entry = self.entries[i]
            if type_filter and entry.mem_type not in type_filter:
                continue
            if sim < min_score:
                continue

            age_hours = (now - entry.timestamp) / 3600.0
            recency_boost = 1.0 / (1.0 + 0.01 * age_hours)
            score = float(sim) * 0.85 + recency_boost * 0.15
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, sc in scores[:k]:
            entry = self.entries[i]
            entry.access_count += 1
            entry.last_accessed = now
            results.append((entry, sc))

        return results

    def retrieve_narrative_chain(
        self,
        query: str,
        max_hops: int = 3,
        top_k_per_hop: int = 3,
    ) -> List[MemoryEntry]:
        """Multi-hop narrative retrieval.

        Implements the bridge-guided chain: retrieve initial memories,
        then follow their connections to build a coherent narrative chain.
        This is the core technique from D-316/D-335.

        Args:
            query: Starting query.
            max_hops: Maximum retrieval depth.
            top_k_per_hop: Results per hop.

        Returns:
            Ordered list of memory entries forming a narrative chain.
        """
        chain = []
        seen_texts = set()
        current_query = query

        for hop in range(max_hops):
            results = self.retrieve(current_query, top_k=top_k_per_hop)
            if not results:
                break

            new_entries = []
            for entry, score in results:
                if entry.text not in seen_texts:
                    chain.append(entry)
                    seen_texts.add(entry.text)
                    new_entries.append(entry)

            if not new_entries:
                break

            narratives = [e.narrative for e in new_entries]
            current_query = f"{query} | Context: {' '.join(narratives[:2])}"

        return chain

    def build_narrative_context(self, entries: List[MemoryEntry]) -> str:
        """Build a coherent narrative from a chain of memory entries.

        Instead of listing facts as bullet points, weave them into a story.
        This is the key insight from D-316: narrative format preserves
        reasoning chains far better than entity lists.
        """
        if not entries:
            return ""

        parts = []
        for i, entry in enumerate(entries):
            if i == 0:
                parts.append(entry.narrative)
            else:
                parts.append(f"Furthermore, {entry.narrative}")

        return " ".join(parts)

    def save(self, json_path: str, pt_path: Optional[str] = None):
        """Save memory to disk."""
        data = [e.to_dict() for e in self.entries]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if pt_path and self._embeddings is not None:
            torch.save(torch.from_numpy(self._embeddings), pt_path)
        log.info("Saved %d memory entries to %s", len(self.entries), json_path)

    def load(self, json_path: str, pt_path: Optional[str] = None):
        """Load memory from disk."""
        import os
        if not os.path.exists(json_path):
            log.info("No memory file at %s, starting fresh.", json_path)
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.entries = [MemoryEntry.from_dict(d) for d in data]

        if pt_path and os.path.exists(pt_path):
            self._embeddings = torch.load(pt_path, weights_only=True).numpy()
            log.info("Loaded %d embeddings from %s", len(self._embeddings), pt_path)
        else:
            self._rebuild_index()

        log.info("Loaded %d memory entries from %s", len(self.entries), json_path)

    def __len__(self):
        return len(self.entries)

    def clear(self):
        self.entries.clear()
        self._embeddings = None
