"""Predict-Then-Compress (PTC) typed tag layer for Nexus-3.

Applies D-397: PTC achieves needle=0.920, commit=1.000, goal=0.800, LH=0.900
at 435 tokens by converting conversation turns into typed entity:value tags
before retrieval, then routing queries to the appropriate tag store or
narrative memory based on query classification.

Applies D-401: Oracle bounds show that perfect typed extraction at 157 tokens
beats full context (508 tokens) on needle recall. Structured format > verbatim.

Core insight: convert raw conversation -> typed tags BEFORE query, then use
tag-type-aware routing (EXACT/COMMITMENT/GOAL/NARRATIVE) to maximize recall
on the specific sub-task.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tag types and extraction patterns
# ---------------------------------------------------------------------------

TAG_TYPES = {
    "entity:name",      # person/entity names
    "entity:number",    # numeric values (phone, age, price, etc.)
    "entity:id",        # identifiers, codes
    "entity:url",       # URLs, links
    "fact:location",    # locations, places
    "fact:attribute",   # general attributes
    "fact:preference",  # user preferences/likes
    "commitment:will",  # user will do something
    "commitment:done",  # user completed something
    "goal:wants",       # user goals/intentions
    "goal:needs",       # user needs
    "correction",       # corrections to prior facts
}

# Patterns: (regex, tag_type, group_index_for_value)
_EXTRACTION_PATTERNS = [
    # Identity / Name
    (r"\bmy name is\s+([A-Za-z][A-Za-z\s'\-]{0,40}?)(?:[,.\n]|$)", "entity:name", 1),
    (r"\bi(?:'m| am)\s+([A-Za-z][A-Za-z\s'\-]{0,40}?)(?:[,.\n]|$)", "entity:name", 1),
    (r"\bcall me\s+([A-Za-z][A-Za-z\s'\-]{0,20}?)(?:[,.\n]|$)", "entity:name", 1),

    # Numbers: phone, age, zip, account
    (r"\bmy (?:phone|number|mobile|cell) (?:number )?is\s+([\d\s\+\-\(\)]{7,20})", "entity:number", 1),
    (r"\bmy (?:age|birthday) is\s+(\d{1,3}[^.]*?)(?:[.\n]|$)", "entity:number", 1),
    (r"\bmy (?:zip|postal) (?:code )?is\s+(\d{4,10})", "entity:number", 1),
    (r"\bmy account (?:number|id) is\s+([\w\-]{3,30})", "entity:id", 1),
    (r"\bmy (?:order|ticket|reference|booking) (?:number|id|#) ?(?:is )?#?([\w\-]{3,30})", "entity:id", 1),
    (r"\b(?:it costs?|price is|costs?)\s+\$?([\d,\.]+)", "entity:number", 1),

    # Location
    (r"\bi live (?:in|at)\s+(.+?)(?:[,.\n]|$)", "fact:location", 1),
    (r"\bi(?:'m| am) (?:from|in|at)\s+(.+?)(?:[,.\n]|$)", "fact:location", 1),
    (r"\bmy (?:address|location) is\s+(.+?)(?:[.\n]|$)", "fact:location", 1),
    (r"\bmy (?:home|office|work) is (?:at|in)\s+(.+?)(?:[.\n]|$)", "fact:location", 1),

    # Attribute / Work
    (r"\bi work (?:at|for|with)\s+(.+?)(?:[,.\n]|$)", "fact:attribute", 1),
    (r"\bmy (?:job|role|title|position|occupation) is\s+(.+?)(?:[,.\n]|$)", "fact:attribute", 1),
    (r"\bmy (?:email|gmail) (?:address )?is\s+([\w\.\+\-@]+)", "entity:id", 1),
    (r"\bmy (?:password|pin) is\s+(\S+)", "entity:id", 1),
    (r"\bmy (?:username|user) is\s+(\S+)", "entity:id", 1),

    # Preferences
    (r"\bi (?:like|love|prefer|enjoy)\s+(.+?)(?:[,.\n]|$)", "fact:preference", 1),
    (r"\bmy (?:favorite|favourite)\s+\w+ is\s+(.+?)(?:[,.\n]|$)", "fact:preference", 1),
    (r"\bi (?:don't|dislike|hate)\s+(.+?)(?:[,.\n]|$)", "fact:preference", 1),

    # Commitments
    (r"\bi(?:'ll| will)\s+(.+?)(?:[,.\n]|$)", "commitment:will", 1),
    (r"\bi(?:'m| am) going to\s+(.+?)(?:[,.\n]|$)", "commitment:will", 1),
    (r"\bpromise(?:d)? (?:to|that)\s+(.+?)(?:[,.\n]|$)", "commitment:will", 1),
    (r"\bi (?:already|just) (?:did|finished|completed|sent|paid|called)\s+(.+?)(?:[,.\n]|$)", "commitment:done", 1),

    # Goals
    (r"\bi (?:want to|wanna)\s+(.+?)(?:[,.\n]|$)", "goal:wants", 1),
    (r"\bmy goal is (?:to\s+)?(.+?)(?:[,.\n]|$)", "goal:wants", 1),
    (r"\bi(?:'m| am) trying to\s+(.+?)(?:[,.\n]|$)", "goal:wants", 1),
    (r"\bi need to\s+(.+?)(?:[,.\n]|$)", "goal:needs", 1),
    (r"\bi need\s+(.+?)(?:[,.\n]|$)", "goal:needs", 1),
    (r"\bplease (?:help me|remind me)\s+(?:to\s+)?(.+?)(?:[,.\n]|$)", "goal:needs", 1),

    # Remember directives
    (r"\bremember (?:that\s+)?(.+?)(?:[,.\n]|$)", "fact:attribute", 1),
    (r"\bdon'?t forget (?:that\s+)?(.+?)(?:[,.\n]|$)", "fact:attribute", 1),
]


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

_EXACT_SIGNALS = re.compile(
    r"\b(?:what(?:'s| is) my (?:phone|number|age|zip|account|order|email|address|password|pin|username|id)|"
    r"how (?:old am i|much does)|"
    r"what (?:number|id|code)|"
    r"\d{3,})\b",
    re.IGNORECASE,
)

_COMMITMENT_SIGNALS = re.compile(
    r"\b(?:did (?:you|i)|have (?:you|i)|did i (?:say|tell|promise)|"
    r"(?:you|i) (?:said|told|promised|agreed)|"
    r"will (?:you|i)|what did (?:you|i) say|"
    r"am i going to)\b",
    re.IGNORECASE,
)

_GOAL_SIGNALS = re.compile(
    r"\b(?:what (?:am i|do i) (?:want|need|try)|"
    r"what(?:'s| is) my (?:goal|plan|intention)|"
    r"remind me what|"
    r"what (?:was i|were we) (?:planning|working on))\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> str:
    """Classify a query into a routing type.

    Returns one of: 'EXACT', 'COMMITMENT', 'GOAL', 'NARRATIVE'.

    EXACT:      numeric/identifier lookups — check tag store first
    COMMITMENT: promise/plan tracking — check commitment tags
    GOAL:       intention/goal queries — check goal tags
    NARRATIVE:  general context — use NarrativeMemory bridge-guided retrieval
    """
    if _EXACT_SIGNALS.search(query):
        return "EXACT"
    if _COMMITMENT_SIGNALS.search(query):
        return "COMMITMENT"
    if _GOAL_SIGNALS.search(query):
        return "GOAL"
    return "NARRATIVE"


# ---------------------------------------------------------------------------
# Tag store
# ---------------------------------------------------------------------------

@dataclass
class TypedTag:
    """A single extracted typed tag."""
    tag_type: str
    value: str
    source_text: str
    timestamp: float = field(default_factory=time.time)
    turn_index: int = 0

    def to_context_str(self) -> str:
        """Format as compact context string (D-397: ~157 tokens targets)."""
        return f"[{self.tag_type}] {self.value}"


class TypedTagStore:
    """Stores and retrieves typed entity:value tags extracted from conversation.

    Implements the PTC tag store from D-397. Tags are extracted per-turn and
    indexed by tag_type for efficient typed retrieval. Unlike NarrativeMemory,
    this store is optimized for exact/typed lookups rather than embedding similarity.
    """

    def __init__(self, max_tags: int = 500):
        self.max_tags = max_tags
        self._tags: List[TypedTag] = []
        self._turn_index: int = 0

    def extract_and_store(self, text: str) -> List[TypedTag]:
        """Extract typed tags from a text turn and store them.

        Returns the list of newly extracted tags.
        """
        new_tags = self._extract_tags(text)
        for tag in new_tags:
            tag.turn_index = self._turn_index
            self._upsert(tag)
        self._turn_index += 1
        return new_tags

    def _extract_tags(self, text: str) -> List[TypedTag]:
        """Run all extraction patterns on text."""
        tags = []
        text_lower = text.lower()
        for pattern, tag_type, group_idx in _EXTRACTION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(group_idx).strip().rstrip(".,;:!?")
                if not value or len(value) < 2:
                    continue
                # Skip suspiciously long values (> 120 chars)
                if len(value) > 120:
                    value = value[:120].rsplit(" ", 1)[0]
                tags.append(TypedTag(
                    tag_type=tag_type,
                    value=value,
                    source_text=text[:200],
                ))
        return tags

    def _upsert(self, new_tag: TypedTag) -> None:
        """Insert or update a tag. For entity/fact types, update if same value prefix found."""
        # For entity:name and entity:id, replace existing same-type tags to avoid stale data
        if new_tag.tag_type in ("entity:name", "entity:id", "fact:location"):
            for i, existing in enumerate(self._tags):
                if existing.tag_type == new_tag.tag_type:
                    # Update in-place — newer info wins
                    self._tags[i] = new_tag
                    return

        # Dedup: skip if identical (type, value) already stored
        for existing in self._tags:
            if existing.tag_type == new_tag.tag_type and existing.value.lower() == new_tag.value.lower():
                existing.timestamp = new_tag.timestamp
                existing.turn_index = new_tag.turn_index
                return

        # Evict oldest if full
        if len(self._tags) >= self.max_tags:
            self._tags.sort(key=lambda t: t.timestamp)
            self._tags.pop(0)

        self._tags.append(new_tag)

    def query(
        self,
        query_type: str,
        query_text: str,
        top_k: int = 5,
    ) -> List[TypedTag]:
        """Retrieve tags relevant to a query.

        Args:
            query_type: One of EXACT/COMMITMENT/GOAL/NARRATIVE
            query_text: The original user query for keyword matching
            top_k: Max results

        Returns:
            List of relevant TypedTag objects, sorted by recency.
        """
        if not self._tags:
            return []

        if query_type == "EXACT":
            target_types = {"entity:number", "entity:id", "entity:name", "fact:location"}
        elif query_type == "COMMITMENT":
            target_types = {"commitment:will", "commitment:done"}
        elif query_type == "GOAL":
            target_types = {"goal:wants", "goal:needs"}
        else:
            # NARRATIVE: return highest-relevance tags across all types
            target_types = TAG_TYPES

        # Filter by type
        candidates = [t for t in self._tags if t.tag_type in target_types]

        # Keyword boost: prefer tags whose value appears in the query
        query_lower = query_text.lower()
        query_words = set(query_lower.split())

        def score(tag: TypedTag) -> float:
            val_lower = tag.value.lower()
            # Keyword overlap: count words from tag value in query
            tag_words = set(val_lower.split())
            overlap = len(tag_words & query_words) / max(len(tag_words), 1)
            # Recency: normalize turn_index (higher = more recent = better)
            recency = tag.turn_index / max(self._turn_index, 1)
            return overlap * 0.6 + recency * 0.4

        candidates.sort(key=score, reverse=True)
        return candidates[:top_k]

    def build_tag_context(self, tags: List[TypedTag], max_tokens_est: int = 160) -> str:
        """Build a compact context string from tags.

        Targets D-401's oracle_gold token budget (~157 tokens).
        """
        if not tags:
            return ""

        parts = []
        char_budget = max_tokens_est * 4  # rough 4 chars/token estimate
        total = 0

        for tag in tags:
            line = tag.to_context_str()
            if total + len(line) > char_budget:
                break
            parts.append(line)
            total += len(line) + 1

        return "\n".join(parts)

    def all_tags(self) -> List[TypedTag]:
        """Return all stored tags sorted by recency."""
        return sorted(self._tags, key=lambda t: t.turn_index, reverse=True)

    def __len__(self) -> int:
        return len(self._tags)


# ---------------------------------------------------------------------------
# PTC orchestrator
# ---------------------------------------------------------------------------

class PredictThenCompress:
    """Main PTC interface combining extraction, storage, and retrieval.

    Usage in the agent:
        ptc = PredictThenCompress()

        # On each user turn:
        ptc.process_turn(user_text)

        # On retrieval:
        query_type = ptc.classify(query)
        if query_type in ('EXACT', 'COMMITMENT', 'GOAL'):
            tag_context = ptc.retrieve_tag_context(query_type, query)
            # Prepend tag_context to narrative context if non-empty
        # Always also use NarrativeMemory for NARRATIVE queries
    """

    def __init__(self, max_tags: int = 500):
        self.store = TypedTagStore(max_tags=max_tags)

    def process_turn(self, text: str) -> List[TypedTag]:
        """Extract and store typed tags from a conversation turn."""
        return self.store.extract_and_store(text)

    def classify(self, query: str) -> str:
        """Classify the query to determine routing strategy."""
        return classify_query(query)

    def retrieve_tag_context(
        self,
        query_type: str,
        query_text: str,
        top_k: int = 8,
        max_tokens_est: int = 160,
    ) -> str:
        """Retrieve a compact typed-tag context string for the query.

        Returns empty string if no relevant tags found.
        """
        tags = self.store.query(query_type, query_text, top_k=top_k)
        if not tags:
            return ""
        return self.store.build_tag_context(tags, max_tokens_est=max_tokens_est)

    def retrieve_tags(self, query_type: str, query_text: str, top_k: int = 8) -> List[TypedTag]:
        """Retrieve tags without formatting them into a string."""
        return self.store.query(query_type, query_text, top_k=top_k)

    @property
    def tag_count(self) -> int:
        return len(self.store)
