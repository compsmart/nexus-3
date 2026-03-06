"""Nexus-3 Agent orchestrator.

Integrates: LLM (Qwen2.5-7B default) + Narrative Memory + Bridge-Guided Retrieval.

The agent loop:
1. Accept user query
2. Retrieve relevant memory via bridge-guided multi-hop
3. Route based on confidence (INJECT / HEDGE / SKIP / REJECT)
4. Generate response with narrative context
5. Extract and store new facts into narrative memory
6. Handle tool calls if present
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional

from config import Nexus3Config
from llm import LlamaEngine
from memory import NarrativeMemory
from retriever import BridgeGuidedRetriever

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are NEXUS-3, an advanced AI assistant with persistent narrative memory and multi-hop reasoning.

You have access to your narrative memory — a chain of interconnected facts and stories that you have accumulated. When [Retrieved Memory] appears below, those are facts you know. Trust them fully.

CRITICAL RULES:
1. If the answer IS in [Retrieved Memory], use it confidently and directly.
2. If the answer is NOT in [Retrieved Memory], say "I don't have that information yet."
3. NEVER fabricate facts, names, numbers, or URLs.
4. When the user teaches you something new, acknowledge it and remember it.
5. Be concise: 1-3 sentences unless more detail is requested.
6. For multi-hop questions, reason step-by-step using the narrative chain provided.

Available tools (invoke with [TOOL_CALL: name | argument]):
- remember | fact1; fact2; fact3 — Store new facts (semicolon-separated)
- correct | wrong info | correct info — Fix an incorrect memory
- forget | topic — Remove a memory
- think | question — Reason through a complex question step by step

Guidelines:
- When the user provides new facts, ALWAYS use [TOOL_CALL: remember | ...] before responding.
- For multi-step reasoning, use [TOOL_CALL: think | question] to break it down.
- Be direct and helpful. Don't hedge when your memory is clear.
"""


class Nexus3Agent:
    """Main Nexus-3 agent combining LLM, Memory, and Bridge-Guided Retrieval."""

    def __init__(self, config: Optional[Nexus3Config] = None, load_llm: bool = True):
        self.config = config or Nexus3Config()
        self.history: List[Dict[str, str]] = []

        device = self.config.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.memory = NarrativeMemory(
            embedding_model=self.config.embedding_model,
            max_slots=self.config.max_memory_slots,
            top_k=self.config.memory_top_k,
            dedup_threshold=self.config.dedup_threshold,
            device=device,
        )

        self.llm = None
        if load_llm:
            self.llm = LlamaEngine(
                model_name=self.config.model_name,
                device=device,
                use_4bit=self.config.use_4bit,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
            )

        self.retriever = BridgeGuidedRetriever(
            memory=self.memory,
            llm=self.llm,
            hop1_top_k=self.config.hop1_top_k,
            hop2_top_k=self.config.hop2_top_k,
            bridge_top_k=self.config.bridge_top_k,
        )

        self._load_memory()

    def _load_memory(self):
        """Load persisted memory if available."""
        if os.path.exists(self.config.memory_path):
            self.memory.load(self.config.memory_path, self.config.memory_pt_path)

    def _save_memory(self):
        """Persist memory to disk."""
        os.makedirs(os.path.dirname(self.config.memory_path) or ".", exist_ok=True)
        self.memory.save(self.config.memory_path, self.config.memory_pt_path)

    def interact(self, user_input: str) -> str:
        """Process a single user turn and return the agent response.

        This is the main entry point for the agent loop:
        1. Extract any facts the user is teaching us
        2. Retrieve relevant memory context
        3. Route based on confidence
        4. Generate response
        5. Handle tool calls
        6. Store interaction in memory
        """
        self._extract_and_store_facts(user_input)

        context, confidence, route = self.retriever.retrieve_with_confidence(user_input)

        log.info("Retrieval: confidence=%.3f route=%s context_len=%d",
                 confidence, route, len(context))

        messages = self._build_messages(user_input, context, route)

        if self.llm is None:
            return "[LLM not loaded — cannot generate response]"

        response = self.llm.generate(messages, greedy=(self.config.temperature == 0))

        response = self._handle_tool_calls(response, user_input)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        if len(self.history) > 20:
            self.history = self.history[-20:]

        self._save_memory()
        return response

    def answer_question(
        self,
        question: str,
        context_paragraphs: Optional[Dict[str, List[str]]] = None,
        greedy: bool = True,
    ) -> str:
        """Answer a question using provided context (for benchmarking).

        This bypasses memory retrieval and directly uses the provided context,
        suitable for evaluating the agent on datasets like HotpotQA where
        context paragraphs are given.
        """
        if context_paragraphs:
            context_parts = []
            for title, sentences in context_paragraphs.items():
                text = " ".join(sentences)
                context_parts.append(f"[{title}]\n{text}")
            context = "\n\n".join(context_parts)
        else:
            retrieval = self.retriever.retrieve_bridge_guided(question)
            context = retrieval["full_context"]

        messages = [
            {"role": "system", "content": (
                "You are a precise question-answering system. "
                "Answer the question using ONLY the provided context. "
                "Give a short, direct answer (a few words or a single sentence). "
                "If the context does not contain the answer, say 'Insufficient information'."
            )},
            {"role": "user", "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )},
        ]

        if self.llm is None:
            return "[LLM not loaded]"

        return self.llm.generate(messages, max_new_tokens=64, greedy=greedy)

    def answer_multihop(
        self,
        question: str,
        context_paragraphs: Dict[str, List[str]],
        supporting_facts: Optional[Dict] = None,
        greedy: bool = True,
    ) -> Dict[str, object]:
        """Answer a multi-hop question using bridge-guided reasoning.

        Implements the 2-call architecture from D-332/D-335:
        Call 1: Identify the bridge entity from initial context
        Call 2: Answer using full context (hop1 + hop2)
        """
        all_titles = list(context_paragraphs.keys())

        call1_messages = [
            {"role": "system", "content": (
                "You are a bridge entity extractor for multi-hop reasoning. "
                "Given a question and context paragraphs, identify the KEY ENTITY "
                "that connects the question to its answer. This entity appears in "
                "one paragraph and links to information in another paragraph.\n"
                "Respond with ONLY the bridge entity name."
            )},
            {"role": "user", "content": self._format_qa_context(
                question, context_paragraphs, all_titles[:5]
            )},
        ]

        if self.llm is None:
            return {"answer": "[LLM not loaded]", "bridge": None}

        bridge = self.llm.generate(call1_messages, max_new_tokens=30, greedy=True).strip()

        bridge_titles = []
        for title in all_titles:
            text = " ".join(context_paragraphs[title])
            if bridge.lower() in text.lower() or bridge.lower() in title.lower():
                bridge_titles.append(title)

        hop2_titles = list(set(all_titles[:5] + bridge_titles))

        call2_messages = [
            {"role": "system", "content": (
                "You are a precise question-answering system. "
                "Answer the question using ONLY the provided context. "
                "Give a short, direct answer (a few words). "
                "Reason step by step if needed, but end with a clear final answer."
            )},
            {"role": "user", "content": self._format_qa_context(
                question, context_paragraphs, hop2_titles
            )},
        ]

        answer = self.llm.generate(call2_messages, max_new_tokens=64, greedy=greedy)

        return {
            "answer": answer.strip(),
            "bridge": bridge,
            "hop1_titles": all_titles[:5],
            "hop2_titles": hop2_titles,
        }

    def _format_qa_context(
        self,
        question: str,
        paragraphs: Dict[str, List[str]],
        titles: List[str],
    ) -> str:
        parts = ["Context:"]
        for title in titles:
            if title in paragraphs:
                text = " ".join(paragraphs[title])
                parts.append(f"\n[{title}]\n{text}")
        parts.append(f"\nQuestion: {question}")
        parts.append("\nAnswer:")
        return "\n".join(parts)

    def _build_messages(
        self,
        user_input: str,
        context: str,
        route: str,
    ) -> List[Dict[str, str]]:
        """Build the chat messages for the LLM."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in self.history[-10:]:
            messages.append(msg)

        if context and route in ("INJECT_FULL", "INJECT_TOP1", "HEDGE"):
            if route == "HEDGE":
                context_block = f"[Retrieved Memory (uncertain — verify before using)]\n{context}"
            else:
                context_block = f"[Retrieved Memory]\n{context}"
            user_msg = f"{context_block}\n\nUser: {user_input}"
        else:
            user_msg = user_input

        messages.append({"role": "user", "content": user_msg})
        return messages

    def _extract_and_store_facts(self, user_input: str):
        """Extract factual statements from user input and store them."""
        fact_patterns = [
            (r"\bmy name is\s+(.+?)(?:\.|$)", "identity"),
            (r"\bi am\s+(.+?)(?:\.|$)", "identity"),
            (r"\bi live in\s+(.+?)(?:\.|$)", "fact"),
            (r"\bi work (?:at|for)\s+(.+?)(?:\.|$)", "fact"),
            (r"\bmy (?:phone|number) is\s+(.+?)(?:\.|$)", "fact"),
            (r"\bmy email is\s+(.+?)(?:\.|$)", "fact"),
            (r"\bremember that\s+(.+?)(?:\.|$)", "fact"),
        ]

        for pattern, mem_type in fact_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                fact = match.group(1).strip()
                narrative = f"The user stated: {user_input.strip()}. Key fact: {fact}."
                self.memory.store(
                    text=fact,
                    narrative=narrative,
                    mem_type=mem_type,
                    source="user_input",
                )

    def _handle_tool_calls(self, response: str, user_input: str) -> str:
        """Process any tool calls in the response."""
        pattern = re.compile(self.config.tool_call_pattern)
        calls_handled = 0

        while calls_handled < self.config.max_tool_calls:
            match = pattern.search(response)
            if not match:
                break

            tool_name = match.group(1).strip().lower()
            tool_arg = match.group(2).strip()
            calls_handled += 1

            result = self._execute_tool(tool_name, tool_arg, user_input)

            response = response[:match.start()] + result + response[match.end():]

        return response.strip()

    def _execute_tool(self, name: str, arg: str, user_input: str) -> str:
        """Execute a tool call and return the result string."""
        if name == "remember":
            facts = [f.strip() for f in arg.split(";") if f.strip()]
            for fact in facts:
                narrative = f"The user taught me: {fact}. Original context: {user_input}"
                self.memory.store(
                    text=fact,
                    narrative=narrative,
                    mem_type="fact",
                    source="tool_remember",
                )
            return f"(Stored {len(facts)} fact(s) in memory)"

        elif name == "correct":
            parts = [p.strip() for p in arg.split("|")]
            if len(parts) >= 2:
                wrong, correct = parts[0], parts[1]
                narrative = f"Correction: '{wrong}' is wrong. The correct information is: {correct}"
                self.memory.store(
                    text=correct,
                    narrative=narrative,
                    mem_type="correction",
                    source="tool_correct",
                )
                return f"(Corrected: {wrong} -> {correct})"
            return "(Correction failed — need 'wrong | correct' format)"

        elif name == "forget":
            results = self.memory.retrieve(arg, top_k=1, min_score=0.5)
            if results:
                entry, score = results[0]
                idx = self.memory.entries.index(entry)
                self.memory.entries.pop(idx)
                if self.memory._embeddings is not None:
                    import numpy as np
                    self.memory._embeddings = np.delete(self.memory._embeddings, idx, axis=0)
                return f"(Forgot: {entry.text[:50]}...)"
            return "(Nothing matching found to forget)"

        elif name == "think":
            chain = self.memory.retrieve_narrative_chain(arg, max_hops=3)
            if chain:
                context = self.memory.build_narrative_context(chain)
                return f"(Reasoning chain: {context[:200]}...)"
            return "(No relevant memories to reason over)"

        log.warning("Unknown tool: %s", name)
        return f"(Unknown tool: {name})"

    def store_knowledge(
        self,
        facts: List[Dict[str, str]],
        source: str = "bulk_load",
    ):
        """Bulk-load knowledge into memory (for benchmarking/seeding).

        Args:
            facts: List of dicts with 'text' and optionally 'narrative', 'type'.
            source: Source label.
        """
        for fact in facts:
            self.memory.store(
                text=fact["text"],
                narrative=fact.get("narrative", fact["text"]),
                mem_type=fact.get("type", "fact"),
                source=source,
            )

    def reset(self):
        """Clear all memory and history (for benchmark isolation)."""
        self.memory.clear()
        self.history.clear()
