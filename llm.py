"""Llama 3 8B inference engine for Nexus-3.

Supports both full-precision and 4-bit quantized inference.
Uses the HuggingFace transformers pipeline with proper chat templating.
"""

import logging
from typing import List, Dict, Optional

import torch

log = logging.getLogger(__name__)


class LlamaEngine:
    """Wrapper around Llama 3 8B for text generation."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "auto",
        use_4bit: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self._model = None
        self._tokenizer = None

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_4bit = use_4bit and self.device == "cuda"

        log.info("LlamaEngine: model=%s device=%s 4bit=%s", model_name, self.device, self.use_4bit)

    def load(self):
        """Load model and tokenizer (lazy, called on first generate)."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading tokenizer %s ...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": True,
        }

        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        log.info("Loading model %s ...", self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if not self.use_4bit and self.device != "cpu":
            self._model = self._model.to(self.device)

        self._model.eval()
        log.info("Model loaded successfully.")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        greedy: bool = False,
    ) -> str:
        """Generate a response from a list of chat messages.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            greedy: If True, use greedy decoding (deterministic).

        Returns:
            Generated text string.
        """
        self.load()

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self._model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "repetition_penalty": self.repetition_penalty,
        }

        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            temp = temperature if temperature is not None else self.temperature
            gen_kwargs["do_sample"] = temp > 0
            gen_kwargs["temperature"] = max(temp, 1e-7)
            gen_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response

    def generate_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
    ) -> tuple:
        """Generate response and return per-token log probabilities.

        Returns:
            (response_text, avg_logprob, min_logprob)
        """
        self.load()

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        new_ids = output.sequences[0, inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        if output.scores:
            import torch.nn.functional as F
            logprobs = []
            for step_idx, score in enumerate(output.scores):
                probs = F.log_softmax(score[0], dim=-1)
                token_id = new_ids[step_idx]
                logprobs.append(probs[token_id].item())
            avg_lp = sum(logprobs) / len(logprobs) if logprobs else 0.0
            min_lp = min(logprobs) if logprobs else 0.0
        else:
            avg_lp, min_lp = 0.0, 0.0

        return response, avg_lp, min_lp
