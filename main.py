#!/usr/bin/env python3
"""Nexus-3 interactive agent.

Usage:
    python main.py                    # Interactive mode
    python main.py --no-llm           # Memory/retrieval only (no GPU needed)
    python main.py --model <path>     # Custom model path
    python main.py --device cpu       # Force CPU
"""

import argparse
import logging
import sys

from colorama import init as colorama_init, Fore, Style

from config import Nexus3Config
from agent import Nexus3Agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def interactive_loop(agent: Nexus3Agent):
    """Run the interactive chat loop."""
    colorama_init()

    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  NEXUS-3 Agent — Narrative Memory + Bridge-Guided Retrieval{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"  Memory slots: {len(agent.memory)}")
    print(f"  Model: {agent.config.model_name}")
    print(f"  Type 'quit' to exit, 'status' for memory stats.")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

    while True:
        try:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "status":
            print(f"  Memory: {len(agent.memory)} entries")
            print(f"  History: {len(agent.history)} messages")
            continue
        if user_input.lower() == "clear":
            agent.reset()
            print("  Memory and history cleared.")
            continue

        response = agent.interact(user_input)
        print(f"{Fore.BLUE}Nexus-3: {Style.RESET_ALL}{response}\n")


def main():
    parser = argparse.ArgumentParser(description="Nexus-3 Interactive Agent")
    parser.add_argument("--model", type=str, default=None, help="Model name or path")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM loading (memory-only mode)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    config = Nexus3Config(device=args.device)
    if args.model:
        config.model_name = args.model
    if args.no_4bit:
        config.use_4bit = False

    agent = Nexus3Agent(config=config, load_llm=not args.no_llm)
    interactive_loop(agent)


if __name__ == "__main__":
    main()
