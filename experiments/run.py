#!/usr/bin/env python3
"""Run a single training experiment — autoresearch style.

This is the file the agent would modify to improve itself.
For now, it runs a benchmark of the agent on test tasks.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.runner import run_single_task, score_trace
from agent.loop import Agent
from llm.client import LLMClient, LLMConfig


# === CONFIGURATION ===
# The agent can modify these values to improve performance
MAX_STEPS = 6          # Max reasoning steps
TEMPERATURE = 0.7      # LLM temperature
MODEL = "phi-3.5-mini-instruct-Q4_K_M.gguf"
URL = "http://127.0.0.1:8080/v1"
# =====================


BENCHMARK_TASKS = [
    "List all Python files in the current directory",
    "Read README.md and tell me the project name",
    "Count files in the agent/ directory",
]


def main():
    print("🧪 Running benchmark experiment...")
    print(f"   Config: steps={MAX_STEPS}, temp={TEMPERATURE}")

    config = LLMConfig(base_url=URL, model=MODEL, temperature=TEMPERATURE)
    llm = LLMClient(config)

    if not llm.is_alive():
        print("❌ LLM server not running!")
        sys.exit(1)

    agent = Agent(llm=llm, max_steps=MAX_STEPS, verbose=False)

    results = []
    total_time = time.time()

    for i, task in enumerate(BENCHMARK_TASKS):
        print(f"\n[{i+1}/{len(BENCHMARK_TASKS)}] {task}")
        result = run_single_task(agent, task, verbose=False)
        results.append(result)

        status = "✅" if result.trace.success else "❌"
        print(f"   {status} reward={result.reward:.2f} steps={len(result.trace.steps)} ({result.duration:.1f}s)")

    total_time = time.time() - total_time

    # Metrics
    avg_reward = sum(r.reward for r in results) / len(results)
    success_rate = sum(1 for r in results if r.trace.success) / len(results)
    avg_steps = sum(len(r.trace.steps) for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"📊 Benchmark Results")
    print(f"   Success rate: {success_rate:.1%}")
    print(f"   Avg reward:   {avg_reward:.3f}")
    print(f"   Avg steps:    {avg_steps:.1f}")
    print(f"   Total time:   {total_time:.1f}s")
    print(f"{'='*50}")

    # Save results
    results_file = Path("experiment_results.jsonl")
    with open(results_file, "a") as f:
        f.write(json.dumps({
            "timestamp": time.time(),
            "config": {"max_steps": MAX_STEPS, "temperature": TEMPERATURE},
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "total_time": total_time,
        }) + "\n")
    print(f"\n📁 Results saved to {results_file}")


if __name__ == "__main__":
    main()
