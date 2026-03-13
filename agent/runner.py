#!/usr/bin/env python3
"""Autonomous experiment runner — Karpathy autoresearch style.

Run tasks autonomously, collect experiences, score them, iterate.

Usage:
    python -m agent.runner --tasks tasks.jsonl
    python -m agent.runner --interactive
"""

import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.loop import Agent, AgentTrace
from llm.client import LLMClient, LLMConfig
from tools.builtin import create_default_registry
from training.experience import Experience, ExperienceBuffer


@dataclass
class TaskResult:
    task: str
    trace: AgentTrace
    reward: float
    duration: float


def score_trace(trace: AgentTrace) -> float:
    """Score an agent trace. Returns 0.0-1.0.

    Simple heuristic scoring:
    - Task completed successfully: +0.5
    - Each successful tool use: +0.1 (max +0.3)
    - Fewer steps is better: bonus for efficiency
    """
    if not trace.success:
        # Partial credit for attempting tools
        tool_uses = sum(1 for s in trace.steps if s.tool_result and not s.tool_result.startswith("ERROR"))
        return min(tool_uses * 0.05, 0.15)

    score = 0.5  # Base for success

    # Tool usage bonus
    successful_tools = sum(
        1 for s in trace.steps
        if s.tool_result and not s.tool_result.startswith("ERROR")
    )
    score += min(successful_tools * 0.1, 0.3)

    # Efficiency bonus (fewer steps = better)
    if len(trace.steps) <= 3:
        score += 0.2
    elif len(trace.steps) <= 5:
        score += 0.1

    return min(score, 1.0)


def run_single_task(agent: Agent, task: str, verbose: bool = True) -> TaskResult:
    """Run a single task and return scored result."""
    start = time.time()
    trace = agent.run(task)
    duration = time.time() - start
    reward = score_trace(trace)

    if verbose:
        status = "✅" if trace.success else "❌"
        print(f"\n{status} Task: {task}")
        print(f"   Reward: {reward:.2f} | Steps: {len(trace.steps)} | Time: {duration:.1f}s")

    return TaskResult(task=task, trace=trace, reward=reward, duration=duration)


def run_task_file(agent: Agent, tasks_path: str, buffer: ExperienceBuffer):
    """Run tasks from a JSONL file. Each line: {"task": "..."} """
    tasks_path = Path(tasks_path)
    if not tasks_path.exists():
        print(f"Tasks file not found: {tasks_path}")
        return

    tasks = []
    with open(tasks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    print(f"🧪 Running {len(tasks)} tasks...")
    results = []

    for i, task_data in enumerate(tasks):
        task = task_data["task"]
        print(f"\n[{i+1}/{len(tasks)}] {task}")

        result = run_single_task(agent, task, verbose=False)
        results.append(result)

        # Save to experience buffer
        exp = Experience(
            task=result.task,
            steps=[{
                "thought": s.thought,
                "tool_name": s.tool_name,
                "tool_args": s.tool_args,
                "tool_result": s.tool_result,
                "final_answer": s.final_answer,
            } for s in result.trace.steps],
            success=result.trace.success,
            reward=result.reward,
            metadata={"duration": result.duration},
        )
        buffer.add(exp)

        status = "✅" if result.trace.success else "❌"
        print(f"   {status} reward={result.reward:.2f} steps={len(result.trace.steps)} ({result.duration:.1f}s)")

    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Results: {len(results)} tasks")
    successes = sum(1 for r in results if r.trace.success)
    avg_reward = sum(r.reward for r in results) / len(results) if results else 0
    avg_steps = sum(len(r.trace.steps) for r in results) / len(results) if results else 0
    print(f"   Success rate: {successes}/{len(results)} ({100*successes/len(results):.0f}%)")
    print(f"   Avg reward:   {avg_reward:.3f}")
    print(f"   Avg steps:    {avg_steps:.1f}")
    print(f"   Experiences:  {len(buffer)} total in buffer")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Agent Runner")
    parser.add_argument("--tasks", help="JSONL file with tasks")
    parser.add_argument("--task", help="Single task to run")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="phi-3.5-mini-instruct-Q4_K_M.gguf")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--experiences", default="experiences.jsonl", help="Experience buffer file")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds to run each task")

    args = parser.parse_args()

    config = LLMConfig(base_url=args.url, model=args.model)
    llm = LLMClient(config)

    if not llm.is_alive():
        print("❌ LLM server not running!")
        sys.exit(1)

    agent = Agent(llm=llm, max_steps=args.max_steps)
    buffer = ExperienceBuffer(path=args.experiences)

    if args.tasks:
        for round_num in range(args.rounds):
            if args.rounds > 1:
                print(f"\n🔄 Round {round_num + 1}/{args.rounds}")
            run_task_file(agent, args.tasks, buffer)
    elif args.task:
        for round_num in range(args.rounds):
            if args.rounds > 1:
                print(f"\n🔄 Round {round_num + 1}/{args.rounds}")
            result = run_single_task(agent, args.task)
            exp = Experience(
                task=result.task,
                steps=[{
                    "thought": s.thought,
                    "tool_name": s.tool_name,
                    "tool_args": s.tool_args,
                    "tool_result": s.tool_result,
                    "final_answer": s.final_answer,
                } for s in result.trace.steps],
                success=result.trace.success,
                reward=result.reward,
            )
            buffer.add(exp)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
