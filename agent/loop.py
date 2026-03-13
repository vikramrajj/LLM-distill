"""Core agent loop with ReAct-style reasoning.

The agent follows this pattern:
1. Think: Reason about the task
2. Act: Call a tool
3. Observe: Get tool result
4. Repeat until task is complete
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from llm.client import LLMClient, LLMConfig
from tools.base import ToolRegistry
from tools.builtin import create_default_registry


# System prompt: loaded from program.md + tool definitions
AGENT_SYSTEM_PROMPT = """{program_instructions}

## Available Tools

{tool_prompt}
"""


@dataclass
class AgentStep:
    """One step in the agent's reasoning."""
    thought: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[str] = None
    final_answer: Optional[str] = None


@dataclass
class AgentTrace:
    """Full trace of an agent run."""
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "steps": [
                {
                    "thought": s.thought,
                    "tool_name": s.tool_name,
                    "tool_args": s.tool_args,
                    "tool_result": s.tool_result,
                    "final_answer": s.final_answer,
                }
                for s in self.steps
            ],
            "success": self.success,
            "error": self.error,
        }


class Agent:
    """ReAct-style agent that uses tools to complete tasks."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        tools: Optional[ToolRegistry] = None,
        max_steps: int = 10,
        verbose: bool = True,
    ):
        self.llm = llm or LLMClient()
        self.tools = tools or create_default_registry()
        self.max_steps = max_steps
        self.verbose = verbose

    def _build_system_prompt(self) -> str:
        # Load instructions from program.md (Karpathy autoresearch pattern)
        program_path = Path(__file__).parent.parent / "program.md"
        program_instructions = ""
        if program_path.exists():
            program_instructions = program_path.read_text()
        return AGENT_SYSTEM_PROMPT.format(
            program_instructions=program_instructions,
            tool_prompt=self.tools.get_tool_prompt(),
        )

    def _parse_tool_call(self, text: str) -> Optional[tuple[str, dict]]:
        """Parse a tool call block from the LLM response."""
        # Match ```tool ... ``` blocks
        pattern = r"```tool\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None

        block = match.group(1)
        name_match = re.search(r"tool_name:\s*(\S+)", block)
        args_match = re.search(r"args:\s*(\{.*?\})\s*$", block, re.DOTALL | re.MULTILINE)

        if not name_match:
            return None

        name = name_match.group(1)
        args = {}
        if args_match:
            try:
                args = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                pass

        return name, args

    def _check_final(self, text: str) -> Optional[str]:
        """Check if the response contains a final answer."""
        # Look for FINAL: marker
        pattern = r"FINAL:\s*(.*?)(?:\n\n|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def run(self, task: str) -> AgentTrace:
        """Run the agent on a task. Returns the full trace."""
        trace = AgentTrace(task=task)
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": task},
        ]

        for step_num in range(self.max_steps):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Step {step_num + 1}/{self.max_steps}")
                print(f"{'='*50}")

            # Get LLM response
            try:
                response = self.llm.chat(messages, max_tokens=512)
            except Exception as e:
                trace.error = f"LLM error: {e}"
                break

            if self.verbose:
                print(f"\nAssistant:\n{response}")

            step = AgentStep(thought=response)

            # Check for tool call FIRST (tool calls take priority over final)
            tool_call = self._parse_tool_call(response)
            if tool_call:
                name, args = tool_call
                step.tool_name = name
                step.tool_args = args

                if self.verbose:
                    print(f"\n🔧 Tool: {name}")
                    print(f"   Args: {json.dumps(args)}")

                # Execute tool
                result = self.tools.execute(name, args)
                step.tool_result = result

                if self.verbose:
                    print(f"   Result: {result[:200]}{'...' if len(result) > 200 else ''}")

                # Add to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Tool result:\n{result}\n\nDo NOT call tools again unless needed. Give your FINAL: answer now, or use one more tool.",
                })
            else:
                # No tool call - check for final answer
                final = self._check_final(response)
                if final:
                    step.final_answer = final
                    trace.steps.append(step)
                    trace.success = True
                    if self.verbose:
                        print(f"\n✅ Final answer: {final}")
                    break

                # No tool call and no final answer - ask model to be clearer
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Use a tool or provide your final answer with FINAL:",
                })

            trace.steps.append(step)

        if not trace.success and not trace.error:
            trace.error = f"Max steps ({self.max_steps}) reached without final answer"

        return trace

    def run_simple(self, task: str) -> str:
        """Run the agent and return just the final answer string."""
        trace = self.run(task)
        if trace.success and trace.steps:
            return trace.steps[-1].final_answer or "No answer produced"
        return f"Failed: {trace.error}"
