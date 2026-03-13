"""Tool definitions and implementations for agentic action."""

import subprocess
import os
import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolDef:
    """Definition of a tool the agent can use."""
    name: str
    description: str
    parameters: dict  # JSON Schema for parameters
    func: Callable[..., str]


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self.tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef):
        self.tools[tool.name] = tool

    def get_tool_prompt(self) -> str:
        """Generate tool descriptions for the system prompt."""
        lines = ["You have access to the following tools. Use them by writing a tool call block."]
        lines.append("")
        lines.append("Format your tool calls exactly like this:")
        lines.append("```tool")
        lines.append("tool_name: <name>")
        lines.append("args: <json arguments>")
        lines.append("```")
        lines.append("")
        for tool in self.tools.values():
            lines.append(f"### {tool.name}")
            lines.append(f"Description: {tool.description}")
            lines.append(f"Parameters: {json.dumps(tool.parameters, indent=2)}")
            lines.append("")
        return "\n".join(lines)

    def execute(self, name: str, args: dict) -> str:
        """Execute a tool by name with given arguments."""
        if name not in self.tools:
            return f"ERROR: Unknown tool '{name}'. Available: {list(self.tools.keys())}"
        try:
            return self.tools[name].func(**args)
        except Exception as e:
            return f"ERROR executing {name}: {e}"

    @property
    def tool_names(self) -> list[str]:
        return list(self.tools.keys())
