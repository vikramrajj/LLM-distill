#!/usr/bin/env python3
"""CLI interface for the local agentic LLM."""

import sys
import json
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.loop import Agent
from llm.client import LLMClient, LLMConfig
from tools.builtin import create_default_registry
from tools.browser import (
    browser_navigate, browser_snapshot, browser_click,
    browser_type, browser_screenshot, browser_list_elements, browser_close,
)
from tools.base import ToolDef


def main():
    parser = argparse.ArgumentParser(description="Local Agentic LLM")
    parser.add_argument("task", nargs="?", help="Task to perform")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1", help="LLM server URL")
    parser.add_argument("--model", default="phi-3.5-mini-instruct-Q4_K_M.gguf", help="Model name")
    parser.add_argument("--max-steps", type=int, default=8, help="Max agent steps")
    parser.add_argument("--browser", action="store_true", help="Enable browser tools")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--trace", action="store_true", help="Save execution trace to JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Set up LLM
    config = LLMConfig(base_url=args.url, model=args.model)
    llm = LLMClient(config)

    if not llm.is_alive():
        print("❌ LLM server not running!")
        print(f"   Expected at: {args.url}")
        print("   Start it with: ./scripts/start_llm.sh start")
        sys.exit(1)

    # Set up tools
    tools = create_default_registry()

    if args.browser:
        tools.register(ToolDef(
            name="browser_navigate",
            description="Navigate browser to a URL.",
            parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            func=browser_navigate,
        ))
        tools.register(ToolDef(
            name="browser_snapshot",
            description="Get text content of the current page.",
            parameters={"type": "object", "properties": {"max_chars": {"type": "integer"}}, "required": []},
            func=browser_snapshot,
        ))
        tools.register(ToolDef(
            name="browser_click",
            description="Click an element by CSS selector.",
            parameters={"type": "object", "properties": {"selector": {"type": "string"}}, "required": ["selector"]},
            func=browser_click,
        ))
        tools.register(ToolDef(
            name="browser_type",
            description="Type text into an element.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                    "press_enter": {"type": "boolean"},
                },
                "required": ["selector", "text"],
            },
            func=browser_type,
        ))
        tools.register(ToolDef(
            name="browser_list",
            description="List interactive elements on the page.",
            parameters={"type": "object", "properties": {}, "required": []},
            func=browser_list_elements,
        ))
        tools.register(ToolDef(
            name="browser_screenshot",
            description="Take a screenshot.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": []},
            func=browser_screenshot,
        ))
        tools.register(ToolDef(
            name="browser_close",
            description="Close the browser.",
            parameters={"type": "object", "properties": {}, "required": []},
            func=browser_close,
        ))

    # Create agent
    agent = Agent(
        llm=llm,
        tools=tools,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )

    if args.interactive:
        print("🤖 Local Agentic LLM (Phi-3.5-mini)")
        print("   Type 'quit' to exit, 'tools' to list available tools")
        print()

        while True:
            try:
                task = input("You: ").strip()
                if not task:
                    continue
                if task.lower() in ("quit", "exit", "q"):
                    break
                if task.lower() == "tools":
                    print("Available tools:", ", ".join(tools.tool_names))
                    continue

                trace = agent.run(task)

                if args.trace:
                    trace_file = Path(f"trace_{len(trace.steps)}steps.json")
                    trace_file.write_text(json.dumps(trace.to_dict(), indent=2))
                    print(f"\n📁 Trace saved to {trace_file}")

            except KeyboardInterrupt:
                print("\n\nBye!")
                break
    elif args.task:
        trace = agent.run(args.task)

        if args.trace:
            trace_file = Path(f"trace_{len(trace.steps)}steps.json")
            trace_file.write_text(json.dumps(trace.to_dict(), indent=2))
            print(f"\n📁 Trace saved to {trace_file}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
