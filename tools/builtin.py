"""Built-in tools for the agent."""

import subprocess
import os
import json
from pathlib import Path

from tools.base import ToolDef, ToolRegistry


def shell_exec(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return stdout/stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        output += f"\n[exit code: {result.returncode}]"
        return output.strip()
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout}s"


def file_read(path: str) -> str:
    """Read a file and return its contents."""
    p = Path(path)
    if not p.exists():
        return f"ERROR: File not found: {path}"
    if p.stat().st_size > 50_000:
        return f"ERROR: File too large ({p.stat().st_size} bytes). Max 50KB."
    return p.read_text()


def file_write(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


def file_list(path: str = ".") -> str:
    """List files in a directory."""
    p = Path(path)
    if not p.exists():
        return f"ERROR: Path not found: {path}"
    if not p.is_dir():
        return f"ERROR: Not a directory: {path}"
    entries = []
    for item in sorted(p.iterdir()):
        type_char = "d" if item.is_dir() else "f"
        size = item.stat().st_size if item.is_file() else 0
        entries.append(f"[{type_char}] {item.name} ({size} bytes)")
    return "\n".join(entries) if entries else "(empty directory)"


def python_exec(code: str, timeout: int = 30) -> str:
    """Execute Python code and return output."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: Code timed out after {timeout}s"


def web_fetch(url: str) -> str:
    """Fetch a URL and return the text content (first 5000 chars)."""
    try:
        import httpx
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        if len(text) > 5000:
            text = text[:5000] + "\n... (truncated)"
        return text
    except Exception as e:
        return f"ERROR fetching {url}: {e}"


def create_default_registry() -> ToolRegistry:
    """Create a registry with all default tools."""
    registry = ToolRegistry()

    registry.register(ToolDef(
        name="shell",
        description="Execute a shell command. Use for system operations, file management, running scripts.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
            },
            "required": ["command"],
        },
        func=shell_exec,
    ))

    registry.register(ToolDef(
        name="read_file",
        description="Read the contents of a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
            },
            "required": ["path"],
        },
        func=file_read,
    ))

    registry.register(ToolDef(
        name="write_file",
        description="Write content to a file. Creates directories if needed.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
        func=file_write,
    ))

    registry.register(ToolDef(
        name="list_files",
        description="List files and directories.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current)"},
            },
            "required": [],
        },
        func=file_list,
    ))

    registry.register(ToolDef(
        name="python",
        description="Execute Python code. Use for calculations, data processing, analysis.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
            },
            "required": ["code"],
        },
        func=python_exec,
    ))

    registry.register(ToolDef(
        name="web_fetch",
        description="Fetch a URL and return text content. Use for reading web pages.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
            },
            "required": ["url"],
        },
        func=web_fetch,
    ))

    return registry
