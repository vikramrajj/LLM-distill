"""OpenAI-compatible LLM client for llama.cpp server."""

import json
import httpx
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "phi-3.5-mini-instruct-Q4_K_M.gguf"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completions API."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client = httpx.Client(timeout=120)

    def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Send chat messages, return assistant response text."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": self.config.top_p,
        }
        if stop:
            payload["stop"] = stop

        resp = self.client.post(
            f"{self.config.base_url}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def is_alive(self) -> bool:
        """Check if the server is running."""
        try:
            resp = self.client.get(f"{self.config.base_url}/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
