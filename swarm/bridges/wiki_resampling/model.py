"""Model client and strict JSON response parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse


class ModelClient(Protocol):
    """Small injection boundary used by the runner and offline tests."""

    def generate(self, prompt: str, *, seed: int) -> str: ...


@dataclass(frozen=True)
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 256
    timeout: float = 120.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Ollama base_url must use http or https")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama base_url must resolve to the local machine")

    def generate(self, prompt: str, *, seed: int) -> str:
        try:
            import httpx
        except ImportError as err:
            raise ImportError("httpx is required for the Ollama client") from err

        url = f"{self.base_url.rstrip('/')}/api/chat"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "seed": seed,
                    },
                },
            )
            response.raise_for_status()
            return str(response.json()["message"]["content"])


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object, tolerating a fenced model response."""

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as err:
        match = re.search(r"\{[\s\S]*\}", candidate)
        if match is None:
            raise ValueError("model response did not contain a JSON object") from err
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value
