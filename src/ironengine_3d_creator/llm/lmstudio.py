"""LMStudio provider — uses the OpenAI-compatible /v1/chat/completions endpoint."""
from __future__ import annotations

import json
import threading
from typing import Iterator, Optional

import requests

from .base import LLMProvider


class LMStudioProvider(LLMProvider):
    name = "lmstudio"

    def __init__(self, model: str, endpoint: str = "http://localhost:1234/v1", api_key: str | None = None) -> None:
        super().__init__(model=model, endpoint=endpoint.rstrip("/"), api_key=api_key)

    def list_models(self) -> list[str]:
        """Probe LMStudio's OpenAI-compatible /models endpoint.

        Returns an empty list if the server isn't reachable — callers fall back
        to whatever the user typed.
        """
        try:
            url = f"{self.endpoint}/models"
            r = requests.get(url, timeout=2)
            r.raise_for_status()
            data = r.json().get("data", [])
            return [m.get("id") for m in data if m.get("id")]
        except Exception:
            return []

    def stream(
        self,
        system: str,
        user: str,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        url = f"{self.endpoint}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model or "local-model",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "temperature": 0.4,
        }
        in_think = False
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=(5, 600)) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if stop_event is not None and stop_event.is_set():
                    r.close()
                    break
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    # Reasoning models on OpenAI-compatible endpoints emit the
                    # chain of thought via `reasoning_content` (LMStudio) or
                    # `reasoning` (some forks) before the answer text.
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        if not in_think:
                            yield "<think>"
                            in_think = True
                        yield reasoning
                    content = delta.get("content")
                    if content:
                        if in_think:
                            yield "</think>"
                            in_think = False
                        yield content
            if in_think:
                yield "</think>"
