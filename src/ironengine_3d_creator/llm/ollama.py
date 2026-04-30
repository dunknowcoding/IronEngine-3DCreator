"""Ollama provider — POST to /api/chat with stream=true."""
from __future__ import annotations

import json
import threading
from typing import Iterator, Optional

import requests

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str,
        endpoint: str = "http://localhost:11434",
        api_key: str | None = None,
        *,
        think_mode: bool = False,
        json_mode: bool = True,
    ) -> None:
        super().__init__(model=model, endpoint=endpoint.rstrip("/"), api_key=api_key)
        self.think_mode = bool(think_mode)
        # When True, ask Ollama to enforce JSON output via grammar. Set False
        # for code-mode where the model emits Python source.
        self.json_mode = bool(json_mode)

    def stream(
        self,
        system: str,
        user: str,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Stream tokens from Ollama, including thinking-mode chunks.

        Reasoning models (qwen3.5, deepseek-r1, glm reasoning, …) split their
        output into a `thinking` field (the chain of thought) and a `content`
        field (the final answer). We surface both: the thinking content is
        wrapped in `<think>…</think>` synthetic tags so the downstream
        `ThinkingFilter` routes it to the dim-italic UI lane and the JSON
        parser strips it before parsing.
        """
        url = f"{self.endpoint}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            # Toggle the model's chain-of-thought. For our JSON-spec use case
            # we keep this off by default — the model produces valid JSON
            # directly without spending tokens "thinking" first, which is much
            # faster on small reasoning models. Older Ollama versions ignore
            # the flag harmlessly.
            "think": self.think_mode,
            # Keep the model resident for 10 minutes so subsequent requests
            # don't pay the load cost again.
            "keep_alive": "10m",
            "options": {
                "temperature": 0.4,
                # Cap output length. Tiny reasoning models occasionally loop
                # under the JSON-format grammar; this prevents them from
                # generating indefinitely. ~2k tokens is more than enough for
                # any sensible GenerationSpec.
                "num_predict": 2048,
                # Penalise immediate repetition. Small models like qwen3.5:0.8b
                # otherwise echo the few-shot example primitive endlessly.
                "repeat_penalty": 1.25,
                "repeat_last_n": 256,
            },
        }
        if self.json_mode:
            # Constrain the model to emit valid JSON. Ollama enforces grammar
            # at decode time so the response is parseable even from tiny
            # models that occasionally drop the JSON structure when free-form.
            payload["format"] = "json"
        # (connect, read) timeouts: 5 s to reach the server, 600 s for the
        # actual response so first-time model loads into VRAM still complete.
        in_think = False
        with requests.post(url, json=payload, stream=True, timeout=(5, 600)) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if stop_event is not None and stop_event.is_set():
                    # Closing the response context will tear down the
                    # underlying socket and free Ollama's worker for the
                    # next request.
                    r.close()
                    break
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message") or {}
                think = msg.get("thinking")
                content = msg.get("content")
                if think:
                    if not in_think:
                        yield "<think>"
                        in_think = True
                    yield think
                if content:
                    if in_think:
                        yield "</think>"
                        in_think = False
                    yield content
                if obj.get("done"):
                    break
            if in_think:
                yield "</think>"

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.endpoint}/api/tags", timeout=2)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []
