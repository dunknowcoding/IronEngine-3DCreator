"""OpenAI cloud provider via the official `openai` SDK."""
from __future__ import annotations

import threading
from typing import Iterator, Optional

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        model: str,
        endpoint: str | None = None,
        api_key: str | None = None,
        *,
        think_mode: bool = False,
        json_mode: bool = True,
    ) -> None:
        super().__init__(model=model, endpoint=endpoint, api_key=api_key)
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError(
                "openai SDK is required for the OpenAI provider. "
                "Install with: `conda run -n IronEngineWorld pip install openai`."
            ) from e
        kwargs = {"api_key": api_key} if api_key else {}
        if endpoint:
            kwargs["base_url"] = endpoint
        self._client = OpenAI(**kwargs)
        # OpenAI chat models expose no server-side chain-of-thought toggle;
        # accepted for signature parity with the local providers so the UI can
        # pass think_mode uniformly without crashing.
        self.think_mode = bool(think_mode)
        self.json_mode = bool(json_mode)

    def _create_completion(self, system: str, user: str):
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "temperature": 0.4,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception:
            if not self.json_mode:
                raise
            # Some OpenAI-compatible endpoints (MiniMax, older forks) reject
            # response_format entirely — retry once without it rather than
            # failing the generation.
            kwargs.pop("response_format", None)
            return self._client.chat.completions.create(**kwargs)

    def stream(
        self,
        system: str,
        user: str,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        completion = self._create_completion(system, user)
        for chunk in completion:
            if stop_event is not None and stop_event.is_set():
                break
            for choice in chunk.choices:
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    yield content
