"""OpenAI cloud provider via the official `openai` SDK."""
from __future__ import annotations

from typing import Iterator

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str, endpoint: str | None = None, api_key: str | None = None) -> None:
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

    def stream(self, system: str, user: str) -> Iterator[str]:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
            temperature=0.4,
        )
        for chunk in completion:
            for choice in chunk.choices:
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    yield content
