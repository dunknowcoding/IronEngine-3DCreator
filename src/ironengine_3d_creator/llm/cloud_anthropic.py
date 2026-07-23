"""Anthropic cloud provider via the official `anthropic` SDK."""
from __future__ import annotations

import threading
from typing import Iterator, Optional

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"

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
            import anthropic  # type: ignore
        except Exception as e:
            raise ImportError(
                "anthropic SDK is required for the Anthropic provider. "
                "Install with: `conda run -n IronEngineWorld pip install anthropic`."
            ) from e
        kwargs = {"api_key": api_key} if api_key else {}
        if endpoint:
            kwargs["base_url"] = endpoint
        self._client = anthropic.Anthropic(**kwargs)
        # Accepted for signature parity with the other providers. Anthropic
        # has no JSON-grammar mode; the prompt + forgiving parser carry that.
        self.think_mode = bool(think_mode)
        self.json_mode = bool(json_mode)

    def stream(
        self,
        system: str,
        user: str,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as s:
            for chunk in s.text_stream:
                if stop_event is not None and stop_event.is_set():
                    break
                if chunk:
                    yield chunk
