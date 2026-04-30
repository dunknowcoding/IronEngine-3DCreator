"""LLMProvider abstraction.

Every provider implements `stream(system, user)` returning an iterator of
text chunks, and `generate(system, user)` which is the joined non-streaming
form. The token-stream UI widget consumes the streaming form via a
QThread signal.

Providers accept an optional `stop_event` (threading.Event). When set, the
provider must promptly abandon any in-flight HTTP request and stop yielding.
This is how callers cancel a runaway generation without leaking a streaming
socket back to the server (which would queue up subsequent requests).
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Iterator, Optional


class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, model: str, endpoint: str | None = None, api_key: str | None = None) -> None:
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key

    @abstractmethod
    def stream(
        self,
        system: str,
        user: str,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive. Bail out promptly when stop_event is set."""

    def generate(self, system: str, user: str) -> str:
        return "".join(self.stream(system, user))

    # Diagnostic — providers may override to do a real liveness probe.
    def test(self) -> tuple[bool, str]:
        try:
            chunk_iter = self.stream("You are a probe responder.", "Reply with just OK.")
            text = "".join(chunk_iter).strip()
            return True, text or "(empty response)"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
