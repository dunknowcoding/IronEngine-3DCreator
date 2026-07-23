"""MiniMax provider — OpenAI-compatible chat completions API.

MiniMax exposes an OpenAI-style `/chat/completions` endpoint, so this is a
thin subclass of OpenAIProvider with the MiniMax defaults baked in. The API
key resolves through `core.secrets` (env `MINIMAX_API_KEY`, the OS keychain,
or the legacy Paperfessor Credential Manager entry).
"""
from __future__ import annotations

from .cloud_openai import OpenAIProvider

DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_MODEL = "MiniMax-M3"


class MiniMaxProvider(OpenAIProvider):
    name = "minimax"

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        *,
        think_mode: bool = False,
        json_mode: bool = True,
    ) -> None:
        super().__init__(
            model=model or DEFAULT_MODEL,
            endpoint=endpoint or DEFAULT_BASE_URL,
            api_key=api_key,
            think_mode=think_mode,
            json_mode=json_mode,
        )
