"""DeepSeek provider — OpenAI-compatible chat completions API.

DeepSeek exposes an OpenAI-style `/chat/completions` endpoint, so this is a
thin subclass of OpenAIProvider with the DeepSeek defaults baked in. The API
key resolves through `core.secrets` (env `DEEPSEEK_API_KEY`, the OS keychain,
or legacy Credential Manager entries predating this app — see
`core.secrets._LEGACY_TARGETS`).
"""
from __future__ import annotations

from .cloud_openai import OpenAIProvider

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekProvider(OpenAIProvider):
    name = "deepseek"

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
