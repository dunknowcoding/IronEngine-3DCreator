"""Provider registry — name → factory.

Cloud providers raise ImportError lazily; the UI catches it and shows an
install hint instead of crashing.
"""
from __future__ import annotations

from .base import LLMProvider
from .lmstudio import LMStudioProvider
from .ollama import OllamaProvider

PROVIDERS = ("ollama", "lmstudio", "anthropic", "openai")


def make_provider(
    name: str,
    *,
    model: str,
    endpoint: str | None = None,
    api_key: str | None = None,
    think_mode: bool = False,
    json_mode: bool = True,
) -> LLMProvider:
    name = name.lower()
    if name == "ollama":
        return OllamaProvider(
            model=model,
            endpoint=endpoint or "http://localhost:11434",
            api_key=api_key,
            think_mode=think_mode,
            json_mode=json_mode,
        )
    if name == "lmstudio":
        return LMStudioProvider(model=model, endpoint=endpoint or "http://localhost:1234/v1", api_key=api_key)
    if name == "anthropic":
        from .cloud_anthropic import AnthropicProvider
        return AnthropicProvider(model=model, endpoint=endpoint, api_key=api_key)
    if name == "openai":
        from .cloud_openai import OpenAIProvider
        return OpenAIProvider(model=model, endpoint=endpoint, api_key=api_key)
    raise KeyError(f"unknown LLM provider: {name!r} (known: {PROVIDERS})")
