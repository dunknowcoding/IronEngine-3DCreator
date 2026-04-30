"""Curated model lists for cloud providers.

These power the model dropdown when the user picks Anthropic or OpenAI. Local
providers (Ollama, LMStudio) populate from their live endpoints instead.
"""
from __future__ import annotations

# Newest-to-oldest within each family. The default per provider (used on
# first-run) is `[0]`.

ANTHROPIC_MODELS = (
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-1",
    "claude-sonnet-4-5",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
)

OPENAI_MODELS = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o1",
    "o1-mini",
)


def for_provider(name: str) -> tuple[str, ...]:
    if name == "anthropic":
        return ANTHROPIC_MODELS
    if name == "openai":
        return OPENAI_MODELS
    return ()
