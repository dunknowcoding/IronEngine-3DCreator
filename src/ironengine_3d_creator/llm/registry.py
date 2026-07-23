"""Provider registry — name → factory.

Cloud providers raise ImportError lazily; the UI catches it and shows an
install hint instead of crashing.
"""
from __future__ import annotations

from .base import LLMProvider
from .lmstudio import LMStudioProvider
from .ollama import OllamaProvider

PROVIDERS = ("ollama", "lmstudio", "anthropic", "openai", "minimax", "deepseek")

# Providers whose model dropdown is fed from `known_models` rather than a
# live local probe. Consumed by `ui.panels.llm_config_panel`.
CLOUD_PROVIDERS = ("anthropic", "openai", "minimax", "deepseek")


def default_endpoint(name: str) -> str:
    """Default base URL for a provider, or "" when the SDK has its own default.

    Mirrors each provider's `DEFAULT_BASE_URL`; imported lazily so the local
    providers never pay for cloud imports. Consumed by the LLM config panel.
    """
    name = name.lower()
    if name == "minimax":
        from .minimax import DEFAULT_BASE_URL
        return DEFAULT_BASE_URL
    if name == "deepseek":
        from .deepseek import DEFAULT_BASE_URL
        return DEFAULT_BASE_URL
    if name == "ollama":
        return "http://localhost:11434"
    if name == "lmstudio":
        return "http://localhost:1234/v1"
    return ""


def credential_env_var(name: str) -> str:
    """Environment variable a provider's API key resolves from.

    Returns "" for local providers (no key needed). Imported lazily from
    `core.secrets` so `llm` never hard-depends on `core` at import time.
    """
    from ..core.secrets import _ENV_VARS

    return _ENV_VARS.get(name.lower(), "")


def credential_hint(name: str) -> str:
    """Human-readable summary of where a provider's credential resolves from.

    Consumed by the LLM config panel to annotate the API-key field, and by
    tests at the non-Qt logic level.
    """
    name = name.lower()
    if name not in CLOUD_PROVIDERS:
        return "no API key needed for a local server"
    env = credential_env_var(name)
    sources = [f"env {env}"] if env else []
    sources.append(f'OS keychain ("{name}" under IronEngine.3DCreator)')
    if name in ("minimax", "deepseek"):
        sources.append("legacy Credential Manager entries")
    return "key resolves from " + " → ".join(sources)


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
        return AnthropicProvider(
            model=model, endpoint=endpoint, api_key=api_key,
            think_mode=think_mode, json_mode=json_mode,
        )
    if name == "openai":
        from .cloud_openai import OpenAIProvider
        return OpenAIProvider(
            model=model, endpoint=endpoint, api_key=api_key,
            think_mode=think_mode, json_mode=json_mode,
        )
    if name == "minimax":
        from .minimax import MiniMaxProvider
        return MiniMaxProvider(
            model=model, endpoint=endpoint, api_key=api_key,
            think_mode=think_mode, json_mode=json_mode,
        )
    if name == "deepseek":
        from .deepseek import DeepSeekProvider
        return DeepSeekProvider(
            model=model, endpoint=endpoint, api_key=api_key,
            think_mode=think_mode, json_mode=json_mode,
        )
    raise KeyError(f"unknown LLM provider: {name!r} (known: {PROVIDERS})")
