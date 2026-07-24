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

# Default provider fallback chain (primary first): MiniMax M3 primary,
# DeepSeek automatic fallback. Runtime execution lives in `llm.chain`.
DEFAULT_CHAIN = ("minimax", "deepseek")


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


def known_models_fallback(name: str) -> str | None:
    """First curated catalog model for a provider, or None when unknown.

    Used when building a provider chain and no per-provider model override
    was saved — each provider falls back to its own catalog default.
    """
    from . import known_models

    catalog = known_models.for_provider(name.lower())
    return catalog[0] if catalog else None


# --------------------------------------------------------------- chain config
def default_chain_config() -> list[dict]:
    """Default ordered fallback chain: MiniMax primary, DeepSeek fallback."""
    return [{"name": name, "enabled": True} for name in DEFAULT_CHAIN]


def normalize_chain_config(raw) -> list[dict]:
    """Coerce a stored chain config into ordered ``{name, enabled}`` dicts.

    Rules:
    - ``None`` / non-list input yields the default chain.
    - Unknown provider names are dropped; order and enabled flags survive.
    - Entries may be dicts (``{"name", "enabled"}``) or bare name strings.
    - Duplicate names are collapsed (first occurrence wins).
    - DEFAULT_CHAIN providers missing from the stored config are appended
      (enabled) so upgraded installs keep the MiniMax → DeepSeek fallback.
    """
    if not isinstance(raw, (list, tuple)):
        return default_chain_config()
    out: list[dict] = []
    seen: set[str] = set()
    for entry in raw:
        if isinstance(entry, str):
            name, enabled = entry.lower(), True
        elif isinstance(entry, dict):
            name = str(entry.get("name", "")).lower()
            enabled = bool(entry.get("enabled", True))
        else:
            continue
        if name not in PROVIDERS or name in seen:
            continue
        seen.add(name)
        out.append({"name": name, "enabled": enabled})
    for name in DEFAULT_CHAIN:
        if name not in seen:
            out.append({"name": name, "enabled": True})
    return out


def chain_status(raw=None, *, key_resolver=None) -> list[dict]:
    """Per-provider status for the chain UI (non-Qt logic, fully testable).

    Returns one dict per chain entry: name, enabled, default endpoint, and
    whether a credential resolves. `key_resolver` defaults to
    `core.secrets.get_api_key` (imported lazily); tests inject a fake.
    """
    cfg = normalize_chain_config(raw)
    if key_resolver is None:
        from ..core.secrets import get_api_key

        key_resolver = get_api_key
    out: list[dict] = []
    for entry in cfg:
        name = entry["name"]
        needs_key = name in CLOUD_PROVIDERS
        out.append({
            "name": name,
            "enabled": entry["enabled"],
            "endpoint": default_endpoint(name),
            "key_resolved": bool(key_resolver(name)) if needs_key else True,
        })
    return out


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
