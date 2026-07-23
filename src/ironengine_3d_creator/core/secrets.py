"""Secret storage via the OS keychain (Windows Credential Manager on this host).

Falls back to an in-memory dict if `keyring` is missing or fails — this means
secrets won't persist across runs but the app stays usable.

Resolution order in `get_api_key`:
  1. in-memory fallback dict (set earlier this session)
  2. provider-specific environment variable (OPENAI_API_KEY, …)
  3. keyring service "IronEngine.3DCreator", username = provider name
  4. legacy Credential Manager entries predating this app (migration shim)
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)
SERVICE = "IronEngine.3DCreator"

# Env-var fallbacks per provider, so CI and headless runs can inject
# credentials without touching the OS keychain.
_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Legacy Windows Credential Manager entries written by older tools. Checked
# last so native IronEngine.3DCreator entries always win. Each provider maps
# to an ordered tuple of (service, username) candidates — the first hit wins.
_LEGACY_TARGETS = {
    "minimax": (("Paperfessor", "api-key:minimax"),),
    "deepseek": (
        ("Paperfessor", "api-key:deepseek"),
        ("Paperfessor", "deepseek"),
        ("DeepSeek", "api-key"),
    ),
}

_fallback: dict[str, str] = {}

try:
    import keyring  # type: ignore
    _AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    keyring = None  # type: ignore
    _AVAILABLE = False
    _log.warning("keyring not available — API keys held in memory only")


def get_api_key(provider: str) -> str | None:
    if provider in _fallback:
        return _fallback[provider]
    env_var = _ENV_VARS.get(provider)
    if env_var:
        key = os.environ.get(env_var)
        if key:
            return key
    if _AVAILABLE:
        try:
            key = keyring.get_password(SERVICE, provider)
            if key:
                return key
        except Exception:
            _log.exception("keyring read failed for %s", provider)
        legacy = _LEGACY_TARGETS.get(provider)
        if legacy:
            for candidate in legacy:
                try:
                    key = keyring.get_password(*candidate)
                    if key:
                        return key
                except Exception:
                    _log.exception("legacy keyring read failed for %s via %s", provider, candidate)
    return None


def set_api_key(provider: str, key: str) -> None:
    if _AVAILABLE:
        try:
            keyring.set_password(SERVICE, provider, key)
            return
        except Exception:
            _log.exception("keyring write failed for %s", provider)
    _fallback[provider] = key


def delete_api_key(provider: str) -> None:
    if _AVAILABLE:
        try:
            keyring.delete_password(SERVICE, provider)
        except Exception:
            pass
        # NOTE: legacy entries (e.g. Paperfessor) belong to other apps and
        # are only ever read, never deleted here.
    _fallback.pop(provider, None)
