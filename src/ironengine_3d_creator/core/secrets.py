"""Secret storage via the OS keychain (Windows Credential Manager on this host).

Falls back to an in-memory dict if `keyring` is missing or fails — this means
secrets won't persist across runs but the app stays usable.
"""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)
SERVICE = "IronEngine.3DCreator"

_fallback: dict[str, str] = {}

try:
    import keyring  # type: ignore
    _AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    keyring = None  # type: ignore
    _AVAILABLE = False
    _log.warning("keyring not available — API keys held in memory only")


def get_api_key(provider: str) -> str | None:
    if _AVAILABLE:
        try:
            return keyring.get_password(SERVICE, provider)
        except Exception:
            _log.exception("keyring read failed for %s", provider)
    return _fallback.get(provider)


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
    _fallback.pop(provider, None)
