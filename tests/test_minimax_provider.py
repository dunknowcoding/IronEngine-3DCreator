"""Tests for the MiniMax provider and the secrets resolution chain.

No real HTTP, SDK, or OS keychain access: the `openai` module and `keyring`
are fakes injected via monkeypatch. The user's real MiniMax key in Windows
Credential Manager is never read by these tests.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest

from ironengine_3d_creator.core import secrets
from ironengine_3d_creator.llm import known_models
from ironengine_3d_creator.llm.minimax import DEFAULT_BASE_URL, DEFAULT_MODEL
from ironengine_3d_creator.llm.registry import PROVIDERS, make_provider


# ---------------------------------------------------------------------- fakes
class _Delta:
    def __init__(self, content):
        self.content = content


class _Chunk:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(delta=_Delta(content))]


class FakeCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter([_Chunk('{"shape": "rock"}')])


def _install_openai(monkeypatch: pytest.MonkeyPatch) -> FakeCompletions:
    completions = FakeCompletions()
    module = types.ModuleType("openai")
    module.OpenAI = lambda **kw: types.SimpleNamespace(  # type: ignore[attr-defined]
        chat=types.SimpleNamespace(completions=completions)
    )
    monkeypatch.setitem(sys.modules, "openai", module)
    return completions


class FakeKeyring:
    """In-memory stand-in keyed by (service, username)."""

    def __init__(self, store: dict[tuple[str, str], str]) -> None:
        self.store = store
        self.reads: list[tuple[str, str]] = []

    def get_password(self, service: str, username: str) -> str | None:
        self.reads.append((service, username))
        return self.store.get((service, username))


@pytest.fixture(autouse=True)
def _clean_secrets(monkeypatch):
    """Isolate each test from the real keyring, env, and session memory."""
    monkeypatch.setattr(secrets, "_fallback", {})
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    yield


def _use_keyring(monkeypatch: pytest.MonkeyPatch, store: dict) -> FakeKeyring:
    fake = FakeKeyring(store)
    monkeypatch.setattr(secrets, "keyring", fake)
    monkeypatch.setattr(secrets, "_AVAILABLE", True)
    return fake


# ------------------------------------------------------------------ provider
class TestMiniMaxProvider:
    def test_registered(self):
        assert "minimax" in PROVIDERS

    def test_catalog_default_is_m3(self):
        models = known_models.for_provider("minimax")
        assert models[0] == "MiniMax-M3"
        assert DEFAULT_MODEL == "MiniMax-M3"

    def test_defaults_applied(self, monkeypatch):
        _install_openai(monkeypatch)
        p = make_provider("minimax", model="", api_key="k")
        from ironengine_3d_creator.llm.minimax import MiniMaxProvider
        assert isinstance(p, MiniMaxProvider)
        assert p.name == "minimax"
        assert p.model == DEFAULT_MODEL
        assert p.endpoint == DEFAULT_BASE_URL

    def test_overrides_respected(self, monkeypatch):
        _install_openai(monkeypatch)
        p = make_provider("minimax", model="MiniMax-M2",
                          endpoint="https://custom.test/v1", api_key="k")
        assert p.model == "MiniMax-M2"
        assert p.endpoint == "https://custom.test/v1"

    def test_stream_accepts_stop_event_and_think_mode(self, monkeypatch):
        completions = _install_openai(monkeypatch)
        p = make_provider("minimax", model="", api_key="k", think_mode=True)
        out = "".join(p.stream("sys", "usr", stop_event=threading.Event()))
        assert out == '{"shape": "rock"}'
        # json_mode defaults on → OpenAI-compatible JSON request.
        assert completions.calls[0]["response_format"] == {"type": "json_object"}


# ------------------------------------------------------------------ secrets
class TestMiniMaxCredentialResolution:
    def test_in_memory_wins_over_env(self, monkeypatch):
        secrets._fallback["minimax"] = "from-memory"
        monkeypatch.setenv("MINIMAX_API_KEY", "from-env")
        _use_keyring(monkeypatch, {("IronEngine.3DCreator", "minimax"): "from-keyring"})
        assert secrets.get_api_key("minimax") == "from-memory"

    def test_env_wins_over_keyring(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "from-env")
        _use_keyring(monkeypatch, {("IronEngine.3DCreator", "minimax"): "from-keyring"})
        assert secrets.get_api_key("minimax") == "from-env"

    def test_keyring_service_wins_over_legacy(self, monkeypatch):
        fake = _use_keyring(monkeypatch, {
            ("IronEngine.3DCreator", "minimax"): "from-keyring",
            ("Paperfessor", "api-key:minimax"): "from-legacy",
        })
        assert secrets.get_api_key("minimax") == "from-keyring"
        assert ("Paperfessor", "api-key:minimax") not in fake.reads

    def test_legacy_fallback_when_service_empty(self, monkeypatch):
        fake = _use_keyring(monkeypatch, {
            ("Paperfessor", "api-key:minimax"): "from-legacy",
        })
        assert secrets.get_api_key("minimax") == "from-legacy"
        # Service entry probed first, then the legacy Paperfessor entry.
        assert fake.reads == [
            ("IronEngine.3DCreator", "minimax"),
            ("Paperfessor", "api-key:minimax"),
        ]

    def test_returns_none_when_nothing_found(self, monkeypatch):
        _use_keyring(monkeypatch, {})
        assert secrets.get_api_key("minimax") is None

    def test_keyring_unavailable_falls_through(self, monkeypatch):
        monkeypatch.setattr(secrets, "_AVAILABLE", False)
        secrets._fallback["minimax"] = "from-memory"
        assert secrets.get_api_key("minimax") == "from-memory"


class TestEnvFallbacks:
    def test_openai_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        _use_keyring(monkeypatch, {})
        assert secrets.get_api_key("openai") == "sk-env"

    def test_anthropic_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        _use_keyring(monkeypatch, {})
        assert secrets.get_api_key("anthropic") == "sk-ant-env"

    def test_keyring_used_when_no_env(self, monkeypatch):
        _use_keyring(monkeypatch, {("IronEngine.3DCreator", "openai"): "sk-keyring"})
        assert secrets.get_api_key("openai") == "sk-keyring"

    def test_set_and_delete_roundtrip_via_fallback(self, monkeypatch):
        monkeypatch.setattr(secrets, "_AVAILABLE", False)
        secrets.set_api_key("minimax", "session-key")
        assert secrets.get_api_key("minimax") == "session-key"
        secrets.delete_api_key("minimax")
        assert secrets.get_api_key("minimax") is None


# --------------------------------------------------------------- real API smoke
@pytest.mark.real_api
class TestMiniMaxRealAPISmoke:
    """ONE real generation against the live MiniMax endpoint.

    This is the only test in the suite allowed to touch the network. It is
    opt-in: set IRONENGINE_REAL_API=1 to run it. It reads the real OS
    keychain via the production secrets chain (the key is never printed or
    persisted), and *skips* — never fails — on missing credentials or a
    401, so CI without the key stays green.
    """

    @pytest.fixture(autouse=True)
    def _require_opt_in(self):
        import os
        if os.environ.get("IRONENGINE_REAL_API") != "1":
            pytest.skip("real-API smoke is opt-in (set IRONENGINE_REAL_API=1)")

    def test_tiny_chair_spec_generation(self, monkeypatch):
        # Undo the file-level isolation: the whole point is the real chain.
        monkeypatch.undo()
        key = secrets.get_api_key("minimax")
        if not key:
            pytest.skip("no MiniMax key resolvable via core.secrets")
        provider = make_provider("minimax", model="", api_key=key, json_mode=True)
        try:
            raw = "".join(provider.stream(
                "You are a 3D spec generator. Return JSON only.",
                "a simple wooden chair with four legs",
            ))
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
            if status == 401 or "401" in str(e):
                pytest.skip(f"MiniMax returned 401 (key rejected): {e}")
            raise
        from ironengine_3d_creator.alignment.parser import parse_spec
        spec = parse_spec(raw)  # raises if the answer isn't a parseable spec
        assert spec is not None
