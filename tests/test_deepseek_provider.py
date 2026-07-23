"""Tests for the DeepSeek provider and its credential resolution chain.

No real HTTP, SDK, or OS keychain access: the `openai` module and `keyring`
are fakes injected via monkeypatch. No real DeepSeek key is read by these
tests.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest

from ironengine_3d_creator.core import secrets
from ironengine_3d_creator.llm import known_models
from ironengine_3d_creator.llm.deepseek import DEFAULT_BASE_URL, DEFAULT_MODEL
from ironengine_3d_creator.llm.registry import (
    CLOUD_PROVIDERS,
    PROVIDERS,
    default_endpoint,
    make_provider,
)


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
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    yield


def _use_keyring(monkeypatch: pytest.MonkeyPatch, store: dict) -> FakeKeyring:
    fake = FakeKeyring(store)
    monkeypatch.setattr(secrets, "keyring", fake)
    monkeypatch.setattr(secrets, "_AVAILABLE", True)
    return fake


# ------------------------------------------------------------------ provider
class TestDeepSeekProvider:
    def test_registered(self):
        assert "deepseek" in PROVIDERS
        assert "deepseek" in CLOUD_PROVIDERS

    def test_catalog(self):
        models = known_models.for_provider("deepseek")
        assert models[0] == "deepseek-chat"
        assert "deepseek-v4-pro" in models
        assert "deepseek-reasoner" in models

    def test_defaults_applied(self, monkeypatch):
        _install_openai(monkeypatch)
        p = make_provider("deepseek", model="", api_key="k")
        from ironengine_3d_creator.llm.deepseek import DeepSeekProvider
        assert isinstance(p, DeepSeekProvider)
        assert p.name == "deepseek"
        assert p.model == DEFAULT_MODEL == "deepseek-chat"
        assert p.endpoint == DEFAULT_BASE_URL == "https://api.deepseek.com"

    def test_default_endpoint_helper(self):
        assert default_endpoint("deepseek") == "https://api.deepseek.com"
        assert default_endpoint("minimax") == "https://api.minimax.io/v1"
        assert default_endpoint("openai") == ""

    def test_overrides_respected(self, monkeypatch):
        _install_openai(monkeypatch)
        p = make_provider("deepseek", model="deepseek-v4-pro",
                          endpoint="https://custom.test", api_key="k")
        assert p.model == "deepseek-v4-pro"
        assert p.endpoint == "https://custom.test"

    def test_stream_accepts_stop_event_and_json_mode(self, monkeypatch):
        completions = _install_openai(monkeypatch)
        p = make_provider("deepseek", model="", api_key="k", think_mode=True)
        out = "".join(p.stream("sys", "usr", stop_event=threading.Event()))
        assert out == '{"shape": "rock"}'
        assert completions.calls[0]["response_format"] == {"type": "json_object"}


# ------------------------------------------------------------------ secrets
class TestDeepSeekCredentialResolution:
    def test_env_wins_over_keyring(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
        _use_keyring(monkeypatch, {("IronEngine.3DCreator", "deepseek"): "from-keyring"})
        assert secrets.get_api_key("deepseek") == "from-env"

    def test_keyring_service_wins_over_legacy(self, monkeypatch):
        fake = _use_keyring(monkeypatch, {
            ("IronEngine.3DCreator", "deepseek"): "from-keyring",
            ("Paperfessor", "api-key:deepseek"): "from-legacy",
        })
        assert secrets.get_api_key("deepseek") == "from-keyring"
        assert ("Paperfessor", "api-key:deepseek") not in fake.reads

    def test_legacy_first_candidate(self, monkeypatch):
        fake = _use_keyring(monkeypatch, {
            ("Paperfessor", "api-key:deepseek"): "legacy-1",
            ("Paperfessor", "deepseek"): "legacy-2",
        })
        assert secrets.get_api_key("deepseek") == "legacy-1"
        # Service probed first, then legacy candidates in order; stops at hit.
        assert fake.reads == [
            ("IronEngine.3DCreator", "deepseek"),
            ("Paperfessor", "api-key:deepseek"),
        ]

    def test_legacy_second_candidate(self, monkeypatch):
        _use_keyring(monkeypatch, {("Paperfessor", "deepseek"): "legacy-2"})
        assert secrets.get_api_key("deepseek") == "legacy-2"

    def test_legacy_third_candidate(self, monkeypatch):
        fake = _use_keyring(monkeypatch, {("DeepSeek", "api-key"): "legacy-3"})
        assert secrets.get_api_key("deepseek") == "legacy-3"
        assert fake.reads == [
            ("IronEngine.3DCreator", "deepseek"),
            ("Paperfessor", "api-key:deepseek"),
            ("Paperfessor", "deepseek"),
            ("DeepSeek", "api-key"),
        ]

    def test_returns_none_when_nothing_found(self, monkeypatch):
        _use_keyring(monkeypatch, {})
        assert secrets.get_api_key("deepseek") is None
