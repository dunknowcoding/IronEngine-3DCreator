"""Tests for the cloud LLM providers (OpenAI / Anthropic).

All SDK objects are fakes injected via sys.modules — no real HTTP or cloud
API is ever touched.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest


# ---------------------------------------------------------------------- fakes
class _Delta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


def _chunks(texts: list[str]):
    return iter([_Chunk(t) for t in texts])


def _infinite_chunks():
    while True:
        yield _Chunk("tok")


class FakeCompletions:
    """Records create() calls; can be told to reject response_format."""

    def __init__(self, texts: list[str] | None = None, *, reject_json: bool = False,
                 infinite: bool = False) -> None:
        self.calls: list[dict] = []
        self._texts = texts or ["hello", " world"]
        self._reject_json = reject_json
        self._infinite = infinite

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._reject_json and "response_format" in kwargs:
            raise RuntimeError("400 response_format is not supported")
        if self._infinite:
            return _infinite_chunks()
        return _chunks(self._texts)


class FakeOpenAIClient:
    def __init__(self, completions: FakeCompletions, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.chat = types.SimpleNamespace(completions=completions)


class FakeAnthropicStream:
    def __init__(self, texts: list[str], *, infinite: bool = False) -> None:
        self.text_stream = _infinite_texts() if infinite else iter(texts)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _infinite_texts():
    while True:
        yield "tok"


class FakeAnthropicMessages:
    def __init__(self, texts: list[str] | None = None, *, infinite: bool = False) -> None:
        self.calls: list[dict] = []
        self._texts = texts or ["hello", " world"]
        self._infinite = infinite

    def stream(self, **kwargs) -> FakeAnthropicStream:
        self.calls.append(kwargs)
        return FakeAnthropicStream(self._texts, infinite=self._infinite)


class FakeAnthropicClient:
    def __init__(self, messages: FakeAnthropicMessages, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.messages = messages


def _install_openai(monkeypatch: pytest.MonkeyPatch, completions: FakeCompletions) -> None:
    module = types.ModuleType("openai")
    module.OpenAI = lambda **kw: FakeOpenAIClient(completions, **kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)


def _install_anthropic(monkeypatch: pytest.MonkeyPatch, messages: FakeAnthropicMessages) -> None:
    module = types.ModuleType("anthropic")
    module.Anthropic = lambda **kw: FakeAnthropicClient(messages, **kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)


# ------------------------------------------------------------------ OpenAI
class TestOpenAIProvider:
    def test_stream_accepts_stop_event_and_yields(self, monkeypatch):
        completions = FakeCompletions(["a", "b"])
        _install_openai(monkeypatch, completions)
        from ironengine_3d_creator.llm.cloud_openai import OpenAIProvider

        p = OpenAIProvider(model="gpt-4o", api_key="sk-fake", think_mode=True, json_mode=False)
        # W1: pipeline always passes stop_event= — must not raise TypeError.
        out = "".join(p.stream("sys", "usr", stop_event=threading.Event()))
        assert out == "ab"

    def test_stream_aborts_when_stop_event_set(self, monkeypatch):
        completions = FakeCompletions(infinite=True)
        _install_openai(monkeypatch, completions)
        from ironengine_3d_creator.llm.cloud_openai import OpenAIProvider

        p = OpenAIProvider(model="gpt-4o", api_key="sk-fake", json_mode=False)
        stop = threading.Event()
        it = p.stream("sys", "usr", stop_event=stop)
        next(it)  # one token flows
        stop.set()
        # Terminates promptly despite an infinite upstream stream.
        assert list(it) == []

    def test_json_mode_requests_response_format(self, monkeypatch):
        completions = FakeCompletions(["{}"])
        _install_openai(monkeypatch, completions)
        from ironengine_3d_creator.llm.cloud_openai import OpenAIProvider

        p = OpenAIProvider(model="gpt-4o", api_key="sk-fake", json_mode=True)
        "".join(p.stream("sys", "usr"))
        assert completions.calls[0]["response_format"] == {"type": "json_object"}

    def test_json_mode_retries_without_response_format_when_rejected(self, monkeypatch):
        completions = FakeCompletions(["{}"], reject_json=True)
        _install_openai(monkeypatch, completions)
        from ironengine_3d_creator.llm.cloud_openai import OpenAIProvider

        p = OpenAIProvider(model="gpt-4o", api_key="sk-fake", json_mode=True)
        out = "".join(p.stream("sys", "usr"))
        assert out == "{}"
        assert len(completions.calls) == 2
        assert "response_format" in completions.calls[0]
        assert "response_format" not in completions.calls[1]

    def test_json_mode_disabled_sends_no_response_format(self, monkeypatch):
        completions = FakeCompletions(["x"])
        _install_openai(monkeypatch, completions)
        from ironengine_3d_creator.llm.cloud_openai import OpenAIProvider

        p = OpenAIProvider(model="gpt-4o", api_key="sk-fake", json_mode=False)
        "".join(p.stream("sys", "usr"))
        assert "response_format" not in completions.calls[0]

    def test_endpoint_sets_base_url(self, monkeypatch):
        completions = FakeCompletions(["x"])
        captured: dict = {}

        def ctor(**kw):
            captured.update(kw)
            return FakeOpenAIClient(completions, **kw)

        module = types.ModuleType("openai")
        module.OpenAI = ctor  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai", module)
        from ironengine_3d_creator.llm.cloud_openai import OpenAIProvider

        OpenAIProvider(model="m", endpoint="https://example.test/v1", api_key="k")
        assert captured["base_url"] == "https://example.test/v1"


# --------------------------------------------------------------- Anthropic
class TestAnthropicProvider:
    def test_stream_accepts_stop_event_and_yields(self, monkeypatch):
        messages = FakeAnthropicMessages(["a", "b"])
        _install_anthropic(monkeypatch, messages)
        from ironengine_3d_creator.llm.cloud_anthropic import AnthropicProvider

        p = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-fake",
                              think_mode=True, json_mode=True)
        out = "".join(p.stream("sys", "usr", stop_event=threading.Event()))
        assert out == "ab"
        assert messages.calls[0]["system"] == "sys"

    def test_stream_aborts_when_stop_event_set(self, monkeypatch):
        messages = FakeAnthropicMessages(infinite=True)
        _install_anthropic(monkeypatch, messages)
        from ironengine_3d_creator.llm.cloud_anthropic import AnthropicProvider

        p = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-fake")
        stop = threading.Event()
        it = p.stream("sys", "usr", stop_event=stop)
        next(it)
        stop.set()
        assert list(it) == []


# --------------------------------------------------------------- registry
class TestRegistry:
    def test_make_provider_passes_modes_without_crash(self, monkeypatch):
        _install_openai(monkeypatch, FakeCompletions(["x"]))
        _install_anthropic(monkeypatch, FakeAnthropicMessages(["x"]))
        from ironengine_3d_creator.llm.registry import make_provider

        p = make_provider("openai", model="gpt-4o", api_key="k",
                          think_mode=True, json_mode=True)
        assert p.think_mode is True and p.json_mode is True
        p = make_provider("anthropic", model="claude-sonnet-4-6", api_key="k",
                          think_mode=True, json_mode=False)
        assert p.think_mode is True and p.json_mode is False
