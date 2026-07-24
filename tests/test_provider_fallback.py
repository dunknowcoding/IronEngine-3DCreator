"""Tests for the provider fallback chain (MiniMax primary → DeepSeek fallback).

Covers, with fully mocked providers (no network):
- chain configuration: default order, normalize/reorder/disable, status map
- failure classification: auth / timeout / rate limit / connection / server
- pipeline fallback on auth-failure, timeout, rate-limit, and invalid-spec
- NO fallback when the primary succeeds (secondary never called)
- chain exhausted → deterministic style engine, offline path unchanged

The real-API class at the bottom is opt-in (IRONENGINE_REAL_API=1): it
discovers the live DeepSeek model ids, forces MiniMax to fail via a bad
endpoint override, and runs one real spec generation through the fallback
path, writing evidence to the e2e proof directory. It reads keys only via
core.secrets.get_api_key and skips (never fails) without credentials.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request
from pathlib import Path

import pytest

from ironengine_3d_creator.core.pipeline import PipelineRequest, run
from ironengine_3d_creator.generation.style_engine import STYLE_FAMILIES
from ironengine_3d_creator.llm.chain import (
    ChainLink,
    ProviderChain,
    classify_failure,
    generate_spec_with_fallback,
)
from ironengine_3d_creator.llm.registry import (
    DEFAULT_CHAIN,
    default_chain_config,
    chain_status,
    make_provider,
    normalize_chain_config,
)
from ironengine_3d_creator.llm.repair import make_spec_validator

_SMALL = 6_000  # keep the compositor fast in tests

_GOOD_CHAIR = """{
  "shape": "chair", "n_points": 5000, "bbox_size": [0.5, 0.9, 0.5],
  "primitives": [
    {"kind": "box", "label": "seat", "params": {"size": [0.45, 0.04, 0.45]},
     "transform": [[1,0,0,0],[0,1,0,0.45],[0,0,1,0],[0,0,0,1]]},
    {"kind": "cylinder", "label": "leg_0", "params": {"radius": 0.03, "height": 0.45},
     "transform": [[1,0,0,-0.2],[0,1,0,0.225],[0,0,1,-0.2],[0,0,0,1]]},
    {"kind": "cylinder", "label": "leg_1", "params": {"radius": 0.03, "height": 0.45},
     "transform": [[1,0,0,0.2],[0,1,0,0.225],[0,0,1,-0.2],[0,0,0,1]]},
    {"kind": "cylinder", "label": "leg_2", "params": {"radius": 0.03, "height": 0.45},
     "transform": [[1,0,0,-0.2],[0,1,0,0.225],[0,0,1,0.2],[0,0,0,1]]},
    {"kind": "cylinder", "label": "leg_3", "params": {"radius": 0.03, "height": 0.45},
     "transform": [[1,0,0,0.2],[0,1,0,0.225],[0,0,1,0.2],[0,0,0,1]]}
  ]
}"""


# ---------------------------------------------------------------------- fakes
class ScriptedProvider:
    """Serves canned answers in order; repeats the last one when exhausted."""

    def __init__(self, answers: list[str], name: str = "scripted") -> None:
        self.name = name
        self._answers = list(answers)
        self._last = ""
        self.calls: list[tuple[str, str]] = []

    def stream(self, system, user, stop_event=None):
        self.calls.append((system, user))
        if self._answers:
            self._last = self._answers.pop(0)
        yield self._last


class FailingProvider:
    """Always raises the given exception when streamed."""

    def __init__(self, exc: BaseException, name: str = "failing") -> None:
        self.name = name
        self.exc = exc
        self.calls = 0

    def stream(self, system, user, stop_event=None):
        self.calls += 1
        raise self.exc
        yield  # pragma: no cover - keeps this a generator


def _http_error(status: int) -> RuntimeError:
    e = RuntimeError(f"Error code: {status}")
    e.status_code = status
    return e


def _req(**kw) -> PipelineRequest:
    kw.setdefault("user_prompt", "a wooden chair")
    kw.setdefault("n_points", _SMALL)
    kw.setdefault("seed", 7)
    return PipelineRequest(**kw)


def _chain(*links) -> ProviderChain:
    return ProviderChain(list(links))


# ------------------------------------------------------------- chain config
class TestChainConfig:
    def test_default_order_is_minimax_then_deepseek(self):
        assert DEFAULT_CHAIN == ("minimax", "deepseek")
        cfg = default_chain_config()
        assert [e["name"] for e in cfg] == ["minimax", "deepseek"]
        assert all(e["enabled"] for e in cfg)

    def test_normalize_none_yields_default(self):
        assert normalize_chain_config(None) == default_chain_config()
        assert normalize_chain_config("not-a-list") == default_chain_config()

    def test_normalize_preserves_order_and_disable(self):
        cfg = normalize_chain_config([
            {"name": "deepseek", "enabled": False},
            {"name": "minimax", "enabled": True},
        ])
        assert cfg == [
            {"name": "deepseek", "enabled": False},
            {"name": "minimax", "enabled": True},
        ]

    def test_normalize_drops_unknown_and_duplicates(self):
        cfg = normalize_chain_config([
            {"name": "bogus-llm", "enabled": True},
            "deepseek",
            {"name": "deepseek", "enabled": False},  # duplicate collapses
        ])
        names = [e["name"] for e in cfg]
        assert "bogus-llm" not in names
        assert names.count("deepseek") == 1
        # Missing DEFAULT_CHAIN members are appended so upgrades keep fallback.
        assert "minimax" in names

    def test_normalize_accepts_bare_strings(self):
        cfg = normalize_chain_config(["deepseek", "minimax"])
        assert [e["name"] for e in cfg] == ["deepseek", "minimax"]
        assert all(e["enabled"] for e in cfg)

    def test_chain_status_reports_key_resolution(self):
        status = chain_status(
            None, key_resolver=lambda n: "sk-x" if n == "deepseek" else None
        )
        assert [s["name"] for s in status] == ["minimax", "deepseek"]
        assert status[0]["key_resolved"] is False
        assert status[1]["key_resolved"] is True
        assert status[0]["endpoint"] == "https://api.minimax.io/v1"
        assert status[1]["endpoint"] == "https://api.deepseek.com"

    def test_chain_status_disabled_keeps_entry(self):
        status = chain_status(
            [{"name": "minimax", "enabled": False}],
            key_resolver=lambda n: "sk-x",
        )
        assert status[0]["enabled"] is False
        assert status[0]["key_resolved"] is True


# ------------------------------------------------------- failure taxonomy
class TestFailureClassification:
    def test_auth(self):
        assert "auth" in classify_failure(_http_error(401))
        assert "auth" in classify_failure(_http_error(403))

    def test_rate_limit(self):
        assert "rate limit" in classify_failure(_http_error(429))

    def test_timeout(self):
        assert "timeout" in classify_failure(TimeoutError("timed out"))
        assert "timeout" in classify_failure(_http_error(408))

    def test_server_error(self):
        assert "server error" in classify_failure(_http_error(502))

    def test_connection_error(self):
        assert "connection" in classify_failure(ConnectionError("refused"))

    def test_message_level_auth_without_status(self):
        assert "auth" in classify_failure(RuntimeError("401 invalid api key"))

    def test_generic(self):
        assert "ValueError" in classify_failure(ValueError("weird"))


# ------------------------------------------------------- fallback behavior
class TestPipelineFallback:
    def test_fallback_on_auth_failure(self):
        mm = FailingProvider(_http_error(401), name="minimax")
        ds = ScriptedProvider([_GOOD_CHAIR], name="deepseek")
        out = run(_req(), None, chain=_chain(("minimax", mm), ("deepseek", ds)))
        assert out.spec_source == "deepseek"
        assert len(out.spec.primitives) == 5
        assert mm.calls == 1  # primary tried once, no repair round on raise
        assert len(ds.calls) == 1  # fallback answer valid on first try
        assert any(
            "provider fallback: minimax → deepseek" in w and "auth" in w
            for w in out.warnings
        )
        assert any("spec source: deepseek (fallback chain)" in w for w in out.warnings)
        assert not any("falling back to style engine" in w for w in out.warnings)
        assert out.generation.positions.shape[0] > 0

    def test_fallback_on_timeout(self):
        mm = FailingProvider(TimeoutError("timed out"), name="minimax")
        ds = ScriptedProvider([_GOOD_CHAIR], name="deepseek")
        out = run(_req(), None, chain=_chain(("minimax", mm), ("deepseek", ds)))
        assert out.spec_source == "deepseek"
        assert any("timeout" in w for w in out.warnings)

    def test_fallback_on_rate_limit(self):
        mm = FailingProvider(_http_error(429), name="minimax")
        ds = ScriptedProvider([_GOOD_CHAIR], name="deepseek")
        out = run(_req(), None, chain=_chain(("minimax", mm), ("deepseek", ds)))
        assert out.spec_source == "deepseek"
        assert any("rate limit" in w for w in out.warnings)

    def test_fallback_on_invalid_spec_after_repair(self):
        # Primary: invalid twice (repair round exhausted). Fallback: valid.
        mm = ScriptedProvider(["garbage-1", "garbage-2"], name="minimax")
        ds = ScriptedProvider([_GOOD_CHAIR], name="deepseek")
        out = run(_req(), None, chain=_chain(("minimax", mm), ("deepseek", ds)))
        assert out.spec_source == "deepseek"
        assert len(mm.calls) == 2  # one repair round on the primary
        assert len(ds.calls) == 1
        assert any(
            "provider fallback: minimax → deepseek" in w and "invalid spec" in w
            for w in out.warnings
        )
        assert len(out.spec.primitives) == 5

    def test_no_fallback_when_primary_succeeds(self):
        mm = ScriptedProvider([_GOOD_CHAIR], name="minimax")
        ds = ScriptedProvider([_GOOD_CHAIR], name="deepseek")
        out = run(_req(), None, chain=_chain(("minimax", mm), ("deepseek", ds)))
        assert out.spec_source == "minimax"
        assert len(mm.calls) == 1
        assert len(ds.calls) == 0  # fallback provider never touched
        assert not any("provider fallback" in w for w in out.warnings)

    def test_chain_exhausted_falls_back_to_style_engine(self):
        mm = ScriptedProvider(["garbage-1", "garbage-2"], name="minimax")
        ds = ScriptedProvider(["junk-a", "junk-b"], name="deepseek")
        out = run(_req(), None, chain=_chain(("minimax", mm), ("deepseek", ds)))
        assert out.spec_source == "style_engine"
        assert out.spec.shape in STYLE_FAMILIES
        assert len(mm.calls) == 2 and len(ds.calls) == 2  # repair each, no third
        assert any("provider fallback: minimax → deepseek" in w for w in out.warnings)
        assert any("provider fallback: deepseek → style engine" in w for w in out.warnings)
        assert any("falling back to style engine" in w for w in out.warnings)
        assert out.generation.positions.shape[0] > 0

    def test_provider_chain_passed_as_provider(self):
        # A ProviderChain in the `provider` slot works the same as chain=.
        mm = FailingProvider(_http_error(401), name="minimax")
        ds = ScriptedProvider([_GOOD_CHAIR], name="deepseek")
        chain = _chain(("minimax", mm), ("deepseek", ds))
        out = run(_req(), chain)
        assert out.spec_source == "deepseek"

    def test_offline_path_unchanged(self):
        out = run(_req(user_prompt=""), provider=None)
        assert out.spec.shape in STYLE_FAMILIES
        assert out.spec_source == "style_engine"
        assert not any("provider fallback" in w for w in out.warnings)

    def test_disabled_all_links_raises_last_error(self):
        # A chain whose only provider raises propagates like a single provider.
        mm = FailingProvider(_http_error(401), name="minimax")
        with pytest.raises(RuntimeError):
            run(_req(), None, chain=_chain(("minimax", mm)))


# ------------------------------------------------- real HTTP, stub servers
class TestFallbackOverRealHTTP:
    """End-to-end over real HTTP + the real openai SDK, against localhost
    stub servers: the 'minimax' stub rejects with 401, the 'deepseek' stub
    streams a valid spec. This exercises the exact production code path —
    OpenAIProvider → SDK → HTTP → classify_failure → fallback — without
    touching the network or any real key."""

    @staticmethod
    def _serve(handler_cls):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        return srv

    def test_401_primary_falls_back_over_real_http(self):
        pytest.importorskip("openai")
        from http.server import BaseHTTPRequestHandler

        class RejectHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"invalid api key"}}')

            def log_message(self, *a):
                pass

        spec_json = _GOOD_CHAIR

        class StreamHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                payload = (
                    b'data: {"choices":[{"delta":{"content":'
                    + json.dumps(spec_json).encode()
                    + b"}}]}\n\ndata: [DONE]\n\n"
                )
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        srv_a = self._serve(RejectHandler)
        srv_b = self._serve(StreamHandler)
        try:
            mm = make_provider(
                "minimax", model="MiniMax-M3",
                endpoint=f"http://127.0.0.1:{srv_a.server_address[1]}/v1",
                api_key="sk-bad", json_mode=True,
            )
            ds = make_provider(
                "deepseek", model="deepseek-chat",
                endpoint=f"http://127.0.0.1:{srv_b.server_address[1]}",
                api_key="sk-fake", json_mode=True,
            )
            out = run(_req(), None, chain=ProviderChain([("minimax", mm), ("deepseek", ds)]))
        finally:
            srv_a.shutdown(); srv_a.server_close()
            srv_b.shutdown(); srv_b.server_close()
        assert out.spec_source == "deepseek"
        assert len(out.spec.primitives) == 5
        assert any(
            "provider fallback: minimax → deepseek" in w and "auth" in w
            for w in out.warnings
        )


class TestChainLoopUnit:
    def test_stop_event_prevents_starting_next_provider(self):
        stop = threading.Event()
        stop.set()
        mm = FailingProvider(_http_error(401), name="minimax")
        ds = ScriptedProvider([_GOOD_CHAIR], name="deepseek")
        outcome = generate_spec_with_fallback(
            [ChainLink("minimax", mm), ChainLink("deepseek", ds)],
            "sys", "usr", make_spec_validator(), stop_event=stop,
        )
        assert len(ds.calls) == 0
        assert not outcome.ok

    def test_empty_chain_raises(self):
        with pytest.raises(ValueError):
            generate_spec_with_fallback([], "s", "u", make_spec_validator())


class TestPanelChainIntegration:
    """Source-level guards: the panel must stay wired to the chain helpers."""

    @staticmethod
    def _panel_source() -> str:
        import ironengine_3d_creator
        root = Path(ironengine_3d_creator.__file__).parent
        return (root / "ui" / "panels" / "llm_config_panel.py").read_text(
            encoding="utf-8"
        )

    def test_panel_uses_chain_helpers(self):
        src = self._panel_source()
        assert "normalize_chain_config" in src
        assert "chain_status" in src
        assert "probe_endpoint" in src
        assert "build_chain" in src

    def test_panel_persists_chain_settings(self):
        src = self._panel_source()
        assert '"llm", "chain"' in src


# --------------------------------------------------------------- real API
@pytest.mark.real_api
class TestDeepSeekFallbackRealAPI:
    """ONE real end-to-end run: MiniMax forced to fail → DeepSeek fallback.

    Opt-in via IRONENGINE_REAL_API=1. Discovers the live DeepSeek model ids
    with the key resolved through core.secrets (never printed/persisted),
    picks the V4-Pro-class chat model, then runs one real spec generation
    through the pipeline fallback path with MiniMax pointed at a dead
    endpoint. Evidence is written to the e2e proof directory (override with
    IRONENGINE_E2E_PROOF_DIR). Skips — never fails — without credentials or
    on a 401.
    """

    @pytest.fixture(autouse=True)
    def _require_opt_in(self):
        if os.environ.get("IRONENGINE_REAL_API") != "1":
            pytest.skip("real-API fallback proof is opt-in (set IRONENGINE_REAL_API=1)")

    @staticmethod
    def _proof_path() -> Path:
        root = os.environ.get(
            "IRONENGINE_E2E_PROOF_DIR", r"G:\Arduino\Tiezhu\e2e_proof"
        )
        return Path(root) / "deepseek_fallback_proof.json"

    @staticmethod
    def _discover_models(key: str) -> list[str]:
        req = urllib.request.Request(
            "https://api.deepseek.com/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        return [m["id"] for m in data.get("data", [])]

    @staticmethod
    def _pick_chat_model(ids: list[str]) -> str:
        # Prefer the V4-Pro-class chat model; fall back to the standard chat id.
        lowered = {m.lower(): m for m in ids}
        for m in ids:
            low = m.lower()
            if "v4" in low and "pro" in low:
                return m
        if "deepseek-v4-pro" in lowered:
            return lowered["deepseek-v4-pro"]
        if "deepseek-chat" in lowered:
            return lowered["deepseek-chat"]
        assert ids, "no DeepSeek models discovered"
        return ids[0]

    def test_minimax_failure_falls_back_to_real_deepseek(self):
        from ironengine_3d_creator.core import secrets

        key = secrets.get_api_key("deepseek")
        if not key:
            pytest.skip("no DeepSeek key resolvable via core.secrets")

        # 1) Discover the exact live model ids with the resolved key.
        try:
            ids = self._discover_models(key)
        except Exception as e:
            status = getattr(e, "code", None)
            if status in (401, 403) or "401" in str(e):
                pytest.skip(f"DeepSeek key rejected on /models: {e}")
            raise
        model = self._pick_chat_model(ids)

        # 2) Force the primary to fail via a bad endpoint override; the
        #    fallback link is the real DeepSeek API. No secrets hardcoded.
        mm = make_provider(
            "minimax", model="MiniMax-M3",
            endpoint="https://minimax.invalid.endpoint.test/v1",
            api_key="sk-deliberately-invalid",
            json_mode=True,
        )
        ds = make_provider("deepseek", model=model, api_key=key, json_mode=True)
        chain = ProviderChain([("minimax", mm), ("deepseek", ds)])

        req = PipelineRequest(
            user_prompt="a simple wooden chair with four legs",
            n_points=4_000, seed=7,
        )
        try:
            out = run(req, None, chain=chain)
        except Exception as e:
            if "401" in str(e) or "invalid api key" in str(e).lower():
                pytest.skip(f"DeepSeek returned 401 (key rejected): {e}")
            raise

        # 3) The fallback path delivered a real DeepSeek spec.
        assert out.spec_source == "deepseek"
        assert len(out.spec.primitives) > 0
        assert any("provider fallback: minimax → deepseek" in w for w in out.warnings)
        assert out.generation.positions.shape[0] > 0

        # 4) Persist the evidence (model id used, primitive count, …).
        proof = {
            "provider_chain": ["minimax", "deepseek"],
            "minimax_forced_failure": "bad endpoint override "
                                      "(https://minimax.invalid.endpoint.test/v1)",
            "fallback_events": [w for w in out.warnings if "fallback" in w],
            "spec_source": out.spec_source,
            "deepseek_models_discovered": ids,
            "deepseek_model_used": model,
            "primitive_count": len(out.spec.primitives),
            "primitive_kinds": sorted({p.kind for p in out.spec.primitives}),
            "points_generated": int(out.generation.positions.shape[0]),
        }
        path = self._proof_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
        assert path.exists()
