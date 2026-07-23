"""Non-Qt tests for the provider-registry → UI mapping.

The LLM config panel must be driven by `llm.registry` constants so new cloud
providers (DeepSeek today, others later) work without touching the panel.
These tests pin the registry helpers the panel consumes, plus a source-level
guard that the panel actually uses them (no PySide6 import — pure logic).
"""
from __future__ import annotations

from pathlib import Path

import ironengine_3d_creator
from ironengine_3d_creator.llm.registry import (
    CLOUD_PROVIDERS,
    PROVIDERS,
    credential_env_var,
    credential_hint,
    default_endpoint,
)


class TestCloudProviderRegistry:
    def test_deepseek_is_a_cloud_provider(self):
        assert "deepseek" in CLOUD_PROVIDERS
        assert set(CLOUD_PROVIDERS) <= set(PROVIDERS)

    def test_default_endpoints(self):
        # MiniMax international platform (China keys override in the panel).
        assert default_endpoint("minimax") == "https://api.minimax.io/v1"
        assert default_endpoint("deepseek") == "https://api.deepseek.com"
        assert default_endpoint("ollama") == "http://localhost:11434"
        assert default_endpoint("lmstudio") == "http://localhost:1234/v1"
        # SDK-native defaults: the panel shows an empty endpoint.
        assert default_endpoint("anthropic") == ""
        assert default_endpoint("openai") == ""

    def test_credential_env_vars(self):
        assert credential_env_var("deepseek") == "DEEPSEEK_API_KEY"
        assert credential_env_var("minimax") == "MINIMAX_API_KEY"
        assert credential_env_var("openai") == "OPENAI_API_KEY"
        assert credential_env_var("anthropic") == "ANTHROPIC_API_KEY"
        # Local providers need no key.
        assert credential_env_var("ollama") == ""
        assert credential_env_var("lmstudio") == ""

    def test_credential_hint_describes_resolution_chain(self):
        hint = credential_hint("deepseek")
        assert "DEEPSEEK_API_KEY" in hint
        assert "keychain" in hint
        assert "Credential Manager" in hint  # legacy entries mentioned
        assert credential_hint("ollama") == "no API key needed for a local server"
        # Every cloud provider gets a non-empty, env-var-led hint.
        for name in CLOUD_PROVIDERS:
            assert credential_env_var(name) in credential_hint(name)


class TestPanelUsesRegistry:
    """Source-level guard: the panel must not drift back to hardcoded lists."""

    @staticmethod
    def _panel_source() -> str:
        root = Path(ironengine_3d_creator.__file__).parent
        return (root / "ui" / "panels" / "llm_config_panel.py").read_text(
            encoding="utf-8"
        )

    def test_panel_consumes_registry_constants(self):
        src = self._panel_source()
        assert "CLOUD_PROVIDERS" in src
        assert "default_endpoint" in src
        assert "credential_hint" in src

    def test_panel_has_no_hardcoded_cloud_list_or_endpoints(self):
        src = self._panel_source()
        assert '("anthropic", "openai", "minimax")' not in src
        assert "api.minimaxi.com" not in src
        assert "api.minimax.io" not in src  # lives in llm/minimax.py only
