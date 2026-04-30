"""Persistent user settings — JSON file in the OS user-config dir.

Secrets live in the OS keychain via core/secrets.py — never in this file.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def settings_dir() -> Path:
    root = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(root) / "IronEngine" / "3d_creator"


def settings_path() -> Path:
    return settings_dir() / "settings.json"


def default_export_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") \
        or str(Path.home() / ".local" / "share")
    return Path(root) / "IronEngine" / "user_models"


_DEFAULTS: dict[str, Any] = {
    "llm": {
        "provider": "ollama",
        "code_mode": False,
        "think_mode": True,
        "endpoints": {
            "ollama": "http://localhost:11434",
            "lmstudio": "http://localhost:1234/v1",
        },
        "models": {
            "ollama": "llama3.1:8b",
            "lmstudio": "",
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-4o-mini",
        },
    },
    "resources": {
        "backend": "auto",
        "prefer_gpu": True,
        "ram_cap_mb": 4096,
        "vram_cap_mb": 2048,
    },
    "ui": {"theme": "cyber_neon"},
    "export": {"target_dir": None},  # None → default_export_dir() at use time
    "generation": {"default_n_points": 50000, "default_seed": 0},
}


@dataclass
class Settings:
    data: dict[str, Any] = field(default_factory=lambda: _deep_copy(_DEFAULTS))

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def set(self, *keys: str, value: Any) -> None:
        if not keys:
            raise ValueError("at least one key required")
        node = self.data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    def export_target_dir(self) -> Path:
        v = self.get("export", "target_dir")
        return Path(os.path.expandvars(v)) if v else default_export_dir()

    def save(self, path: Path | None = None) -> None:
        path = path or settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or settings_path()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _log.exception("failed to read settings; using defaults")
            return cls()
        merged = _deep_merge(_deep_copy(_DEFAULTS), data)
        return cls(data=merged)


def _deep_copy(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _deep_copy(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_deep_copy(v) for v in d]
    return d


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base
