"""Shared pytest configuration for the IronEngine 3D Creator test suite."""
from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_api: tests that hit a live LLM endpoint. Opt-in via "
        "IRONENGINE_REAL_API=1; they skip (never fail) on missing keys or 401.",
    )
