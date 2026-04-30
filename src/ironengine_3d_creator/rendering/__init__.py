"""Public rendering API.

These functions are the single entry point both the UI viewport and external
callers (SceneEditor, Sim, scripts) should use. They keep the rendering
choices the user makes — render mode, color mode, point size — collected in
one place.

See `api.py` for the actual implementation.
"""
from __future__ import annotations

from .api import (
    RenderOptions,
    colorize_plain,
    render_mesh_offscreen,
    render_points_offscreen,
    reconstruct_mesh,
)

__all__ = (
    "RenderOptions",
    "colorize_plain",
    "render_mesh_offscreen",
    "render_points_offscreen",
    "reconstruct_mesh",
)
