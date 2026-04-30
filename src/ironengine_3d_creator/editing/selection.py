"""Screen-space picking helpers.

Given world-space points and a 4x4 view-projection matrix, pick the points
nearest a (mouse_x, mouse_y) in normalized device coordinates. We avoid building
a permanent KD-tree because point arrays change in place during edits — we
just project and run a single radius mask.
"""
from __future__ import annotations

import numpy as np


def project_points(positions: np.ndarray, mvp: np.ndarray) -> np.ndarray:
    """Return (N, 3) NDC coordinates (x, y in [-1, 1], z in [-1, 1])."""
    n = positions.shape[0]
    h = np.concatenate([positions, np.ones((n, 1), dtype=positions.dtype)], axis=1)
    clip = h @ mvp.T
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return clip[:, :3] / w


def pick_radius(
    positions: np.ndarray,
    mvp: np.ndarray,
    cursor_ndc: tuple[float, float],
    radius: float,
) -> np.ndarray:
    """Boolean mask of points whose NDC distance to the cursor is < radius."""
    if positions.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    ndc = project_points(positions, mvp)
    in_front = ndc[:, 2] < 1.0
    dx = ndc[:, 0] - cursor_ndc[0]
    dy = ndc[:, 1] - cursor_ndc[1]
    return in_front & ((dx * dx + dy * dy) < radius * radius)


def falloff(
    positions: np.ndarray,
    mvp: np.ndarray,
    cursor_ndc: tuple[float, float],
    radius: float,
) -> np.ndarray:
    """Per-point [0, 1] weight following a smoothstep from radius outward."""
    if positions.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    ndc = project_points(positions, mvp)
    dx = ndc[:, 0] - cursor_ndc[0]
    dy = ndc[:, 1] - cursor_ndc[1]
    d = np.sqrt(dx * dx + dy * dy)
    t = np.clip(1.0 - d / max(radius, 1e-6), 0.0, 1.0)
    in_front = (ndc[:, 2] < 1.0).astype(np.float32)
    return (t * t * (3 - 2 * t)).astype(np.float32) * in_front
