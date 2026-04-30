"""Real-time edit operations on a (positions, colors) point cloud.

Operations mutate the arrays in place and return a tuple of (changed_mask,
op_record). The op_record captures enough info to (a) replay the op and
(b) reverse it via the snapshot stack in history.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .selection import falloff


@dataclass
class OpRecord:
    name: str
    params: dict[str, Any]


def brush_move(
    positions: np.ndarray,
    mvp: np.ndarray,
    cursor_ndc: tuple[float, float],
    radius: float,
    direction: np.ndarray,
    strength: float,
) -> OpRecord:
    """Move points within a screen-space radius along a world-space direction."""
    w = falloff(positions, mvp, cursor_ndc, radius)
    if not w.any():
        return OpRecord("brush_move", {"affected": 0})
    direction = np.asarray(direction, dtype=np.float32)
    positions += direction[None, :] * (strength * w)[:, None]
    return OpRecord("brush_move", {"radius": radius, "strength": strength, "affected": int((w > 0).sum())})


def radial_warp(
    positions: np.ndarray,
    mvp: np.ndarray,
    cursor_ndc: tuple[float, float],
    radius: float,
    factor: float,
) -> OpRecord:
    """Scale points within the brush radially around their local centroid."""
    w = falloff(positions, mvp, cursor_ndc, radius)
    if not w.any():
        return OpRecord("radial_warp", {"affected": 0})
    in_brush = w > 0
    sel = positions[in_brush]
    centroid = sel.mean(axis=0)
    radial = sel - centroid
    scaled = radial * (1.0 + (factor - 1.0) * w[in_brush, None])
    positions[in_brush] = centroid + scaled
    return OpRecord("radial_warp", {"radius": radius, "factor": factor, "affected": int(in_brush.sum())})


def point_paint(
    colors: np.ndarray,
    mvp: np.ndarray,
    positions: np.ndarray,
    cursor_ndc: tuple[float, float],
    radius: float,
    rgb: tuple[float, float, float],
    strength: float,
) -> OpRecord:
    w = falloff(positions, mvp, cursor_ndc, radius)
    if not w.any():
        return OpRecord("point_paint", {"affected": 0})
    rgb_arr = np.asarray(rgb, dtype=np.float32)
    weight = (strength * w)[:, None]
    colors[:] = np.clip(colors * (1 - weight) + rgb_arr[None, :] * weight, 0.0, 1.0)
    return OpRecord("point_paint", {"radius": radius, "rgb": list(rgb), "strength": strength,
                                    "affected": int((w > 0).sum())})


def smooth(
    positions: np.ndarray,
    mvp: np.ndarray,
    cursor_ndc: tuple[float, float],
    radius: float,
    strength: float,
    k: int = 8,
) -> OpRecord:
    """Move each affected point toward the mean of its k nearest neighbours."""
    w = falloff(positions, mvp, cursor_ndc, radius)
    affected = np.where(w > 0)[0]
    if affected.size == 0:
        return OpRecord("smooth", {"affected": 0})
    # Naive kNN — fine for the brush region (typically < 1000 affected points).
    pts = positions[affected]
    diffs = pts[:, None, :] - positions[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
    k = min(k, positions.shape[0])
    idx = np.argpartition(d2, k, axis=1)[:, :k]
    means = positions[idx].mean(axis=1)
    blend = (strength * w[affected])[:, None]
    positions[affected] = pts * (1 - blend) + means * blend
    return OpRecord("smooth", {"radius": radius, "strength": strength, "affected": int(affected.size)})
