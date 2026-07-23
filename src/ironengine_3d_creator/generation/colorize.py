"""Initial color assignment for the unified point cloud.

A simple model: each primitive starts at the spec's base color (or a
deterministic per-shape default), then we add a soft per-point shading variation
based on the local surface direction so the cloud doesn't look flat in the
viewport.
"""
from __future__ import annotations

import numpy as np


_SHAPE_DEFAULT = {
    "chair": (0.55, 0.40, 0.32),
    "table": (0.50, 0.35, 0.25),
    "vase": (0.30, 0.55, 0.65),
    "lamp": (0.85, 0.80, 0.60),
    "creature": (0.65, 0.55, 0.35),
    "tree": (0.30, 0.50, 0.25),
    "rock": (0.45, 0.45, 0.42),
    "vehicle": (0.30, 0.40, 0.50),
    "abstract": (0.40, 0.70, 1.00),
}


def base_color(shape: str, override: tuple[float, float, float] | None) -> np.ndarray:
    if override is not None:
        return np.asarray(override, dtype=np.float32)
    return np.asarray(_SHAPE_DEFAULT.get(shape, (0.6, 0.6, 0.65)), dtype=np.float32)


def albedo_colors(
    positions: np.ndarray,
    base: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unbaked albedo variation — per-point noise only, no lighting terms.

    Exported assets must carry raw albedo (W8): downstream renderers apply
    their own lighting, so any baked Lambert term would double-shade. The
    interactive viewport shades with its own GL lighting, so unbaked colors
    still look lit in preview.
    """
    if positions.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    noise = rng.uniform(0.94, 1.06, positions.shape[0])
    return np.clip(base[None, :] * noise[:, None], 0.0, 1.0).astype(np.float32)


def shaded_colors(
    positions: np.ndarray,
    base: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Preview-only faux shading. Do NOT export these colors (W8) — they bake
    a fixed directional light into albedo and double-shade downstream."""
    if positions.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    centroid = positions.mean(axis=0)
    radial = positions - centroid
    rnorm = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9
    n = radial / rnorm
    # Faux directional light from +Y +Z.
    light = np.asarray([0.0, 0.7, 0.7], dtype=np.float32)
    light /= np.linalg.norm(light)
    intensity = 0.65 + 0.35 * np.clip(n @ light, 0.0, 1.0)
    noise = rng.uniform(0.92, 1.08, positions.shape[0])
    factor = (intensity * noise)[:, None]
    return np.clip(base[None, :] * factor, 0.0, 1.0).astype(np.float32)
