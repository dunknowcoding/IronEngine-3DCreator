"""Procedural surface textures applied per-primitive.

Each primitive in a `GenerationSpec` may carry a `material` param picked by
the LLM (or inferred from the shape/label). At sampling time we look up the
material's texture function and modulate the per-point colour with the
result. This gives richer-looking point clouds without needing UVs or any
shader changes.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


_MaterialFn = Callable[[np.ndarray, tuple, np.random.Generator], np.ndarray]


def _grad_noise(positions: np.ndarray, freq: float, rng: np.random.Generator) -> np.ndarray:
    """Cheap value-noise via random projection — fine for visual variation."""
    seeds = rng.uniform(-1.0, 1.0, (4, 3)).astype(np.float32)
    a = np.sin(positions @ seeds[0] * freq)
    b = np.sin(positions @ seeds[1] * freq * 1.7)
    c = np.sin(positions @ seeds[2] * freq * 2.3)
    d = np.sin(positions @ seeds[3] * freq * 0.5)
    return ((a + b * 0.5 + c * 0.25 + d * 0.5) * 0.25 + 0.5).astype(np.float32)  # in [0, 1]


def wood(positions: np.ndarray, base: tuple, rng: np.random.Generator) -> np.ndarray:
    """Concentric ring grain along the local Y axis with tangential streaks."""
    rad = np.sqrt(positions[:, 0] ** 2 + positions[:, 2] ** 2)
    rings = (np.sin(rad * 24.0) * 0.5 + 0.5)
    streaks = _grad_noise(positions * 0.2, 8.0, rng)
    factor = 0.7 + 0.45 * rings * streaks
    light = (0.55, 0.40, 0.27)
    dark = (0.30, 0.18, 0.10)
    base = np.asarray(base, dtype=np.float32)
    blend = factor[:, None]
    out = (1 - blend) * np.asarray(dark) + blend * np.asarray(light)
    return np.clip(out * 0.5 + base * 0.5, 0, 1).astype(np.float32)


def stone(positions: np.ndarray, base: tuple, rng: np.random.Generator) -> np.ndarray:
    """Speckled granite-like with veins."""
    speckle = _grad_noise(positions, 30.0, rng)
    vein = (np.sin(positions[:, 0] * 4.0 + positions[:, 2] * 6.0) > 0.7).astype(np.float32)
    base = np.asarray(base, dtype=np.float32)
    out = base[None, :] * (0.65 + 0.35 * speckle[:, None])
    out -= 0.18 * vein[:, None]
    return np.clip(out, 0, 1).astype(np.float32)


def fabric(positions: np.ndarray, base: tuple, rng: np.random.Generator) -> np.ndarray:
    """Cross-hatched weave + soft per-thread variation."""
    warp = (np.sin(positions[:, 0] * 90.0) * 0.5 + 0.5)
    weft = (np.sin(positions[:, 2] * 90.0) * 0.5 + 0.5)
    weave = warp * weft
    noise = _grad_noise(positions, 25.0, rng) * 0.15
    base = np.asarray(base, dtype=np.float32)
    out = base[None, :] * (0.85 + 0.20 * weave[:, None] + noise[:, None])
    return np.clip(out, 0, 1).astype(np.float32)


def metal(positions: np.ndarray, base: tuple, rng: np.random.Generator) -> np.ndarray:
    """Brushed-metal anisotropic streaks along Y."""
    streak = (np.sin(positions[:, 1] * 220.0) * 0.5 + 0.5)
    spec = _grad_noise(positions, 5.0, rng)
    base = np.asarray(base, dtype=np.float32)
    sheen = 0.4 + 0.6 * streak * spec
    out = base[None, :] * sheen[:, None] + 0.25 * np.array([0.85, 0.85, 0.95]) * sheen[:, None]
    return np.clip(out, 0, 1).astype(np.float32)


def leather(positions: np.ndarray, base: tuple, rng: np.random.Generator) -> np.ndarray:
    """Cell-like pebbled surface."""
    cells = _grad_noise(positions, 60.0, rng)
    cracks = (cells > 0.85).astype(np.float32)
    base = np.asarray(base, dtype=np.float32)
    out = base[None, :] * (0.85 + 0.20 * cells[:, None]) - 0.18 * cracks[:, None]
    return np.clip(out, 0, 1).astype(np.float32)


def ceramic(positions: np.ndarray, base: tuple, rng: np.random.Generator) -> np.ndarray:
    """Smooth glaze with subtle highlights."""
    micro = _grad_noise(positions, 80.0, rng)
    base = np.asarray(base, dtype=np.float32)
    out = base[None, :] * (0.92 + 0.10 * micro[:, None]) + 0.08
    return np.clip(out, 0, 1).astype(np.float32)


def organic(positions: np.ndarray, base: tuple, rng: np.random.Generator) -> np.ndarray:
    """Skin/bark — large-scale tonal variation + small bumps."""
    big = _grad_noise(positions, 3.0, rng)
    small = _grad_noise(positions, 18.0, rng)
    base = np.asarray(base, dtype=np.float32)
    factor = 0.7 + 0.30 * (0.6 * big + 0.4 * small)
    out = base[None, :] * factor[:, None]
    return np.clip(out, 0, 1).astype(np.float32)


MATERIALS: dict[str, _MaterialFn] = {
    "wood": wood,
    "stone": stone,
    "fabric": fabric,
    "metal": metal,
    "leather": leather,
    "ceramic": ceramic,
    "organic": organic,
}


def apply_texture(
    positions: np.ndarray,
    base_color: tuple,
    material: str | None,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Return per-point colors for the requested material, or None if unknown."""
    if not material:
        return None
    fn = MATERIALS.get(material.lower())
    if fn is None:
        return None
    return fn(positions, base_color, rng)


def shape_default_material(shape: str, primitive_label: str | None) -> str | None:
    """Heuristic default when the LLM didn't pick one."""
    label = (primitive_label or "").lower()
    if any(t in label for t in ("seat", "back", "cushion", "fabric")):
        return "fabric"
    if any(t in label for t in ("leg", "frame", "rod", "axle", "spring")):
        return "metal"
    if any(t in label for t in ("trunk", "branch", "bark")):
        return "wood"
    s = shape.lower()
    return {
        "chair": "wood", "table": "wood", "vase": "ceramic",
        "lamp": "metal", "creature": "organic", "tree": "wood",
        "rock": "stone", "vehicle": "metal", "abstract": None,
    }.get(s)
