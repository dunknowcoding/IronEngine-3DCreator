"""Surface features applied after primitive sampling.

Each `apply_*` operates in place on a (positions, colors, labels) tuple where
positions: (N, 3) float32, colors: (N, 3) float32, labels: (N,) int (the index
of the originating primitive — used to scope features to a region).
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Region masks
# ---------------------------------------------------------------------------


def region_mask(region, labels: np.ndarray, label_lookup: dict[str, int]) -> np.ndarray:
    """Resolve a feature.region descriptor into a boolean mask.

    Supported regions:
    - "all" → every point
    - "<label>" → only points from a primitive with this label
    - {"labels": ["a", "b"]} → union of those labels
    - {"axis": "y", "min": ..., "max": ...} → world-space slab
    """
    if region == "all" or region is None:
        return np.ones(labels.shape, dtype=bool)
    if isinstance(region, str):
        idx = label_lookup.get(region)
        if idx is None:
            return np.zeros(labels.shape, dtype=bool)
        return labels == idx
    if isinstance(region, dict):
        if "labels" in region:
            mask = np.zeros(labels.shape, dtype=bool)
            for name in region["labels"]:
                idx = label_lookup.get(name)
                if idx is not None:
                    mask |= labels == idx
            return mask
    return np.ones(labels.shape, dtype=bool)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def apply_scratch(
    positions: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> None:
    count = int(params.get("count", 6))
    depth = float(params.get("depth", 0.005))
    if count <= 0 or not mask.any():
        return
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return
    pmin, pmax = positions[idxs].min(0), positions[idxs].max(0)
    extents = pmax - pmin + 1e-6
    for _ in range(count):
        # Pick a random line through the region and carve a thin band.
        a = rng.uniform(pmin, pmax)
        b = rng.uniform(pmin, pmax)
        d = b - a
        d /= np.linalg.norm(d) + 1e-9
        # Project all points onto the line; carve points within `band` of the line.
        proj = (positions[idxs] - a) @ d
        closest = a + proj[:, None] * d
        offset = positions[idxs] - closest
        radial = np.linalg.norm(offset, axis=1)
        band = float(extents.max()) * 0.01
        scratch_mask = radial < band
        # Push points inward along their displacement from the line.
        shrink = (1.0 - depth)
        positions[idxs[scratch_mask]] = closest[scratch_mask] + offset[scratch_mask] * shrink
        # Slightly darken the scratched points.
        colors[idxs[scratch_mask]] *= 0.7


def apply_curve_pattern(
    positions: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> None:
    freq = float(params.get("frequency", 4.0))
    amp = float(params.get("amplitude", 0.01))
    if not mask.any():
        return
    idxs = np.where(mask)[0]
    pts = positions[idxs]
    centroid = pts.mean(axis=0)
    radial = pts - centroid
    rnorm = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9
    n = radial / rnorm
    # Sinusoidal radial perturbation along Y → "ribbed" / wavy bands.
    phase = freq * pts[:, 1]
    positions[idxs] = pts + n * (amp * np.sin(phase))[:, None]


def apply_bump_field(
    positions: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> None:
    count = int(params.get("count", 30))
    radius = float(params.get("radius", 0.04))
    height = float(params.get("height", radius * 0.6))
    if count <= 0 or not mask.any():
        return
    idxs = np.where(mask)[0]
    pts = positions[idxs]
    centroid = pts.mean(axis=0)
    radial = pts - centroid
    rnorm = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9
    nrm = radial / rnorm
    # Pick `count` random "bump centers" from the surface itself, then push points within radius outward.
    centers = pts[rng.choice(pts.shape[0], size=count, replace=False)]
    for c in centers:
        d = np.linalg.norm(pts - c, axis=1)
        within = d < radius
        if not within.any():
            continue
        falloff = (1.0 - d[within] / radius) ** 2
        pts[within] = pts[within] + nrm[within] * (height * falloff[:, None])
    positions[idxs] = pts


def apply_dent(
    positions: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> None:
    # Dent = inverse of a single bump.
    count = int(params.get("count", 3))
    radius = float(params.get("radius", 0.08))
    depth = float(params.get("depth", radius * 0.5))
    apply_bump_field(positions, colors, mask, {
        "count": count, "radius": radius, "height": -depth,
    }, rng)


def apply_erosion(
    positions: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> None:
    """Random small inward jitter — looks like weathered stone."""
    strength = float(params.get("strength", 0.01))
    if not mask.any():
        return
    idxs = np.where(mask)[0]
    pts = positions[idxs]
    centroid = pts.mean(axis=0)
    radial = pts - centroid
    rnorm = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9
    nrm = radial / rnorm
    jitter = rng.uniform(-strength, 0.0, idxs.size)  # only inward
    positions[idxs] = pts + nrm * jitter[:, None]
    # Random small color variation for visual richness.
    colors[idxs] *= rng.uniform(0.85, 1.0, idxs.size)[:, None]


def apply_ridges(
    positions: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> None:
    count = int(params.get("count", 8))
    depth = float(params.get("depth", 0.01))
    if not mask.any():
        return
    idxs = np.where(mask)[0]
    pts = positions[idxs]
    centroid = pts.mean(axis=0)
    radial = pts - centroid
    rnorm = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9
    nrm = radial / rnorm
    # Vertical ridges: rotate around Y, partition by angular bin.
    angle = np.arctan2(radial[:, 2], radial[:, 0])
    band = np.cos(angle * count)
    positions[idxs] = pts + nrm * (depth * band)[:, None]


def apply_holes(
    positions: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Returns a *keep* mask — points to retain after deletion.

    Unlike the other features, holes change `len(positions)`. The compositor
    handles the resulting filter outside this function.
    """
    count = int(params.get("count", 5))
    radius = float(params.get("radius", 0.06))
    keep = np.ones(positions.shape[0], dtype=bool)
    if count <= 0 or not mask.any():
        return keep
    idxs = np.where(mask)[0]
    pts = positions[idxs]
    centers = pts[rng.choice(pts.shape[0], size=min(count, pts.shape[0]), replace=False)]
    for c in centers:
        d = np.linalg.norm(positions - c, axis=1)
        keep &= ~(d < radius)
    return keep


def apply_fur(
    positions: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Add additional points "above" the surface to suggest fur/grass.

    Returns (extra_positions, extra_colors). The compositor concatenates them
    after running.
    """
    density = float(params.get("density", 0.5))
    length = float(params.get("length", 0.02))
    if not mask.any() or density <= 0.0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    idxs = np.where(mask)[0]
    pts = positions[idxs]
    centroid = pts.mean(axis=0)
    radial = pts - centroid
    rnorm = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9
    nrm = radial / rnorm
    n_extra = int(idxs.size * density)
    if n_extra == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    pick = rng.choice(idxs.size, size=n_extra, replace=False)
    base = pts[pick]; n = nrm[pick]
    t = rng.uniform(0, 1, n_extra)[:, None]
    new_pos = base + n * (length * t)
    new_col = colors[idxs[pick]] * (0.6 + 0.4 * (1 - t.squeeze()))[:, None]
    return new_pos.astype(np.float32), new_col.astype(np.float32)


FEATURE_FUNCS = {
    "scratch": apply_scratch,
    "curve_pattern": apply_curve_pattern,
    "bump_field": apply_bump_field,
    "dent": apply_dent,
    "erosion": apply_erosion,
    "ridges": apply_ridges,
    # `holes` and `fur` are handled specially by the compositor.
}
