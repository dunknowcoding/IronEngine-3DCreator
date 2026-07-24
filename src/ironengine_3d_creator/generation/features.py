"""Surface features applied after primitive sampling.

Each `apply_*` operates in place on a (positions, colors, labels) tuple where
positions: (N, 3) float32, colors: (N, 3) float32, labels: (N,) int (the index
of the originating primitive — used to scope features to a region).

Displacement features move points along *estimated local surface normals*
(KDTree + PCA per neighborhood, W11) instead of the centroid-radial direction,
so boxes/cylinders deform perpendicular to their actual surface rather than
pinching or ballooning sideways.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Local surface-normal estimation
# ---------------------------------------------------------------------------


def estimate_surface_normals(pts: np.ndarray, k: int = 12) -> np.ndarray:
    """Per-point normals via KDTree + PCA (smallest-covariance-eigenvector).

    Normals are oriented to agree with the centroid-radial direction so
    outward displacements stay outward. Falls back to centroid-radial when
    scipy is unavailable or the set is too small.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n_pts = pts.shape[0]
    centroid = pts.mean(axis=0)
    radial = pts - centroid
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9
    if n_pts < 4:
        return radial.astype(np.float32)
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:  # pragma: no cover - scipy always present in the env
        return radial.astype(np.float32)

    k = int(min(max(k, 4), n_pts))
    _, idx = cKDTree(pts).query(pts, k=k, workers=-1)
    nbrs = pts[idx]                                       # (N, k, 3)
    centered = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    _, evecs = np.linalg.eigh(cov)                        # ascending eigenvalues
    normals = evecs[:, :, 0]                              # smallest eigenvalue
    sign = np.sign(np.einsum("ij,ij->i", normals, radial))
    sign[sign == 0.0] = 1.0
    return (normals * sign[:, None]).astype(np.float32)


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
    n = estimate_surface_normals(pts)
    # Sinusoidal displacement along Y → "ribbed" / wavy bands.
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
    nrm = estimate_surface_normals(pts)
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
    nrm = estimate_surface_normals(pts)
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
    nrm = estimate_surface_normals(pts)
    # Vertical ridges: rotate around Y, partition by angular bin. The angle is
    # measured from the centroid (a radial property), but displacement follows
    # the estimated surface normal.
    radial = pts - pts.mean(axis=0)
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
    nrm = estimate_surface_normals(pts)
    n_extra = int(idxs.size * density)
    if n_extra == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    pick = rng.choice(idxs.size, size=n_extra, replace=False)
    base = pts[pick]; n = nrm[pick]
    t = rng.uniform(0, 1, n_extra)[:, None]
    new_pos = base + n * (length * t)
    new_col = colors[idxs[pick]] * (0.6 + 0.4 * (1 - t.squeeze()))[:, None]
    return new_pos.astype(np.float32), new_col.astype(np.float32)


def _value_noise(coords: np.ndarray, freq: float, rng: np.random.Generator,
                 octaves: int = 3) -> np.ndarray:
    """Seeded multi-octave value noise in [-1, 1] for a (N, 3) coord array.

    Cheap sin-hash noise (same family as generation.textures._grad_noise but
    signed and octave-stacked): deterministic under the compositor's seeded
    RNG, no scipy / external noise lib needed.
    """
    seeds = rng.uniform(-1.0, 1.0, (octaves, 3)).astype(np.float64)
    out = np.zeros(coords.shape[0], dtype=np.float64)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        out += amp * np.sin(coords @ seeds[o] * freq * (1.7 ** o)
                            + float(rng.uniform(0.0, 2.0 * math.pi)))
        total += amp
        amp *= 0.5
    return out / max(total, 1e-12)


def apply_asperity(
    positions: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> None:
    """Micro surface roughness — subtle seeded bidirectional normal noise.

    Breaks the razor-perfect CG look on stone / wood / soil / cast metal.
    `strength` is the peak displacement in metres (default 1 mm); `frequency`
    scales the noise field (higher = finer grain). Displacement follows the
    estimated local surface normal so slabs roughen perpendicular to their
    faces instead of sliding sideways.
    """
    strength = float(params.get("strength", 0.001))
    frequency = float(params.get("frequency", 35.0))
    if strength <= 0.0 or not mask.any():
        return
    idxs = np.where(mask)[0]
    pts = positions[idxs]
    nrm = estimate_surface_normals(pts)
    noise = _value_noise(pts.astype(np.float64), frequency, rng, octaves=2)
    positions[idxs] = pts + nrm * (strength * noise).astype(np.float32)[:, None]
    # Grain shows in the colour too: ±4 % albedo modulation.
    colors[idxs] *= (1.0 + 0.04 * noise).astype(np.float32)[:, None]


def apply_relief(
    positions: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> None:
    """Large-scale terrain-style relief — ground/soil is never a flat sheet.

    Multi-octave displacement along the surface normal (for a ground plane
    that is +Y) with a few octave layers so the result reads as clods and
    undulation rather than uniform jitter. Optional `pebbles` adds a handful
    of small local mounds. Colour darkens in the dips for readability.
    """
    amplitude = float(params.get("amplitude", 0.02))
    frequency = float(params.get("frequency", 6.0))
    octaves = int(params.get("octaves", 3))
    pebbles = int(params.get("pebbles", 0))
    if amplitude <= 0.0 or not mask.any():
        return
    idxs = np.where(mask)[0]
    pts = positions[idxs]
    nrm = estimate_surface_normals(pts)
    noise = _value_noise(pts.astype(np.float64), frequency, rng, octaves=octaves)
    disp = amplitude * noise
    if pebbles > 0:
        # A few compact mounds on top of the base undulation.
        centres = pts[rng.choice(pts.shape[0], size=min(pebbles, pts.shape[0]),
                                 replace=False)]
        pr = amplitude * 1.6
        for c in centres:
            d = np.linalg.norm(pts - c, axis=1)
            fall = np.clip(1.0 - d / pr, 0.0, None) ** 2
            disp += amplitude * 0.8 * fall
    positions[idxs] = pts + nrm * disp.astype(np.float32)[:, None]
    # Dips read darker (ambient-occlusion cheat).
    colors[idxs] *= (0.92 + 0.16 * (noise * 0.5 + 0.5)).astype(np.float32)[:, None]


FEATURE_FUNCS = {
    "scratch": apply_scratch,
    "curve_pattern": apply_curve_pattern,
    "bump_field": apply_bump_field,
    "dent": apply_dent,
    "erosion": apply_erosion,
    "ridges": apply_ridges,
    "relief": apply_relief,
    "asperity": apply_asperity,
    # `holes` and `fur` are handled specially by the compositor.
}
