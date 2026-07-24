"""Slicing builder — '3D-printing style' lofting of 2D cross-section profiles.

A solid is described as a stack of 2D cross-section *slices* along an axis,
exactly like a 3D printer lays down layers. Each slice carries the same base
profile plus an independent scale / in-plane rotation / in-plane offset, so
tapered, twisted, and drifting silhouettes (vases, towers, hulls, fuselages,
lamp shades) fall out of one mechanism.

The output is a watertight analytic triangle mesh:
- ring vertices are shared between adjacent slice bands (no T-junctions),
- every edge is used by exactly two triangles (2-manifold),
- smooth per-vertex normals oriented outward (centroid test),
- UVs with u running around the profile and v along the axis,
- caps are centroid fans — perfect for convex and star-shaped profiles
  (keep cap profiles star-shaped w.r.t. their centroid).

Axis mapping: the profile lives in a 2D (u, w) plane; ``axis="y"`` embeds it
as (u, position, w), ``axis="z"`` as (u, w, position), ``axis="x"`` as
(position, u, w). All dimensions are metres, real-world scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TAU = 2.0 * math.pi

Mesh = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# profile library — closed CCW 2D loops (first point NOT repeated)
# ---------------------------------------------------------------------------


def profile_circle(radius: float = 0.5, segments: int = 32) -> np.ndarray:
    """Regular N-gon approximation of a circle (CCW)."""
    th = np.linspace(0.0, TAU, max(3, int(segments)), endpoint=False)
    return np.stack([radius * np.cos(th), radius * np.sin(th)], axis=-1)


def profile_rounded_rect(
    width: float = 1.0,
    height: float = 1.0,
    corner_radius: float = 0.1,
    seg_per_corner: int = 6,
) -> np.ndarray:
    """Centred rounded rectangle (CCW), straight edges + quarter-arc corners."""
    hx, hy = width / 2.0, height / 2.0
    r = min(corner_radius, hx - 1e-9, hy - 1e-9)
    if r <= 1e-9:
        return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
    pts: list[list[float]] = []
    # Corner centres in CCW order, sweeping angles from -x/-y around.
    corners = [
        (hx - r, -hy + r, -math.pi / 2),
        (hx - r, hy - r, 0.0),
        (-hx + r, hy - r, math.pi / 2),
        (-hx + r, -hy + r, math.pi),
    ]
    n = max(2, int(seg_per_corner))
    for cx, cy, a0 in corners:
        for i in range(n):
            a = a0 + (math.pi / 2.0) * (i / n)
            pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    return np.asarray(pts, dtype=np.float64)


def profile_superellipse(
    a: float = 0.5,
    b: float = 0.5,
    exponent: float = 2.5,
    segments: int = 32,
) -> np.ndarray:
    """|x/a|^n + |y/b|^n = 1 (n=2 ellipse, n→∞ rectangle). CCW."""
    n = max(0.2, float(exponent))
    th = np.linspace(0.0, TAU, max(8, int(segments)), endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    x = a * np.sign(ct) * np.abs(ct) ** (2.0 / n)
    y = b * np.sign(st) * np.abs(st) ** (2.0 / n)
    return np.stack([x, y], axis=-1)


def profile_polygon(points) -> np.ndarray:
    """Custom polygon from an iterable of (x, y); reordered CCW if needed."""
    poly = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 3:
        raise ValueError("profile_polygon needs at least 3 points")
    if profile_area(poly) < 0.0:  # clockwise → flip to CCW
        poly = poly[::-1].copy()
    return poly


def profile_area(profile: np.ndarray) -> float:
    """Signed shoelace area (> 0 for CCW)."""
    p = np.asarray(profile, dtype=np.float64).reshape(-1, 2)
    return float(
        0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    )


def profile_centroid(profile: np.ndarray) -> np.ndarray:
    """Area-weighted centroid of a (CCW) polygon."""
    p = np.asarray(profile, dtype=np.float64).reshape(-1, 2)
    a = profile_area(p)
    if abs(a) < 1e-15:
        return p.mean(axis=0)
    p1 = np.roll(p, -1, axis=0)
    cross = p[:, 0] * p1[:, 1] - p1[:, 0] * p[:, 1]
    c = ((p + p1) * cross[:, None]).sum(axis=0) / (6.0 * a)
    return c


# ---------------------------------------------------------------------------
# slices
# ---------------------------------------------------------------------------


@dataclass
class Slice:
    """One cross-section layer: the base profile transformed in its plane.

    Attributes
    ----------
    position:
        Coordinate along the loft axis (metres).
    scale:
        (su, sw) in-plane scale applied to the profile.
    rotation:
        In-plane rotation in radians (CCW in the profile plane).
    offset:
        (du, dw) in-plane translation applied after scale + rotation —
        this is how a slice drifts off the axis (e.g. a banana curve).
    """

    position: float
    scale: tuple[float, float] = (1.0, 1.0)
    rotation: float = 0.0
    offset: tuple[float, float] = (0.0, 0.0)


def radius_slices(positions, radii) -> list[Slice]:
    """Convenience: lathe-style slices for a circular profile."""
    return [Slice(position=float(p), scale=(float(r), float(r)))
            for p, r in zip(positions, radii)]


def _slice_ring(profile: np.ndarray, sl: Slice) -> np.ndarray:
    """Transform the 2D profile into the slice's plane (still 2D coords)."""
    su, sw = sl.scale
    c, s = math.cos(sl.rotation), math.sin(sl.rotation)
    p = profile * np.array([su, sw])
    rot = np.array([[c, -s], [s, c]])
    return p @ rot.T + np.asarray(sl.offset, dtype=np.float64)


def _embed(plane: np.ndarray, position: float, axis: str) -> np.ndarray:
    """Map 2D plane points to 3D at `position` along `axis`."""
    if axis == "y":
        return np.stack([plane[:, 0], np.full(len(plane), position), plane[:, 1]], axis=-1)
    if axis == "z":
        return np.stack([plane[:, 0], plane[:, 1], np.full(len(plane), position)], axis=-1)
    if axis == "x":
        return np.stack([np.full(len(plane), position), plane[:, 0], plane[:, 1]], axis=-1)
    raise ValueError(f"unknown loft axis {axis!r} (expected 'x', 'y' or 'z')")


# ---------------------------------------------------------------------------
# the loft
# ---------------------------------------------------------------------------


def loft(
    profile: np.ndarray,
    slices: list[Slice],
    axis: str = "y",
    caps: bool = True,
    uv_repeat: tuple[float, float] = (1.0, 1.0),
) -> Mesh:
    """Stack `slices` of `profile` into a watertight analytic mesh.

    Returns (vertices, normals, uvs, faces): float32 (V,3) / (V,3) / (V,2)
    and int64 (F,3) — the same convention as `generation.analytic_mesh`.
    """
    profile = profile_polygon(profile)  # guarantees CCW
    if len(slices) < 2:
        raise ValueError("loft needs at least 2 slices")
    order = np.argsort([sl.position for sl in slices])
    slices = [slices[int(i)] for i in order]

    n = profile.shape[0]
    axis = axis.lower()
    rings2d = [_slice_ring(profile, sl) for sl in slices]
    pos_axis = np.array([sl.position for sl in slices], dtype=np.float64)
    span = max(pos_axis[-1] - pos_axis[0], 1e-12)

    n_rings = len(slices)
    v = np.zeros((n_rings * n, 3), dtype=np.float64)
    for i, ring in enumerate(rings2d):
        v[i * n : (i + 1) * n] = _embed(ring, pos_axis[i], axis)

    # --- smooth normals: cross(profile tangent, axis tangent), oriented out.
    nrm = np.zeros_like(v)
    for i in range(n_rings):
        ring3 = v[i * n : (i + 1) * n]
        centroid = ring3.mean(axis=0)
        prev3 = v[max(i - 1, 0) * n : (max(i - 1, 0) + 1) * n]
        next3 = v[min(i + 1, n_rings - 1) * n : (min(i + 1, n_rings - 1) + 1) * n]
        t_axis = next3 - prev3                                   # (n, 3)
        t_prof = np.roll(ring3, -1, axis=0) - np.roll(ring3, 1, axis=0)
        nn = np.cross(t_axis, t_prof)
        out = ring3 - centroid[None, :]
        flip = np.einsum("ij,ij->i", nn, out) < 0.0
        nn[flip] *= -1.0
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-12
        nrm[i * n : (i + 1) * n] = nn

    uv = np.zeros((n_rings * n, 2), dtype=np.float64)
    u_col = np.linspace(0.0, 1.0, n, endpoint=False) * uv_repeat[0]
    for i in range(n_rings):
        v_coord = (pos_axis[i] - pos_axis[0]) / span * uv_repeat[1]
        uv[i * n : (i + 1) * n, 0] = u_col
        uv[i * n : (i + 1) * n, 1] = v_coord

    # --- side bands (shared ring vertices → watertight seams). Winding is
    # chosen so geometric normals agree with the outward vertex normals:
    # (a, d, c) / (a, c, b) for ring i → ring i+1.
    bands: list[np.ndarray] = []
    j = np.arange(n, dtype=np.int64)
    j1 = (j + 1) % n
    for i in range(n_rings - 1):
        a, b = i * n + j, i * n + j1
        c, d = (i + 1) * n + j1, (i + 1) * n + j
        bands.append(np.stack([a, d, c], axis=1))
        bands.append(np.stack([a, c, b], axis=1))
    faces = np.concatenate(bands, axis=0)

    # --- caps: centroid fan on the first/last ring. The fan REUSES the ring
    # vertex indices (only the centroid is new) so the mesh stays watertight.
    extra_v: list[np.ndarray] = []
    extra_n: list[np.ndarray] = []
    extra_u: list[np.ndarray] = []
    cap_chunks: list[np.ndarray] = []
    base = n_rings * n
    if caps:
        for ci, (ring_i, sign) in enumerate(((0, -1.0), (n_rings - 1, +1.0))):
            ring3 = v[ring_i * n : (ring_i + 1) * n]
            centre3 = ring3.mean(axis=0)
            cidx = base + ci
            cap_n = np.zeros(3)
            cap_n[{"x": 0, "y": 1, "z": 2}[axis]] = sign
            jj = np.arange(n, dtype=np.int64)
            jj1 = (jj + 1) % n
            cf = np.stack(
                [np.full(n, cidx, dtype=np.int64), ring_i * n + jj, ring_i * n + jj1],
                axis=1,
            )
            # Orient the fan so its geometric normal matches `sign`.
            probe = np.cross(
                v[cf[:, 1]] - centre3[None, :], v[cf[:, 2]] - centre3[None, :]
            )
            if float((probe @ cap_n).mean()) < 0.0:
                cf = cf[:, [0, 2, 1]]
            extra_v.append(centre3)
            extra_n.append(cap_n)
            extra_u.append(np.array([0.5, 0.5]))
            cap_chunks.append(cf)

    # --- merge. Winding is derived from the vertex normals (which are
    # authoritative and outward everywhere): any face whose geometric normal
    # disagrees with its vertex normals is flipped. This is uniform across
    # the three axis choices, whose profile-plane handedness differs.
    v_all = np.concatenate([v, np.asarray(extra_v).reshape(-1, 3)], axis=0)
    n_all = np.concatenate([nrm, np.asarray(extra_n).reshape(-1, 3)], axis=0)
    u_all = np.concatenate([uv, np.asarray(extra_u).reshape(-1, 2)], axis=0)
    f_all = np.concatenate([faces, *cap_chunks], axis=0) if cap_chunks else faces
    v0 = v_all[f_all[:, 0]]
    v1 = v_all[f_all[:, 1]]
    v2 = v_all[f_all[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = n_all[f_all].mean(axis=1)
    flip = np.einsum("ij,ij->i", fn, vn) < 0.0
    f_all[flip] = f_all[flip][:, [0, 2, 1]]
    # Global winding sanity: if the volume still comes out negative, flip the
    # faces AND the vertex normals so they never disagree.
    if _signed_volume(v_all, f_all) < 0.0:
        f_all = f_all[:, [0, 2, 1]]
        n_all = -n_all
    return (
        v_all.astype(np.float32),
        n_all.astype(np.float32),
        u_all.astype(np.float32),
        f_all.astype(np.int64),
    )


def loft_volume(profile: np.ndarray, slices: list[Slice]) -> float:
    """Trapezoidal volume estimate: area(profile)·su·sw integrated along axis."""
    profile = profile_polygon(profile)
    area = abs(profile_area(profile))
    order = np.argsort([sl.position for sl in slices])
    pos = np.array([slices[int(i)].position for i in order], dtype=np.float64)
    areas = np.array(
        [area * abs(slices[int(i)].scale[0] * slices[int(i)].scale[1]) for i in order]
    )
    return float(np.sum((areas[:-1] + areas[1:]) * 0.5 * np.diff(pos)))


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    v0 = vertices[faces[:, 0]].astype(np.float64)
    v1 = vertices[faces[:, 1]].astype(np.float64)
    v2 = vertices[faces[:, 2]].astype(np.float64)
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def signed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Public signed-volume helper (positive = outward winding)."""
    return _signed_volume(vertices, faces)
