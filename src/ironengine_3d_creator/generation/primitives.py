"""Surface samplers for the 10 primitive kinds.

Each `sample_*(n, params, rng)` returns an (N, 3) float32 array of points on
the *unit* primitive's surface in local space. The compositor multiplies by
the primitive's transform afterwards. Sampling is uniform on the surface
(area-weighted where the primitive has multiple faces).

When a CUDA backend is active (see `core.resources.active_backend`), the
heavyweight unit-sphere and unit-ellipsoid samplers run on the GPU via CuPy.
Smaller / non-bulk primitives stay on the CPU because the transfer cost would
swallow the savings for typical point counts.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.resources import active_backend


def _stack(*arrs):
    return np.concatenate([a for a in arrs if len(a) > 0], axis=0).astype(np.float32, copy=False)


def _gpu_threshold() -> int:
    """Below this point count the CPU path is faster (CPU↔GPU transfer dominates)."""
    return 8_000


def _try_cupy():
    if active_backend() != "cuda_cupy":
        return None
    try:
        import cupy as cp  # type: ignore
        return cp
    except Exception:
        return None


def sample_box(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    sx, sy, sz = params.get("size", [1.0, 1.0, 1.0])
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    # Area-weighted face sampling.
    a_xy, a_xz, a_yz = sx * sy, sx * sz, sy * sz
    weights = np.array([a_xy, a_xy, a_xz, a_xz, a_yz, a_yz])
    weights = weights / weights.sum()
    counts = rng.multinomial(n, weights)
    out = []
    # ±Z faces (XY plane)
    u = rng.uniform(-hx, hx, counts[0]); v = rng.uniform(-hy, hy, counts[0])
    out.append(np.stack([u, v, np.full_like(u, hz)], axis=-1))
    u = rng.uniform(-hx, hx, counts[1]); v = rng.uniform(-hy, hy, counts[1])
    out.append(np.stack([u, v, np.full_like(u, -hz)], axis=-1))
    # ±Y faces
    u = rng.uniform(-hx, hx, counts[2]); v = rng.uniform(-hz, hz, counts[2])
    out.append(np.stack([u, np.full_like(u, hy), v], axis=-1))
    u = rng.uniform(-hx, hx, counts[3]); v = rng.uniform(-hz, hz, counts[3])
    out.append(np.stack([u, np.full_like(u, -hy), v], axis=-1))
    # ±X faces
    u = rng.uniform(-hy, hy, counts[4]); v = rng.uniform(-hz, hz, counts[4])
    out.append(np.stack([np.full_like(u, hx), u, v], axis=-1))
    u = rng.uniform(-hy, hy, counts[5]); v = rng.uniform(-hz, hz, counts[5])
    out.append(np.stack([np.full_like(u, -hx), u, v], axis=-1))
    return _stack(*out)


def sample_sphere(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    r = float(params.get("radius", 0.5))
    cp = _try_cupy()
    if cp is not None and n >= _gpu_threshold():
        seed = int(rng.integers(0, 2**31 - 1))
        gpu_rng = cp.random.default_rng(seed)
        pts = gpu_rng.standard_normal((n, 3), dtype=cp.float32)
        pts /= cp.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
        pts *= r
        return cp.asnumpy(pts)
    # CPU fallback: Marsaglia — uniform on the unit sphere.
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    return (pts * r).astype(np.float32)


def sample_cylinder(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    r = float(params.get("radius", 0.4))
    h = float(params.get("height", 1.0))
    caps = bool(params.get("caps", True))
    side_area = 2 * math.pi * r * h
    cap_area = math.pi * r * r if caps else 0.0
    total = side_area + 2 * cap_area
    n_side = int(round(n * side_area / total))
    n_cap = (n - n_side) // 2 if caps else 0
    n_extra = n - n_side - 2 * n_cap

    theta = rng.uniform(0, 2 * math.pi, n_side + n_extra)
    z = rng.uniform(-h / 2, h / 2, n_side + n_extra)
    side = np.stack([r * np.cos(theta), z, r * np.sin(theta)], axis=-1)

    out = [side]
    if caps and n_cap > 0:
        for sign in (+1, -1):
            # Disk sample: uniform via sqrt(u).
            u = rng.uniform(0, 1, n_cap); v = rng.uniform(0, 2 * math.pi, n_cap)
            rr = r * np.sqrt(u)
            cap = np.stack([rr * np.cos(v), np.full_like(rr, sign * h / 2), rr * np.sin(v)], axis=-1)
            out.append(cap)
    return _stack(*out)


def sample_capsule(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    r = float(params.get("radius", 0.3))
    h = float(params.get("height", 1.0))
    side_area = 2 * math.pi * r * h
    sphere_area = 4 * math.pi * r * r
    total = side_area + sphere_area
    n_side = int(round(n * side_area / total))
    n_hemi = (n - n_side) // 2

    theta = rng.uniform(0, 2 * math.pi, n_side)
    z = rng.uniform(-h / 2, h / 2, n_side)
    side = np.stack([r * np.cos(theta), z, r * np.sin(theta)], axis=-1)

    pts = rng.standard_normal((n_hemi * 2 + (n - n_side - 2 * n_hemi), 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    sphere = pts * r
    # Push the upper half up by h/2, lower half down — gives proper hemispherical caps.
    sphere[:, 1] = np.where(sphere[:, 1] >= 0, sphere[:, 1] + h / 2, sphere[:, 1] - h / 2)
    return _stack(side, sphere)


def sample_cone(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    r = float(params.get("radius", 0.5))
    h = float(params.get("height", 1.0))
    side_l = math.sqrt(r * r + h * h)
    side_area = math.pi * r * side_l
    cap_area = math.pi * r * r
    total = side_area + cap_area
    n_side = int(round(n * side_area / total))
    n_cap = n - n_side

    # Side: parameterize by t ∈ [0, 1] along height (apex at t=1) and angle θ.
    t = np.sqrt(rng.uniform(0, 1, n_side))   # area-weighted toward the base
    theta = rng.uniform(0, 2 * math.pi, n_side)
    rr = r * (1.0 - t)
    side = np.stack([rr * np.cos(theta), -h / 2 + h * t, rr * np.sin(theta)], axis=-1)

    u = rng.uniform(0, 1, n_cap); v = rng.uniform(0, 2 * math.pi, n_cap)
    rr = r * np.sqrt(u)
    cap = np.stack([rr * np.cos(v), np.full_like(rr, -h / 2), rr * np.sin(v)], axis=-1)
    return _stack(side, cap)


def sample_torus(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    R = float(params.get("major_radius", 0.5))
    r = float(params.get("minor_radius", 0.15))
    # Rejection sampling for area-uniform distribution on the torus.
    out = []
    needed = n
    while needed > 0:
        m = int(needed * 1.4) + 16
        u = rng.uniform(0, 2 * math.pi, m)
        v = rng.uniform(0, 2 * math.pi, m)
        accept = rng.uniform(0, R + r, m) <= (R + r * np.cos(v))
        u, v = u[accept], v[accept]
        x = (R + r * np.cos(v)) * np.cos(u)
        z = (R + r * np.cos(v)) * np.sin(u)
        y = r * np.sin(v)
        out.append(np.stack([x, y, z], axis=-1))
        needed -= u.size
    return np.concatenate(out, axis=0)[:n].astype(np.float32)


def sample_ellipsoid(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    rx, ry, rz = params.get("radii", [0.5, 0.5, 0.5])
    cp = _try_cupy()
    if cp is not None and n >= _gpu_threshold():
        seed = int(rng.integers(0, 2**31 - 1))
        gpu_rng = cp.random.default_rng(seed)
        pts = gpu_rng.standard_normal((n, 3), dtype=cp.float32)
        pts /= cp.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
        # Broadcast multiply: (N,3) * (3,) hits one elementwise kernel instead
        # of three strided column updates.
        pts *= cp.asarray([rx, ry, rz], dtype=cp.float32)
        return cp.asnumpy(pts)
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    pts *= np.asarray([rx, ry, rz], dtype=np.float32)
    return pts.astype(np.float32)


def sample_prism(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    sides = max(3, int(params.get("sides", 6)))
    r = float(params.get("radius", 0.5))
    h = float(params.get("height", 1.0))
    angles = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    verts = np.stack([r * np.cos(angles), np.zeros_like(angles), r * np.sin(angles)], axis=-1)

    side_area_per_face = np.linalg.norm(verts[1] - verts[0]) * h
    cap_area = 0.5 * sides * r * r * math.sin(2 * math.pi / sides)
    total = sides * side_area_per_face + 2 * cap_area
    n_side_each = int(round(n * side_area_per_face / total))
    n_cap = (n - sides * n_side_each) // 2

    out = []
    for i in range(sides):
        a, b = verts[i], verts[(i + 1) % sides]
        t = rng.uniform(0, 1, n_side_each)
        z = rng.uniform(-h / 2, h / 2, n_side_each)
        face = a * (1 - t)[:, None] + b * t[:, None]
        face[:, 1] = z
        out.append(face)
    if n_cap > 0:
        for sign in (+1, -1):
            u = rng.uniform(0, 1, n_cap); v = rng.uniform(0, 2 * math.pi, n_cap)
            rr = r * np.sqrt(u)
            cap = np.stack([rr * np.cos(v), np.full_like(rr, sign * h / 2), rr * np.sin(v)], axis=-1)
            out.append(cap)
    return _stack(*out)


def sample_helix(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    R = float(params.get("radius", 0.4))
    pitch = float(params.get("pitch", 0.2))
    turns = float(params.get("turns", 3.0))
    thickness = float(params.get("thickness", 0.05))

    # Sample (t along helix, θ around the tube cross-section).
    t = rng.uniform(0, turns, n)
    theta = rng.uniform(0, 2 * math.pi, n)
    cx = R * np.cos(2 * math.pi * t)
    cz = R * np.sin(2 * math.pi * t)
    cy = pitch * t - (pitch * turns) / 2  # centered vertically

    # Local frame: tangent T, normal N (pointing outward radially), binormal B.
    tan = np.stack([
        -R * 2 * math.pi * np.sin(2 * math.pi * t),
        np.full_like(t, pitch),
        R * 2 * math.pi * np.cos(2 * math.pi * t),
    ], axis=-1)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    nrm = np.stack([np.cos(2 * math.pi * t), np.zeros_like(t), np.sin(2 * math.pi * t)], axis=-1)
    bnm = np.cross(tan, nrm)

    offset = thickness * (np.cos(theta)[:, None] * nrm + np.sin(theta)[:, None] * bnm)
    pts = np.stack([cx, cy, cz], axis=-1) + offset
    return pts.astype(np.float32)


def sample_plane(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    sx, sz = params.get("size", [1.0, 1.0])
    x = rng.uniform(-sx / 2, sx / 2, n)
    z = rng.uniform(-sz / 2, sz / 2, n)
    return np.stack([x, np.zeros_like(x), z], axis=-1).astype(np.float32)


SAMPLERS = {
    "box": sample_box,
    "sphere": sample_sphere,
    "cylinder": sample_cylinder,
    "capsule": sample_capsule,
    "cone": sample_cone,
    "torus": sample_torus,
    "ellipsoid": sample_ellipsoid,
    "prism": sample_prism,
    "helix": sample_helix,
    "plane": sample_plane,
}


def sample_primitive(kind: str, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    if kind not in SAMPLERS:
        raise KeyError(f"unknown primitive kind: {kind!r}")
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    return SAMPLERS[kind](n, params, rng)


def primitive_area(kind: str, params: dict) -> float:
    """Approximate surface area for budget allocation."""
    if kind == "box":
        sx, sy, sz = params.get("size", [1, 1, 1])
        return 2 * (sx * sy + sx * sz + sy * sz)
    if kind == "sphere":
        r = params.get("radius", 0.5)
        return 4 * math.pi * r * r
    if kind == "cylinder":
        r = params.get("radius", 0.4); h = params.get("height", 1.0)
        return 2 * math.pi * r * h + (2 * math.pi * r * r if params.get("caps", True) else 0)
    if kind == "capsule":
        r = params.get("radius", 0.3); h = params.get("height", 1.0)
        return 2 * math.pi * r * h + 4 * math.pi * r * r
    if kind == "cone":
        r = params.get("radius", 0.5); h = params.get("height", 1.0)
        return math.pi * r * math.sqrt(r * r + h * h) + math.pi * r * r
    if kind == "torus":
        R = params.get("major_radius", 0.5); r = params.get("minor_radius", 0.15)
        return 4 * math.pi * math.pi * R * r
    if kind == "ellipsoid":
        rx, ry, rz = params.get("radii", [0.5, 0.5, 0.5])
        # Knud Thomsen approximation.
        p = 1.6075
        return 4 * math.pi * (((rx * ry) ** p + (rx * rz) ** p + (ry * rz) ** p) / 3) ** (1 / p)
    if kind == "prism":
        sides = max(3, params.get("sides", 6)); r = params.get("radius", 0.5); h = params.get("height", 1.0)
        side_len = 2 * r * math.sin(math.pi / sides)
        cap = 0.5 * sides * r * r * math.sin(2 * math.pi / sides)
        return sides * side_len * h + 2 * cap
    if kind == "helix":
        R = params.get("radius", 0.4); pitch = params.get("pitch", 0.2)
        turns = params.get("turns", 3.0); t = params.get("thickness", 0.05)
        length_per_turn = math.sqrt((2 * math.pi * R) ** 2 + pitch * pitch)
        return 2 * math.pi * t * length_per_turn * turns
    if kind == "plane":
        sx, sz = params.get("size", [1, 1])
        return sx * sz
    return 1.0
