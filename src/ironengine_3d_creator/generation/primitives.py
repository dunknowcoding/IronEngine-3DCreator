"""Surface samplers for the 10 primitive kinds.

Each `sample_*(n, params, rng)` returns an (N, 3) float32 array of points on
the *unit* primitive's surface in local space. The compositor multiplies by
the primitive's transform afterwards. Sampling is uniform on the surface
(area-weighted where the primitive has multiple faces).

When a CUDA backend is active (see `core.resources.active_backend`), the
heavyweight unit-sphere sampler runs on the GPU via CuPy. The ellipsoid
sampler stays on the CPU because area-uniform rejection sampling (W12) is
cheap and correctness matters more than throughput here. Smaller / non-bulk
primitives stay on the CPU because the transfer cost would swallow the
savings for typical point counts.
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


def _auto_bevel(half_extents, params: dict) -> float:
    """Resolve the edge micro-chamfer width (metres).

    ``params["bevel"]`` wins when present (0 disables). Otherwise a small
    default chamfer is derived from the part size so hard-surface slabs no
    longer have razor-perfect CG edges: 6 % of the smallest half-extent,
    clamped to [0.5 mm, 4 mm].
    """
    b = params.get("bevel")
    if b is not None:
        return max(0.0, float(b))
    h = float(np.min(half_extents))
    if h <= 1e-6:
        return 0.0
    return float(min(0.004, max(0.0005, h * 0.06)))


def _chamfer_box_points(pts: np.ndarray, half: np.ndarray, bevel: float) -> np.ndarray:
    """In-place 45° micro-chamfer of box surface points near the 12 edges.

    Points that sit within `bevel` of *two* face boundaries (i.e. on an edge
    strip) are pushed inward onto the chamfer plane. Corners (three
    boundaries) are handled by running the three axis pairs in sequence.
    """
    if bevel <= 0.0 or pts.shape[0] == 0:
        return pts
    for i, j in ((0, 1), (0, 2), (1, 2)):
        over_i = np.abs(pts[:, i]) - (half[i] - bevel)
        over_j = np.abs(pts[:, j]) - (half[j] - bevel)
        sel = (over_i > 0.0) & (over_j > 0.0)
        if not sel.any():
            continue
        excess = (over_i[sel] + over_j[sel]) * 0.5
        push_i = np.minimum(excess, over_i[sel])
        push_j = np.minimum(excess, over_j[sel])
        pts[sel, i] -= np.sign(pts[sel, i]) * push_i
        pts[sel, j] -= np.sign(pts[sel, j]) * push_j
    return pts


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
    pts = _stack(*out)
    bevel = _auto_bevel(np.array([hx, hy, hz], dtype=np.float64), params)
    return _chamfer_box_points(pts, np.array([hx, hy, hz], dtype=np.float32), bevel)


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
    """Area-uniform sampling on the ellipsoid surface (W12).

    Mapping a uniform sphere direction u through D = diag(radii) induces a
    surface density proportional to |D^-1 u|. Rejection-sampling u with
    probability |D^-1 u| / max(|D^-1 u|) = |D^-1 u| * min(radii) cancels that
    bias, giving uniform density per unit area even on elongated ellipsoids
    (for spheres every candidate is accepted).
    """
    rx, ry, rz = (float(v) for v in params.get("radii", [0.5, 0.5, 0.5]))
    radii = np.asarray([rx, ry, rz], dtype=np.float64)
    inv = 1.0 / radii
    accept_max = inv.max()  # max of |D^-1 u| over the unit sphere
    out: list[np.ndarray] = []
    needed = n
    while needed > 0:
        m = int(needed * 1.6) + 16
        u = rng.standard_normal((m, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        w = np.linalg.norm(u * inv, axis=1)
        keep = rng.uniform(0.0, accept_max, m) <= w
        pts = u[keep] * radii
        if pts.shape[0]:
            out.append(pts.astype(np.float32))
            needed -= pts.shape[0]
    if not out:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(out, axis=0)[:n]


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


# ---------------------------------------------------------------------------
# complex-shape kinds (F6): superellipsoid / tube / arch / panel
# ---------------------------------------------------------------------------


def _signed_pow(v: np.ndarray, e: float) -> np.ndarray:
    return np.sign(v) * np.abs(v) ** e


def _superellipsoid_unit(dirs: np.ndarray, e1: float, e2: float) -> np.ndarray:
    """Map unit-sphere directions radially onto the unit superellipsoid.

    The implicit surface (|x|^(2/e2) + |z|^(2/e2))^(e2/e1) + |y|^(2/e1) = 1 is
    homogeneous of degree 2/e1, so the radial scale factor has a closed form.
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    a = (np.abs(x) ** (2.0 / e2) + np.abs(z) ** (2.0 / e2)) ** (e2 / e1)
    t = (a + np.abs(y) ** (2.0 / e1)) ** (-e1 / 2.0)
    return dirs * t[:, None]


def sample_superellipsoid(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """Rejection sampler: candidate directions weighted by the local area
    element (exact for ellipsoids, a close approximation elsewhere)."""
    rx, ry, rz = (float(v) for v in params.get("radii", [0.5, 0.5, 0.5]))
    e1, e2 = (float(v) for v in params.get("exponents", [1.0, 1.0]))
    radii = np.asarray([rx, ry, rz], dtype=np.float64)
    inv = 1.0 / radii
    accept_max = inv.max()
    out: list[np.ndarray] = []
    needed = n
    while needed > 0:
        m = int(needed * 1.6) + 16
        u = rng.standard_normal((m, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        w = _superellipsoid_unit(u, e1, e2)
        # Approximate area weight via the unit-surface normal mapped by D^-1.
        g = np.stack([
            np.abs(w[:, 0]) ** max(2.0 / e2 - 1.0, 0.0) * np.sign(w[:, 0]),
            np.abs(w[:, 1]) ** max(2.0 / e1 - 1.0, 0.0) * np.sign(w[:, 1]),
            np.abs(w[:, 2]) ** max(2.0 / e2 - 1.0, 0.0) * np.sign(w[:, 2]),
        ], axis=-1)
        gn = np.linalg.norm(g, axis=1, keepdims=True)
        nrm = g / np.where(gn > 1e-12, gn, 1.0)
        weight = np.linalg.norm(nrm * inv, axis=1)
        keep = rng.uniform(0.0, accept_max, m) <= weight
        pts = w[keep] * radii
        if pts.shape[0]:
            out.append(pts.astype(np.float32))
            needed -= pts.shape[0]
    if not out:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(out, axis=0)[:n]


def tube_path_and_radii(params: dict) -> tuple[np.ndarray, float, float]:
    """Resolve a tube's path polyline + (start, end) radii from params.

    `path` is the canonical form; when absent we fall back to a straight
    vertical bar of `height` so `tube` is a drop-in curvable cylinder.
    """
    path = params.get("path")
    if not path or len(path) < 2:
        h = float(params.get("height", 1.0))
        path = [[0.0, -h / 2.0, 0.0], [0.0, h / 2.0, 0.0]]
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    r1 = float(params.get("radius", 0.05))
    r2 = float(params.get("radius2", r1))
    return pts, r1, r2


def path_length(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _path_frames(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel-transport frames (tangent, normal, binormal) per path vertex."""
    k = pts.shape[0]
    tan = np.zeros_like(pts)
    tan[0] = pts[1] - pts[0]
    tan[-1] = pts[-1] - pts[-2]
    if k > 2:
        tan[1:-1] = pts[2:] - pts[:-2]
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    # Seed normal: least-aligned world axis crossed with the first tangent.
    axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ tan[0]))]
    nrm = np.zeros_like(pts)
    nrm[0] = seed - np.dot(seed, tan[0]) * tan[0]
    nrm[0] /= np.linalg.norm(nrm[0]) + 1e-12
    for i in range(1, k):
        v = nrm[i - 1] - np.dot(nrm[i - 1], tan[i]) * tan[i]
        n = np.linalg.norm(v)
        if n < 1e-9:
            v = nrm[i - 1]
            n = 1.0
        nrm[i] = v / n
    bnm = np.cross(tan, nrm)
    bnm /= np.linalg.norm(bnm, axis=1, keepdims=True) + 1e-12
    return tan, nrm, bnm


def sample_tube(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    pts, r1, r2 = tube_path_and_radii(params)
    caps = bool(params.get("caps", True))
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    k = pts.shape[0]
    # Radius at each vertex, linearly interpolated along the path.
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1] if cum[-1] > 0 else 1.0
    r_vert = r1 + (r2 - r1) * (cum / total)
    # Side area per segment ≈ π (r_a + r_b) L (frustum); caps as disks.
    seg_area = math.pi * (r_vert[:-1] + r_vert[1:]) * seg_len
    cap_area = (math.pi * r1 * r1 + math.pi * r2 * r2) if caps else 0.0
    area_all = np.concatenate([seg_area, [cap_area]])
    weights = area_all / area_all.sum()
    counts = rng.multinomial(n, weights)
    tan, nrm, bnm = _path_frames(pts)

    out: list[np.ndarray] = []
    for s in range(k - 1):
        m = int(counts[s])
        if m <= 0 or seg_len[s] <= 0:
            continue
        t = rng.uniform(0.0, 1.0, m)
        r_s = r_vert[s] * (1.0 - t) + r_vert[s + 1] * t
        # Taper rejection: density along the segment ∝ local radius.
        r_max = max(r_vert[s], r_vert[s + 1])
        if r_max > 0:
            keep = rng.uniform(0.0, r_max, m) <= r_s
            t, r_s = t[keep], r_s[keep]
        if t.size == 0:
            continue
        center = pts[s] * (1.0 - t)[:, None] + pts[s + 1] * t[:, None]
        # Interpolate the frame along the segment (short-arc blend).
        frame_n = nrm[s] * (1.0 - t)[:, None] + nrm[s + 1] * t[:, None]
        frame_b = bnm[s] * (1.0 - t)[:, None] + bnm[s + 1] * t[:, None]
        frame_n /= np.linalg.norm(frame_n, axis=1, keepdims=True) + 1e-12
        frame_b /= np.linalg.norm(frame_b, axis=1, keepdims=True) + 1e-12
        th = rng.uniform(0.0, 2 * math.pi, t.size)
        offset = r_s[:, None] * (np.cos(th)[:, None] * frame_n + np.sin(th)[:, None] * frame_b)
        out.append(center + offset)
    n_cap = int(counts[-1])
    if caps and n_cap > 0:
        n_each = n_cap // 2
        for row, rr, sign, m in ((0, r1, -1.0, n_each), (k - 1, r2, 1.0, n_cap - n_each)):
            if m <= 0 or rr <= 0:
                continue
            u = rng.uniform(0.0, 1.0, m)
            v = rng.uniform(0.0, 2 * math.pi, m)
            rr_s = rr * np.sqrt(u)
            cap = pts[row] + rr_s[:, None] * (
                np.cos(v)[:, None] * nrm[row] + np.sin(v)[:, None] * bnm[row]
            )
            out.append(cap)
    if not out:
        return np.empty((0, 3), dtype=np.float32)
    return _stack(*out)[:n]


def arch_angles(params: dict) -> tuple[float, float, float, float]:
    """(R, r, start_angle, arc) for the arch primitive."""
    R = float(params.get("major_radius", 0.5))
    r = float(params.get("minor_radius", 0.1))
    start = float(params.get("start_angle", 0.0))
    arc = float(params.get("arc", math.pi))
    return R, r, start, arc


def sample_arch(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """Area-uniform sampling on a torus *segment* standing in the XY plane.

    Centre curve C(u) = (R cos u, R sin u, 0), u ∈ [start, start+arc]; the
    default (arc = π) is a proper ∩ arch with both feet at y = 0.
    """
    R, r, start, arc = arch_angles(params)
    caps = bool(params.get("caps", True))
    side_area = 2 * math.pi * r * R * arc
    cap_area = 2 * math.pi * r * r if caps else 0.0
    total = side_area + cap_area
    n_side = int(round(n * side_area / total))
    n_cap = n - n_side

    out: list[np.ndarray] = []
    needed = n_side
    while needed > 0:
        m = int(needed * 1.4) + 16
        u = rng.uniform(start, start + arc, m)
        v = rng.uniform(0.0, 2 * math.pi, m)
        # Area element ∝ (R + r cos v): rejection as on the full torus.
        accept = rng.uniform(0.0, R + r, m) <= (R + r * np.cos(v))
        u, v = u[accept], v[accept]
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v) * np.ones_like(x)
        if u.size:
            out.append(np.stack([x, y, z], axis=-1))
            needed -= u.size
    if caps and n_cap > 0:
        n_each = n_cap // 2
        for u_end, m in ((start, n_each), (start + arc, n_cap - n_each)):
            if m <= 0:
                continue
            # Disk ⊥ tangent in the plane spanned by radial/binormal.
            radial = np.array([math.cos(u_end), math.sin(u_end), 0.0])
            binormal = np.array([0.0, 0.0, 1.0])
            centre = np.array([R * math.cos(u_end), R * math.sin(u_end), 0.0])
            a = rng.uniform(0.0, 1.0, m)
            b = rng.uniform(0.0, 2 * math.pi, m)
            rr = r * np.sqrt(a)
            cap = centre + rr[:, None] * (
                np.cos(b)[:, None] * radial + np.sin(b)[:, None] * binormal
            )
            out.append(cap)
    if not out:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(out, axis=0)[:n].astype(np.float32)


def panel_geometry(params: dict) -> tuple[float, float, float, float]:
    """(width, height, thickness, bend_radians)."""
    w, h = (float(v) for v in params.get("size", [1.0, 1.0]))
    t = float(params.get("thickness", 0.02))
    bend = float(params.get("bend", 0.0))
    return w, h, t, bend


def _panel_face_areas(w: float, h: float, t: float, bend: float) -> np.ndarray:
    """Areas of (front, back, u0 edge, u1 edge, top, bottom). Exact for any bend."""
    if abs(bend) < 1e-9:
        return np.array([w * h, w * h, h * t, h * t, w * t, w * t], dtype=np.float64)
    rc = w / abs(bend)
    ab = abs(bend)
    return np.array([
        (rc + t / 2) * ab * h,
        (rc - t / 2) * ab * h,
        h * t,
        h * t,
        rc * ab * t,
        rc * ab * t,
    ], dtype=np.float64)


def sample_panel(n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    w, h, t, bend = panel_geometry(params)
    if abs(bend) < 1e-9:
        return sample_box(n, {"size": [w, h, t], "bevel": params.get("bevel")}, rng)
    rc = w / abs(bend)
    areas = _panel_face_areas(w, h, t, bend)
    counts = rng.multinomial(n, areas / areas.sum())

    # Arc centred at (0, ·, rc): mid-surface passes through the origin and
    # curves toward +z, so the panel stays centred like every other kind.
    # bend < 0 mirrors the arc through the z=0 plane (curves toward −z).
    zsign = -1.0 if bend < 0 else 1.0

    def sheet(m: int, s: float) -> np.ndarray:
        th = (rng.uniform(0.0, 1.0, m) - 0.5) * bend
        y = rng.uniform(-h / 2, h / 2, m)
        rho = rc + s
        return np.stack([rho * np.sin(th), y, zsign * (rc - rho * np.cos(th))], axis=-1)

    out = []
    if counts[0]:
        out.append(sheet(counts[0], +t / 2))
    if counts[1]:
        out.append(sheet(counts[1], -t / 2))
    for idx, u_end in ((2, -0.5), (3, +0.5)):
        if counts[idx]:
            th = np.full(int(counts[idx]), u_end * bend)
            y = rng.uniform(-h / 2, h / 2, int(counts[idx]))
            s = rng.uniform(-t / 2, t / 2, int(counts[idx]))
            rho = rc + s
            out.append(np.stack([rho * np.sin(th), y, zsign * (rc - rho * np.cos(th))], axis=-1))
    for idx, y_end in ((4, +h / 2), (5, -h / 2)):
        if counts[idx]:
            th = (rng.uniform(0.0, 1.0, int(counts[idx])) - 0.5) * bend
            s = rng.uniform(-t / 2, t / 2, int(counts[idx]))
            rho = rc + s
            out.append(np.stack(
                [rho * np.sin(th), np.full_like(th, y_end), zsign * (rc - rho * np.cos(th))], axis=-1))
    return _stack(*out)


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
    "superellipsoid": sample_superellipsoid,
    "tube": sample_tube,
    "sweep": sample_tube,
    "arch": sample_arch,
    "panel": sample_panel,
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
    if kind == "superellipsoid":
        # No closed form; estimate from a fixed tessellation (deterministic).
        rx, ry, rz = (float(v) for v in params.get("radii", [0.5, 0.5, 0.5]))
        e1, e2 = (float(v) for v in params.get("exponents", [1.0, 1.0]))
        phi = np.linspace(-math.pi / 2, math.pi / 2, 13)
        theta = np.linspace(0.0, 2 * math.pi, 25)
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        x = rx * np.outer(_signed_pow(cp, e1), _signed_pow(ct, e2))
        y = ry * np.outer(_signed_pow(sp, e1), np.ones_like(theta))
        z = rz * np.outer(_signed_pow(cp, e1), _signed_pow(st, e2))
        # Cell areas via cross products of grid diagonals.
        du = np.stack([x[:-1, 1:] - x[:-1, :-1], y[:-1, 1:] - y[:-1, :-1], z[:-1, 1:] - z[:-1, :-1]], axis=-1)
        dv = np.stack([x[1:, :-1] - x[:-1, :-1], y[1:, :-1] - y[:-1, :-1], z[1:, :-1] - z[:-1, :-1]], axis=-1)
        return float(np.linalg.norm(np.cross(du, dv), axis=-1).sum())
    if kind in ("tube", "sweep"):
        pts, r1, r2 = tube_path_and_radii(params)
        L = path_length(pts)
        area = math.pi * (r1 + r2) * L
        if params.get("caps", True):
            area += math.pi * r1 * r1 + math.pi * r2 * r2
        return area
    if kind == "arch":
        R, r, _, arc = arch_angles(params)
        area = 2 * math.pi * r * R * arc
        if params.get("caps", True):
            area += 2 * math.pi * r * r
        return area
    if kind == "panel":
        w, h, t, bend = panel_geometry(params)
        return float(_panel_face_areas(w, h, t, bend).sum())
    return 1.0


# ---------------------------------------------------------------------------
# inside tests (CSG-lite subtraction, point-cloud level)
# ---------------------------------------------------------------------------


def inside_primitive(kind: str, params: dict, pts_local: np.ndarray) -> np.ndarray:
    """Boolean mask: which *local-space* points lie strictly inside the solid.

    Used by the compositor to carve `role: "subtract"` cutters out of host
    parts. Falls back to the primitive's AABB for kinds without an exact test.
    """
    if pts_local.size == 0:
        return np.zeros((0,), dtype=bool)
    p = np.asarray(pts_local, dtype=np.float64).reshape(-1, 3)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    if kind == "box":
        sx, sy, sz = (float(v) for v in params.get("size", [1.0, 1.0, 1.0]))
        return (np.abs(x) <= sx / 2) & (np.abs(y) <= sy / 2) & (np.abs(z) <= sz / 2)
    if kind == "sphere":
        r = float(params.get("radius", 0.5))
        return (x * x + y * y + z * z) <= r * r
    if kind == "ellipsoid":
        rx, ry, rz = (float(v) for v in params.get("radii", [0.5, 0.5, 0.5]))
        return (x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2 <= 1.0
    if kind == "superellipsoid":
        rx, ry, rz = (float(v) for v in params.get("radii", [0.5, 0.5, 0.5]))
        e1, e2 = (float(v) for v in params.get("exponents", [1.0, 1.0]))
        a = (np.abs(x / rx) ** (2.0 / e2) + np.abs(z / rz) ** (2.0 / e2)) ** (e2 / e1)
        return (a + np.abs(y / ry) ** (2.0 / e1)) <= 1.0
    if kind == "cylinder":
        r = float(params.get("radius", 0.4))
        h = float(params.get("height", 1.0))
        return (x * x + z * z <= r * r) & (np.abs(y) <= h / 2)
    if kind == "capsule":
        r = float(params.get("radius", 0.3))
        h = float(params.get("height", 1.0))
        yc = np.clip(y, -h / 2, h / 2)
        return (x * x + (y - yc) ** 2 + z * z) <= r * r
    if kind == "cone":
        r = float(params.get("radius", 0.5))
        h = float(params.get("height", 1.0))
        t = (y + h / 2) / h
        rr = r * np.clip(1.0 - t, 0.0, None)
        return (np.abs(y) <= h / 2) & (x * x + z * z <= rr * rr)
    if kind == "prism":
        sides = max(3, int(params.get("sides", 6)))
        r = float(params.get("radius", 0.5))
        h = float(params.get("height", 1.0))
        ang = np.linspace(0.0, 2 * math.pi, sides, endpoint=False)
        poly = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=-1)
        # Point-in-convex-polygon: consistent side of every edge.
        inside = np.ones(p.shape[0], dtype=bool)
        for i in range(sides):
            a = poly[i]
            b = poly[(i + 1) % sides]
            edge = b - a
            cross = edge[0] * (z - a[1]) - edge[1] * (x - a[0])
            inside &= cross >= -1e-12
        return inside & (np.abs(y) <= h / 2)
    if kind in ("tube", "sweep"):
        pts, r1, r2 = tube_path_and_radii(params)
        # Distance from each point to every segment ≤ interpolated radius.
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        total = cum[-1] if cum[-1] > 0 else 1.0
        r_vert = r1 + (r2 - r1) * (cum / total)
        inside = np.zeros(p.shape[0], dtype=bool)
        for s in range(pts.shape[0] - 1):
            a, b = pts[s], pts[s + 1]
            ab = b - a
            denom = float(ab @ ab)
            t = np.clip(((p - a) @ ab) / (denom + 1e-12), 0.0, 1.0)
            d = np.linalg.norm(p - (a + t[:, None] * ab), axis=1)
            r_s = r_vert[s] * (1.0 - t) + r_vert[s + 1] * t
            inside |= d <= r_s
        return inside
    if kind == "arch":
        R, r, start, arc = arch_angles(params)
        # Closest point on the centre curve, then compare against tube radius.
        u = np.arctan2(y, x)
        # Unwrap u into the [start, start+arc] interval branch.
        twopi = 2 * math.pi
        u_rel = np.mod(u - start, twopi)
        u_clamped = np.clip(u_rel, 0.0, arc) + start
        cx, cy = R * np.cos(u_clamped), R * np.sin(u_clamped)
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + z * z)
        return d <= r
    if kind == "panel":
        w, h, t, bend = panel_geometry(params)
        if abs(bend) < 1e-9:
            return (np.abs(x) <= w / 2) & (np.abs(y) <= h / 2) & (np.abs(z) <= t / 2)
        rc = w / abs(bend)
        zz = -z if bend < 0 else z  # negative bend mirrors through z=0
        # Arc centred at (0, ·, rc): radius in the xz plane from the arc centre.
        rho = np.sqrt(x * x + (zz - rc) ** 2)
        th = np.arctan2(x, rc - zz)
        eps = 1e-6  # float32 surface points sit a few ulp outside the boundary
        return (
            (np.abs(y) <= h / 2 + eps)
            & (np.abs(th) <= abs(bend) / 2 + eps)
            & (np.abs(rho - rc) <= t / 2 + eps)
        )
    # Fallback: torus / helix / plane / unknown — AABB test.
    lo, hi = _fallback_aabb(kind, params)
    return (
        (x >= lo[0]) & (x <= hi[0])
        & (y >= lo[1]) & (y <= hi[1])
        & (z >= lo[2]) & (z <= hi[2])
    )


def _fallback_aabb(kind: str, params: dict) -> tuple[np.ndarray, np.ndarray]:
    if kind == "torus":
        R = float(params.get("major_radius", 0.5))
        r = float(params.get("minor_radius", 0.15))
        e = np.array([R + r, r, R + r])
    elif kind == "helix":
        R = float(params.get("radius", 0.4))
        hh = float(params.get("pitch", 0.2)) * float(params.get("turns", 3.0)) / 2
        t = float(params.get("thickness", 0.05))
        e = np.array([R + t, hh + t, R + t])
    elif kind == "plane":
        sx, sz = (float(v) for v in params.get("size", [1.0, 1.0]))
        e = np.array([sx / 2, 0.0, sz / 2])
    else:
        e = np.full(3, 0.5)
    return -e, e
