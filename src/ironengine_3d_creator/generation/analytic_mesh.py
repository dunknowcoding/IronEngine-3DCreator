"""Exact analytic triangle meshes for the 10 primitive kinds (F5).

The generator samples point clouds from parametric primitives; triangulating
those clouds afterwards (ball-pivot / Poisson) degrades what were exact
analytic surfaces. This module builds the exact triangle mesh per primitive
kind instead — smooth analytic normals, UVs, watertight faces, and solid
volumes — then applies each primitive's 4x4 transform.

Spec-driven GLB/OBJ export uses these meshes; point-cloud reconstruction
(`generation.reconstruct`) remains only as a fallback for code-mode /
freeform clouds.

UV conventions per kind:
- box / plane / prism caps: box-projection (the two varying axes, normalized)
- cylinder / capsule / cone side, prism sides, helix: cylindrical (theta, axis)
- sphere / ellipsoid / torus: spherical / parametric (u, v)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

_log = logging.getLogger(__name__)

# Tessellation density. Analytic primitives look exact even at modest segment
# counts because normals are analytic rather than averaged.
SEG_U = 32                 # around (longitude)
SEG_V = 16                 # latitudinal / parametric v
HELIX_SEG_PER_TURN = 24    # rings per helix turn
TUBE_SEG = 8               # helix tube cross-section segments

TAU = 2.0 * math.pi


@dataclass
class AnalyticPart:
    """One transformed, fully-described mesh for a spec primitive."""
    label: str
    kind: str
    material: str
    vertices: np.ndarray      # (V, 3) float32, world space
    normals: np.ndarray       # (V, 3) float32, world space
    uvs: np.ndarray           # (V, 2) float32
    faces: np.ndarray         # (F, 3) int64
    aabb_min: np.ndarray      # (3,)
    aabb_max: np.ndarray      # (3,)
    solid_volume_m3: float


# ---------------------------------------------------------------------------
# grid helpers
# ---------------------------------------------------------------------------


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed enclosed volume via the divergence theorem (> 0 = outward)."""
    v0 = vertices[faces[:, 0]].astype(np.float64)
    v1 = vertices[faces[:, 1]].astype(np.float64)
    v2 = vertices[faces[:, 2]].astype(np.float64)
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def _grid_mesh(
    pos: np.ndarray,
    nrm: np.ndarray,
    uv: np.ndarray,
    *,
    fix_winding: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a (R, C, 3) parametric grid into vertices + quad faces."""
    rows, cols = pos.shape[0], pos.shape[1]
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[1:, :-1].ravel()
    faces = np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=0
    )
    v = pos.reshape(-1, 3).astype(np.float32)
    n = nrm.reshape(-1, 3).astype(np.float32)
    u = uv.reshape(-1, 2).astype(np.float32)
    if fix_winding and signed_volume(v, faces) < 0.0:
        faces = faces[:, [0, 2, 1]]
    return v, n, u, faces


def _disk_cap(
    r: float,
    y: float,
    sign: float,
    seg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center + ring fan for a horizontal disk cap at height `y`."""
    th = np.linspace(0.0, TAU, seg + 1)
    ring = np.stack([r * np.cos(th), np.full_like(th, y), r * np.sin(th)], axis=-1)
    v = np.concatenate([np.array([[0.0, y, 0.0]]), ring], axis=0).astype(np.float32)
    n = np.tile(np.array([[0.0, sign, 0.0]], dtype=np.float32), (seg + 2, 1))
    u = np.concatenate(
        [np.array([[0.5, 0.5]]), np.stack([0.5 + 0.5 * np.cos(th), 0.5 + 0.5 * np.sin(th)], axis=-1)],
        axis=0,
    ).astype(np.float32)
    j = np.arange(1, seg + 1, dtype=np.int64)
    if sign > 0:
        faces = np.stack([np.zeros(seg, dtype=np.int64), j, j + 1], axis=1)
    else:
        faces = np.stack([np.zeros(seg, dtype=np.int64), j + 1, j], axis=1)
    return v, n, u, faces


def _merge(
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate (v, n, uv, f) chunks with face-index offsets."""
    vs, ns, us, fs = [], [], [], []
    offset = 0
    for v, n, u, f in chunks:
        vs.append(v)
        ns.append(n)
        us.append(u)
        fs.append(f + offset)
        offset += v.shape[0]
    return (
        np.concatenate(vs, axis=0),
        np.concatenate(ns, axis=0),
        np.concatenate(us, axis=0),
        np.concatenate(fs, axis=0),
    )


def _orient_like_normals(
    v: np.ndarray, n: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """Flip any face whose geometric normal disagrees with its vertex normals.

    Analytic normals are authoritative; this makes winding consistent with
    them chunk-by-chunk (a global signed-volume flip cannot fix mixed chunks).
    """
    v0 = v[f[:, 0]].astype(np.float64)
    v1 = v[f[:, 1]].astype(np.float64)
    v2 = v[f[:, 2]].astype(np.float64)
    fn = np.cross(v1 - v0, v2 - v0)
    vn = n[f].astype(np.float64).mean(axis=1)
    flip = np.einsum("ij,ij->i", fn, vn) < 0.0
    f = f.copy()
    f[flip] = f[flip][:, [0, 2, 1]]
    return f


# ---------------------------------------------------------------------------
# per-kind local mesh builders: (vertices, normals, uvs, faces)
# ---------------------------------------------------------------------------


def mesh_box(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sx, sy, sz = (float(v) for v in params.get("size", [1.0, 1.0, 1.0]))
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    # 6 faces x 4 verts, CCW seen from outside; uv = box projection per face.
    face_defs = [
        # (normal, 4 corners, uv basis axes)
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
        ((0, 1, 0), [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        ((1, 0, 0), [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)]),
        ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
    ]
    half = np.array([hx, hy, hz], dtype=np.float64)
    size = np.array([sx, sy, sz], dtype=np.float64)
    verts, norms, uvs, faces = [], [], [], []
    for fi, (normal, corners) in enumerate(face_defs):
        base = fi * 4
        c = np.asarray(corners, dtype=np.float64)
        n_arr = np.abs(np.asarray(normal, dtype=np.float64))
        # Project onto the two varying axes for a box-projection UV.
        varying = np.where(n_arr < 0.5)[0]
        uv = (c[:, varying] + half[varying]) / size[varying]
        verts.append(c)
        norms.append(np.tile(np.asarray(normal, dtype=np.float64), (4, 1)))
        uvs.append(uv)
        faces.append(np.array([[base, base + 1, base + 2], [base, base + 2, base + 3]]))
    v = np.concatenate(verts).astype(np.float32)
    n = np.concatenate(norms).astype(np.float32)
    u = np.concatenate(uvs).astype(np.float32)
    f = np.concatenate(faces).astype(np.int64)
    if signed_volume(v, f) < 0.0:
        f = f[:, [0, 2, 1]]
    return v, n, u, f


def _sphere_grid(
    rx: float, ry: float, rz: float, seg_u: int, seg_v: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """UV-sphere grid scaled by radii; ellipsoid-correct smooth normals."""
    phi = np.linspace(-math.pi / 2, math.pi / 2, seg_v + 1)
    theta = np.linspace(0.0, TAU, seg_u + 1)
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    # (seg_v+1, seg_u+1, 3) unit-sphere directions
    d = np.stack(
        [
            cp[:, None] * ct[None, :],
            np.broadcast_to(sp[:, None], (seg_v + 1, seg_u + 1)),
            cp[:, None] * st[None, :],
        ],
        axis=-1,
    )
    radii = np.array([rx, ry, rz])
    pos = d * radii
    # Ellipsoid normal = normalize(D^-2 q) = normalize(d / radii).
    n = d / radii
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    uv = np.stack(
        [
            np.broadcast_to(theta[None, :] / TAU, (seg_v + 1, seg_u + 1)),
            np.broadcast_to((phi[:, None] + math.pi / 2) / math.pi, (seg_v + 1, seg_u + 1)),
        ],
        axis=-1,
    )
    return _grid_mesh(pos, n, uv)


def mesh_sphere(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = float(params.get("radius", 0.5))
    return _sphere_grid(r, r, r, SEG_U, SEG_V)


def mesh_ellipsoid(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rx, ry, rz = (float(v) for v in params.get("radii", [0.5, 0.5, 0.5]))
    return _sphere_grid(rx, ry, rz, SEG_U, SEG_V)


def mesh_cylinder(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = float(params.get("radius", 0.4))
    h = float(params.get("height", 1.0))
    caps = bool(params.get("caps", True))
    th = np.linspace(0.0, TAU, SEG_U + 1)
    ct, st = np.cos(th), np.sin(th)
    ys = np.array([-h / 2, h / 2])
    pos = np.stack(
        [
            np.broadcast_to(r * ct[None, :], (2, SEG_U + 1)),
            np.broadcast_to(ys[:, None], (2, SEG_U + 1)),
            np.broadcast_to(r * st[None, :], (2, SEG_U + 1)),
        ],
        axis=-1,
    )
    nrm = np.stack(
        [
            np.broadcast_to(ct[None, :], (2, SEG_U + 1)),
            np.zeros((2, SEG_U + 1)),
            np.broadcast_to(st[None, :], (2, SEG_U + 1)),
        ],
        axis=-1,
    )
    uv = np.stack(
        [
            np.broadcast_to((th / TAU)[None, :], (2, SEG_U + 1)),
            np.broadcast_to((ys / h + 0.5)[:, None], (2, SEG_U + 1)),
        ],
        axis=-1,
    )
    chunks = [_grid_mesh(pos, nrm, uv, fix_winding=False)]
    if caps:
        chunks.append(_disk_cap(r, h / 2, +1.0, SEG_U))
        chunks.append(_disk_cap(r, -h / 2, -1.0, SEG_U))
    v, n, u, f = _merge(chunks)
    if caps and signed_volume(v, f) < 0.0:
        f = f[:, [0, 2, 1]]
    return v, n, u, f


def mesh_capsule(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = float(params.get("radius", 0.3))
    h = float(params.get("height", 1.0))
    half = SEG_V // 2
    th = np.linspace(0.0, TAU, SEG_U + 1)
    ct, st = np.cos(th), np.sin(th)
    # Bottom hemisphere rings (phi -pi/2..0) then top hemisphere (phi 0..pi/2);
    # the two equator rings at y = ±h/2 form the cylinder band between them.
    phi = np.concatenate(
        [np.linspace(-math.pi / 2, 0.0, half + 1), np.linspace(0.0, math.pi / 2, half + 1)]
    )
    yoff = np.where(phi >= 0.0, h / 2, -h / 2)
    rows = phi.shape[0]
    cp, sp = np.cos(phi), np.sin(phi)
    pos = np.stack(
        [
            (r * cp)[:, None] * ct[None, :],
            (yoff + r * sp)[:, None] * np.ones(SEG_U + 1),
            (r * cp)[:, None] * st[None, :],
        ],
        axis=-1,
    )
    nrm = np.stack(
        [
            cp[:, None] * ct[None, :],
            np.broadcast_to(sp[:, None], (rows, SEG_U + 1)),
            cp[:, None] * st[None, :],
        ],
        axis=-1,
    )
    v_coord = (yoff + r * sp + h / 2 + r) / (h + 2.0 * r)
    uv = np.stack(
        [
            np.broadcast_to((th / TAU)[None, :], (rows, SEG_U + 1)),
            np.broadcast_to(v_coord[:, None], (rows, SEG_U + 1)),
        ],
        axis=-1,
    )
    return _grid_mesh(pos, nrm, uv)


def mesh_cone(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = float(params.get("radius", 0.5))
    h = float(params.get("height", 1.0))
    th = np.linspace(0.0, TAU, SEG_U + 1)
    ct, st = np.cos(th), np.sin(th)
    t = np.array([0.0, 1.0])  # base → apex
    rr = r * (1.0 - t)
    ys = -h / 2 + h * t
    pos = np.stack(
        [
            rr[:, None] * ct[None, :],
            np.broadcast_to(ys[:, None], (2, SEG_U + 1)),
            rr[:, None] * st[None, :],
        ],
        axis=-1,
    )
    sl = math.hypot(h, r)
    nrm = np.stack(
        [
            np.broadcast_to((h / sl) * ct[None, :], (2, SEG_U + 1)),
            np.full((2, SEG_U + 1), r / sl),
            np.broadcast_to((h / sl) * st[None, :], (2, SEG_U + 1)),
        ],
        axis=-1,
    )
    uv = np.stack(
        [
            np.broadcast_to((th / TAU)[None, :], (2, SEG_U + 1)),
            np.broadcast_to(t[:, None], (2, SEG_U + 1)),
        ],
        axis=-1,
    )
    chunks = [_grid_mesh(pos, nrm, uv, fix_winding=False), _disk_cap(r, -h / 2, -1.0, SEG_U)]
    v, n, u, f = _merge(chunks)
    if signed_volume(v, f) < 0.0:
        f = f[:, [0, 2, 1]]
    return v, n, u, f


def mesh_torus(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = float(params.get("major_radius", 0.5))
    r = float(params.get("minor_radius", 0.15))
    u = np.linspace(0.0, TAU, SEG_U + 1)
    vv = np.linspace(0.0, TAU, SEG_V + 1)
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(vv), np.sin(vv)
    # rows = v (minor), cols = u (major)
    pos = np.stack(
        [
            (R + r * cv[:, None]) * cu[None, :],
            np.broadcast_to((r * sv)[:, None], (SEG_V + 1, SEG_U + 1)),
            (R + r * cv[:, None]) * su[None, :],
        ],
        axis=-1,
    )
    nrm = np.stack(
        [
            cv[:, None] * cu[None, :],
            np.broadcast_to(sv[:, None], (SEG_V + 1, SEG_U + 1)),
            cv[:, None] * su[None, :],
        ],
        axis=-1,
    )
    uv = np.stack(
        [
            np.broadcast_to((u / TAU)[None, :], (SEG_V + 1, SEG_U + 1)),
            np.broadcast_to((vv / TAU)[:, None], (SEG_V + 1, SEG_U + 1)),
        ],
        axis=-1,
    )
    return _grid_mesh(pos, nrm, uv)


def mesh_prism(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sides = max(3, int(params.get("sides", 6)))
    r = float(params.get("radius", 0.5))
    h = float(params.get("height", 1.0))
    ang = np.linspace(0.0, TAU, sides, endpoint=False)
    poly = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=-1)  # (sides, 2) xz
    chunks = []
    for i in range(sides):
        a, b = poly[i], poly[(i + 1) % sides]
        mid = (a + b) / 2.0
        n2 = mid / (np.linalg.norm(mid) + 1e-12)
        v = np.array(
            [
                [a[0], -h / 2, a[1]],
                [b[0], -h / 2, b[1]],
                [b[0], h / 2, b[1]],
                [a[0], h / 2, a[1]],
            ],
            dtype=np.float32,
        )
        n = np.tile(np.array([n2[0], 0.0, n2[1]], dtype=np.float32), (4, 1))
        u = np.array(
            [[i / sides, 0.0], [(i + 1) / sides, 0.0], [(i + 1) / sides, 1.0], [i / sides, 1.0]],
            dtype=np.float32,
        )
        f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        chunks.append((v, n, u, f))
    # Caps: center + polygon ring fans with planar UVs.
    for sign in (+1.0, -1.0):
        y = sign * h / 2
        ring = np.stack([poly[:, 0], np.full(sides, y), poly[:, 1]], axis=-1)
        v = np.concatenate([np.array([[0.0, y, 0.0]]), ring], axis=0).astype(np.float32)
        n = np.tile(np.array([0.0, sign, 0.0], dtype=np.float32), (sides + 1, 1))
        u = np.concatenate(
            [np.array([[0.5, 0.5]]), 0.5 + poly / (2.0 * r)], axis=0
        ).astype(np.float32)
        j = np.arange(1, sides + 1, dtype=np.int64)
        j_next = np.where(j + 1 > sides, 1, j + 1)
        if sign > 0:
            f = np.stack([np.zeros(sides, dtype=np.int64), j, j_next], axis=1)
        else:
            f = np.stack([np.zeros(sides, dtype=np.int64), j_next, j], axis=1)
        chunks.append((v, n, u, f))
    v, n, u, f = _merge(chunks)
    if signed_volume(v, f) < 0.0:
        f = f[:, [0, 2, 1]]
    return v, n, u, f


def mesh_helix(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = float(params.get("radius", 0.4))
    pitch = float(params.get("pitch", 0.2))
    turns = float(params.get("turns", 3.0))
    thick = float(params.get("thickness", 0.05))
    n_rings = max(2, int(round(turns * HELIX_SEG_PER_TURN)) + 1)
    t = np.linspace(0.0, turns, n_rings)
    ang = TAU * t
    center = np.stack(
        [R * np.cos(ang), pitch * t - pitch * turns / 2.0, R * np.sin(ang)], axis=-1
    )
    tan = np.stack(
        [-R * TAU * np.sin(ang), np.full_like(t, pitch), R * TAU * np.cos(ang)], axis=-1
    )
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    nrm_rad = np.stack([np.cos(ang), np.zeros_like(t), np.sin(ang)], axis=-1)
    bnm = np.cross(tan, nrm_rad)
    bnm /= np.linalg.norm(bnm, axis=1, keepdims=True) + 1e-12

    th = np.linspace(0.0, TAU, TUBE_SEG + 1)
    ct, st = np.cos(th), np.sin(th)
    # (n_rings, TUBE_SEG+1, 3) tube cross-section frames.
    ring_n = (
        ct[None, :, None] * nrm_rad[:, None, :] + st[None, :, None] * bnm[:, None, :]
    )
    pos = center[:, None, :] + thick * ring_n
    uv = np.stack(
        [
            np.broadcast_to((t / turns)[:, None], (n_rings, TUBE_SEG + 1)),
            np.broadcast_to((th / TAU)[None, :], (n_rings, TUBE_SEG + 1)),
        ],
        axis=-1,
    )
    tube = _grid_mesh(pos, ring_n, uv, fix_winding=False)
    # Cap both tube ends so the mesh is geometrically closed (physics-friendly).
    caps = []
    for row, sign in ((0, -1.0), (n_rings - 1, +1.0)):
        c = center[row]
        tangent = tan[row] * sign
        ring = pos[row]  # (TUBE_SEG+1, 3)
        v = np.concatenate([c[None, :], ring], axis=0).astype(np.float32)
        n = np.tile(tangent.astype(np.float32)[None, :], (TUBE_SEG + 2, 1))
        u = np.concatenate(
            [np.array([[0.5, 0.5]]),
             np.stack([0.5 + 0.5 * ct, 0.5 + 0.5 * st], axis=-1)],
            axis=0,
        ).astype(np.float32)
        j = np.arange(1, TUBE_SEG + 1, dtype=np.int64)
        if sign > 0:
            f = np.stack([np.zeros(TUBE_SEG, dtype=np.int64), j, j + 1], axis=1)
        else:
            f = np.stack([np.zeros(TUBE_SEG, dtype=np.int64), j + 1, j], axis=1)
        caps.append((v, n, u, f))
    v, n, u, f = _merge([tube, *caps])
    if signed_volume(v, f) < 0.0:
        f = f[:, [0, 2, 1]]
    return v, n, u, f


def mesh_plane(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sx, sz = (float(v) for v in params.get("size", [1.0, 1.0]))
    hx, hz = sx / 2, sz / 2
    v = np.array(
        [[-hx, 0.0, -hz], [-hx, 0.0, hz], [hx, 0.0, hz], [hx, 0.0, -hz]],
        dtype=np.float32,
    )
    n = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (4, 1))
    u = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return v, n, u, f


# ---------------------------------------------------------------------------
# complex-shape kinds (F6)
# ---------------------------------------------------------------------------


def _signed_pow(v: np.ndarray, e: float) -> np.ndarray:
    return np.sign(v) * np.abs(v) ** e


def mesh_superellipsoid(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(cosη)^e1(cosω)^e2 parametrization: rounded box ↔ ellipsoid continuum."""
    rx, ry, rz = (float(v) for v in params.get("radii", [0.5, 0.5, 0.5]))
    e1, e2 = (float(v) for v in params.get("exponents", [1.0, 1.0]))
    eta = np.linspace(-math.pi / 2, math.pi / 2, SEG_V + 1)      # rows (latitude)
    omega = np.linspace(0.0, TAU, SEG_U + 1)                     # cols (longitude)
    ce, se = np.cos(eta), np.sin(eta)
    co, so = np.cos(omega), np.sin(omega)

    sp_ce, sp_se = _signed_pow(ce, e1), _signed_pow(se, e1)
    sp_co, sp_so = _signed_pow(co, e2), _signed_pow(so, e2)
    pos = np.stack(
        [
            rx * sp_ce[:, None] * sp_co[None, :],
            np.broadcast_to(ry * sp_se[:, None], (SEG_V + 1, SEG_U + 1)),
            rz * sp_ce[:, None] * sp_so[None, :],
        ],
        axis=-1,
    )
    # Snap the pole rows exactly (cos(π/2)^e is 1e-10-ish for e < 1, which
    # breaks geometric watertightness at the pole fan).
    pos[0, :] = np.array([0.0, -ry, 0.0])
    pos[-1, :] = np.array([0.0, ry, 0.0])
    # Implicit-gradient normals: ∇F with the common factor dropped,
    # n ∝ (sign(x)|x/rx|^(2/e2−1)/rx, sign(y)|y/ry|^(2/e1−1)/ry, sign(z)|z/rz|^(2/e2−1)/rz).
    # Exact for ellipsoids; well-behaved at sharp edges (e ≥ 2) and flat
    # faces (e < 1) alike. Star-shaped ⇒ the gradient always points outward.
    xr = np.clip(np.abs(pos[..., 0]) / rx, 1e-9, None)
    yr = np.clip(np.abs(pos[..., 1]) / ry, 1e-9, None)
    zr = np.clip(np.abs(pos[..., 2]) / rz, 1e-9, None)
    n = np.stack(
        [
            np.sign(pos[..., 0]) * xr ** (2.0 / e2 - 1.0) / rx,
            np.sign(pos[..., 1]) * yr ** (2.0 / e1 - 1.0) / ry,
            np.sign(pos[..., 2]) * zr ** (2.0 / e2 - 1.0) / rz,
        ],
        axis=-1,
    )
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    # Exact pole normals (the parametrization degenerates there).
    n[0, :] = np.array([0.0, -1.0, 0.0])
    n[-1, :] = np.array([0.0, 1.0, 0.0])
    uv = np.stack(
        [
            np.broadcast_to((omega / TAU)[None, :], (SEG_V + 1, SEG_U + 1)),
            np.broadcast_to(((eta + math.pi / 2) / math.pi)[:, None], (SEG_V + 1, SEG_U + 1)),
        ],
        axis=-1,
    )
    return _grid_mesh(pos, n, uv)


def mesh_tube(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pipe following a polyline path with parallel-transport frames and caps."""
    from .primitives import path_length, tube_path_and_radii, _path_frames

    pts, r1, r2 = tube_path_and_radii(params)
    caps = bool(params.get("caps", True))
    k = pts.shape[0]
    total = path_length(pts) or 1.0
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    r_vert = r1 + (r2 - r1) * (cum / total)
    tan, nrm, bnm = _path_frames(pts)

    seg = 12  # denser ring than the helix: 12-gon keeps ≥95% of the volume
    th = np.linspace(0.0, TAU, seg + 1)
    ct, st = np.cos(th), np.sin(th)
    ring_n = ct[None, :, None] * nrm[:, None, :] + st[None, :, None] * bnm[:, None, :]
    pos = pts[:, None, :] + r_vert[:, None, None] * ring_n
    uv = np.stack(
        [
            np.broadcast_to((cum / total)[:, None], (k, seg + 1)),
            np.broadcast_to((th / TAU)[None, :], (k, seg + 1)),
        ],
        axis=-1,
    )
    tube = _grid_mesh(pos, ring_n, uv, fix_winding=False)
    chunks = [tube]
    if caps:
        for row, sign in ((0, -1.0), (k - 1, +1.0)):
            c = pts[row]
            tangent = tan[row] * sign
            ring = pos[row]
            v = np.concatenate([c[None, :], ring], axis=0).astype(np.float32)
            n = np.tile(tangent.astype(np.float32)[None, :], (seg + 2, 1))
            u = np.concatenate(
                [np.array([[0.5, 0.5]]),
                 np.stack([0.5 + 0.5 * ct, 0.5 + 0.5 * st], axis=-1)],
                axis=0,
            ).astype(np.float32)
            j = np.arange(1, seg + 1, dtype=np.int64)
            if sign > 0:
                f = np.stack([np.zeros(seg, dtype=np.int64), j, j + 1], axis=1)
            else:
                f = np.stack([np.zeros(seg, dtype=np.int64), j + 1, j], axis=1)
            chunks.append((v, n, u, f))
    v, n, u, f = _merge(chunks)
    if signed_volume(v, f) < 0.0:
        f = f[:, [0, 2, 1]]
    return v, n, u, f


def mesh_arch(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A true torus *segment* standing in the XY plane (default: half-torus ∩)."""
    from .primitives import arch_angles

    R, r, start, arc = arch_angles(params)
    caps = bool(params.get("caps", True))
    n_u = max(8, int(round(SEG_U * arc / TAU)))
    u = np.linspace(start, start + arc, n_u + 1)
    vv = np.linspace(0.0, TAU, SEG_V + 1)
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(vv), np.sin(vv)
    # rows = v (tube), cols = u (along the arc)
    pos = np.stack(
        [
            (R + r * cv[:, None]) * cu[None, :],
            (R + r * cv[:, None]) * su[None, :],
            np.broadcast_to((r * sv)[:, None], (SEG_V + 1, n_u + 1)),
        ],
        axis=-1,
    )
    nrm = np.stack(
        [
            cv[:, None] * cu[None, :],
            cv[:, None] * su[None, :],
            np.broadcast_to(sv[:, None], (SEG_V + 1, n_u + 1)),
        ],
        axis=-1,
    )
    uv = np.stack(
        [
            np.broadcast_to(((u - start) / arc)[None, :], (SEG_V + 1, n_u + 1)),
            np.broadcast_to((vv / TAU)[:, None], (SEG_V + 1, n_u + 1)),
        ],
        axis=-1,
    )
    chunks = [_grid_mesh(pos, nrm, uv, fix_winding=False)]
    if caps:
        for col, sign in ((0, -1.0), (n_u, +1.0)):
            u_end = u[col]
            radial = np.array([math.cos(u_end), math.sin(u_end), 0.0])
            centre = np.array([R * math.cos(u_end), R * math.sin(u_end), 0.0])
            # Outward cap normal = ±tangent = ±(−sin u, cos u, 0).
            tangent = sign * np.array([-math.sin(u_end), math.cos(u_end), 0.0])
            ring = pos[:, col]  # (SEG_V+1, 3)
            v = np.concatenate([centre[None, :], ring], axis=0).astype(np.float32)
            n = np.tile(tangent.astype(np.float32)[None, :], (SEG_V + 2, 1))
            uv_cap = np.concatenate(
                [np.array([[0.5, 0.5]]),
                 np.stack([0.5 + 0.5 * cv, 0.5 + 0.5 * sv], axis=-1)],
                axis=0,
            ).astype(np.float32)
            j = np.arange(1, SEG_V + 1, dtype=np.int64)
            if sign > 0:
                f = np.stack([np.zeros(SEG_V, dtype=np.int64), j, j + 1], axis=1)
            else:
                f = np.stack([np.zeros(SEG_V, dtype=np.int64), j + 1, j], axis=1)
            chunks.append((v, n, uv_cap, f))
    v, n, u_, f = _merge(chunks)
    if signed_volume(v, f) < 0.0:
        f = f[:, [0, 2, 1]]
    return v, n, u_, f


def mesh_panel(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Thin sheet of width w, height h, thickness t, bent around the Y axis.

    `bend` is the arc angle in radians (0 → flat plate). The arc is centred
    at (0, ·, Rc) so the mid-surface passes through the origin.
    """
    from .primitives import panel_geometry

    w, h, t, bend = panel_geometry(params)
    if abs(bend) < 1e-9:
        return mesh_box({"size": [w, h, t]})
    ab = abs(bend)
    rc = w / ab
    th = np.linspace(-ab / 2, ab / 2, SEG_U + 1)
    ys = np.linspace(-h / 2, h / 2, 3)

    def sheet(s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = rc + s
        rows = ys.shape[0]
        pos = np.stack(
            [
                np.broadcast_to((rho * np.sin(th))[None, :], (rows, SEG_U + 1)),
                np.broadcast_to(ys[:, None], (rows, SEG_U + 1)),
                np.broadcast_to((rc - rho * np.cos(th))[None, :], (rows, SEG_U + 1)),
            ],
            axis=-1,
        )
        nrm = np.stack(
            [
                np.broadcast_to((np.sign(s) * np.sin(th))[None, :], (rows, SEG_U + 1)),
                np.zeros((rows, SEG_U + 1)),
                np.broadcast_to((-np.sign(s) * np.cos(th))[None, :], (rows, SEG_U + 1)),
            ],
            axis=-1,
        )
        uv = np.stack(
            [
                np.broadcast_to(((th + ab / 2) / ab)[None, :], (rows, SEG_U + 1)),
                np.broadcast_to((ys / h + 0.5)[:, None], (rows, SEG_U + 1)),
            ],
            axis=-1,
        )
        return pos, nrm, uv

    chunks = []
    # Front (s = +t/2) and back (s = −t/2) curved faces.
    chunks.append(_grid_mesh(*sheet(+t / 2), fix_winding=False))
    chunks.append(_grid_mesh(*sheet(-t / 2), fix_winding=False))
    # u-edges at θ = ±ab/2 (flat rectangles h × t).
    for u_end, n_sign in ((-ab / 2, -1.0), (+ab / 2, +1.0)):
        edge_n = np.array(
            [n_sign * math.cos(u_end), 0.0, n_sign * math.sin(u_end)], dtype=np.float64
        )
        s_vals = np.array([-t / 2, t / 2])
        rows, cols = ys.shape[0], 2
        pos = np.zeros((rows, cols, 3))
        for ci, s in enumerate(s_vals):
            rho = rc + s
            pos[:, ci, 0] = rho * math.sin(u_end)
            pos[:, ci, 1] = ys
            pos[:, ci, 2] = rc - rho * math.cos(u_end)
        n = np.tile(edge_n, (rows, cols, 1))
        u = np.stack(
            [
                np.broadcast_to(((s_vals + t / 2) / t)[None, :], (rows, cols)),
                np.broadcast_to((ys / h + 0.5)[:, None], (rows, cols)),
            ],
            axis=-1,
        )
        chunks.append(_grid_mesh(pos, n, u, fix_winding=False))
    # y-edges at y = ±h/2 (curved strips, arc length w × t).
    for y_end, n_sign in ((+h / 2, +1.0), (-h / 2, -1.0)):
        s_vals = np.array([t / 2, -t / 2]) if n_sign > 0 else np.array([-t / 2, t / 2])
        rows, cols = 2, SEG_U + 1
        pos = np.zeros((rows, cols, 3))
        for ri, s in enumerate(s_vals):
            rho = rc + s
            pos[ri, :, 0] = rho * np.sin(th)
            pos[ri, :, 1] = y_end
            pos[ri, :, 2] = rc - rho * np.cos(th)
        n = np.tile(np.array([0.0, n_sign, 0.0]), (rows, cols, 1))
        u = np.stack(
            [
                np.broadcast_to(((th + ab / 2) / ab)[None, :], (rows, cols)),
                np.broadcast_to(((s_vals + t / 2) / t)[:, None], (rows, cols)),
            ],
            axis=-1,
        )
        chunks.append(_grid_mesh(pos, n, u, fix_winding=False))
    # Orient every chunk so its faces agree with the analytic normals, then
    # assemble (normals are authoritative; winding follows).
    oriented = [(v, n, u, _orient_like_normals(v, n, f)) for v, n, u, f in chunks]
    v, n, u_, f = _merge(oriented)
    # bend < 0 mirrors the arc through the z=0 plane (curves toward −z).
    if bend < 0:
        v = v.copy()
        n = n.copy()
        v[:, 2] *= -1.0
        n[:, 2] *= -1.0
        f = f[:, [0, 2, 1]]
    if signed_volume(v, f) < 0.0:
        f = f[:, [0, 2, 1]]
    return v, n, u_, f


MESH_BUILDERS = {
    "box": mesh_box,
    "sphere": mesh_sphere,
    "cylinder": mesh_cylinder,
    "capsule": mesh_capsule,
    "cone": mesh_cone,
    "torus": mesh_torus,
    "ellipsoid": mesh_ellipsoid,
    "prism": mesh_prism,
    "helix": mesh_helix,
    "plane": mesh_plane,
    "superellipsoid": mesh_superellipsoid,
    "tube": mesh_tube,
    "sweep": mesh_tube,
    "arch": mesh_arch,
    "panel": mesh_panel,
}


# ---------------------------------------------------------------------------
# analytic solid volumes (local space)
# ---------------------------------------------------------------------------


def primitive_solid_volume(kind: str, params: dict) -> float:
    """Exact solid volume of the local (untransformed) primitive in m^3."""
    if kind == "box":
        sx, sy, sz = (float(v) for v in params.get("size", [1.0, 1.0, 1.0]))
        return sx * sy * sz
    if kind == "sphere":
        r = float(params.get("radius", 0.5))
        return 4.0 / 3.0 * math.pi * r ** 3
    if kind == "cylinder":
        r = float(params.get("radius", 0.4))
        h = float(params.get("height", 1.0))
        return math.pi * r * r * h
    if kind == "capsule":
        r = float(params.get("radius", 0.3))
        h = float(params.get("height", 1.0))
        return math.pi * r * r * h + 4.0 / 3.0 * math.pi * r ** 3
    if kind == "cone":
        r = float(params.get("radius", 0.5))
        h = float(params.get("height", 1.0))
        return math.pi * r * r * h / 3.0
    if kind == "torus":
        R = float(params.get("major_radius", 0.5))
        r = float(params.get("minor_radius", 0.15))
        return 2.0 * math.pi * math.pi * R * r * r
    if kind == "ellipsoid":
        rx, ry, rz = (float(v) for v in params.get("radii", [0.5, 0.5, 0.5]))
        return 4.0 / 3.0 * math.pi * rx * ry * rz
    if kind == "prism":
        sides = max(3, int(params.get("sides", 6)))
        r = float(params.get("radius", 0.5))
        h = float(params.get("height", 1.0))
        base = 0.5 * sides * r * r * math.sin(TAU / sides)
        return base * h
    if kind == "helix":
        R = float(params.get("radius", 0.4))
        pitch = float(params.get("pitch", 0.2))
        turns = float(params.get("turns", 3.0))
        thick = float(params.get("thickness", 0.05))
        length = turns * math.hypot(TAU * R, pitch)
        return math.pi * thick * thick * length
    if kind == "plane":
        return 0.0  # zero-thickness surface
    if kind == "superellipsoid":
        rx, ry, rz = (float(v) for v in params.get("radii", [0.5, 0.5, 0.5]))
        e1, e2 = (float(v) for v in params.get("exponents", [1.0, 1.0]))
        # V = (2/3) e1 e2 rx ry rz B(e2/2, e2/2) B(e1, e1/2).
        # (e=1 → 4π/3 r³ sphere; e=2 → 4/3 octahedron; e→0 → 8 rx ry rz box.)
        b1 = math.gamma(e2 / 2) ** 2 / math.gamma(e2)
        b2 = math.gamma(e1) * math.gamma(e1 / 2) / math.gamma(3 * e1 / 2)
        return (2.0 / 3.0) * e1 * e2 * rx * ry * rz * b1 * b2
    if kind in ("tube", "sweep"):
        from .primitives import path_length, tube_path_and_radii

        pts, r1, r2 = tube_path_and_radii(params)
        L = path_length(pts)
        # Frustum volume (exact for constant radius).
        return math.pi * L * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0
    if kind == "arch":
        from .primitives import arch_angles

        R, r, _, arc = arch_angles(params)
        return math.pi * r * r * R * arc
    if kind == "panel":
        from .primitives import panel_geometry

        w, h, t, _ = panel_geometry(params)
        return w * h * t  # exact for any bend
    return 0.0


def local_aabb(kind: str, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Exact local-space AABB of a primitive (before its transform)."""
    if kind == "box":
        sx, sy, sz = (float(v) for v in params.get("size", [1.0, 1.0, 1.0]))
        e = np.array([sx, sy, sz]) / 2.0
    elif kind == "sphere":
        e = np.full(3, float(params.get("radius", 0.5)))
    elif kind == "cylinder":
        r = float(params.get("radius", 0.4))
        h = float(params.get("height", 1.0))
        e = np.array([r, h / 2, r])
    elif kind == "capsule":
        r = float(params.get("radius", 0.3))
        h = float(params.get("height", 1.0))
        e = np.array([r, h / 2 + r, r])
    elif kind == "cone":
        r = float(params.get("radius", 0.5))
        h = float(params.get("height", 1.0))
        e = np.array([r, h / 2, r])
    elif kind == "torus":
        R = float(params.get("major_radius", 0.5))
        r = float(params.get("minor_radius", 0.15))
        e = np.array([R + r, r, R + r])
    elif kind == "ellipsoid":
        e = np.asarray(params.get("radii", [0.5, 0.5, 0.5]), dtype=np.float64)
    elif kind == "prism":
        r = float(params.get("radius", 0.5))
        h = float(params.get("height", 1.0))
        e = np.array([r, h / 2, r])
    elif kind == "helix":
        R = float(params.get("radius", 0.4))
        pitch = float(params.get("pitch", 0.2))
        turns = float(params.get("turns", 3.0))
        thick = float(params.get("thickness", 0.05))
        e = np.array([R + thick, pitch * turns / 2.0 + thick, R + thick])
    elif kind == "plane":
        sx, sz = (float(v) for v in params.get("size", [1.0, 1.0]))
        e = np.array([sx / 2, 0.0, sz / 2])
    elif kind == "superellipsoid":
        e = np.asarray(params.get("radii", [0.5, 0.5, 0.5]), dtype=np.float64)
    elif kind in ("tube", "sweep"):
        from .primitives import tube_path_and_radii

        pts, r1, r2 = tube_path_and_radii(params)
        rmax = max(r1, r2)
        return pts.min(axis=0) - rmax, pts.max(axis=0) + rmax
    elif kind == "arch":
        from .primitives import arch_angles

        R, r, start, arc = arch_angles(params)
        us = np.linspace(start, start + arc, 65)
        xs = R * np.cos(us)
        ys = R * np.sin(us)
        lo = np.array([xs.min() - r, ys.min() - r, -r])
        hi = np.array([xs.max() + r, ys.max() + r, r])
        return lo, hi
    elif kind == "panel":
        from .primitives import panel_geometry

        w, h, t, bend = panel_geometry(params)
        if abs(bend) < 1e-9:
            e = np.array([w / 2, h / 2, t / 2])
        else:
            rc = w / abs(bend)
            lo = np.array([
                -(rc + t / 2) * math.sin(abs(bend) / 2),
                -h / 2,
                -t / 2,
            ])
            hi = np.array([
                (rc + t / 2) * math.sin(abs(bend) / 2),
                h / 2,
                rc - (rc - t / 2) * math.cos(abs(bend) / 2),
            ])
            if bend < 0:  # mirrored through z=0
                lo[2], hi[2] = -hi[2], -lo[2]
            return lo, hi
    else:
        e = np.zeros(3)
    return -e, e


# ---------------------------------------------------------------------------
# CSG-lite subtraction (F7): straight-through tunnels in extruded hosts
# ---------------------------------------------------------------------------


def is_cutter(prim) -> bool:
    """A primitive marked `params["role"] == "subtract"` carves its host(s)."""
    return str((getattr(prim, "params", None) or {}).get("role", "")).lower() == "subtract"


_ALIGN_TOL = 0.999       # cutter axis must be within ~2.6° of a host axis
_MARGIN_REL = 1e-3       # hole must stay this far inside the host cross-section
HOLE_SEG_PER_EDGE = 8    # boundary subdivisions per host cross-section edge


def _ray_rect_distance(theta: float, hx: float, hy: float) -> float:
    """Distance from the origin to a centred rectangle boundary along `theta`."""
    dx, dy = math.cos(theta), math.sin(theta)
    t = math.inf
    if dx > 1e-12:
        t = min(t, hx / dx)
    elif dx < -1e-12:
        t = min(t, -hx / dx)
    if dy > 1e-12:
        t = min(t, hy / dy)
    elif dy < -1e-12:
        t = min(t, -hy / dy)
    return t


def _ray_polygon_distance(theta: float, centre: np.ndarray, poly: np.ndarray) -> float:
    """Distance from an interior point to a convex polygon boundary along `theta`."""
    d = np.array([math.cos(theta), math.sin(theta)])
    best = math.inf
    k = len(poly)
    for i in range(k):
        p0 = poly[i]
        e = poly[(i + 1) % k] - p0
        det = d[0] * (-e[1]) - (-e[0]) * d[1]
        if abs(det) < 1e-15:
            continue
        rhs = p0 - centre
        t = (rhs[0] * (-e[1]) - (-e[0]) * rhs[1]) / det
        u = (d[0] * rhs[1] - rhs[0] * d[1]) / det
        if t > 1e-12 and -1e-9 <= u <= 1.0 + 1e-9:
            best = min(best, t)
    return best


def _point_in_convex_polygon(pt: np.ndarray, poly: np.ndarray, margin: float = 0.0) -> bool:
    """Strict containment test for a CCW convex polygon (with optional margin)."""
    k = len(poly)
    for i in range(k):
        a = poly[i]
        b = poly[(i + 1) % k]
        e = b - a
        if e[0] * (pt[1] - a[1]) - e[1] * (pt[0] - a[0]) < margin:
            return False
    return True


def _edge_distance(pt: np.ndarray, poly: np.ndarray) -> float:
    """Minimum distance from an interior point to any polygon edge."""
    k = len(poly)
    best = math.inf
    for i in range(k):
        a = poly[i]
        e = poly[(i + 1) % k] - a
        t = float(np.clip(np.dot(pt - a, e) / (np.dot(e, e) + 1e-30), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(pt - (a + t * e))))
    return best


class _Hole:
    """Cross-section of a straight tunnel: ellipse or axis-aligned rect."""

    def __init__(self, shape: str, centre, a: float, b: float):
        self.shape = shape
        self.centre = np.asarray(centre, dtype=np.float64)
        self.a = float(a)   # semi-axis / half extent along u
        self.b = float(b)   # along v

    @property
    def area(self) -> float:
        return math.pi * self.a * self.b if self.shape == "ellipse" else 4.0 * self.a * self.b

    @property
    def max_radius(self) -> float:
        return max(self.a, self.b)

    def radius(self, theta: float) -> float:
        if self.shape == "ellipse":
            return 1.0 / math.hypot(math.cos(theta) / self.a, math.sin(theta) / self.b)
        return _ray_rect_distance(theta, self.a, self.b)

    def outward_normal(self, theta: float) -> np.ndarray:
        """Boundary normal pointing out of the hole (into the solid)."""
        if self.shape == "ellipse":
            r = self.radius(theta)
            n = np.array([r * math.cos(theta) / (self.a * self.a),
                          r * math.sin(theta) / (self.b * self.b)])
            return n / (np.linalg.norm(n) + 1e-30)
        # Rect: sector lookup between the corner angles (CCW).
        corners = [(-self.a, -self.b), (self.a, -self.b), (self.a, self.b), (-self.a, self.b)]
        normals = [np.array([0.0, -1.0]), np.array([1.0, 0.0]),
                   np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
        cang = np.unwrap(np.arctan2([c[1] for c in corners], [c[0] for c in corners]))
        th = theta
        while th < cang[0]:
            th += TAU
        while th >= cang[0] + TAU:
            th -= TAU
        for i in range(4):
            a1 = cang[i + 1] if i < 3 else cang[0] + TAU
            if th <= a1 + 1e-12:
                near0 = abs(th - cang[i]) < 1e-6
                near1 = abs(th - a1) < 1e-6
                if near0 and not near1:
                    return (normals[i - 1] + normals[i]) / 2 ** 0.5
                if near1:
                    return (normals[i] + normals[(i + 1) % 4]) / 2 ** 0.5
                return normals[i]
        return normals[0]


def _boundary_loops(
    poly: np.ndarray, hole: _Hole, seg_per_edge: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Subdivided outer (polygon) + inner (hole) boundary loops on a shared
    angle list. Loops are explicitly closed (last entry repeats the first).
    Returns (outer, inner, thetas, edge_spans)."""
    centre = hole.centre
    k = len(poly)
    rel = poly - centre
    ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    outer: list[np.ndarray] = []
    inner: list[np.ndarray] = []
    thetas: list[float] = []
    spans: list[tuple[int, int]] = []
    for i in range(k):
        a0 = ang[i]
        a1 = ang[i + 1] if i + 1 < k else ang[0] + TAU
        start = len(thetas)
        for th in np.linspace(a0, a1, seg_per_edge + 2):
            d = np.array([math.cos(th), math.sin(th)])
            thetas.append(float(th))
            outer.append(centre + _ray_polygon_distance(float(th), centre, poly) * d)
            inner.append(centre + hole.radius(float(th)) * d)
        spans.append((start, len(thetas) - 1))
    return (
        np.asarray(outer, dtype=np.float64),
        np.asarray(inner, dtype=np.float64),
        np.asarray(thetas, dtype=np.float64),
        spans,
    )


def _mesh_extruded_with_hole(
    poly: np.ndarray,
    half_len: float,
    hole: _Hole,
    seg_per_edge: int = HOLE_SEG_PER_EDGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Closed mesh of `poly` × [−half_len, +half_len] with a straight tunnel.

    Canonical frame: cross-section in (u, v), extrusion along w. Returns
    (vertices, normals, uvs, faces, carved solid volume).
    """
    outer, inner, thetas, spans = _boundary_loops(poly, hole, seg_per_edge)
    n_loop = outer.shape[0]
    poly_min = poly.min(axis=0)
    poly_ext = np.maximum(poly.max(axis=0) - poly_min, 1e-12)

    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    # --- caps: bridge outer ↔ inner loops at w = ±half_len -----------------
    for w_val, n_sign in ((half_len, 1.0), (-half_len, -1.0)):
        ring = np.concatenate([outer, inner], axis=0)  # (2L, 2)
        v = np.stack([ring[:, 0], ring[:, 1], np.full(2 * n_loop, w_val)], axis=-1)
        n = np.tile(np.array([0.0, 0.0, n_sign]), (2 * n_loop, 1))
        uv = (ring - poly_min) / poly_ext
        j = np.arange(n_loop - 1, dtype=np.int64)
        f = np.concatenate(
            [
                np.stack([j, j + 1, n_loop + j + 1], axis=1),
                np.stack([j, n_loop + j + 1, n_loop + j], axis=1),
            ],
            axis=0,
        )
        chunks.append((v, n, uv.astype(np.float32), f))

    # --- side strips: one per polygon edge, subdivided to match the caps ---
    k = len(poly)
    for i in range(k):
        s0, s1 = spans[i]
        strip = outer[s0 : s1 + 1]  # (m, 2) along the edge, corners included
        m = strip.shape[0]
        a = poly[i]
        e = poly[(i + 1) % k] - a
        edge_len = float(np.linalg.norm(e)) + 1e-30
        outward = np.array([e[1], -e[0]]) / edge_len  # CCW polygon → right-hand normal
        t_par = ((strip - a) @ e) / (edge_len * edge_len)
        v = np.concatenate(
            [
                np.stack([strip[:, 0], strip[:, 1], np.full(m, half_len)], axis=-1),
                np.stack([strip[:, 0], strip[:, 1], np.full(m, -half_len)], axis=-1),
            ],
            axis=0,
        )
        n = np.tile(np.array([outward[0], outward[1], 0.0]), (2 * m, 1))
        uv = np.concatenate(
            [
                np.stack([t_par, np.ones(m)], axis=-1),
                np.stack([t_par, np.zeros(m)], axis=-1),
            ],
            axis=0,
        )
        j = np.arange(m - 1, dtype=np.int64)
        f = np.concatenate(
            [
                np.stack([j, j + 1, m + j + 1], axis=1),
                np.stack([j, m + j + 1, m + j], axis=1),
            ],
            axis=0,
        )
        chunks.append((v, n, uv.astype(np.float32), f))

    # --- tunnel wall: inward normals ---------------------------------------
    m = n_loop
    v = np.concatenate(
        [
            np.stack([inner[:, 0], inner[:, 1], np.full(m, half_len)], axis=-1),
            np.stack([inner[:, 0], inner[:, 1], np.full(m, -half_len)], axis=-1),
        ],
        axis=0,
    )
    n2d = np.stack([hole.outward_normal(float(th)) for th in thetas], axis=0)
    n = np.concatenate([-n2d, np.zeros((m, 1))], axis=-1)
    n = np.tile(n, (2, 1))
    th0 = thetas[0]
    u_par = (thetas - th0) / TAU
    uv = np.concatenate(
        [
            np.stack([u_par, np.ones(m)], axis=-1),
            np.stack([u_par, np.zeros(m)], axis=-1),
        ],
        axis=0,
    )
    j = np.arange(m - 1, dtype=np.int64)
    f = np.concatenate(
        [
            np.stack([j, j + 1, m + j + 1], axis=1),
            np.stack([j, m + j + 1, m + j], axis=1),
        ],
        axis=0,
    )
    chunks.append((v, n, uv.astype(np.float32), f))

    # Orient every chunk to its analytic normals, then assemble.
    oriented = [(cv, cn, cu, _orient_like_normals(cv, cn, cf)) for cv, cn, cu, cf in chunks]
    v, n, u, f = _merge(oriented)

    poly_area = 0.5 * abs(
        float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
    )
    volume = (poly_area - hole.area) * 2.0 * half_len
    return (
        v.astype(np.float32),
        n.astype(np.float32),
        u.astype(np.float32),
        f,
        volume,
    )


def _cutter_hole_candidates(
    cutter_kind: str, cutter_params: dict, T_rel: np.ndarray
) -> list[tuple[int, _Hole, float, float]]:
    """Describe possible straight tunnels a cutter makes, in the host's local
    frame. Returns (axis, hole, cutter_length_along_axis, centre_along_axis)."""
    M = np.asarray(T_rel, dtype=np.float64)[:3, :3]
    ctr = np.asarray(T_rel, dtype=np.float64)[:3, 3]
    out: list[tuple[int, _Hole, float, float]] = []
    if cutter_kind == "cylinder":
        r = float(cutter_params.get("radius", 0.4))
        h = float(cutter_params.get("height", 1.0))
        d = M @ np.array([0.0, 1.0, 0.0])
        axis = int(np.argmax(np.abs(d)))
        if abs(d[axis]) < _ALIGN_TOL * (np.linalg.norm(d) + 1e-30):
            return []
        others = [i for i in range(3) if i != axis]
        ra = r * max(abs(M[others[0], 0]), abs(M[others[0], 2]))
        rb = r * max(abs(M[others[1], 0]), abs(M[others[1], 2]))
        length = h * abs(M[axis, 1])
        if min(ra, rb) <= 1e-9 or length <= 1e-9:
            return []
        out.append((axis, _Hole("ellipse", [ctr[others[0]], ctr[others[1]]], ra, rb),
                    length, float(ctr[axis])))
    elif cutter_kind == "box":
        size = [float(v) for v in cutter_params.get("size", [1.0, 1.0, 1.0])]
        col_axis: dict[int, int] = {}
        for j in range(3):
            col = M[:, j]
            nrm = float(np.linalg.norm(col))
            if nrm < 1e-12:
                return []
            a = int(np.argmax(np.abs(col)))
            if abs(col[a]) < _ALIGN_TOL * nrm or a in col_axis.values():
                return []
            col_axis[j] = a
        for j, a in col_axis.items():
            others = [i for i in range(3) if i != a]
            jothers = [jj for jj in range(3) if jj != j]
            length = size[j] * abs(M[a, j])
            ha = size[jothers[0]] / 2 * abs(M[others[0], jothers[0]])
            hb = size[jothers[1]] / 2 * abs(M[others[1], jothers[1]])
            if min(ha, hb) <= 1e-9 or length <= 1e-9:
                continue
            out.append((a, _Hole("rect", [ctr[others[0]], ctr[others[1]]], ha, hb),
                        length, float(ctr[a])))
    return out


def carve_straight_hole(
    host_kind: str,
    host_params: dict,
    cutter_kind: str,
    cutter_params: dict,
    T_rel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Carve a straight tunnel through an extruded host (box / flat panel /
    prism) with an axis-aligned cylinder or box cutter.

    Returns (vertices, normals, uvs, faces, carved volume) in the host's
    local frame, or None when the case is unsupported (the caller then falls
    back to point-cloud-level subtraction). Containment is validated with a
    margin so the tunnel can never sever the host into orphan fragments.
    """
    candidates = _cutter_hole_candidates(cutter_kind, cutter_params, T_rel)
    if not candidates:
        return None

    # Canonical host cross-sections per supported axis.
    host_sections: list[tuple[int, np.ndarray, float]] = []
    if host_kind == "box":
        sx, sy, sz = (float(v) for v in host_params.get("size", [1.0, 1.0, 1.0]))
        half = [sx / 2, sy / 2, sz / 2]
        for a in range(3):
            b, c = [i for i in range(3) if i != a]
            poly = np.array(
                [[-half[b], -half[c]], [half[b], -half[c]],
                 [half[b], half[c]], [-half[b], half[c]]],
                dtype=np.float64,
            )
            host_sections.append((a, poly, half[a]))
    elif host_kind == "panel":
        from .primitives import panel_geometry

        w, h, t, bend = panel_geometry(host_params)
        if abs(bend) >= 1e-9:
            return None  # bent panel: no straight tunnel — point level only
        poly = np.array(
            [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]],
            dtype=np.float64,
        )
        host_sections.append((2, poly, t / 2))
    elif host_kind == "prism":
        sides = max(3, int(host_params.get("sides", 6)))
        r = float(host_params.get("radius", 0.5))
        h = float(host_params.get("height", 1.0))
        ang = np.linspace(0.0, TAU, sides, endpoint=False)
        poly = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=-1)
        host_sections.append((1, poly, h / 2))
    else:
        return None

    for axis, hole, length, ctr_a in candidates:
        for host_axis, poly, half_len in host_sections:
            if host_axis != axis:
                continue
            # Must pass straight through (blind holes unsupported).
            if ctr_a - length / 2 > -half_len + 1e-9 or ctr_a + length / 2 < half_len - 1e-9:
                continue
            # Containment with margin → the host is never severed.
            margin = max(1e-6, _MARGIN_REL * float(np.abs(poly).max()))
            if not _point_in_convex_polygon(hole.centre, poly, margin=margin):
                continue
            if _edge_distance(hole.centre, poly) < hole.max_radius + margin:
                continue
            v, n, u, f, vol = _mesh_extruded_with_hole(poly, half_len, hole)
            # Map canonical (u, v, w) → host (x, y, z).
            b, c = [i for i in range(3) if i != axis]
            vw = np.zeros_like(v)
            nw = np.zeros_like(n)
            vw[:, b], vw[:, c], vw[:, axis] = v[:, 0], v[:, 1], v[:, 2]
            nw[:, b], nw[:, c], nw[:, axis] = n[:, 0], n[:, 1], n[:, 2]
            if signed_volume(vw, f) < 0.0:
                f = f[:, [0, 2, 1]]
            return vw, nw, u, f, vol
    return None


def apply_transform(
    vertices: np.ndarray, normals: np.ndarray, T: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 4x4 row-major transform; normals via the inverse-transpose."""
    T = np.asarray(T, dtype=np.float64)
    h = np.concatenate([vertices.astype(np.float64), np.ones((vertices.shape[0], 1))], axis=1)
    w = (h @ T.T)[:, :3]
    M = T[:3, :3]
    det = float(np.linalg.det(M))
    if abs(det) < 1e-12:
        n = normals.astype(np.float64).copy()
    else:
        # Row-vector form of n' = (M^-1)^T n is n @ M^-1.
        n = normals.astype(np.float64) @ np.linalg.inv(M)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return w.astype(np.float32), n.astype(np.float32)


def build_part_mesh(kind: str, params: dict, transform, label: str, material: str) -> AnalyticPart:
    """Build one transformed analytic mesh for a single primitive."""
    builder = MESH_BUILDERS[kind]
    v, n, uv, f = builder(params)
    T = np.asarray(transform, dtype=np.float64)
    vw, nw = apply_transform(v, n, T)
    det = abs(float(np.linalg.det(T[:3, :3])))
    volume = primitive_solid_volume(kind, params) * (det if det > 1e-12 else 1.0)
    return AnalyticPart(
        label=label,
        kind=kind,
        material=material,
        vertices=vw,
        normals=nw,
        uvs=uv,
        faces=f,
        aabb_min=vw.min(axis=0),
        aabb_max=vw.max(axis=0),
        solid_volume_m3=volume,
    )


def part_material_name(spec, prim) -> str:
    """Resolve a primitive's material hint exactly like the compositor does."""
    from .textures import shape_default_material

    mat = (prim.params or {}).get("material")
    if not isinstance(mat, str) or not mat.strip():
        mat = shape_default_material(getattr(spec, "shape", ""), prim.label)
    if not mat:
        return "default"
    return str(mat).strip().lower()


def build_spec_meshes_with_report(spec) -> tuple[list[AnalyticPart], list[str]]:
    """Build analytic part meshes, applying `role: "subtract"` cutters.

    A cutter whose `params["target"]` names a part carves only that part;
    otherwise it carves the first supported host whose frame contains it.
    Unsupported combinations fall back to point-cloud-level subtraction in
    the compositor (a warning explains which path was taken).
    """
    warnings: list[str] = []
    prims = list(getattr(spec, "primitives", []) or [])
    cutters = [(i, p) for i, p in enumerate(prims) if is_cutter(p)]
    consumed: set[int] = set()
    parts: list[AnalyticPart] = []
    for i, prim in enumerate(prims):
        if is_cutter(prim) or prim.kind not in MESH_BUILDERS:
            continue
        label = prim.label or f"{prim.kind}_{i}"
        material = part_material_name(spec, prim)
        T = prim.transform_matrix()
        carved: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None = None
        carved_ci: int | None = None
        for ci, cut in cutters:
            if ci in consumed:
                continue
            target = str((cut.params or {}).get("target", "") or "")
            if target and target != label:
                continue
            T_rel = np.linalg.inv(np.asarray(T, dtype=np.float64)) @ np.asarray(
                cut.transform_matrix(), dtype=np.float64
            )
            carved = carve_straight_hole(
                prim.kind, prim.params or {}, cut.kind, cut.params or {}, T_rel
            )
            if carved is not None:
                carved_ci = ci
                break
        if carved is not None and carved_ci is not None:
            v, n, uv, f, vol = carved
            vw, nw = apply_transform(v, n, T)
            det = abs(float(np.linalg.det(np.asarray(T, dtype=np.float64)[:3, :3])))
            parts.append(
                AnalyticPart(
                    label=label,
                    kind=prim.kind,
                    material=material,
                    vertices=vw,
                    normals=nw,
                    uvs=uv,
                    faces=f,
                    aabb_min=vw.min(axis=0),
                    aabb_max=vw.max(axis=0),
                    solid_volume_m3=vol * (det if det > 1e-12 else 1.0),
                )
            )
            consumed.add(carved_ci)
            cut_label = prims[carved_ci].label or f"{prims[carved_ci].kind}_{carved_ci}"
            warnings.append(f"subtract: {cut_label!r} carved a tunnel through {label!r}")
            continue
        parts.append(
            build_part_mesh(prim.kind, prim.params or {}, T, label, material)
        )
    for ci, cut in cutters:
        if ci in consumed:
            continue
        cut_label = cut.label or f"{cut.kind}_{ci}"
        if _cutter_overlaps_any_host(cut, prims):
            warnings.append(
                f"subtract: cutter {cut_label!r} has no supported mesh-level host "
                "(unsupported shape, misaligned axis, or containment margin) — "
                "point-cloud subtraction still applies"
            )
        else:
            warnings.append(
                f"subtract: cutter {cut_label!r} overlaps no host part — nothing carved"
            )
    return parts, warnings


def _cutter_overlaps_any_host(cutter, prims: list) -> bool:
    """World-AABB overlap between a cutter and any non-cutter primitive."""

    def world_aabb(prim) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = local_aabb(prim.kind, prim.params or {})
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        T = np.asarray(prim.transform_matrix(), dtype=np.float64)
        world = (np.concatenate([corners, np.ones((8, 1))], axis=1) @ T.T)[:, :3]
        return world.min(axis=0), world.max(axis=0)

    clo, chi = world_aabb(cutter)
    for prim in prims:
        if prim is cutter or is_cutter(prim):
            continue
        lo, hi = world_aabb(prim)
        if np.all(clo <= hi) and np.all(chi >= lo):
            return True
    return False


def build_spec_meshes(spec) -> list[AnalyticPart]:
    """Build all analytic part meshes for a GenerationSpec (unknown kinds skipped)."""
    parts, warnings = build_spec_meshes_with_report(spec)
    for w in warnings:
        _log.info("build_spec_meshes: %s", w)
    return parts
