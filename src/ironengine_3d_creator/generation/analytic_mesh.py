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

import math
from dataclasses import dataclass

import numpy as np

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
    else:
        e = np.zeros(3)
    return -e, e


# ---------------------------------------------------------------------------
# transform + spec-level assembly
# ---------------------------------------------------------------------------


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


def build_spec_meshes(spec) -> list[AnalyticPart]:
    """Build all analytic part meshes for a GenerationSpec (unknown kinds skipped)."""
    parts: list[AnalyticPart] = []
    for i, prim in enumerate(getattr(spec, "primitives", []) or []):
        if prim.kind not in MESH_BUILDERS:
            continue
        label = prim.label or f"{prim.kind}_{i}"
        parts.append(
            build_part_mesh(
                prim.kind,
                prim.params or {},
                prim.transform_matrix(),
                label,
                part_material_name(spec, prim),
            )
        )
    return parts
