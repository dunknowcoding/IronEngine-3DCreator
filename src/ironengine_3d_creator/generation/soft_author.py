"""Non-rigid object authoring: cloth sheets, ropes/cables, frangible vessels,
and articulated humanoid ragdolls (iemodel/3).

Each generator returns a :class:`SoftAuthorResult` bundling the standard
pipeline outputs:

- ``positions`` / ``colors`` / ``labels`` — the point cloud, exactly as
  ``generation.compositor.generate`` would produce,
- ``parts`` — exact analytic meshes (:class:`AnalyticPart`) for GLB export via
  ``core.exporter.write_glb_parts`` (one named node per part, per-part
  materials),
- ``spec`` — a :class:`GenerationSpec` facade so
  ``core.manifest.build_manifest`` measures AABBs / parts / materials through
  the existing code path,
- ``extras`` — the iemodel/3 non-rigid manifest blocks: ``physics.body_type``
  plus ``soft_body`` / ``fracture`` / ``articulation``.

Typical usage::

    result = author_cloth(material="cotton", resolution=(24, 16))
    result.write_glb("towel.glb")
    manifest = result.build_manifest(mesh_path="towel.glb")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..alignment.schema import GenerationSpec, Primitive
from .analytic_mesh import (
    AnalyticPart,
    build_spec_meshes,
    signed_volume,
)
from .colorize import albedo_colors, base_color
from .materials import MATERIAL_PRESETS, default_preset

TAU = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Material parameter tables (soft-body tuning lives here; the PBR/physics
# presets in generation.materials stay untouched).
# ---------------------------------------------------------------------------

# Cloth fabrics: area density and normalized (0..1) solver stiffness values.
CLOTH_FABRICS: dict[str, dict[str, float]] = {
    #            area_kg_m2  stretch  bend  damping
    "silk":   {"area_density_kg_m2": 0.08, "stretch_stiffness": 0.60, "bend_stiffness": 0.15, "damping": 0.04},
    "nylon":  {"area_density_kg_m2": 0.12, "stretch_stiffness": 0.70, "bend_stiffness": 0.20, "damping": 0.05},
    "linen":  {"area_density_kg_m2": 0.25, "stretch_stiffness": 0.85, "bend_stiffness": 0.25, "damping": 0.05},
    "cotton": {"area_density_kg_m2": 0.35, "stretch_stiffness": 0.80, "bend_stiffness": 0.30, "damping": 0.06},
    "wool":   {"area_density_kg_m2": 0.40, "stretch_stiffness": 0.50, "bend_stiffness": 0.35, "damping": 0.10},
    "denim":  {"area_density_kg_m2": 0.55, "stretch_stiffness": 0.95, "bend_stiffness": 0.60, "damping": 0.08},
}

# Rope / cable materials: linear density, stiffness, and the closest PBR
# preset from generation.materials used for rendering + the manifest facade.
ROPE_MATERIALS: dict[str, dict] = {
    #             kg/m    stretch  bend  damping  manifest preset
    "nylon": {"linear_density_kg_m": 0.06, "stretch_stiffness": 0.35, "bend_stiffness": 0.25, "damping": 0.06, "preset": "fabric"},
    "hemp":  {"linear_density_kg_m": 0.08, "stretch_stiffness": 0.60, "bend_stiffness": 0.40, "damping": 0.08, "preset": "fabric"},
    "steel": {"linear_density_kg_m": 0.35, "stretch_stiffness": 0.98, "bend_stiffness": 0.90, "damping": 0.03, "preset": "iron"},
}

# Brittleness 0..1: drives fracture threshold (lower for brittle materials),
# shard count (higher), and the fracture pattern.
MATERIAL_BRITTLENESS: dict[str, float] = {
    "wood": 0.10,
    "plastic": 0.15,
    "stone": 0.40,
    "terracotta": 0.70,
    "ceramic": 0.80,
    "porcelain": 0.90,
    "glass": 1.00,
}

# Human range-of-motion limits actually emitted per ragdoll joint
# (degrees; [lo, hi]). Hinge axes are the flexion axis; ball-joint axes are
# the primary twist axis.
RAGDOLL_JOINTS: list[dict] = [
    {"name": "waist",       "kind": "ball",  "parent": "pelvis",      "child": "abdomen",     "axis": [0.0, 1.0, 0.0],  "limits_deg": [-30.0, 45.0]},
    {"name": "spine",       "kind": "ball",  "parent": "abdomen",     "child": "chest",       "axis": [0.0, 1.0, 0.0],  "limits_deg": [-25.0, 30.0]},
    {"name": "neck",        "kind": "ball",  "parent": "chest",       "child": "head",        "axis": [0.0, 1.0, 0.0],  "limits_deg": [-55.0, 60.0]},
    {"name": "shoulder_l",  "kind": "ball",  "parent": "chest",       "child": "upper_arm_l", "axis": [0.0, -1.0, 0.0], "limits_deg": [-60.0, 180.0]},
    {"name": "shoulder_r",  "kind": "ball",  "parent": "chest",       "child": "upper_arm_r", "axis": [0.0, -1.0, 0.0], "limits_deg": [-60.0, 180.0]},
    {"name": "elbow_l",     "kind": "hinge", "parent": "upper_arm_l", "child": "forearm_l",   "axis": [1.0, 0.0, 0.0],  "limits_deg": [0.0, 145.0]},
    {"name": "elbow_r",     "kind": "hinge", "parent": "upper_arm_r", "child": "forearm_r",   "axis": [1.0, 0.0, 0.0],  "limits_deg": [0.0, 145.0]},
    {"name": "wrist_l",     "kind": "hinge", "parent": "forearm_l",   "child": "hand_l",      "axis": [1.0, 0.0, 0.0],  "limits_deg": [-70.0, 80.0]},
    {"name": "wrist_r",     "kind": "hinge", "parent": "forearm_r",   "child": "hand_r",      "axis": [1.0, 0.0, 0.0],  "limits_deg": [-70.0, 80.0]},
    {"name": "hip_l",       "kind": "ball",  "parent": "pelvis",      "child": "thigh_l",     "axis": [0.0, -1.0, 0.0], "limits_deg": [-30.0, 120.0]},
    {"name": "hip_r",       "kind": "ball",  "parent": "pelvis",      "child": "thigh_r",     "axis": [0.0, -1.0, 0.0], "limits_deg": [-30.0, 120.0]},
    {"name": "knee_l",      "kind": "hinge", "parent": "thigh_l",     "child": "shin_l",      "axis": [1.0, 0.0, 0.0],  "limits_deg": [0.0, 135.0]},
    {"name": "knee_r",      "kind": "hinge", "parent": "thigh_r",     "child": "shin_r",      "axis": [1.0, 0.0, 0.0],  "limits_deg": [0.0, 135.0]},
    {"name": "ankle_l",     "kind": "hinge", "parent": "shin_l",      "child": "foot_l",      "axis": [1.0, 0.0, 0.0],  "limits_deg": [-50.0, 20.0]},
    {"name": "ankle_r",     "kind": "hinge", "parent": "shin_r",      "child": "foot_r",      "axis": [1.0, 0.0, 0.0],  "limits_deg": [-50.0, 20.0]},
]

_REFERENCE_HEIGHT_M = 1.75


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class SoftAuthorResult:
    """Bundle of pipeline outputs + iemodel/3 extras for one authored object."""

    positions: np.ndarray            # (N, 3) float32
    colors: np.ndarray               # (N, 3) float32 in [0, 1]
    labels: np.ndarray               # (N,) int32
    label_names: list[str]
    parts: list[AnalyticPart]        # exact meshes for GLB export
    spec: GenerationSpec             # facade for the manifest builder
    extras: dict                     # physics.body_type + non-rigid blocks

    def build_manifest(self, **kwargs) -> dict:
        """Manifest via core.manifest.build_manifest with extras attached."""
        from ..core.manifest import build_manifest

        return build_manifest(
            self.spec,
            self.positions,
            self.colors,
            labels=self.labels,
            extras=self.extras,
            **kwargs,
        )

    def write_glb(self, path, **kwargs):
        """GLB via core.exporter.write_glb_parts (named nodes, per-part PBR)."""
        from ..core.exporter import write_glb_parts

        return write_glb_parts(path, self.parts, self.positions, self.colors, **kwargs)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _cloud_colors(positions: np.ndarray, color, rng: np.random.Generator) -> np.ndarray:
    """Unbaked albedo colors, same convention as the compositor (W8)."""
    base = base_color("abstract", tuple(color) if color is not None else None)
    return albedo_colors(np.asarray(positions, dtype=np.float32), base, rng)


def _make_part(
    label: str,
    kind: str,
    material: str,
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    solid_volume_m3: float,
) -> AnalyticPart:
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    return AnalyticPart(
        label=label,
        kind=kind,
        material=material,
        vertices=v,
        normals=np.asarray(normals, dtype=np.float32).reshape(-1, 3),
        uvs=np.asarray(uvs, dtype=np.float32).reshape(-1, 2),
        faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3),
        aabb_min=v.min(axis=0),
        aabb_max=v.max(axis=0),
        solid_volume_m3=float(solid_volume_m3),
    )


def _primitive(kind: str, params: dict, label: str, transform: np.ndarray | None = None) -> Primitive:
    T = np.eye(4, dtype=np.float32) if transform is None else np.asarray(transform, dtype=np.float32)
    return Primitive(kind=kind, transform=T.tolist(), params=params, label=label)


# ---------------------------------------------------------------------------
# cloth
# ---------------------------------------------------------------------------


def cloth_grid_mesh(
    width: float,
    depth: float,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flat (h * w, 3) vertex grid in the XZ plane (+Y normals) with quad faces.

    Vertex index = row * w + col (row along Z, col along X), so the four
    corner indices are ``[0, w-1, (h-1)*w, h*w-1]``.
    """
    w, h = int(resolution[0]), int(resolution[1])
    if w < 2 or h < 2:
        raise ValueError(f"cloth resolution must be >= 2x2, got {resolution!r}")
    xs = np.linspace(-width / 2.0, width / 2.0, w)
    zs = np.linspace(-depth / 2.0, depth / 2.0, h)
    zz, xx = np.meshgrid(zs, xs, indexing="ij")  # (h, w)
    vertices = np.stack([xx, np.zeros_like(xx), zz], axis=-1).reshape(-1, 3)
    normals = np.tile(np.array([[0.0, 1.0, 0.0]]), (h * w, 1))
    uus = np.linspace(0.0, 1.0, w)
    vvs = np.linspace(0.0, 1.0, h)
    vv, uu = np.meshgrid(vvs, uus, indexing="ij")
    uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)

    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()   # +Z neighbour
    c = idx[1:, 1:].ravel()    # +Z +X
    d = idx[:-1, 1:].ravel()   # +X neighbour
    # (a, b, c) / (a, c, d) winds counter-clockwise seen from +Y.
    faces = np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=0
    )
    return (
        vertices.astype(np.float32),
        normals.astype(np.float32),
        uvs.astype(np.float32),
        faces,
    )


def cloth_corner_pins(resolution: tuple[int, int]) -> list[int]:
    """Grid-vertex indices of the four sheet corners."""
    w, h = int(resolution[0]), int(resolution[1])
    return [0, w - 1, (h - 1) * w, h * w - 1]


def _drape_cylinder(
    vertices: np.ndarray,
    normals: np.ndarray,
    width: float,
    depth: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap the flat XZ grid around a vertical cylinder (towel over a bucket).

    The grid's X extent becomes arc length around the cylinder (angle
    preserved: theta = x / radius) and the Z extent hangs vertically
    downward from y = 0 (top edge) to y = -depth (free hem).
    """
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError(f"drape_radius must be > 0, got {radius}")
    theta = vertices[:, 0] / radius
    r = radius + vertices[:, 1]  # allow pre-existing thickness offsets
    out = np.stack(
        [
            r * np.sin(theta),
            -(vertices[:, 2] + depth / 2.0),
            r * np.cos(theta) - radius,
        ],
        axis=-1,
    )
    nrm = np.stack(
        [
            normals[:, 0] * np.cos(theta) + normals[:, 1] * np.sin(theta),
            normals[:, 2] * 0.0,
            -normals[:, 0] * np.sin(theta) + normals[:, 1] * np.cos(theta),
        ],
        axis=-1,
    )
    return out.astype(np.float32), nrm.astype(np.float32)


def _solidify_sheet(
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Give a zero-thickness sheet real volume: a back layer offset along
    -normal with reversed winding, stitched to the front along the border."""
    thickness = float(thickness)
    if thickness <= 0.0:
        return vertices, normals, uvs, faces
    n = vertices.shape[0]
    back_v = vertices - thickness * normals
    back_n = -normals
    # Border stitching: deduplicated boundary edges from the front faces.
    edge_count: dict[tuple[int, int], int] = {}
    directed: dict[tuple[int, int], tuple[int, int]] = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edge_count[key] = edge_count.get(key, 0) + 1
            directed.setdefault(key, (int(a), int(b)))
    stitch: list[list[int]] = []
    for key, cnt in edge_count.items():
        if cnt == 1:
            a, b = directed[key]
            stitch.append([a, b, n + b])
            stitch.append([a, n + b, n + a])
    all_v = np.concatenate([vertices, back_v], axis=0)
    all_n = np.concatenate([normals, back_n], axis=0)
    all_uv = np.concatenate([uvs, uvs], axis=0)
    back_f = faces[:, [0, 2, 1]] + n
    all_f = np.concatenate([faces, back_f] + ([np.asarray(stitch, dtype=np.int64)] if stitch else []), axis=0)
    return (
        all_v.astype(np.float32),
        all_n.astype(np.float32),
        all_uv.astype(np.float32),
        all_f.astype(np.int64),
    )


def author_cloth(
    material: str = "cotton",
    width: float = 0.6,
    depth: float = 0.4,
    resolution: tuple[int, int] = (24, 16),
    pins="corners",
    n_points: int = 4000,
    color: tuple[float, float, float] | None = None,
    seed: int = 0,
    drape: str | None = None,
    drape_radius: float | None = None,
    thickness: float = 0.0,
    weave: bool = False,
) -> SoftAuthorResult:
    """Cloth sheet (towel): grid mesh + iemodel/3 ``soft_body`` block.

    ``pins`` is ``"corners"`` (default), ``"top_edge"``, ``"none"``, or an
    explicit iterable of grid-vertex indices. Stiffness / damping / mass come
    from :data:`CLOTH_FABRICS`.

    Surface-realism options (all default off, fully backward compatible):

    - ``drape="cylinder"`` + ``drape_radius``: wrap the sheet around a
      vertical cylinder — towel over a bucket rim (top edge at y=0, hem
      hanging to y=-depth).
    - ``thickness``: real sheet volume in meters — a back layer offset along
      -normal, stitched at the borders (default 0 = single surface).
    - ``weave=True``: warp/weft albedo modulation (±4 %) on the point cloud
      so the textile reads as woven instead of flat plastic.
    """
    fabric = CLOTH_FABRICS.get(str(material).lower())
    if fabric is None:
        raise ValueError(
            f"unknown cloth fabric {material!r}; choose from {sorted(CLOTH_FABRICS)}"
        )
    rng = np.random.default_rng(seed or None)
    w, h = int(resolution[0]), int(resolution[1])
    vertices, normals, uvs, faces = cloth_grid_mesh(width, depth, (w, h))

    if isinstance(pins, str):
        if pins == "corners":
            pin_indices = cloth_corner_pins((w, h))
        elif pins == "top_edge":
            pin_indices = list(range(w))
        elif pins == "none":
            pin_indices = []
        else:
            raise ValueError(f"unknown pins mode {pins!r}")
    else:
        pin_indices = [int(i) for i in pins]

    area = float(width) * float(depth)
    mass_kg = fabric["area_density_kg_m2"] * area

    # --- optional shaping (surface realism; all default off) ----------------
    do_drape = drape is not None
    if do_drape:
        if str(drape).lower() != "cylinder":
            raise ValueError(f"unknown drape mode {drape!r}; supported: 'cylinder'")
        vertices, normals = _drape_cylinder(
            vertices, normals, width, depth,
            drape_radius if drape_radius is not None else width / math.pi,
        )
    if thickness > 0.0:
        vertices, normals, uvs, faces = _solidify_sheet(
            vertices, normals, uvs, faces, thickness
        )

    # Point cloud: uniform samples on the sheet (a textured plane), shaped the
    # same way as the mesh so renders match the analytic surface.
    px = rng.uniform(-width / 2.0, width / 2.0, n_points)
    pz = rng.uniform(-depth / 2.0, depth / 2.0, n_points)
    positions = np.stack([px, np.zeros_like(px), pz], axis=-1).astype(np.float32)
    if do_drape:
        positions, _ = _drape_cylinder(
            positions, np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (n_points, 1)),
            width, depth,
            drape_radius if drape_radius is not None else width / math.pi,
        )
    colors = _cloud_colors(positions, color, rng)
    if weave:
        # Warp/weft checkerboard albedo modulation (±4 %), period ~4 mm.
        u = px / float(width) + 0.5
        v = pz / float(depth) + 0.5
        cell = 0.004
        check = (np.floor(u * float(width) / cell) + np.floor(v * float(depth) / cell)) % 2
        colors = np.clip(colors * (1.0 + 0.04 * (2.0 * check - 1.0))[:, None], 0.0, 1.0)
    if thickness > 0.0:
        # Second (back) layer of points so the hem reads as a solid edge.
        back = positions.copy()
        if do_drape:
            theta = px / (drape_radius if drape_radius is not None else width / math.pi)
            radial = np.stack([np.sin(theta), np.zeros(n_points), np.cos(theta)], axis=-1)
        else:
            radial = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (n_points, 1))
        back = back - float(thickness) * radial
        positions = np.concatenate([positions, back.astype(np.float32)], axis=0)
        colors = np.concatenate([colors, colors], axis=0)
    labels = np.zeros(positions.shape[0], dtype=np.int32)

    eff_thickness = float(thickness) if thickness > 0.0 else 8e-4
    part = _make_part(
        "cloth", "cloth_sheet", "fabric", vertices, normals, uvs, faces,
        solid_volume_m3=area * eff_thickness,
    )
    spec = GenerationSpec(
        shape="abstract",
        n_points=int(n_points),
        bbox_size=(float(width), eff_thickness, float(depth)),
        primitives=[
            _primitive("plane", {"size": [float(width), float(depth)], "material": "fabric"}, "cloth")
        ],
        features=[],
        color=color,
        seed=int(seed),
    )
    extras = {
        "physics": {"body_type": "soft", "mass_kg": mass_kg},
        "soft_body": {
            "kind": "cloth",
            "resolution": [w, h],
            "mass_kg": mass_kg,
            "stretch_stiffness": fabric["stretch_stiffness"],
            "bend_stiffness": fabric["bend_stiffness"],
            "damping": fabric["damping"],
            "pin_indices": pin_indices,
        },
        # World dimensions so Sim can size the sheet without guessing (B1-CR):
        # width_m × height_m are the two in-plane extents, resolution the grid.
        "cloth": {
            "width_m": float(width),
            "height_m": float(depth),
            "resolution": [w, h],
        },
    }
    # Non-default realism options are reported, defaults keep the legacy dict.
    if do_drape:
        extras["cloth"]["drape"] = "cylinder"
        extras["cloth"]["drape_radius_m"] = float(
            drape_radius if drape_radius is not None else width / math.pi
        )
    if thickness > 0.0:
        extras["cloth"]["thickness_m"] = float(thickness)
    if weave:
        extras["cloth"]["weave"] = True
    return SoftAuthorResult(positions, colors, labels, ["cloth"], [part], spec, extras)


# ---------------------------------------------------------------------------
# rope / cable
# ---------------------------------------------------------------------------


def _tube_mesh(
    centers: np.ndarray,
    radius: float,
    tube_seg: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Capped tube along a polyline centerline (periodic cross-section rings)."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    n_rings = centers.shape[0]
    if n_rings < 2:
        raise ValueError("tube centerline needs at least 2 points")
    tangents = np.gradient(centers, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12

    th = np.linspace(0.0, TAU, tube_seg, endpoint=False)
    ct, st = np.cos(th), np.sin(th)

    verts: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    for i in range(n_rings):
        tan = tangents[i]
        ref = np.array([0.0, 1.0, 0.0]) if abs(tan[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        nrm = np.cross(ref, tan)
        nrm /= np.linalg.norm(nrm) + 1e-12
        bnm = np.cross(tan, nrm)
        ring_n = ct[:, None] * nrm[None, :] + st[:, None] * bnm[None, :]
        verts.append(centers[i][None, :] + radius * ring_n)
        norms.append(ring_n)

    vertices = np.concatenate(verts, axis=0)
    normals = np.concatenate(norms, axis=0)
    u_coord = np.linspace(0.0, 1.0, n_rings)
    uvs = np.stack(
        [
            np.repeat(u_coord, tube_seg),
            np.tile(th / TAU, n_rings),
        ],
        axis=-1,
    )

    idx = np.arange(n_rings * tube_seg, dtype=np.int64).reshape(n_rings, tube_seg)
    faces: list[np.ndarray] = []
    for i in range(n_rings - 1):
        r0, r1 = idx[i], idx[i + 1]
        j = np.arange(tube_seg)
        jn = (j + 1) % tube_seg
        # Wound so the face normal points radially outward.
        faces.append(np.stack([r0[j], r0[jn], r1[jn]], axis=1))
        faces.append(np.stack([r0[j], r1[jn], r1[j]], axis=1))

    # End caps: one center vertex per end + fan with ±tangent normals.
    extra_v: list[np.ndarray] = []
    extra_n: list[np.ndarray] = []
    base_offset = vertices.shape[0]
    for row, sign in ((0, -1.0), (n_rings - 1, +1.0)):
        c = centers[row]
        tangent = tangents[row] * sign
        base = base_offset + len(extra_v)
        ring = idx[row]
        j = np.arange(tube_seg)
        jn = (j + 1) % tube_seg
        if sign > 0:
            cap_f = np.stack([np.full(tube_seg, base), ring[j], ring[jn]], axis=1)
        else:
            cap_f = np.stack([np.full(tube_seg, base), ring[jn], ring[j]], axis=1)
        faces.append(cap_f)
        extra_v.append(c)
        extra_n.append(tangent)

    vertices = np.concatenate([vertices, np.asarray(extra_v)], axis=0)
    normals = np.concatenate([normals, np.asarray(extra_n)], axis=0)
    uvs = np.concatenate(
        [uvs, np.full((len(extra_v), 2), 0.5)], axis=0
    )
    faces = np.concatenate(faces, axis=0)
    return (
        vertices.astype(np.float32),
        normals.astype(np.float32),
        uvs.astype(np.float32),
        faces.astype(np.int64),
    )


def author_rope(
    material: str = "hemp",
    length: float = 1.0,
    radius: float = 0.012,
    segment_count: int = 24,
    sag: float | None = None,
    pins=(0, -1),
    n_points: int = 3000,
    color: tuple[float, float, float] | None = None,
    seed: int = 0,
) -> SoftAuthorResult:
    """Rope / cable: capped tube along a sagging (or hanging) centerline.

    With both ends pinned (default ``pins=(0, -1)``) the centerline is a
    parabola between two anchors at the same height, sagging by ``sag``
    (default 15 % of length). With a single pin at segment 0 the rope hangs
    straight down from the origin. ``segment_count`` soft particles map 1:1 to
    equally spaced centerline samples.
    """
    rope_mat = ROPE_MATERIALS.get(str(material).lower())
    if rope_mat is None:
        raise ValueError(
            f"unknown rope material {material!r}; choose from {sorted(ROPE_MATERIALS)}"
        )
    rng = np.random.default_rng(seed or None)
    segment_count = int(segment_count)
    if segment_count < 2:
        raise ValueError(f"segment_count must be >= 2, got {segment_count}")
    length = float(length)
    radius = float(radius)

    pin_indices = sorted({int(p) % segment_count for p in pins})
    t = np.linspace(0.0, 1.0, segment_count)
    if len(pin_indices) >= 2:
        sag = (0.15 * length) if sag is None else float(sag)
        span = max(length - sag, 0.1 * length)
        centers = np.stack(
            [
                -span / 2.0 + span * t,
                -sag * (1.0 - (2.0 * t - 1.0) ** 2),
                np.zeros_like(t),
            ],
            axis=-1,
        )
    else:
        # Single-pin rope hangs straight down from the origin.
        centers = np.stack(
            [np.zeros_like(t), -length * t, np.zeros_like(t)], axis=-1
        )

    vertices, normals, uvs, faces = _tube_mesh(centers, radius)
    arc_len = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    mass_kg = rope_mat["linear_density_kg_m"] * arc_len

    # Point cloud on the tube surface.
    tangents = np.gradient(centers, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    ti = rng.uniform(0.0, 1.0, n_points)
    seg_f = ti * (segment_count - 1)
    i0 = np.clip(seg_f.astype(np.int64), 0, segment_count - 2)
    lam = seg_f - i0
    c0 = centers[i0] * (1.0 - lam[:, None]) + centers[i0 + 1] * lam[:, None]
    tan = tangents[i0]
    ref = np.where(
        (np.abs(tan[:, 1]) < 0.9)[:, None],
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    nrm = np.cross(ref, tan)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    bnm = np.cross(tan, nrm)
    ang = rng.uniform(0.0, TAU, n_points)
    positions = (
        c0 + radius * (np.cos(ang)[:, None] * nrm + np.sin(ang)[:, None] * bnm)
    ).astype(np.float32)
    colors = _cloud_colors(positions, color, rng)
    labels = np.zeros(n_points, dtype=np.int32)

    part = _make_part(
        "rope", "rope_tube", rope_mat["preset"], vertices, normals, uvs, faces,
        solid_volume_m3=math.pi * radius * radius * arc_len,
    )
    spec = GenerationSpec(
        shape="abstract",
        n_points=int(n_points),
        bbox_size=(max(length, 2 * radius), 2 * radius, 2 * radius),
        primitives=[
            _primitive(
                "cylinder",
                {"radius": radius, "height": length, "material": rope_mat["preset"]},
                "rope",
            )
        ],
        features=[],
        color=color,
        seed=int(seed),
    )
    extras = {
        "physics": {"body_type": "soft", "mass_kg": mass_kg},
        "soft_body": {
            "kind": "rope",
            "segment_count": segment_count,
            "mass_kg": mass_kg,
            "stretch_stiffness": rope_mat["stretch_stiffness"],
            "bend_stiffness": rope_mat["bend_stiffness"],
            "damping": rope_mat["damping"],
            "pin_indices": pin_indices,
        },
    }
    return SoftAuthorResult(positions, colors, labels, ["rope"], [part], spec, extras)


# ---------------------------------------------------------------------------
# frangible vessel
# ---------------------------------------------------------------------------


def _lathe_closed(
    profile: list[tuple[float, float]],
    seg: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Watertight solid of revolution from a *closed* (r, y) profile loop.

    The loop must be CCW in the (r, y) half-plane with the solid interior on
    the left of the direction of travel. On-axis points (r == 0) become pole
    vertices; bands between two on-axis points generate no faces. Theta is
    periodic (no duplicated seam column), so every edge is shared by exactly
    two triangles — the mesh is a closed manifold.
    """
    pts = [np.array([float(r), float(y)], dtype=np.float64) for r, y in profile]
    n = len(pts)
    eps = 1e-9

    # Per-point tangents (average of adjacent edge directions) and outward
    # normals (right side of travel for a CCW loop).
    normals2d: list[np.ndarray] = []
    for i in range(n):
        d0 = pts[i] - pts[i - 1]
        d1 = pts[(i + 1) % n] - pts[i]
        t = d0 + d1
        if np.linalg.norm(t) < 1e-12:
            t = d1
        t = t / (np.linalg.norm(t) + 1e-12)
        normals2d.append(np.array([t[1], -t[0]]))

    # Arc-length v coordinate.
    arc = [0.0]
    for i in range(1, n + 1):
        arc.append(arc[-1] + float(np.linalg.norm(pts[i % n] - pts[i - 1])))
    total_arc = arc[-1]

    th = np.linspace(0.0, TAU, seg, endpoint=False)
    ct, st = np.cos(th), np.sin(th)

    verts: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    uvs: list[list[float]] = []
    rings: list[np.ndarray] = []  # vertex indices per profile point
    for i, p in enumerate(pts):
        n2 = normals2d[i]
        v_coord = arc[i] / total_arc
        if p[0] < eps:
            base = len(verts)
            verts.append(np.array([0.0, p[1], 0.0]))
            norms.append(np.array([0.0, n2[1], 0.0]))
            uvs.append([0.5, v_coord])
            rings.append(np.array([base], dtype=np.int64))  # pole: 1 vertex
        else:
            base = len(verts)
            ring_v = np.stack([p[0] * ct, np.full(seg, p[1]), p[0] * st], axis=-1)
            ring_n = np.stack([n2[0] * ct, np.full(seg, n2[1]), n2[0] * st], axis=-1)
            ring_n /= np.linalg.norm(ring_n, axis=1, keepdims=True) + 1e-12
            verts.extend(ring_v)
            norms.extend(ring_n)
            uvs.extend([[j / seg, v_coord] for j in range(seg)])
            rings.append(np.arange(base, base + seg, dtype=np.int64))

    faces: list[list[int]] = []
    for i in range(n):
        r0, r1 = rings[i], rings[(i + 1) % n]
        p0, p1 = pts[i], pts[(i + 1) % n]
        if p0[0] < eps and p1[0] < eps:
            continue  # axis segment: no surface
        if r0.shape[0] == 1 and r1.shape[0] == 1:
            continue
        if r0.shape[0] == 1:
            j = np.arange(seg)
            jn = (j + 1) % seg
            for a, b in zip(j, jn):
                faces.append([int(r0[0]), int(r1[a]), int(r1[b])])
        elif r1.shape[0] == 1:
            j = np.arange(seg)
            jn = (j + 1) % seg
            for a, b in zip(j, jn):
                faces.append([int(r1[0]), int(r0[b]), int(r0[a])])
        else:
            j = np.arange(seg)
            jn = (j + 1) % seg
            for a, b in zip(j, jn):
                faces.append([int(r0[a]), int(r1[b]), int(r0[b])])
                faces.append([int(r0[a]), int(r1[a]), int(r1[b])])

    vertices = np.asarray(verts, dtype=np.float32)
    normals = np.asarray(norms, dtype=np.float32)
    uv_arr = np.asarray(uvs, dtype=np.float32)
    face_arr = np.asarray(faces, dtype=np.int64)
    if signed_volume(vertices, face_arr) < 0.0:
        face_arr = face_arr[:, [0, 2, 1]]
    return vertices, normals, uv_arr, face_arr


def vessel_profile(radius: float, height: float, wall_thickness: float) -> list[tuple[float, float]]:
    """Closed (r, y) lathe profile for a hollow pot with real wall thickness.

    Outer bottom pole → outer wall → rim → inner wall → inner floor pole and
    back down the axis (the axis segment generates no faces; it only closes
    the loop).
    """
    R, H, t = float(radius), float(height), float(wall_thickness)
    if not (0.0 < t < R * 0.9):
        raise ValueError(f"wall_thickness must be in (0, {0.9 * R:.4f}), got {t}")
    if not (t < H * 0.5):
        raise ValueError(f"wall_thickness {t} too thick for height {H}")
    Ri = R - t
    return [(0.0, 0.0), (R, 0.0), (R, H), (Ri, H), (Ri, t), (0.0, t)]


def vessel_solid_volume(radius: float, height: float, wall_thickness: float) -> float:
    """Exact analytic wall volume: outer cylinder minus inner cavity."""
    R, H, t = float(radius), float(height), float(wall_thickness)
    Ri = R - t
    return math.pi * R * R * H - math.pi * Ri * Ri * (H - t)


def fracture_block(
    material: str,
    fragment_count: int | None = None,
) -> dict:
    """iemodel/3 ``fracture`` block scaled by material brittleness."""
    key = str(material).lower()
    b = MATERIAL_BRITTLENESS.get(key, 0.5)
    return {
        "threshold_impulse": round(6.0 / max(b, 0.05), 2),  # N*s; brittle → low
        "fragment_count": int(fragment_count) if fragment_count else 8 + int(round(40 * b)),
        "pattern": "shatter" if b >= 0.85 else "voronoi",
        "debris_material": key,
    }


def author_frangible_vessel(
    material: str = "ceramic",
    radius: float = 0.09,
    height: float = 0.16,
    wall_thickness: float = 0.006,
    fragment_count: int | None = None,
    n_points: int = 5000,
    color: tuple[float, float, float] | None = None,
    seed: int = 0,
    lathe_seg: int = 32,
) -> SoftAuthorResult:
    """Hollow vessel (ceramic pot): watertight inner+outer shells with real
    wall thickness, plus an iemodel/3 ``fracture`` block scaled by brittleness.
    """
    key = str(material).lower()
    rng = np.random.default_rng(seed or None)
    R, H, t = float(radius), float(height), float(wall_thickness)
    profile = vessel_profile(R, H, t)
    vertices, normals, uvs, faces = _lathe_closed(profile, seg=int(lathe_seg))
    volume = vessel_solid_volume(R, H, t)
    density = MATERIAL_PRESETS.get(key, default_preset())["density_kg_m3"]
    mass_kg = volume * density

    # Point cloud: sample lathe bands proportional to band area.
    pts = [np.array(p, dtype=np.float64) for p in profile]
    bands = []
    for i in range(len(pts)):
        p0, p1 = pts[i], pts[(i + 1) % len(pts)]
        seg_len = float(np.linalg.norm(p1 - p0))
        area = math.pi * (p0[0] + p1[0]) * seg_len
        if area > 1e-12:
            bands.append((p0, p1, area))
    areas = np.array([b[2] for b in bands])
    counts = np.floor(n_points * areas / areas.sum()).astype(np.int64)
    counts[areas.argmax()] += n_points - counts.sum()
    chunks = []
    for (p0, p1, _), m in zip(bands, counts):
        if m <= 0:
            continue
        lam = rng.uniform(0.0, 1.0, m)
        ang = rng.uniform(0.0, TAU, m)
        r = p0[0] + lam * (p1[0] - p0[0])
        y = p0[1] + lam * (p1[1] - p0[1])
        chunks.append(np.stack([r * np.cos(ang), y, r * np.sin(ang)], axis=-1))
    positions = np.concatenate(chunks, axis=0).astype(np.float32)
    colors = _cloud_colors(positions, color, rng)
    labels = np.zeros(positions.shape[0], dtype=np.int32)

    part = _make_part(
        "vessel", "vessel_lathe", key, vertices, normals, uvs, faces,
        solid_volume_m3=volume,
    )
    spec = GenerationSpec(
        shape="vase",
        n_points=int(n_points),
        bbox_size=(2 * R, H, 2 * R),
        primitives=[
            _primitive("cylinder", {"radius": R, "height": H, "material": key}, "vessel")
        ],
        features=[],
        color=color,
        seed=int(seed),
    )
    extras = {
        "physics": {"body_type": "frangible", "mass_kg": mass_kg},
        "fracture": fracture_block(key, fragment_count),
    }
    return SoftAuthorResult(positions, colors, labels, ["vessel"], [part], spec, extras)


# ---------------------------------------------------------------------------
# humanoid ragdoll
# ---------------------------------------------------------------------------


def _rotation_about_z_90() -> np.ndarray:
    """Maps the capsule's local +Y axis onto +X (det = +1)."""
    return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _rotation_about_x_90() -> np.ndarray:
    """Maps the capsule's local +Y axis onto +Z (det = +1)."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _capsule_transform(center, rot: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if rot is not None:
        T[:3, :3] = rot
    T[:3, 3] = np.asarray(center, dtype=np.float64)
    return T


# label, radius, cylinder height, center, rotation — at reference height 1.75 m.
_RAGDOLL_BODY: list[tuple[str, float, float, tuple[float, float, float], str]] = [
    ("pelvis",      0.095, 0.14, (0.0,   0.98, 0.0),  "x"),
    ("abdomen",     0.100, 0.12, (0.0,   1.12, 0.0),  "y"),
    ("chest",       0.115, 0.18, (0.0,   1.31, 0.0),  "y"),
    ("head",        0.105, 0.06, (0.0,   1.55, 0.0),  "y"),
    ("upper_arm_l", 0.040, 0.24, (0.22,  1.28, 0.0),  "y"),
    ("forearm_l",   0.035, 0.22, (0.22,  1.01, 0.0),  "y"),
    ("hand_l",      0.032, 0.08, (0.22,  0.84, 0.0),  "y"),
    ("upper_arm_r", 0.040, 0.24, (-0.22, 1.28, 0.0),  "y"),
    ("forearm_r",   0.035, 0.22, (-0.22, 1.01, 0.0),  "y"),
    ("hand_r",      0.032, 0.08, (-0.22, 0.84, 0.0),  "y"),
    ("thigh_l",     0.070, 0.36, (0.10,  0.71, 0.0),  "y"),
    ("shin_l",      0.055, 0.34, (0.10,  0.30, 0.0),  "y"),
    ("foot_l",      0.045, 0.10, (0.10,  0.045, 0.05), "z"),
    ("thigh_r",     0.070, 0.36, (-0.10, 0.71, 0.0),  "y"),
    ("shin_r",      0.055, 0.34, (-0.10, 0.30, 0.0),  "y"),
    ("foot_r",      0.045, 0.10, (-0.10, 0.045, 0.05), "z"),
]


def ragdoll_spec(height: float = _REFERENCE_HEIGHT_M, n_points: int = 12000,
                 color=None, seed: int = 0) -> GenerationSpec:
    """GenerationSpec with one capsule primitive per ragdoll body part."""
    s = float(height) / _REFERENCE_HEIGHT_M
    prims = []
    for label, r, h, center, axis in _RAGDOLL_BODY:
        rot = None
        if axis == "x":
            rot = _rotation_about_z_90()
        elif axis == "z":
            rot = _rotation_about_x_90()
        T = _capsule_transform([c * s for c in center], rot)
        prims.append(
            _primitive(
                "capsule",
                {"radius": r * s, "height": h * s, "material": "organic"},
                label,
                T,
            )
        )
    return GenerationSpec(
        shape="creature",
        n_points=int(n_points),
        bbox_size=(0.55 * s, float(height), 0.35 * s),
        primitives=prims,
        features=[],
        color=color,
        seed=int(seed),
    )


def author_ragdoll(
    height: float = _REFERENCE_HEIGHT_M,
    n_points: int = 12000,
    color: tuple[float, float, float] | None = None,
    seed: int = 0,
) -> SoftAuthorResult:
    """Humanoid ragdoll: 16 capsule body parts (reusing the exact analytic
    capsule pipeline) + an iemodel/3 ``articulation`` block with 15 joints at
    human range-of-motion limits.
    """
    spec = ragdoll_spec(height=height, n_points=n_points, color=color, seed=seed)

    # Reuse the existing pipeline for points / colors / labels and meshes.
    from .compositor import generate

    result = generate(spec)
    parts = build_spec_meshes(spec)

    joints = [
        {
            "name": j["name"],
            "kind": j["kind"],
            "parent": j["parent"],
            "child": j["child"],
            "axis": list(j["axis"]),
            "limits_deg": list(j["limits_deg"]),
        }
        for j in RAGDOLL_JOINTS
    ]
    extras = {
        "physics": {"body_type": "articulated"},
        "articulation": {"joints": joints, "ragdoll": True},
    }
    return SoftAuthorResult(
        result.positions,
        result.colors,
        result.labels,
        result.label_names,
        parts,
        spec,
        extras,
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

SOFT_GENERATORS = {
    "cloth": author_cloth,
    "rope": author_rope,
    "frangible_vessel": author_frangible_vessel,
    "ragdoll": author_ragdoll,
}
