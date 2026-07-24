"""Part-graph complex builder — declarative assemblies with instancing.

A `PartGraph` describes a complex object as a set of named parts (nodes) and
attachment transforms (edges). Nodes are analytic primitives (any kind from
`generation.analytic_mesh.MESH_BUILDERS`) or slicing-builder lofts
(`generation.slicer`). Edges compose a child's local transform under its
parent, so whole sub-assemblies move together.

Symmetry is expressed as *instancing*: `mirror` and `array_radial` register
extra world transforms on a node instead of duplicating geometry. When the
graph is built, every instance of a node SHARES the same vertex / normal /
uv / face numpy arrays — one definition, many instances, minimal memory.

Every built part carries metadata for downstream boundary display/tracking:
per-instance conservative world AABBs (corner-transformed local AABB),
per-name merged AABBs, triangle counts, materials, and solid volumes.

`BuildResult.bake()` produces world-space `AnalyticPart` copies (with winding
flipped for mirror instances) for consumers that expect flattened meshes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import slicer
from .analytic_mesh import (
    AnalyticPart,
    MESH_BUILDERS,
    apply_transform,
    local_aabb,
    primitive_solid_volume,
)

TAU = 2.0 * math.pi


def T(translate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
      ry: float = 0.0, rx: float = 0.0, rz: float = 0.0) -> np.ndarray:
    """4x4 transform = T · Ry · Rx · Rz · S (angles in radians)."""
    cy, sy = math.cos(ry), math.sin(ry)
    cx, sx = math.cos(rx), math.sin(rx)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    m = rot_y @ rot_x @ rot_z @ np.diag(np.asarray(scale, dtype=np.float64))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = m
    out[:3, 3] = np.asarray(translate, dtype=np.float64)
    return out


def mirror_matrix(axis: str = "x", coord: float = 0.0) -> np.ndarray:
    """Reflection through the plane `axis` = `coord`."""
    m = np.eye(4, dtype=np.float64)
    i = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    m[i, i] = -1.0
    m[i, 3] = 2.0 * coord
    return m


def rotation_matrix(axis: str = "y", angle: float = 0.0,
                    center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Rotation of `angle` radians around `axis` through `center`."""
    i = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    j, k = (i + 1) % 3, (i + 2) % 3
    c, s = math.cos(angle), math.sin(angle)
    r = np.eye(4, dtype=np.float64)
    r[j, j], r[k, k] = c, c
    r[j, k], r[k, j] = -s, s
    ctr = np.asarray(center, dtype=np.float64)
    r[:3, 3] = ctr - r[:3, :3] @ ctr
    return r


# ---------------------------------------------------------------------------
# graph nodes
# ---------------------------------------------------------------------------


@dataclass
class PartNode:
    """One named part definition (a single mesh, possibly many instances)."""

    name: str
    kind: str                                  # primitive kind or "loft"
    params: dict = field(default_factory=dict)
    loft_spec: dict | None = None              # {profile, slices, axis, caps}
    material: str = "metal"
    parent: str | None = None
    local: np.ndarray = field(default_factory=lambda: np.eye(4))
    metadata: dict = field(default_factory=dict)
    # Extra world-space transforms left-multiplied onto the node's world
    # transform (mirror / array instances). Instance 0 is always identity.
    instances: list[np.ndarray] = field(default_factory=list)


@dataclass
class BuiltPart:
    """One placed instance of a node; mesh arrays are SHARED with siblings."""

    name: str                # node name (shared across instances)
    instance: int            # 0 = base placement
    label: str               # "name" for instance 0, "name#i" otherwise
    kind: str
    material: str
    transform: np.ndarray    # world transform (local mesh → world)
    vertices: np.ndarray     # node-local, shared across instances
    normals: np.ndarray
    uvs: np.ndarray
    faces: np.ndarray
    local_aabb_min: np.ndarray
    local_aabb_max: np.ndarray
    aabb_min: np.ndarray     # conservative world AABB (corner transform)
    aabb_max: np.ndarray
    solid_volume_m3: float
    metadata: dict = field(default_factory=dict)


@dataclass
class BuildResult:
    name: str
    parts: list[BuiltPart]

    def aabbs(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Merged world AABB per named part (all its instances)."""
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for p in self.parts:
            if p.name in out:
                lo, hi = out[p.name]
                out[p.name] = (np.minimum(lo, p.aabb_min), np.maximum(hi, p.aabb_max))
            else:
                out[p.name] = (p.aabb_min.copy(), p.aabb_max.copy())
        return out

    def triangle_count(self) -> int:
        """Total triangles across all instances."""
        return int(sum(p.faces.shape[0] for p in self.parts))

    def stats(self) -> dict[str, dict]:
        """Per-named-part stats: instances, unique/shared tris, world AABB."""
        merged = self.aabbs()
        out: dict[str, dict] = {}
        for p in self.parts:
            st = out.setdefault(p.name, {
                "instances": 0, "tris_unique": int(p.faces.shape[0]),
                "tris_total": 0, "material": p.material,
                "aabb_min": merged[p.name][0], "aabb_max": merged[p.name][1],
                "volume_m3": 0.0,
            })
            st["instances"] += 1
            st["tris_total"] += int(p.faces.shape[0])
            st["volume_m3"] += p.solid_volume_m3
        return out

    def bake(self) -> list[AnalyticPart]:
        """World-space AnalyticPart copies (mirror instances flip winding)."""
        baked: list[AnalyticPart] = []
        for p in self.parts:
            vw, nw = apply_transform(p.vertices, p.normals, p.transform)
            f = p.faces
            det = float(np.linalg.det(np.asarray(p.transform)[:3, :3]))
            if det < 0.0:  # reflection: restore outward winding
                f = f[:, [0, 2, 1]]
            baked.append(AnalyticPart(
                label=p.label, kind=p.kind, material=p.material,
                vertices=vw, normals=nw, uvs=p.uvs.copy(), faces=f.copy(),
                aabb_min=vw.min(axis=0), aabb_max=vw.max(axis=0),
                solid_volume_m3=p.solid_volume_m3,
            ))
        return baked


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


class PartGraph:
    """Declarative part graph: named nodes + attachment edges + instancing."""

    def __init__(self, name: str = "assembly"):
        self.name = name
        self.nodes: dict[str, PartNode] = {}

    # -- node creation ------------------------------------------------------
    def add_primitive(
        self,
        name: str,
        kind: str,
        params: dict | None = None,
        material: str = "metal",
        parent: str | None = None,
        translate=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        ry: float = 0.0,
        rx: float = 0.0,
        rz: float = 0.0,
        metadata: dict | None = None,
    ) -> PartNode:
        """Add an analytic-primitive node (kind must be in MESH_BUILDERS)."""
        if kind not in MESH_BUILDERS:
            raise ValueError(f"unknown primitive kind {kind!r}")
        node = PartNode(
            name=name, kind=kind, params=dict(params or {}), material=material,
            parent=parent, local=T(translate, scale, ry=ry, rx=rx, rz=rz),
            metadata=dict(metadata or {}),
        )
        self._register(node)
        return node

    def add_loft(
        self,
        name: str,
        profile: np.ndarray,
        slices: list[slicer.Slice],
        axis: str = "y",
        caps: bool = True,
        material: str = "metal",
        parent: str | None = None,
        translate=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        ry: float = 0.0,
        rx: float = 0.0,
        rz: float = 0.0,
        metadata: dict | None = None,
    ) -> PartNode:
        """Add a slicing-builder loft node (stacked cross-sections)."""
        node = PartNode(
            name=name, kind="loft", material=material, parent=parent,
            loft_spec={"profile": np.asarray(profile, dtype=np.float64),
                       "slices": list(slices), "axis": axis, "caps": bool(caps)},
            local=T(translate, scale, ry=ry, rx=rx, rz=rz),
            metadata=dict(metadata or {}),
        )
        self._register(node)
        return node

    def _register(self, node: PartNode) -> None:
        if node.name in self.nodes:
            raise ValueError(f"duplicate part name {node.name!r}")
        if node.parent is not None and node.parent not in self.nodes:
            raise ValueError(f"parent {node.parent!r} must be added before {node.name!r}")
        self.nodes[node.name] = node

    # -- edges ---------------------------------------------------------------
    def attach(self, child: str, parent: str, transform: np.ndarray | None = None) -> None:
        """Re-parent `child` under `parent`; optionally replace its local T."""
        c, p = self.nodes[child], self.nodes[parent]
        c.parent = p.name
        if transform is not None:
            c.local = np.asarray(transform, dtype=np.float64).reshape(4, 4)

    # -- instancing -----------------------------------------------------------
    def mirror(self, name: str, axis: str = "x", coord: float = 0.0) -> PartNode:
        """Add a mirrored instance of `name` (reflection plane axis=coord)."""
        node = self.nodes[name]
        node.instances.append(mirror_matrix(axis, coord))
        return node

    def array_radial(
        self,
        name: str,
        count: int,
        axis: str = "y",
        center=(0.0, 0.0, 0.0),
        start_angle: float = 0.0,
    ) -> PartNode:
        """Add `count - 1` rotational instances of `name` around `axis`."""
        node = self.nodes[name]
        for i in range(1, int(count)):
            node.instances.append(
                rotation_matrix(axis, start_angle + TAU * i / int(count), center)
            )
        return node

    # -- build -----------------------------------------------------------------
    def _world(self, node: PartNode, memo: dict[str, np.ndarray]) -> np.ndarray:
        if node.name in memo:
            return memo[node.name]
        if node.parent is None:
            w = node.local
        else:
            w = self._world(self.nodes[node.parent], memo) @ node.local
        memo[node.name] = w
        return w

    def _build_mesh(self, node: PartNode) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if node.kind == "loft":
            spec = node.loft_spec
            assert spec is not None
            return slicer.loft(spec["profile"], spec["slices"],
                               axis=spec["axis"], caps=spec["caps"])
        return MESH_BUILDERS[node.kind](node.params)

    def _local_volume(self, node: PartNode, v: np.ndarray, f: np.ndarray) -> float:
        if node.kind == "loft":
            return abs(slicer.signed_mesh_volume(v, f))
        return primitive_solid_volume(node.kind, node.params)

    def build(self) -> BuildResult:
        memo: dict[str, np.ndarray] = {}
        parts: list[BuiltPart] = []
        for node in self.nodes.values():
            v, n, uv, f = self._build_mesh(node)          # built ONCE per node
            if node.kind == "loft":
                lo, hi = v.min(axis=0), v.max(axis=0)
            else:
                lo, hi = local_aabb(node.kind, node.params)
            lo = np.asarray(lo, dtype=np.float64)
            hi = np.asarray(hi, dtype=np.float64)
            corners = np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                 for z in (lo[2], hi[2])]
            )
            world = self._world(node, memo)
            base_vol = self._local_volume(node, v, f)
            for inst_i, inst_m in enumerate([np.eye(4), *node.instances]):
                w = inst_m @ world
                det = abs(float(np.linalg.det(w[:3, :3])))
                wc = (np.concatenate([corners, np.ones((8, 1))], axis=1) @ w.T)[:, :3]
                parts.append(BuiltPart(
                    name=node.name,
                    instance=inst_i,
                    label=node.name if inst_i == 0 else f"{node.name}#{inst_i}",
                    kind=node.kind,
                    material=node.material,
                    transform=w,
                    vertices=v, normals=n, uvs=uv, faces=f,   # shared, no copy
                    local_aabb_min=lo, local_aabb_max=hi,
                    aabb_min=wc.min(axis=0), aabb_max=wc.max(axis=0),
                    solid_volume_m3=base_vol * (det if det > 1e-12 else 1.0),
                    metadata=dict(node.metadata),
                ))
        return BuildResult(name=self.name, parts=parts)
