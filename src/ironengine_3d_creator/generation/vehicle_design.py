"""Parametric vehicle designer — real body curves, detailed running gear,
openable closures, complete interiors, and per-vertex color zones.

Built on the same analytic-mesh conventions as `generation.analytic_mesh` /
`generation.slicer` (float32 (V,3) / (V,2), int64 (F,3), outward normals,
watertight 2-manifold parts), but self-contained: every panel is generated
here so car-specific features (wheel-arch cutouts, door shut lines, hinged
closures, interior trim) are first-class.

Vehicle frame
-------------
- X: longitudinal — nose at −X, tail at +X, front axle at x = 0.
- Y: vertical — ground plane at y = 0.
- Z: lateral — +Z is the vehicle LEFT side (driver side for LHD default).

Design notes
------------
- **Body**: the body tub is a multi-section loft (`loft_sections`) whose
  cross-section topology is fixed (22 points) while every coordinate is
  driven per station — hood slope, windshield/cowl step, beltline rise,
  shoulder crease, plan-view taper, tumblehome, engine bay, cabin floor
  tub, trunk well and pickup bed all fall out of one parameterised loop.
- **Wheel arches** are a true 2-D CSG subtraction: at each station inside a
  wheel cutter the section loop is rebuilt as ``section − cutter`` (the
  bottom/flank points below the cutter height are lifted onto the arch).
  The loop stays a single simple polygon, so the loft is guaranteed free of
  orphan geometry — see `subtract_arch_from_section`.
- **Closures** (doors, hood, trunk/hatch) are separate named parts with
  `HingeSpec` articulation (front-hinged doors, ROM 0–65°; hood 0–60°;
  trunk/hatch 0–70°). `VehicleSpec.bake(state)` applies any opening state.
- **Interior** is a named-part set (floor, seats, steering, dashboard,
  console, door cards) visible through the thin glass panels and open
  doors.
- **Finer surface detail**: every part carries `(V, 3)` vertex colors with
  small zones (paint gradients, glass tint bands, chrome, rubber, lens
  colors, interior trim). IronEngine-BonaFide consumes per-vertex colors
  directly as albedo.

API
---
``build_vehicle(params) -> VehicleSpec`` with params::

    class:           sedan | hatchback | suv | sports | pickup | van
    color:           named color or (r, g, b) in [0, 1]
    livery:          None | "racing_stripes" | "two_tone"
    doors_open:      False | True | float 0..1 | {assembly: 0..1}
    interior_detail: "low" | "high"
    lod:             "high" | "mid" | "low"   (triangle-density knobs)
    seed:            int (reserved; generation is deterministic)

All dimensions are metres, real-world scale (sedan length 4.88 m).
High-detail triangle budget: <= 45 000 (see `VehicleSpec.triangle_count`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TAU = 2.0 * math.pi

Mesh = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# colors / materials
# ---------------------------------------------------------------------------

PAINT_COLORS: dict[str, tuple[float, float, float]] = {
    "candy_red": (0.62, 0.05, 0.07),
    "deep_blue": (0.06, 0.14, 0.38),
    "silver": (0.72, 0.74, 0.77),
    "black": (0.045, 0.05, 0.06),
    "white": (0.92, 0.92, 0.90),
    "british_green": (0.05, 0.25, 0.15),
    "sunset_orange": (0.80, 0.28, 0.06),
    "sand": (0.70, 0.62, 0.46),
    "steel_grey": (0.36, 0.39, 0.42),
}

RUBBER = (0.045, 0.045, 0.05)
CHROME = (0.83, 0.85, 0.88)
GLASS_TINT = (0.30, 0.38, 0.44)
LENS_CLEAR = (0.86, 0.90, 0.94)
LENS_AMBER = (0.95, 0.55, 0.08)
LENS_RED = (0.60, 0.03, 0.04)
INTERIOR_DARK = (0.10, 0.10, 0.11)
INTERIOR_TRIM = (0.16, 0.155, 0.15)
SEAT_FABRIC = (0.17, 0.17, 0.19)
CARPET = (0.07, 0.07, 0.075)
ENGINE_BAY = (0.13, 0.13, 0.14)


def _rgb(c) -> tuple[float, float, float]:
    if isinstance(c, str):
        key = c.strip().lower().replace(" ", "_")
        if key not in PAINT_COLORS:
            raise ValueError(f"unknown paint color {c!r}; pick one of {sorted(PAINT_COLORS)}")
        return PAINT_COLORS[key]
    v = tuple(float(x) for x in c)
    if len(v) != 3 or any(not (0.0 <= x <= 1.0) for x in v):
        raise ValueError(f"color must be an (r, g, b) triple in [0, 1], got {c!r}")
    return v


# ---------------------------------------------------------------------------
# vehicle-class proportion table (metres, real-world scale)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassParams:
    name: str
    body_style: str          # notchback | hatch | wedge | bed | van
    length: float
    width: float
    height: float            # roof outer height
    wheelbase: float
    front_overhang: float
    track: float
    clearance: float
    wheel_diameter: float
    tire_width: float
    rim_diameter: float
    # heights
    y_belt_front: float      # beltline at cowl
    y_belt_rear: float       # beltline at C-pillar base
    y_hood_nose: float       # fender/hood line at the nose
    y_deck_tail: float       # deck / bed-rail height at the tail
    # plan (x measured from the front axle)
    x_cowl: float            # A-pillar base (end of hood)
    cabin_len: float         # cowl → C-pillar base
    # greenhouse
    windshield_rake: float   # degrees from vertical
    backlight_rake: float
    tumblehome: float        # roof-rail half width as a fraction of half width
    doors: int               # side doors (2 or 4)
    rows: int                # seating rows
    nose_taper: float        # half-width fraction at the very nose
    tail_taper: float        # half-width fraction at the very tail
    roof_crown: float        # roof transverse crown
    # pickups: bed length (cabin rear → tail); vans: extra glass rows
    bed_len: float = 0.0
    engine: str = "front"    # engine bay location hint


VEHICLE_CLASSES: dict[str, ClassParams] = {
    "sedan": ClassParams(
        name="sedan", body_style="notchback", length=4.88, width=1.86,
        height=1.45, wheelbase=2.88, front_overhang=0.98, track=1.60,
        clearance=0.140, wheel_diameter=0.652, tire_width=0.215,
        rim_diameter=0.406, y_belt_front=0.90, y_belt_rear=0.94,
        y_hood_nose=0.74, y_deck_tail=0.86, x_cowl=0.34, cabin_len=1.85,
        windshield_rake=30.0, backlight_rake=33.0, tumblehome=0.70,
        doors=4, rows=2, nose_taper=0.84, tail_taper=0.90, roof_crown=0.045,
    ),
    "hatchback": ClassParams(
        name="hatchback", body_style="hatch", length=4.35, width=1.80,
        height=1.47, wheelbase=2.65, front_overhang=0.90, track=1.55,
        clearance=0.145, wheel_diameter=0.620, tire_width=0.205,
        rim_diameter=0.381, y_belt_front=0.92, y_belt_rear=0.95,
        y_hood_nose=0.76, y_deck_tail=0.93, x_cowl=0.30, cabin_len=1.80,
        windshield_rake=32.0, backlight_rake=30.0, tumblehome=0.71,
        doors=4, rows=2, nose_taper=0.85, tail_taper=0.88, roof_crown=0.04,
    ),
    "suv": ClassParams(
        name="suv", body_style="hatch", length=4.78, width=1.92,
        height=1.72, wheelbase=2.85, front_overhang=0.98, track=1.64,
        clearance=0.210, wheel_diameter=0.750, tire_width=0.245,
        rim_diameter=0.457, y_belt_front=1.06, y_belt_rear=1.08,
        y_hood_nose=0.94, y_deck_tail=1.06, x_cowl=0.36, cabin_len=2.05,
        windshield_rake=33.0, backlight_rake=28.0, tumblehome=0.76,
        doors=4, rows=2, nose_taper=0.88, tail_taper=0.92, roof_crown=0.04,
    ),
    "sports": ClassParams(
        name="sports", body_style="wedge", length=4.45, width=1.88,
        height=1.24, wheelbase=2.60, front_overhang=1.00, track=1.62,
        clearance=0.110, wheel_diameter=0.670, tire_width=0.255,
        rim_diameter=0.432, y_belt_front=0.82, y_belt_rear=0.90,
        y_hood_nose=0.62, y_deck_tail=0.86, x_cowl=0.30, cabin_len=1.55,
        windshield_rake=38.0, backlight_rake=34.0, tumblehome=0.66,
        doors=2, rows=2, nose_taper=0.82, tail_taper=0.94, roof_crown=0.05,
    ),
    "pickup": ClassParams(
        name="pickup", body_style="bed", length=5.35, width=1.95,
        height=1.82, wheelbase=3.25, front_overhang=1.00, track=1.68,
        clearance=0.240, wheel_diameter=0.780, tire_width=0.265,
        rim_diameter=0.432, y_belt_front=1.10, y_belt_rear=1.10,
        y_hood_nose=0.98, y_deck_tail=1.12, x_cowl=0.40, cabin_len=1.70,
        windshield_rake=30.0, backlight_rake=15.0, tumblehome=0.78,
        doors=2, rows=2, nose_taper=0.90, tail_taper=0.96, roof_crown=0.035,
        bed_len=1.55,
    ),
    "van": ClassParams(
        name="van", body_style="van", length=5.10, width=1.95,
        height=2.05, wheelbase=3.20, front_overhang=0.95, track=1.66,
        clearance=0.165, wheel_diameter=0.700, tire_width=0.225,
        rim_diameter=0.406, y_belt_front=1.02, y_belt_rear=1.04,
        y_hood_nose=0.92, y_deck_tail=1.04, x_cowl=0.18, cabin_len=3.30,
        windshield_rake=24.0, backlight_rake=10.0, tumblehome=0.86,
        doors=2, rows=3, nose_taper=0.90, tail_taper=0.97, roof_crown=0.05,
    ),
}

# LOD knobs — segment-density scaling and feature gates.
LOD_PRESETS: dict[str, dict] = {
    "high": {"station_step": 0.10, "arch_step": 0.045, "lathe_seg": 28,
             "lugs": 40, "tube_sides": 10, "panel_grid": 12, "spokes": True,
             "lug_nuts": True, "engine": True, "fog_lamps": True},
    "mid": {"station_step": 0.16, "arch_step": 0.07, "lathe_seg": 20,
            "lugs": 24, "tube_sides": 8, "panel_grid": 8, "spokes": True,
            "lug_nuts": False, "engine": True, "fog_lamps": False},
    "low": {"station_step": 0.24, "arch_step": 0.11, "lathe_seg": 14,
            "lugs": 0, "tube_sides": 6, "panel_grid": 5, "spokes": False,
            "lug_nuts": False, "engine": False, "fog_lamps": False},
}


# ---------------------------------------------------------------------------
# part / spec dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HingeSpec:
    """One hinged closure: rotation axis through `origin`, ROM in degrees."""

    assembly: str            # e.g. "door_fl", "hood", "trunk"
    kind: str                # "door" | "hood" | "trunk" | "hatch"
    axis: np.ndarray         # unit axis (3,)
    origin: np.ndarray       # hinge pivot point (3,)
    rom_deg: tuple[float, float]   # (min, max), e.g. (0, 65)
    open_sign: float         # +1/-1: sign of angle that opens the closure
    gap_mm: float = 4.0      # design panel gap used for clearance checks


@dataclass
class VehiclePart:
    """One named part (shared mesh arrays are allowed via `transform`).

    `vertices`/`normals`/`uvs`/`faces`/`vertex_colors` are the part's LOCAL
    definition; `transform` places it into vehicle space (right-hand side
    parts share the left definition with a mirror transform, and tread lugs
    use `instances` for the around-the-tire array). Zero-copy sharing keeps
    memory flat exactly like `generation.complex_builder`.
    """

    name: str
    material: str
    vertices: np.ndarray     # (V, 3) float32, vehicle space when transform=I
    normals: np.ndarray
    uvs: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray  # (V, 3) float32 albedo zones
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    solid_volume_m3: float
    metadata: dict = field(default_factory=dict)
    transform: np.ndarray | None = None       # extra 4x4 placement (mirror…)
    instances: list[np.ndarray] = field(default_factory=list)

    @property
    def tri_count(self) -> int:
        return int(self.faces.shape[0]) * max(1, 1 + len(self.instances))


@dataclass
class VehicleSpec:
    """The full vehicle: named parts + articulations + dimensions."""

    vehicle_class: str
    params: dict
    parts: list[VehiclePart]
    articulations: dict[str, HingeSpec]
    assemblies: dict[str, list[int]]          # assembly name → part indices
    dimensions: dict[str, float]
    default_state: dict[str, float]           # assembly → opening fraction
    geometry_cache: dict = field(default_factory=dict, repr=False)

    # -- queries -----------------------------------------------------------
    def triangle_count(self) -> int:
        return int(sum(p.tri_count for p in self.parts))

    def part(self, name: str) -> VehiclePart:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(name)

    def aabbs(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {p.name: (p.aabb_min, p.aabb_max) for p in self.parts}

    # -- articulation ------------------------------------------------------
    def resolve_state(self, state=None) -> dict[str, float]:
        """Merge an explicit state over the default opening fractions."""
        out = dict(self.default_state)
        if state is None:
            return out
        if isinstance(state, dict):
            for k, v in state.items():
                out[k] = float(v)
        return out

    def assembly_transform(self, assembly: str, fraction: float) -> np.ndarray:
        """4x4 world rotation for an assembly opened by `fraction` (0..1)."""
        h = self.articulations[assembly]
        frac = float(np.clip(fraction, 0.0, 1.0))
        ang = math.radians(h.rom_deg[0] + frac * (h.rom_deg[1] - h.rom_deg[0]))
        ang *= h.open_sign
        return _axis_angle_matrix(h.axis, ang, h.origin)

    def bake(self, state=None) -> list[VehiclePart]:
        """World-space part copies with articulation `state` applied.

        Parts not in an articulated assembly pass through with their
        placement transform applied. Mirrors (det < 0) get winding flipped.
        Vertex colors are carried through untouched (they are albedo zones,
        not lighting).
        """
        resolved = self.resolve_state(state)
        asm_rot: dict[int, np.ndarray] = {}
        for asm, idxs in self.assemblies.items():
            frac = resolved.get(asm, 0.0)
            if abs(frac) < 1e-9:
                continue
            rot = self.assembly_transform(asm, frac)
            for i in idxs:
                asm_rot[i] = rot @ asm_rot[i] if i in asm_rot else rot

        baked: list[VehiclePart] = []
        for i, p in enumerate(self.parts):
            base = np.eye(4) if p.transform is None else np.asarray(p.transform)
            placements = [base] + [base @ m for m in p.instances]
            for inst_i, m in enumerate(placements):
                world = asm_rot.get(i, np.eye(4)) @ m
                v, n = _apply_transform(p.vertices, p.normals, world)
                f = p.faces
                if float(np.linalg.det(world[:3, :3])) < 0.0:
                    f = f[:, [0, 2, 1]]
                label = p.name if inst_i == 0 else f"{p.name}#{inst_i}"
                baked.append(VehiclePart(
                    name=label, material=p.material, vertices=v, normals=n,
                    uvs=p.uvs.copy(), faces=f.copy(),
                    vertex_colors=p.vertex_colors.copy(),
                    aabb_min=v.min(axis=0), aabb_max=v.max(axis=0),
                    solid_volume_m3=p.solid_volume_m3,
                    metadata=dict(p.metadata),
                ))
        return baked

    def to_analytic_parts(self, state=None):
        """Adapter for `generation.analytic_mesh.AnalyticPart` consumers."""
        from .analytic_mesh import AnalyticPart

        out = []
        for p in self.bake(state):
            out.append(AnalyticPart(
                label=p.name, kind="vehicle", material=p.material,
                vertices=p.vertices, normals=p.normals, uvs=p.uvs,
                faces=p.faces, aabb_min=p.aabb_min, aabb_max=p.aabb_max,
                solid_volume_m3=p.solid_volume_m3,
            ))
        return out

    def summary(self) -> dict:
        return {
            "class": self.vehicle_class,
            "dimensions": dict(self.dimensions),
            "triangles": self.triangle_count(),
            "parts": len(self.parts),
            "assemblies": {k: len(v) for k, v in self.assemblies.items()},
            "articulations": {
                k: {"kind": h.kind, "rom_deg": list(h.rom_deg)}
                for k, h in self.articulations.items()
            },
        }


# ---------------------------------------------------------------------------
# small linear-algebra helpers
# ---------------------------------------------------------------------------


def _apply_transform(vertices: np.ndarray, normals: np.ndarray, T: np.ndarray):
    """Apply a 4x4; normals via the inverse-transpose (float32 out)."""
    T = np.asarray(T, dtype=np.float64)
    h = np.concatenate([vertices.astype(np.float64),
                        np.ones((vertices.shape[0], 1))], axis=1)
    w = (h @ T.T)[:, :3]
    M = T[:3, :3]
    if abs(float(np.linalg.det(M))) < 1e-12:
        n = normals.astype(np.float64).copy()
    else:
        n = normals.astype(np.float64) @ np.linalg.inv(M)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return w.astype(np.float32), n.astype(np.float32)


def _translation(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


def _axis_angle_matrix(axis, angle: float, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Rotation of `angle` rad about unit `axis` through `origin` (4x4)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-30)
    x, y, z = a
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    o = np.asarray(origin, dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = o - R @ o
    return m


def _mirror_z() -> np.ndarray:
    m = np.eye(4)
    m[2, 2] = -1.0
    return m


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    v0 = vertices[faces[:, 0]].astype(np.float64)
    v1 = vertices[faces[:, 1]].astype(np.float64)
    v2 = vertices[faces[:, 2]].astype(np.float64)
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


# ---------------------------------------------------------------------------
# mesh helpers
# ---------------------------------------------------------------------------


def _finalize(v: np.ndarray, n: np.ndarray, u: np.ndarray, f: np.ndarray) -> Mesh:
    """Enforce outward consistency: faces agree with vertex normals; if the
    enclosed volume is still negative, flip faces AND normals together."""
    v0 = v[f[:, 0]].astype(np.float64)
    v1 = v[f[:, 1]].astype(np.float64)
    v2 = v[f[:, 2]].astype(np.float64)
    fn = np.cross(v1 - v0, v2 - v0)
    vn = n[f].astype(np.float64).mean(axis=1)
    flip = np.einsum("ij,ij->i", fn, vn) < 0.0
    f = f.copy()
    f[flip] = f[flip][:, [0, 2, 1]]
    if _signed_volume(v, f) < 0.0:
        f = f[:, [0, 2, 1]]
        n = -n
    return (v.astype(np.float32), n.astype(np.float32),
            u.astype(np.float32), f.astype(np.int64))


def _merge(chunks: list[Mesh]) -> Mesh:
    vs, ns, us, fs = [], [], [], []
    off = 0
    for v, n, u, f in chunks:
        vs.append(v)
        ns.append(n)
        us.append(u)
        fs.append(f + off)
        off += v.shape[0]
    return (np.concatenate(vs), np.concatenate(ns),
            np.concatenate(us), np.concatenate(fs))


def loft_sections(profiles: list[np.ndarray], positions: list[float],
                  caps: bool = True) -> Mesh:
    """Loft a stack of custom cross-sections along +X.

    Every profile is an (N, 2) loop in (y, z) with identical point count and
    cyclic correspondence (the vehicle section generators guarantee this).
    Unlike `slicer.loft` (single base profile, convexity-friendly centroid
    heuristic), normals here come from the profile winding, so non-convex
    sections — engine bays, cabin tubs, trunk wells — shade correctly.
    """
    if len(profiles) < 2:
        raise ValueError("loft_sections needs at least 2 stations")
    n = profiles[0].shape[0]
    if any(p.shape != (n, 2) for p in profiles):
        raise ValueError("all sections must share the same point count")
    order = np.argsort(positions)
    profiles = [profiles[int(i)] for i in order]
    pos = np.asarray([positions[int(i)] for i in order], dtype=np.float64)
    nr = len(profiles)

    v = np.zeros((nr * n, 3))
    for i, p in enumerate(profiles):
        v[i * n:(i + 1) * n] = np.stack(
            [np.full(n, pos[i]), p[:, 0], p[:, 1]], axis=-1)

    # Smooth normals from the two tangent directions. cross(t_prof, t_axis)
    # is outward for a CCW (in y,z) profile; the global _finalize pass makes
    # the whole mesh consistent regardless of input handedness.
    nrm = np.zeros_like(v)
    for i in range(nr):
        ring = v[i * n:(i + 1) * n]
        prev3 = v[max(i - 1, 0) * n:(max(i - 1, 0) + 1) * n]
        next3 = v[min(i + 1, nr - 1) * n:(min(i + 1, nr - 1) + 1) * n]
        t_axis = next3 - prev3
        t_prof = np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0)
        nn = np.cross(t_prof, t_axis)
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-12
        nrm[i * n:(i + 1) * n] = nn

    uv = np.zeros((nr * n, 2))
    u_col = np.linspace(0.0, 1.0, n, endpoint=False)
    span = max(pos[-1] - pos[0], 1e-12)
    for i in range(nr):
        uv[i * n:(i + 1) * n, 0] = u_col
        uv[i * n:(i + 1) * n, 1] = (pos[i] - pos[0]) / span

    j = np.arange(n, dtype=np.int64)
    j1 = (j + 1) % n
    bands = []
    for i in range(nr - 1):
        a, b = i * n + j, i * n + j1
        c, d = (i + 1) * n + j1, (i + 1) * n + j
        bands.append(np.stack([a, d, c], axis=1))
        bands.append(np.stack([a, c, b], axis=1))
    faces = np.concatenate(bands, axis=0)

    extra_v, extra_n, extra_u, cap_chunks = [], [], [], []
    base = nr * n
    if caps:
        for ci, (ri, sign) in enumerate(((0, -1.0), (nr - 1, 1.0))):
            ring3 = v[ri * n:(ri + 1) * n]
            centre3 = ring3.mean(axis=0)
            cidx = base + ci
            cap_n = np.array([sign, 0.0, 0.0])
            cf = np.stack([np.full(n, cidx, dtype=np.int64),
                           ri * n + j, ri * n + j1], axis=1)
            probe = np.cross(v[cf[:, 1]] - centre3[None, :],
                             v[cf[:, 2]] - centre3[None, :])
            if float((probe @ cap_n).mean()) < 0.0:
                cf = cf[:, [0, 2, 1]]
            extra_v.append(centre3)
            extra_n.append(cap_n)
            extra_u.append(np.array([0.5, 0.5]))
            cap_chunks.append(cf)
    if cap_chunks:
        v = np.concatenate([v, np.asarray(extra_v).reshape(-1, 3)], axis=0)
        nrm = np.concatenate([nrm, np.asarray(extra_n).reshape(-1, 3)], axis=0)
        uv = np.concatenate([uv, np.asarray(extra_u).reshape(-1, 2)], axis=0)
        faces = np.concatenate([faces, *cap_chunks], axis=0)
    return _finalize(v, nrm, uv, faces)


def lathe(profile: np.ndarray, segments: int = 28) -> Mesh:
    """Revolve a closed (N, 2) (radius, y) profile loop around the Y axis.

    A closed profile loop yields a watertight solid of revolution (tire
    carcass, rim barrel) with correct smooth normals from the profile
    winding. The angular seam shares vertices (wrapped indexing), so the
    result is a true 2-manifold, not a duplicated-edge strip.
    """
    profile = np.asarray(profile, dtype=np.float64).reshape(-1, 2)
    npts = profile.shape[0]
    seg = max(6, int(segments))
    th = np.linspace(0.0, TAU, seg, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    r = profile[:, 0]
    y = profile[:, 1]
    # (npts, seg, 3) — ring per profile point.
    pos = np.stack([r[:, None] * ct[None, :],
                    np.broadcast_to(y[:, None], (npts, seg)),
                    r[:, None] * st[None, :]], axis=-1)
    # In-plane outward normal of the profile loop (perp to tangent).
    tang = np.roll(profile, -1, axis=0) - np.roll(profile, 1, axis=0)
    n2 = np.stack([tang[:, 1], -tang[:, 0]], axis=-1)
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12
    # Orient against the profile centroid (radially outward reference).
    cent = profile.mean(axis=0)
    out = profile - cent[None, :]
    flip = np.einsum("ij,ij->i", n2, out) < 0.0
    n2[flip] *= -1.0
    nrm = np.stack([n2[:, 0][:, None] * ct[None, :],
                    np.broadcast_to(n2[:, 1][:, None], (npts, seg)),
                    n2[:, 0][:, None] * st[None, :]], axis=-1)
    uv = np.stack([np.broadcast_to((th / TAU)[None, :], (npts, seg)),
                   np.broadcast_to((np.arange(npts) / max(npts - 1, 1))[:, None],
                                   (npts, seg))], axis=-1)
    idx = np.arange(npts * seg).reshape(npts, seg)
    j1 = (np.arange(seg) + 1) % seg                    # wrapped seam
    a = idx[:, :].ravel()
    b = idx[:, j1].ravel()
    c = np.roll(idx[:, j1], -1, axis=0).ravel()
    d = np.roll(idx[:, :], -1, axis=0).ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return _finalize(pos.reshape(-1, 3), nrm.reshape(-1, 3),
                     uv.reshape(-1, 2), faces)


def ring_mesh(major_r: float, minor_r: float, seg_u: int = 20,
              seg_v: int = 8) -> Mesh:
    """Low-segment torus (steering-wheel rim) in the local XY plane."""
    su, sv = max(6, seg_u), max(4, seg_v)
    u = np.linspace(0.0, TAU, su + 1)
    vv = np.linspace(0.0, TAU, sv + 1)
    cu, su_ = np.cos(u), np.sin(u)
    cv, sv_ = np.cos(vv), np.sin(vv)
    pos = np.stack([(major_r + minor_r * cv[:, None]) * cu[None, :],
                    (major_r + minor_r * cv[:, None]) * su_[None, :],
                    np.broadcast_to((minor_r * sv_)[:, None], (sv + 1, su + 1))],
                   axis=-1)
    nrm = np.stack([cv[:, None] * cu[None, :],
                    cv[:, None] * su_[None, :],
                    np.broadcast_to(sv_[:, None], (sv + 1, su + 1))], axis=-1)
    uv = np.stack([np.broadcast_to((u / TAU)[None, :], (sv + 1, su + 1)),
                   np.broadcast_to((vv / TAU)[:, None], (sv + 1, su + 1))],
                  axis=-1)
    idx = np.arange((sv + 1) * (su + 1)).reshape(sv + 1, su + 1)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[1:, :-1].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return _finalize(pos.reshape(-1, 3), nrm.reshape(-1, 3),
                     uv.reshape(-1, 2), faces)


def box_mesh(size=(1.0, 1.0, 1.0)) -> Mesh:
    """Axis-aligned unit box builder (24 verts, flat normals, 12 tris)."""
    sx, sy, sz = (float(s) / 2 for s in size)
    defs = [
        ((0, 0, 1), [(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]),
        ((0, 0, -1), [(sx, -sy, -sz), (-sx, -sy, -sz), (-sx, sy, -sz), (sx, sy, -sz)]),
        ((0, 1, 0), [(-sx, sy, -sz), (-sx, sy, sz), (sx, sy, sz), (sx, sy, -sz)]),
        ((0, -1, 0), [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz)]),
        ((1, 0, 0), [(sx, -sy, sz), (sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz)]),
        ((-1, 0, 0), [(-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz), (-sx, sy, -sz)]),
    ]
    vs, ns, us, fs = [], [], [], []
    for fi, (nrm, corners) in enumerate(defs):
        base = fi * 4
        vs.append(np.asarray(corners, dtype=np.float64))
        ns.append(np.tile(np.asarray(nrm, dtype=np.float64), (4, 1)))
        us.append(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64))
        fs.append(np.array([[base, base + 1, base + 2], [base, base + 2, base + 3]]))
    return _finalize(np.concatenate(vs), np.concatenate(ns),
                     np.concatenate(us), np.concatenate(fs))


def rounded_box(size=(1.0, 1.0, 1.0), exponent: float = 3.0,
                seg_u: int = 12, seg_v: int = 6) -> Mesh:
    """Low-segment superellipsoid — a softly rounded box (seats, consoles,
    mirror housings). exponent ~2.5–4 gives a car-interior 'padded' look."""
    rx, ry, rz = (float(s) / 2 for s in size)
    e1 = e2 = float(exponent)

    def sp(v_, e):
        return np.sign(v_) * np.abs(v_) ** e

    eta = np.linspace(-math.pi / 2, math.pi / 2, seg_v + 1)
    omega = np.linspace(0.0, TAU, seg_u + 1)
    ce, se = np.cos(eta), np.sin(eta)
    co, so = np.cos(omega), np.sin(omega)
    pos = np.stack([rx * sp(ce, e1)[:, None] * sp(co, e2)[None, :],
                    np.broadcast_to(ry * sp(se, e1)[:, None], (seg_v + 1, seg_u + 1)),
                    rz * sp(ce, e1)[:, None] * sp(so, e2)[None, :]], axis=-1)
    pos[0, :] = np.array([0.0, -ry, 0.0])
    pos[-1, :] = np.array([0.0, ry, 0.0])
    xr = np.clip(np.abs(pos[..., 0]) / max(rx, 1e-9), 1e-9, None)
    yr = np.clip(np.abs(pos[..., 1]) / max(ry, 1e-9), 1e-9, None)
    zr = np.clip(np.abs(pos[..., 2]) / max(rz, 1e-9), 1e-9, None)
    n = np.stack([np.sign(pos[..., 0]) * xr ** (2.0 / e2 - 1.0) / max(rx, 1e-9),
                  np.sign(pos[..., 1]) * yr ** (2.0 / e1 - 1.0) / max(ry, 1e-9),
                  np.sign(pos[..., 2]) * zr ** (2.0 / e2 - 1.0) / max(rz, 1e-9)],
                 axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n[0, :] = np.array([0.0, -1.0, 0.0])
    n[-1, :] = np.array([0.0, 1.0, 0.0])
    uv = np.stack([np.broadcast_to((omega / TAU)[None, :], (seg_v + 1, seg_u + 1)),
                   np.broadcast_to(((eta + math.pi / 2) / math.pi)[:, None],
                                   (seg_v + 1, seg_u + 1))], axis=-1)
    idx = np.arange((seg_v + 1) * (seg_u + 1)).reshape(seg_v + 1, seg_u + 1)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[1:, :-1].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return _finalize(pos.reshape(-1, 3), n.reshape(-1, 3),
                     uv.reshape(-1, 2), faces)


def cylinder_mesh(radius: float, height: float, segments: int = 16,
                  caps: bool = True) -> Mesh:
    """Low-segment Y-axis cylinder (brake discs, hubs, columns, vents)."""
    seg = max(6, int(segments))
    th = np.linspace(0.0, TAU, seg + 1)
    ct, st = np.cos(th), np.sin(th)
    ys = np.array([-height / 2, height / 2])
    pos = np.stack([np.broadcast_to(radius * ct[None, :], (2, seg + 1)),
                    np.broadcast_to(ys[:, None], (2, seg + 1)),
                    np.broadcast_to(radius * st[None, :], (2, seg + 1))], axis=-1)
    nrm = np.stack([np.broadcast_to(ct[None, :], (2, seg + 1)),
                    np.zeros((2, seg + 1)),
                    np.broadcast_to(st[None, :], (2, seg + 1))], axis=-1)
    uv = np.stack([np.broadcast_to((th / TAU)[None, :], (2, seg + 1)),
                   np.broadcast_to((ys / height + 0.5)[:, None], (2, seg + 1))],
                  axis=-1)
    idx = np.arange(2 * (seg + 1)).reshape(2, seg + 1)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[1:, :-1].ravel()
    faces = [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
    chunks = [(pos.reshape(-1, 3), nrm.reshape(-1, 3), uv.reshape(-1, 2),
               np.concatenate(faces))]
    if caps:
        for yv, sign in ((height / 2, 1.0), (-height / 2, -1.0)):
            ring = np.stack([radius * ct, np.full(seg + 1, yv), radius * st], -1)
            vv = np.concatenate([np.array([[0.0, yv, 0.0]]), ring])
            nn = np.tile(np.array([0.0, sign, 0.0]), (seg + 2, 1))
            uu = np.concatenate([np.array([[0.5, 0.5]]),
                                 np.stack([0.5 + 0.5 * ct, 0.5 + 0.5 * st], -1)])
            jj = np.arange(1, seg + 1, dtype=np.int64)
            if sign > 0:
                ff = np.stack([np.zeros(seg, dtype=np.int64), jj, jj + 1], 1)
            else:
                ff = np.stack([np.zeros(seg, dtype=np.int64), jj + 1, jj], 1)
            chunks.append((vv, nn, uu, ff))
    v, n, u, f = _merge(chunks)
    return _finalize(v, n, u, f)


def tube_along(points, radius: float, sides: int = 8, caps: bool = True,
               radius_end: float | None = None) -> Mesh:
    """Pipe swept along a 3-D polyline with parallel-transport frames
    (pillars, seals, steering column). Tapers toward `radius_end` if set."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    k = pts.shape[0]
    if k < 2:
        raise ValueError("tube_along needs at least 2 points")
    sides = max(4, int(sides))
    r2 = radius if radius_end is None else float(radius_end)
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    # Parallel-transport a stable normal along the path.
    n0 = np.array([0.0, 0.0, 1.0])
    if abs(float(tang[0] @ n0)) > 0.9:
        n0 = np.array([1.0, 0.0, 0.0])
    nrms = np.zeros((k, 3))
    nrms[0] = n0 - tang[0] * (tang[0] @ n0)
    nrms[0] /= np.linalg.norm(nrms[0]) + 1e-12
    for i in range(1, k):
        p = nrms[i - 1] - tang[i] * (tang[i] @ nrms[i - 1])
        nrms[i] = p / (np.linalg.norm(p) + 1e-12)
    bnm = np.cross(tang, nrms)
    th = np.linspace(0.0, TAU, sides + 1)
    ct, st = np.cos(th), np.sin(th)
    t_par = np.linspace(0.0, 1.0, k)
    r_vert = radius + (r2 - radius) * t_par
    ring_n = ct[None, :, None] * nrms[:, None, :] + st[None, :, None] * bnm[:, None, :]
    pos = pts[:, None, :] + r_vert[:, None, None] * ring_n
    uv = np.stack([np.broadcast_to(t_par[:, None], (k, sides + 1)),
                   np.broadcast_to((th / TAU)[None, :], (k, sides + 1))], axis=-1)
    idx = np.arange(k * (sides + 1)).reshape(k, sides + 1)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[1:, :-1].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    chunks = [(pos.reshape(-1, 3), ring_n.reshape(-1, 3), uv.reshape(-1, 2), faces)]
    if caps:
        for row, sign in ((0, -1.0), (k - 1, 1.0)):
            ctr = pts[row]
            tv = tang[row] * sign
            ring = pos[row]
            vv = np.concatenate([ctr[None, :], ring])
            nn = np.tile(tv[None, :], (sides + 2, 1))
            uu = np.concatenate([np.array([[0.5, 0.5]]),
                                 np.stack([0.5 + 0.5 * ct, 0.5 + 0.5 * st], -1)])
            jj = np.arange(1, sides + 1, dtype=np.int64)
            if sign > 0:
                ff = np.stack([np.zeros(sides, dtype=np.int64), jj, jj + 1], 1)
            else:
                ff = np.stack([np.zeros(sides, dtype=np.int64), jj + 1, jj], 1)
            chunks.append((vv, nn, uu, ff))
    v, n, u, f = _merge(chunks)
    return _finalize(v, n, u, f)


def thin_panel(grid: np.ndarray, thickness: float = 0.004,
               grid_uv: bool = True) -> Mesh:
    """Turn an (R, C, 3) surface grid into a watertight thin solid.

    Two layers offset ±thickness/2 along the smooth grid normals plus edge
    walls around the boundary. Used for glass, hood, deck lids, door skins,
    roof, dash pads — any panel that must read as a *sheet*, never a block.
    """
    g = np.asarray(grid, dtype=np.float64)
    rows, cols = g.shape[0], g.shape[1]
    t_c = np.gradient(g, axis=1)
    t_r = np.gradient(g, axis=0)
    nrm = np.cross(t_r, t_c)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
    h = thickness / 2.0
    top = g + nrm * h
    bot = g - nrm * h

    layers = [top, bot]
    layer_n = [nrm, -nrm]
    vs, ns, us, fs = [], [], [], []
    off = 0
    for li, (layer, ln) in enumerate(zip(layers, layer_n)):
        vs.append(layer.reshape(-1, 3))
        ns.append(ln.reshape(-1, 3))
        if grid_uv:
            uu = np.stack([np.broadcast_to(
                (np.arange(cols) / max(cols - 1, 1))[None, :], (rows, cols)),
                np.broadcast_to(
                (np.arange(rows) / max(rows - 1, 1))[:, None], (rows, cols))],
                axis=-1)
        else:
            uu = np.zeros((rows, cols, 2))
        us.append(uu.reshape(-1, 2))
        idx = np.arange(rows * cols).reshape(rows, cols) + off
        a = idx[:-1, :-1].ravel()
        b = idx[:-1, 1:].ravel()
        c = idx[1:, 1:].ravel()
        d = idx[1:, :-1].ravel()
        f = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
        if li == 1:  # bottom layer faces the other way
            f = f[:, [0, 2, 1]]
        fs.append(f)
        off += rows * cols
    # Boundary walls: walk the grid perimeter, stitch top ↔ bottom.
    perim = ( [(0, c_) for c_ in range(cols)] +
              [(r_, cols - 1) for r_ in range(1, rows)] +
              [(rows - 1, c_) for c_ in range(cols - 2, -1, -1)] +
              [(r_, 0) for r_ in range(rows - 2, 0, -1)] )
    pidx = np.array([r_ * cols + c_ for r_, c_ in perim], dtype=np.int64)
    m = len(pidx)
    t_idx = pidx                    # top layer indices (offset 0)
    b_idx = pidx + rows * cols      # bottom layer indices
    wall_v_idx = np.concatenate([t_idx, b_idx])
    wall_pos = np.concatenate([top.reshape(-1, 3)[t_idx],
                               bot.reshape(-1, 3)[b_idx - rows * cols]])
    # Wall normals: outward in the surface plane (⊥ to the boundary tangent).
    bnd = g.reshape(-1, 3)[pidx]
    tang = np.roll(bnd, -1, axis=0) - np.roll(bnd, 1, axis=0)
    n_bnd = nrm.reshape(-1, 3)[pidx]
    wn = np.cross(tang, n_bnd)
    wn /= np.linalg.norm(wn, axis=1, keepdims=True) + 1e-12
    wall_n = np.concatenate([wn, wn])
    wall_u = np.stack([np.arange(2 * m) % max(m - 1, 1) / max(m - 1, 1),
                       np.concatenate([np.ones(m), np.zeros(m)])], axis=-1)
    j = np.arange(m, dtype=np.int64)
    j1 = (j + 1) % m
    wf = np.concatenate([
        np.stack([j, j1, m + j1], 1),
        np.stack([j, m + j1, m + j], 1),
    ]) + off
    vs.append(wall_pos)
    ns.append(wall_n)
    us.append(wall_u)
    fs.append(wf)
    # Offsets are tracked manually above (layers + wall share one index
    # space) — concatenate directly, do NOT re-offset via _merge.
    v = np.concatenate(vs)
    n = np.concatenate(ns)
    u = np.concatenate(us)
    f = np.concatenate(fs)
    return _finalize(v, n, u, f)


def _color_fill(mesh: Mesh, color) -> np.ndarray:
    """Uniform (V, 3) float32 vertex-color array for a mesh."""
    c = np.asarray(color, dtype=np.float32).reshape(1, 3)
    return np.repeat(c, mesh[0].shape[0], axis=0)


def _panel_colors(grid: np.ndarray, grid_colors: np.ndarray) -> np.ndarray:
    """Expand (rows*cols, 3) grid-point colors into `thin_panel` vertex
    order: top layer, bottom layer, then the boundary wall (top, bottom)."""
    rows, cols = grid.shape[0], grid.shape[1]
    gc = np.asarray(grid_colors, dtype=np.float32).reshape(rows * cols, 3)
    perim_idx = ([(0, c) for c in range(cols)]
                 + [(r, cols - 1) for r in range(1, rows)]
                 + [(rows - 1, c) for c in range(cols - 2, -1, -1)]
                 + [(r, 0) for r in range(rows - 2, 0, -1)])
    wall = gc[[r * cols + c for r, c in perim_idx]]
    return np.concatenate([gc, gc, wall, wall], axis=0)


def _volume_of(mesh: Mesh) -> float:
    return abs(_signed_volume(mesh[0], mesh[3]))


# ---------------------------------------------------------------------------
# body layout
# ---------------------------------------------------------------------------

# Section point layout (22 points, fixed correspondence across stations).
# Order: bottom centre → right flank (z<0) up → top → left flank down → close.
# Indices are used by the vertex-color zoning rules.
N_SECTION_POINTS = 22
J_UNDER = (0, 1, 2, 20, 21)          # underbody / floor edge
J_ARCH_R = (2, 3, 4, 5)              # right arch-blend zone
J_ARCH_L = (17, 18, 19, 20)          # left arch-blend zone
J_FLANK_LO = (5, 6, 16, 17)          # lower flank (gradient shade)
J_BELT = (7, 8, 14, 15)              # shoulder crease / beltline
J_TOP = (9, 10, 11, 12, 13)          # top band (bay / tub / deck / well)


@dataclass
class _Layout:
    """All key longitudinal positions/heights derived from ClassParams."""
    cp: ClassParams
    x_nose: float
    x_tail: float
    x_rear_axle: float
    x_cabin_rear: float
    x_roof_front: float
    x_roof_rear: float
    x_b_pillar: float
    y_roof: float
    y_wc: float                      # wheel centre height
    r_arch: float                    # wheel-arch cutter radius
    z_wheel: float                   # tyre centre-plane |z|
    y_bay: float                     # engine-bay floor
    y_trunk: float                   # trunk-well floor
    y_floor_int: float               # cabin interior floor
    w_max: float                     # half width

    @classmethod
    def build(cls, cp: ClassParams) -> "_Layout":
        x_nose = -cp.front_overhang
        x_tail = cp.wheelbase + (cp.length - cp.wheelbase - cp.front_overhang)
        y_roof = cp.height - 0.012
        x_cabin_rear = cp.x_cowl + cp.cabin_len
        dz_f = y_roof - cp.y_belt_front
        dz_r = y_roof - cp.y_belt_rear
        x_roof_front = cp.x_cowl + dz_f * math.tan(math.radians(cp.windshield_rake))
        x_roof_rear = x_cabin_rear - dz_r * math.tan(math.radians(cp.backlight_rake))
        # Guard: the roof must keep a sane length even for upright vans.
        x_roof_rear = max(x_roof_rear, x_roof_front + 0.45)
        y_wc = cp.wheel_diameter / 2.0
        return cls(
            cp=cp, x_nose=x_nose, x_tail=x_tail, x_rear_axle=cp.wheelbase,
            x_cabin_rear=x_cabin_rear, x_roof_front=x_roof_front,
            x_roof_rear=x_roof_rear,
            x_b_pillar=cp.x_cowl + 0.55 * cp.cabin_len,
            y_roof=y_roof, y_wc=y_wc,
            r_arch=y_wc + 0.045, z_wheel=cp.track / 2.0,
            y_bay=max(cp.clearance + 0.26, cp.y_hood_nose - 0.30),
            y_trunk=max(cp.clearance + 0.26, cp.y_deck_tail - 0.38),
            y_floor_int=cp.clearance + 0.17,
            w_max=cp.width / 2.0,
        )

    # -- longitudinal curves -------------------------------------------------
    def y_belt(self, x: float) -> float:
        """Beltline height — gentle wedge rise from cowl to C-pillar."""
        cp = self.cp
        t = float(np.clip((x - cp.x_cowl) / max(self.x_cabin_rear - cp.x_cowl, 1e-9),
                          0.0, 1.0))
        return cp.y_belt_front + t * (cp.y_belt_rear - cp.y_belt_front)

    def y_hood(self, x: float) -> float:
        """Hood centre line: nose height → cowl (beltline front − cap)."""
        cp = self.cp
        t = float(np.clip((x - self.x_nose) / max(cp.x_cowl - self.x_nose, 1e-9),
                          0.0, 1.0))
        # Slight S-curve: the last third climbs toward the cowl.
        t = t * t * (3.0 - 2.0 * t)
        return cp.y_hood_nose + t * (cp.y_belt_front - 0.045 - cp.y_hood_nose)

    def y_deck(self, x: float) -> float:
        """Rear deck line: C-pillar base (beltline rear) → tail."""
        cp = self.cp
        t = float(np.clip((x - self.x_cabin_rear) /
                          max(self.x_tail - self.x_cabin_rear, 1e-9), 0.0, 1.0))
        t = t * t * (3.0 - 2.0 * t)
        return cp.y_belt_rear + t * (cp.y_deck_tail - cp.y_belt_rear)

    def half_width(self, x: float) -> float:
        """Plan-view taper: narrow nose, full-width cabin, tucked tail."""
        cp = self.cp
        pts = [
            (self.x_nose, cp.nose_taper),
            (self.x_nose + 0.35 * cp.front_overhang, 0.94),
            (-0.10, 1.0),
            (cp.x_cowl + 0.4 * cp.cabin_len, 1.0),
            (cp.wheelbase, 0.985),
            (self.x_tail - 0.25, 0.96 if cp.body_style != "van" else 0.985),
            (self.x_tail, cp.tail_taper),
        ]
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        return float(np.interp(x, xs, ys)) * self.w_max

    # -- body regions ---------------------------------------------------------
    def region(self, x: float) -> str:
        """Longitudinal body region driving the section's top openness."""
        cp = self.cp
        if x < self.x_nose + 0.30:
            return "nose"
        if x < cp.x_cowl - 0.02:
            return "bay"
        if x < cp.x_cowl + 0.14:
            return "cowl"                       # plenum panel at windshield base
        if cp.body_style == "bed":
            return "bed" if x < self.x_tail - 0.055 else "tail_bed"
        if x < self.x_cabin_rear + 0.07:
            return "cabin"
        if x > self.x_tail - 0.28:
            return "tail"
        return "trunk"

    def top_profile(self, x: float) -> tuple[float, float, float, float, float]:
        """(y_top_outer, y_top_mid, y_top_centre, w_top, w_mid_frac).

        Closed decks (nose/cowl/tail) put all three at the local deck
        height; open regions (bay/cabin/trunk/bed) dip to their well floors
        so the section is a U — that is what makes the interior / engine
        bay / trunk / pickup bed real volumes instead of painted humps.
        `w_mid_frac` sets how far outboard the mid-wall point sits: engine
        bays and pickup beds need near-vertical walls so the hood/tailgate
        edges clear the fender tops; cabins slope inboard to the floor.
        """
        cp = self.cp
        reg = self.region(x)
        w = self.half_width(x)
        if reg == "nose":
            # fixed front clip panel: sits a panel-gap below the hood's
            # front edge so the two never fight for the same space
            y = self.y_hood(x) - 0.012
            return y, y, y, 0.90 * w, 0.55
        if reg == "bay":
            yt = self.y_hood(x) + 0.035            # fender line proud of hood
            return yt, 0.5 * (yt + self.y_bay), self.y_bay, 0.86 * w, 0.90
        if reg == "cowl":
            y = cp.y_belt_front - 0.04             # plenum below the beltline
            return y, y, y, 0.88 * w, 0.90
        if reg == "cabin":
            yb = self.y_belt(x)
            # interior wall drops nearly vertically from sill to floor pan
            return (yb - 0.015, self.y_floor_int + 0.06,
                    self.y_floor_int, 0.80 * w, 0.93)
        if reg == "trunk":
            yt = self.y_deck(x)
            return yt, 0.5 * (yt + self.y_trunk), self.y_trunk, 0.90 * w, 0.70
        if reg == "bed":
            yt = cp.y_belt_rear + 0.30             # bed rail above beltline
            yf = self.y_floor_int + 0.16           # bed floor
            return yt, 0.5 * (yt + yf), yf, 0.90 * w, 0.92
        if reg == "tail_bed":
            y = self.y_floor_int + 0.21            # rear valance below gate
            return y, y, y, 0.92 * w, 0.90
        # tail: closed deck (hatch/wedge/van tailgate area included)
        y = self.y_deck(x)
        return y, y, y, 0.90 * w, 0.55


def _door_spans(cp: ClassParams, lay: "_Layout") -> list[tuple[float, float]]:
    """Longitudinal spans of the side doors (shared by the tub's door
    cavity, the door skins, the cards and the seals)."""
    gap = 0.004
    if cp.doors >= 4:
        return [(cp.x_cowl + 0.14, lay.x_b_pillar - gap),
                (lay.x_b_pillar + gap, lay.x_cabin_rear - 0.30)]
    if cp.body_style == "van":
        return [(cp.x_cowl + 0.14, cp.x_cowl + 1.30)]
    if cp.body_style == "bed":
        return [(cp.x_cowl + 0.16, lay.x_cabin_rear - 0.24)]
    return [(cp.x_cowl + 0.14, lay.x_cabin_rear - 0.24)]


def make_section(lay: _Layout, x: float, recess: float = 0.0,
                 door_span: bool = False) -> np.ndarray:
    """The 22-point body cross-section at station `x` (before arch cutters).

    `recess` pulls the cabin-span flanks inboard. At `door_span` stations
    the flank above the rocker steps all the way inboard to the interior
    wall (w_top + 20 mm) — a real door cavity: with the door open you see
    the sill and interior wall, not a shallow dimple.
    """
    cp = lay.cp
    w = lay.half_width(x)
    y0 = cp.clearance
    yr = y0 + 0.055                       # rocker top
    reg = lay.region(x)
    if reg == "cabin":
        yb = lay.y_belt(x)
    elif reg == "cowl":
        yb = cp.y_belt_front              # plenum flanks meet the beltline
    elif x < cp.x_cowl:
        yb = lay.y_hood(x) + 0.06
    else:
        yb = lay.y_deck(x) + 0.02
    yb = max(yb, y0 + 0.30)
    yt_out, yt_mid, yt_ctr, w_top, w_mid_frac = lay.top_profile(x)
    # Flank width logic: at door spans the body opens (door cavity); in
    # the rest of the cabin the flank steps inboard modestly (trim gap).
    in_cabin = reg == "cabin"
    rec_lo = recess if in_cabin else 0.0            # rocker / sill
    rec_hi = 3.0 * recess if in_cabin else 0.0      # flank above the sill
    wr_lo = w - rec_lo
    if in_cabin and door_span:
        wr_hi = w_top + 0.020                       # deep door cavity
    else:
        wr_hi = w - rec_hi
    hs = max(yb - y0, 0.2)                # flank vertical span
    p = np.array([
        [y0, 0.0],                        # 0  underbody centre
        [y0, -0.45 * w],                  # 1
        [y0, -0.72 * w],                  # 2  floor edge (arch blend start)
        [yr, -0.85 * wr_lo],              # 3  rocker lower
        [yr + 0.04, -0.96 * wr_lo],       # 4  rocker outer
        [yb - 0.24 * hs, -1.00 * wr_hi],  # 5  flank low
        [yb - 0.10 * hs, -1.00 * wr_hi],  # 6  flank mid
        [yb - 0.03 * hs, -0.995 * wr_hi], # 7  shoulder crease lower
        [yb, -0.965 * wr_hi],             # 8  beltline (crease kink)
        [yt_out, -w_top],                 # 9  top outer (fender / sill / rail)
        [yt_mid, -w_mid_frac * w_top],    # 10 top mid (well wall)
        [yt_ctr, 0.0],                    # 11 top centre (well floor / deck)
        [yt_mid, w_mid_frac * w_top],     # 12
        [yt_out, w_top],                  # 13
        [yb, 0.965 * wr_hi],             # 14
        [yb - 0.03 * hs, 0.995 * wr_hi], # 15
        [yb - 0.10 * hs, 1.00 * wr_hi],  # 16
        [yb - 0.24 * hs, 1.00 * wr_hi],  # 17
        [yr + 0.04, 0.96 * wr_lo],       # 18
        [yr, 0.85 * wr_lo],              # 19
        [y0, 0.72 * w],                  # 20
        [y0, 0.45 * w],                  # 21
    ], dtype=np.float64)
    return p


def subtract_arch_from_section(section: np.ndarray, y_cut: float,
                               z_inner_frac: float = 0.55,
                               z_full_frac: float = 0.85) -> np.ndarray:
    """2-D CSG subtraction of a wheel-arch cutter from one cross-section.

    The cutter is a half-cylinder whose axis runs laterally (Z) through the
    wheel centre; at this station it removes everything below `y_cut` on the
    flanks (|z| beyond the blend band). Points are lifted onto the arch —
    never deleted — so the loop stays a single simple polygon and the loft
    provably contains no orphan geometry.
    """
    out = section.copy()
    w = float(np.abs(section[:, 1]).max()) + 1e-12
    z_in, z_full = z_inner_frac * w, z_full_frac * w
    for j in J_ARCH_R + J_ARCH_L:
        z = abs(float(section[j, 1]))
        t = float(np.clip((z - z_in) / max(z_full - z_in, 1e-9), 0.0, 1.0))
        t = t * t * (3.0 - 2.0 * t)                # smoothstep blend
        y_lift = section[j, 0] + (y_cut - section[j, 0]) * t
        out[j, 0] = max(section[j, 0], min(y_lift, y_cut))
    # Never let a lifted point cross the point above it (simple polygon).
    for a, b in ((5, 6), (17, 16)):
        out[a, 0] = min(out[a, 0], out[b, 0] - 0.004)
    return out


def arch_cut_height(lay: _Layout, x: float, x_wheel: float) -> float | None:
    """Cutter height at station `x`, or None outside the cutter span."""
    dx = x - x_wheel
    if abs(dx) >= lay.r_arch:
        return None
    return lay.y_wc + math.sqrt(max(lay.r_arch ** 2 - dx * dx, 0.0))


def build_tub(lay: _Layout, lod: dict) -> tuple[Mesh, dict]:
    """The main body shell: multi-section loft with wheel-arch subtraction.

    Returns (mesh, cache) where the cache carries the station positions, the
    structural (recessed + arched) sections and the unrecessed *surface*
    sections — the latter are the reference skin used to place door skins,
    glass, seals and mirrors exactly on the body surface.
    """
    cp = lay.cp
    step = lod["station_step"]
    xs = set(np.arange(lay.x_nose, lay.x_tail + 1e-9, step).tolist())
    xs.update([lay.x_nose, lay.x_tail, cp.x_cowl, lay.x_cabin_rear,
               lay.x_roof_front, lay.x_roof_rear, 0.0, cp.wheelbase])
    # Dense stations inside the arch cutters for a smooth round arch.
    for xw in (0.0, cp.wheelbase):
        span = lay.r_arch + 0.02
        xs.update(np.arange(xw - span, xw + span + 1e-9,
                            lod["arch_step"]).tolist())
    positions = sorted(xs)
    spans = _door_spans(cp, lay)

    struct, surface = [], []
    for x in positions:
        in_span = any(a - 0.005 <= x <= b + 0.005 for a, b in spans)
        sec = make_section(lay, x, recess=0.025, door_span=in_span)
        surf = make_section(lay, x, recess=0.0)
        for xw in (0.0, cp.wheelbase):
            y_cut = arch_cut_height(lay, x, xw)
            if y_cut is not None:
                sec = subtract_arch_from_section(sec, y_cut)
                surf = subtract_arch_from_section(surf, y_cut)
        struct.append(sec)
        surface.append(surf)
    mesh = loft_sections(struct, positions, caps=True)
    cache = {
        "positions": np.asarray(positions),
        "sections": np.stack(struct),        # (S, 22, 2) structural
        "surface": np.stack(surface),        # (S, 22, 2) unrecessed skin
        "arch": {"x_front": 0.0, "x_rear": cp.wheelbase, "r": lay.r_arch,
                 "y_wc": lay.y_wc},
    }
    return mesh, cache


def surface_z(cache: dict, x: float, y: float, side: float = 1.0) -> float:
    """Outer skin |z| of the unrecessed body at (x, y).

    Interpolates between bracketing stations (sections share point
    correspondence), then walks the flank/top portion of the loop for the
    segment bracketing `y`. Used to flush-mount door skins, glass and seals.
    """
    pos = cache["positions"]
    surf = cache["surface"]
    i = int(np.clip(np.searchsorted(pos, x) - 1, 0, len(pos) - 2))
    t = float(np.clip((x - pos[i]) / max(pos[i + 1] - pos[i], 1e-12), 0.0, 1.0))
    sec = surf[i] * (1.0 - t) + surf[i + 1] * t      # (22, 2) lerp
    # Scan the right half (indices 3..13) which spans flank → top outer.
    best = None
    for j in range(3, 13):
        y0, z0 = sec[j]
        y1, z1 = sec[j + 1]
        if (y0 - y) * (y1 - y) <= 0.0 and abs(y1 - y0) > 1e-9:
            s = (y - y0) / (y1 - y0)
            z = z0 + s * (z1 - z0)
            if best is None or abs(z) > abs(best):
                best = z
    if best is None:                                  # above/below the loop
        best = float(np.abs(sec[:, 1]).max())
    return math.copysign(abs(best), side)


# ---------------------------------------------------------------------------
# vehicle builder
# ---------------------------------------------------------------------------


def _beam_between(p0, p1, width: float, thick: float) -> tuple[Mesh, np.ndarray]:
    """A rectangular beam spanning p0 → p1 (frame strips, slats, spokes).

    Returns the local box mesh plus its placement transform (box local X
    runs along the beam).
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        raise ValueError("beam endpoints coincide")
    x_ax = d / length
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(x_ax @ up)) > 0.92:
        up = np.array([0.0, 0.0, 1.0])
    z_ax = np.cross(x_ax, up)
    z_ax /= np.linalg.norm(z_ax) + 1e-12
    y_ax = np.cross(z_ax, x_ax)
    m = np.eye(4)
    m[:3, 0] = x_ax
    m[:3, 1] = y_ax
    m[:3, 2] = z_ax
    m[:3, 3] = (p0 + p1) / 2.0
    return box_mesh((length, thick, width)), m


class _VehicleBuilder:
    """Accumulates named parts, assemblies and hinge specs for one vehicle."""

    def __init__(self, cp: ClassParams, lay: _Layout, lod: dict,
                 paint: tuple[float, float, float], livery: str | None):
        self.cp = cp
        self.lay = lay
        self.lod = lod
        self.paint = np.asarray(paint, dtype=np.float32)
        self.livery = livery
        self.parts: list[VehiclePart] = []
        self.assemblies: dict[str, list[int]] = {}
        self.articulations: dict[str, HingeSpec] = {}
        # Livery helpers.
        lum = 0.2126 * paint[0] + 0.7152 * paint[1] + 0.0722 * paint[2]
        self.stripe_color = (0.90, 0.90, 0.88) if lum < 0.45 else (0.10, 0.10, 0.11)
        self.two_tone_color = (0.06, 0.06, 0.07)

    # -- registration --------------------------------------------------------
    def add(self, name: str, mesh: Mesh, material: str, colors: np.ndarray,
            *, metadata: dict | None = None, transform: np.ndarray | None = None,
            instances: list[np.ndarray] | None = None,
            assembly: str | None = None) -> int:
        v, n, u, f = mesh
        world = np.eye(4) if transform is None else np.asarray(transform)
        vw = (np.concatenate([v.astype(np.float64),
                              np.ones((v.shape[0], 1))], 1) @ world.T)[:, :3]
        inst = list(instances or [])
        lo, hi = vw.min(axis=0), vw.max(axis=0)
        for m in inst:
            cw = (np.concatenate([v.astype(np.float64),
                                  np.ones((v.shape[0], 1))], 1) @ (world @ m).T)[:, :3]
            lo = np.minimum(lo, cw.min(axis=0))
            hi = np.maximum(hi, cw.max(axis=0))
        part = VehiclePart(
            name=name, material=material, vertices=v, normals=n, uvs=u, faces=f,
            vertex_colors=np.asarray(colors, dtype=np.float32),
            aabb_min=lo, aabb_max=hi, solid_volume_m3=_volume_of(mesh),
            metadata=dict(metadata or {}), transform=transform, instances=inst,
        )
        self.parts.append(part)
        idx = len(self.parts) - 1
        if assembly is not None:
            self.assemblies.setdefault(assembly, []).append(idx)
        return idx

    def add_hinge(self, assembly: str, kind: str, axis, origin,
                  rom_deg: tuple[float, float], open_sign: float) -> None:
        self.articulations[assembly] = HingeSpec(
            assembly=assembly, kind=kind,
            axis=np.asarray(axis, dtype=np.float64),
            origin=np.asarray(origin, dtype=np.float64),
            rom_deg=rom_deg, open_sign=float(open_sign),
        )

    # -- color helpers --------------------------------------------------------
    def paint_colors(self, mesh: Mesh, shade: float = 1.0) -> np.ndarray:
        return _color_fill(mesh, self.paint * shade)

    def livery_grid_colors(self, grid: np.ndarray, base=None) -> np.ndarray:
        """Per-vertex colors (thin_panel vertex order) for a top panel
        grid; applies racing stripes."""
        base = self.paint if base is None else np.asarray(base, dtype=np.float32)
        rows, cols = grid.shape[0], grid.shape[1]
        gc = np.repeat(base[None, :], rows * cols, axis=0)
        if self.livery == "racing_stripes":
            z = grid[:, :, 2].reshape(-1)
            mask = (np.abs(np.abs(z) - 0.13) < 0.055)
            gc[mask] = np.asarray(self.stripe_color, dtype=np.float32)
        return _panel_colors(grid, gc)

    def two_tone(self, colors: np.ndarray) -> np.ndarray:
        if self.livery == "two_tone":
            colors = np.broadcast_to(
                np.asarray(self.two_tone_color, dtype=np.float32)[None, :],
                colors.shape).copy()
        return colors


# ---------------------------------------------------------------------------
# fascia: bumpers, grille, lamps, mirrors
# ---------------------------------------------------------------------------


def _band_section(y_lo: float, y_hi: float, w: float) -> np.ndarray:
    """12-point stadium (rounded-rect) loop in (y, z) for bumper lofts."""
    r = min((y_hi - y_lo) * 0.5, 0.09)
    pts = [
        (y_lo, -0.55 * w), (y_lo, -0.85 * w),
        (y_lo + r, -0.98 * w), (y_hi - r, -0.98 * w),
        (y_hi, -0.85 * w), (y_hi, -0.55 * w),
        (y_hi, 0.55 * w), (y_hi, 0.85 * w),
        (y_hi - r, 0.98 * w), (y_lo + r, 0.98 * w),
        (y_lo, 0.85 * w), (y_lo, 0.55 * w),
    ]
    return np.asarray(pts, dtype=np.float64)


def build_bumpers(b: _VehicleBuilder) -> None:
    cp, lay = b.cp, b.lay
    y0 = cp.clearance
    for tag, x0, dirn in (("front", lay.x_nose, 1.0), ("rear", lay.x_tail, -1.0)):
        # bumpers are low valances, not full-height walls
        y_hi = cp.clearance + 0.42
        if tag == "front":
            y_hi = min(y_hi, lay.y_hood(lay.x_nose) - 0.02)
        else:
            y_hi = min(y_hi, lay.y_deck(lay.x_tail) + 0.01)
            if cp.body_style == "bed":
                y_hi = min(y_hi, lay.y_floor_int + 0.25)
        stations = [x0 - dirn * 0.015, x0 + dirn * 0.12, x0 + dirn * 0.26]
        sections = []
        for sx in stations:
            w = lay.half_width(float(np.clip(sx, lay.x_nose, lay.x_tail))) + 0.014
            sections.append(_band_section(y0 + 0.02, y_hi, w))
        mesh = loft_sections(sections, stations, caps=True)
        colors = b.paint_colors(mesh, shade=0.96)
        b.add(f"bumper_{tag}", mesh, "plastic", colors,
              metadata={"zone": "fascia"})


def build_fascia_details(b: _VehicleBuilder, cache: dict) -> None:
    """Grille + slats, head/taillamps, fog lamps, mirrors, panel-gap seals."""
    cp, lay = b.cp, b.lay
    y0 = cp.clearance
    w_n = lay.half_width(lay.x_nose)

    # -- grille: dark inset + chrome slats -----------------------------------
    gy0, gy1 = y0 + 0.16, lay.y_hood(lay.x_nose) - 0.10
    gz = 0.34 * w_n
    grid = np.zeros((3, 7, 3))
    for ri, yy in enumerate(np.linspace(gy0, gy1, 3)):
        for ci, zz in enumerate(np.linspace(-gz, gz, 7)):
            grid[ri, ci] = [lay.x_nose - 0.004 - 0.02 * (zz / gz) ** 2, yy, zz]
    mesh = thin_panel(grid, 0.006)
    b.add("grille", mesh, "plastic", _color_fill(mesh, (0.05, 0.05, 0.055)),
          metadata={"zone": "grille"})
    n_slats = 4 if b.lod["panel_grid"] > 6 else 3
    for si in range(n_slats):
        yy = gy0 + (gy1 - gy0) * (si + 0.5) / n_slats
        beam, m = _beam_between((lay.x_nose - 0.012, yy, -gz * 0.92),
                                (lay.x_nose - 0.012, yy, gz * 0.92), 0.012, 0.008)
        b.add(f"grille_slat_{si}", beam, "metal", _color_fill(beam, CHROME),
              transform=m, metadata={"zone": "grille"})

    # -- headlamps: clear lens pod + amber indicator --------------------------
    hy = lay.y_hood(lay.x_nose) - 0.055
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        hz = sgn * 0.60 * w_n
        pod = rounded_box((0.17, 0.085, 0.20), exponent=2.6, seg_u=10, seg_v=5)
        m = np.eye(4)
        m[:3, :3] = _axis_angle_matrix((0, 1, 0), sgn * math.radians(-16.0))[:3, :3]
        m[:3, 3] = [lay.x_nose + 0.115, hy, hz]
        cols = _color_fill(pod, LENS_CLEAR)
        b.add(f"headlamp_{side}", pod, "glass", cols, transform=m,
              metadata={"zone": "lens_clear"})
        ind = rounded_box((0.15, 0.06, 0.05), exponent=2.6, seg_u=8, seg_v=4)
        mi = np.eye(4)
        mi[:3, :3] = m[:3, :3]
        mi[:3, 3] = [lay.x_nose + 0.115, hy - 0.005, hz + sgn * 0.125]
        b.add(f"indicator_{side}", ind, "glass", _color_fill(ind, LENS_AMBER),
              transform=mi, metadata={"zone": "lens_amber"})

    # -- tail lamps: red lens + clear reverse strip ---------------------------
    w_t = lay.half_width(lay.x_tail)
    ty = lay.y_deck(lay.x_tail) - 0.085
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        tz = sgn * 0.60 * w_t
        pod = rounded_box((0.14, 0.09, 0.22), exponent=2.6, seg_u=10, seg_v=5)
        m = np.eye(4)
        m[:3, :3] = _axis_angle_matrix((0, 1, 0), sgn * math.radians(14.0))[:3, :3]
        m[:3, 3] = [lay.x_tail - 0.075, ty, tz]
        b.add(f"taillamp_{side}", pod, "glass", _color_fill(pod, LENS_RED),
              transform=m, metadata={"zone": "lens_red"})
        rev = rounded_box((0.12, 0.05, 0.06), exponent=2.6, seg_u=8, seg_v=4)
        mr = np.eye(4)
        mr[:3, :3] = m[:3, :3]
        mr[:3, 3] = [lay.x_tail - 0.075, ty - 0.005, tz - sgn * 0.13]
        b.add(f"reverse_{side}", rev, "glass", _color_fill(rev, LENS_CLEAR),
              transform=mr, metadata={"zone": "lens_clear"})

    # -- fog lamps (high detail) ----------------------------------------------
    if b.lod["fog_lamps"]:
        for side, sgn in (("l", 1.0), ("r", -1.0)):
            lamp = cylinder_mesh(0.045, 0.03, segments=12)
            # cylinder local Y → world X via rz 90
            m = _translation(lay.x_nose + 0.005, y0 + 0.12, sgn * 0.42 * w_n) @ \
                _axis_angle_matrix((0, 0, 1), math.radians(90.0))
            b.add(f"foglamp_{side}", lamp, "glass",
                  _color_fill(lamp, (0.35, 0.38, 0.42)), transform=m,
                  metadata={"zone": "lens_clear"})

    # -- side mirrors -----------------------------------------------------------
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        xm = cp.x_cowl + 0.05
        ym = lay.y_belt(xm) + 0.045
        zm = surface_z(cache, xm, ym - 0.04, sgn)
        stalk, ms = _beam_between((xm, ym - 0.02, zm),
                                  (xm + 0.01, ym + 0.01, zm + sgn * 0.075),
                                  0.028, 0.02)
        b.add(f"mirror_stalk_{side}", stalk, "plastic",
              b.paint_colors(stalk, 0.9), transform=ms, metadata={"zone": "mirror"})
        housing = rounded_box((0.10, 0.065, 0.12), exponent=2.8, seg_u=10, seg_v=5)
        mh = np.eye(4)
        mh[:3, 3] = [xm + 0.015, ym + 0.025, zm + sgn * 0.13]
        cols = b.two_tone(b.paint_colors(housing))
        b.add(f"mirror_{side}", housing, "plastic", cols, transform=mh,
              metadata={"zone": "mirror"})
        glass = box_mesh((0.008, 0.05, 0.095))
        mg = np.eye(4)
        mg[:3, 3] = [xm + 0.062, ym + 0.025, zm + sgn * 0.13]
        b.add(f"mirror_glass_{side}", glass, "glass",
              _color_fill(glass, (0.55, 0.60, 0.65)), transform=mg,
              metadata={"zone": "mirror"})


# ---------------------------------------------------------------------------
# greenhouse: roof, pillars, glass
# ---------------------------------------------------------------------------


def _rail_z(b: _VehicleBuilder, x: float) -> float:
    """Roof-rail |z| (greenhouse is narrower than the body — tumblehome)."""
    return b.cp.tumblehome * b.lay.half_width(x)


def build_greenhouse(b: _VehicleBuilder, cache: dict) -> None:
    cp, lay = b.cp, b.lay
    sides = b.lod["tube_sides"]
    xr0, xr1 = lay.x_roof_front, lay.x_roof_rear
    y_r = lay.y_roof

    # -- roof panel: thin curved sheet between the rails ----------------------
    rows = max(3, b.lod["panel_grid"] // 2)
    cols = max(5, b.lod["panel_grid"])
    grid = np.zeros((rows, cols, 3))
    for ri, xx in enumerate(np.linspace(xr0 + 0.02, xr1 - 0.02, rows)):
        zr = _rail_z(b, xx)
        for ci, zz in enumerate(np.linspace(-zr, zr, cols)):
            crown = cp.roof_crown * (1.0 - (zz / max(zr, 1e-9)) ** 2)
            grid[ri, ci] = [xx, y_r + crown, zz]
    mesh = thin_panel(grid, 0.006)
    colors = b.two_tone(b.livery_grid_colors(grid))
    b.add("roof", mesh, "metal", colors, metadata={"zone": "roof"})

    # -- roof rails -------------------------------------------------------------
    for sgn in (1.0, -1.0):
        pts = [(xx, y_r + cp.roof_crown * 0.4 - 0.012, sgn * _rail_z(b, xx))
               for xx in np.linspace(xr0 + 0.01, xr1 - 0.01, 4)]
        tube = tube_along(pts, 0.016, sides=sides)
        b.add(f"roof_rail_{'l' if sgn > 0 else 'r'}", tube, "metal",
              b.two_tone(b.paint_colors(tube)), metadata={"zone": "pillar"})

    # -- pillars ----------------------------------------------------------------
    def pillar(name, p_bot, p_top):
        tube = tube_along([p_bot,
                           [(p_bot[0] + p_top[0]) / 2, (p_bot[1] + p_top[1]) / 2,
                            (p_bot[2] + p_top[2]) / 2],
                           p_top], 0.021, sides=sides, radius_end=0.016)
        b.add(name, tube, "metal", b.two_tone(b.paint_colors(tube)),
              metadata={"zone": "pillar"})

    for sgn, tag in ((1.0, "l"), (-1.0, "r")):
        zb = surface_z(cache, cp.x_cowl, cp.y_belt_front, sgn)
        pillar(f"pillar_a_{tag}",
               (cp.x_cowl + 0.03, cp.y_belt_front - 0.02, zb - sgn * 0.012),
               (xr0 + 0.02, y_r - 0.008, sgn * (_rail_z(b, xr0) + 0.004)))
        if cp.doors >= 4:
            xb = lay.x_b_pillar
            zbb = surface_z(cache, xb, lay.y_belt(xb), sgn)
            pillar(f"pillar_b_{tag}",
                   (xb, lay.y_belt(xb) - 0.02, zbb - sgn * 0.012),
                   (xb, y_r - 0.008, sgn * (_rail_z(b, xb) + 0.004)))
        xc = lay.x_cabin_rear
        zcb = surface_z(cache, xc, cp.y_belt_rear, sgn)
        pillar(f"pillar_c_{tag}",
               (xc - 0.02, cp.y_belt_rear - 0.02, zcb - sgn * 0.012),
               (xr1 - 0.02, y_r - 0.008, sgn * (_rail_z(b, xr1) + 0.004)))

    # -- windshield: thin raked sheet with a dark sun strip ---------------------
    rows = max(4, b.lod["panel_grid"] // 2)
    cols = max(6, b.lod["panel_grid"])
    grid = np.zeros((rows, cols, 3))
    for ri, t in enumerate(np.linspace(0.0, 1.0, rows)):
        xx = cp.x_cowl + 0.045 + t * (xr0 - cp.x_cowl - 0.03)
        yy = cp.y_belt_front + 0.005 + t * (y_r - cp.y_belt_front - 0.012)
        z_lo = surface_z(cache, xx, yy, 1.0) - 0.010
        z_hi = _rail_z(b, xr0) + 0.006
        z_half = (1.0 - t) * z_lo + t * z_hi
        for ci, s in enumerate(np.linspace(-1.0, 1.0, cols)):
            bulge = 0.018 * (1.0 - s * s)      # transverse crown
            grid[ri, ci] = [xx - bulge, yy, s * z_half]
    mesh = thin_panel(grid, 0.004)
    gc = np.repeat(np.asarray(GLASS_TINT, dtype=np.float32)[None, :],
                   rows * cols, axis=0)
    top_rows = np.repeat(np.linspace(0.0, 1.0, rows)[:, None] > 0.80, cols, axis=1)
    gc[top_rows.reshape(-1)] *= 0.45         # factory tint band
    b.add("windshield", mesh, "glass", _panel_colors(grid, gc),
          metadata={"zone": "glass"})

    # -- backlight (rear glass) --------------------------------------------------
    grid = np.zeros((rows, cols, 3))
    for ri, t in enumerate(np.linspace(0.0, 1.0, rows)):
        xx = xr1 - 0.02 - t * (xr1 - lay.x_cabin_rear - 0.02)
        xx = max(xx, lay.x_cabin_rear + 0.02)
        yy = y_r - 0.012 - t * (y_r - cp.y_belt_rear - 0.02)
        z_lo = surface_z(cache, xx, yy, 1.0) - 0.010
        z_hi = _rail_z(b, xr1) + 0.006
        z_half = (1.0 - t) * z_hi + t * z_lo
        for ci, s in enumerate(np.linspace(-1.0, 1.0, cols)):
            bulge = 0.016 * (1.0 - s * s)
            grid[ri, ci] = [xx + bulge, yy, s * z_half]
    mesh = thin_panel(grid, 0.004)
    b.add("backlight", mesh, "glass", _color_fill(mesh, GLASS_TINT),
          metadata={"zone": "glass"})

    # -- fixed rear-quarter glass (2-door bodies) / van cargo windows ----------
    if cp.doors == 2 and cp.body_style not in ("bed", "van"):
        for sgn, tag in ((1.0, "l"), (-1.0, "r")):
            x0 = lay.x_b_pillar if cp.doors >= 4 else cp.x_cowl + 0.62 * cp.cabin_len
            _quarter_glass(b, cache, x0 + 0.05, lay.x_cabin_rear - 0.06, sgn, tag,
                           y_base=lay.y_belt((x0 + lay.x_cabin_rear) / 2))
    if cp.body_style == "van":
        for sgn, tag in ((1.0, "l"), (-1.0, "r")):
            _quarter_glass(b, cache, lay.x_b_pillar + 0.10,
                           lay.x_cabin_rear - 0.15, sgn, tag,
                           y_base=cp.y_belt_rear, tall=True)


def _quarter_glass(b: _VehicleBuilder, cache: dict, x0: float, x1: float,
                   sgn: float, tag: str, y_base: float, tall: bool = False) -> None:
    lay = b.lay
    y0 = y_base + 0.02
    y1 = lay.y_roof - (0.10 if tall else 0.14)
    rows, cols = 4, max(4, b.lod["panel_grid"] // 2)
    grid = np.zeros((rows, cols, 3))
    for ri, yy in enumerate(np.linspace(y0, y1, rows)):
        t = (yy - y0) / max(y1 - y0, 1e-9)
        for ci, xx in enumerate(np.linspace(x0, x1, cols)):
            zb = surface_z(cache, xx, yy, sgn) - sgn * 0.010
            zr = sgn * (_rail_z(b, xx) + 0.006)
            grid[ri, ci] = [xx, yy, (1.0 - t) * zb + t * zr]
    mesh = thin_panel(grid, 0.004)
    b.add(f"quarter_glass_{tag}", mesh, "glass", _color_fill(mesh, GLASS_TINT),
          metadata={"zone": "glass"})


# ---------------------------------------------------------------------------
# closures: doors, hood, trunk / hatch / tailgate — with hinge articulation
# ---------------------------------------------------------------------------


def _panel_gap_seal(b: _VehicleBuilder, name: str, pts) -> None:
    """A dark rubber tube along a panel-gap line (door shut lines, hood and
    trunk parting lines) — cheap geometry that reads as a real shut line."""
    tube = tube_along(pts, 0.0045, sides=6, caps=True)
    b.add(name, tube, "plastic", _color_fill(tube, RUBBER),
          metadata={"zone": "seal"})


def build_doors(b: _VehicleBuilder, cache: dict) -> None:
    cp, lay = b.cp, b.lay
    gap = 0.004
    names = ["fl", "rl"][: len(_door_spans(cp, lay))]
    spans = [(f"door_{n}", a, b_) for n, (a, b_) in
             zip(names, _door_spans(cp, lay))]
    n_rows = max(5, b.lod["panel_grid"] // 2)
    n_cols = max(6, b.lod["panel_grid"] - 2)

    for asm, x_f, x_r in spans:
        y_bot = cp.clearance + 0.075
        # Door front edge is raked (relieved toward the top) so the top
        # corner clears the fender/cowl flank through the swing arc.
        rake = 0.045
        # --- left side (driver side, +Z) -------------------------------------
        grid = np.zeros((n_rows, n_cols, 3))
        for ri, t in enumerate(np.linspace(0.0, 1.0, n_rows)):
            xf_t = x_f + rake * t
            for ci, xx in enumerate(np.linspace(xf_t, x_r, n_cols)):
                yy = y_bot + t * (lay.y_belt(xx) + 0.008 - y_bot)
                grid[ri, ci] = [xx, yy, surface_z(cache, xx, yy, 1.0) + 0.0015]
        shell = thin_panel(grid, 0.008)
        # Subtle lower-flank shading gradient (finer surface detail).
        gc = np.repeat(b.paint[None, :], n_rows * n_cols, axis=0)
        shade = np.linspace(0.90, 1.0, n_rows)[:, None]
        gc = (gc.reshape(n_rows, n_cols, 3) * shade[:, :, None]).reshape(-1, 3)
        b.add(f"{asm}_shell", shell, "metal", _panel_colors(grid, gc),
              assembly=asm, metadata={"zone": "door_skin", "side": "left"})

        # window frame: two verticals + chrome top rail
        y_top = lay.y_roof - 0.055
        xf_top = x_f + 0.10 if asm.startswith("door_f") else x_f + 0.03
        xr_top = x_r - 0.06
        zb_f = surface_z(cache, x_f, lay.y_belt(x_f), 1.0) - 0.006
        zb_r = surface_z(cache, x_r, lay.y_belt(x_r), 1.0) - 0.006
        zr_f = _rail_z(b, xf_top) + 0.004
        zr_r = _rail_z(b, xr_top) + 0.004
        for tag, p0, p1, mat_c in (
                ("frame_front",
                 (x_f + rake + 0.01, lay.y_belt(x_f + rake) + 0.01,
                  surface_z(cache, x_f + rake, lay.y_belt(x_f + rake), 1.0) - 0.006),
                 (xf_top, y_top, zr_f), b.paint),
                ("frame_rear", (x_r - 0.01, lay.y_belt(x_r) + 0.01, zb_r),
                 (xr_top, y_top, zr_r), b.paint),
                ("frame_top", (xf_top, y_top, zr_f), (xr_top, y_top, zr_r),
                 np.asarray(CHROME, dtype=np.float32))):
            beam, m = _beam_between(p0, p1, 0.034, 0.013)
            b.add(f"{asm}_{tag}", beam, "metal", _color_fill(beam, mat_c),
                  transform=m, assembly=asm, metadata={"zone": "door_frame"})

        # door glass: thin sheet on the greenhouse plane
        g_rows, g_cols = 4, n_cols
        ggrid = np.zeros((g_rows, g_cols, 3))
        for ri, t in enumerate(np.linspace(0.0, 1.0, g_rows)):
            for ci, s in enumerate(np.linspace(0.0, 1.0, g_cols)):
                xx = (x_f + rake + 0.03) + s * ((x_r - 0.045) - (x_f + rake + 0.03))
                y_lo = lay.y_belt(xx) + 0.015
                y_hi = y_top - 0.015
                yy = y_lo + t * (y_hi - y_lo)
                z_lo = surface_z(cache, xx, y_lo, 1.0) - 0.009
                z_hi = _rail_z(b, xx) + 0.004
                ggrid[ri, ci] = [xx, yy, (1.0 - t) * z_lo + t * z_hi]
        glass = thin_panel(ggrid, 0.004)
        b.add(f"{asm}_glass", glass, "glass", _color_fill(glass, GLASS_TINT),
              assembly=asm, metadata={"zone": "glass", "side": "left"})

        # handle + recess pocket
        xh = x_r - 0.17
        yh = lay.y_belt(xh) - 0.055
        zh = surface_z(cache, xh, yh, 1.0)
        handle = rounded_box((0.13, 0.026, 0.032), exponent=3.0, seg_u=8, seg_v=4)
        mh = np.eye(4)
        mh[:3, 3] = [xh, yh, zh + 0.010]
        b.add(f"{asm}_handle", handle, "metal", _color_fill(handle, CHROME),
              transform=mh, assembly=asm, metadata={"zone": "handle"})
        pocket = box_mesh((0.15, 0.04, 0.012))
        mp = np.eye(4)
        mp[:3, 3] = [xh, yh, zh + 0.002]
        b.add(f"{asm}_handle_pocket", pocket, "plastic",
              _color_fill(pocket, (0.06, 0.06, 0.065)), transform=mp,
              assembly=asm, metadata={"zone": "handle"})

        # hinge spec (front-hinged, ROM 0–65°, opens outward)
        hinge_z = surface_z(cache, x_f, (y_bot + lay.y_belt(x_f)) / 2, 1.0) - 0.004
        b.add_hinge(asm, "door", (0.0, 1.0, 0.0),
                    (x_f + 0.008, (y_bot + lay.y_belt(x_f)) / 2, hinge_z),
                    (0.0, 65.0), open_sign=-1.0)

        # --- mirror to the right side ----------------------------------------
        asm_r = asm.replace("_fl", "_fr").replace("_rl", "_rr")
        mirror = _mirror_z()
        for idx in list(b.assemblies[asm]):
            p = b.parts[idx]
            rp = VehiclePart(
                name=p.name.replace("_fl_", "_fr_").replace("_rl_", "_rr_")
                     .replace("_fl", "_fr").replace("_rl", "_rr"),
                material=p.material, vertices=p.vertices, normals=p.normals,
                uvs=p.uvs, faces=p.faces, vertex_colors=p.vertex_colors,
                aabb_min=np.zeros(3), aabb_max=np.zeros(3),
                solid_volume_m3=p.solid_volume_m3,
                metadata={**p.metadata, "side": "right"},
                transform=mirror, instances=[])
            vw, _ = _apply_transform(p.vertices, p.normals, mirror)
            rp.aabb_min = vw.min(axis=0)
            rp.aabb_max = vw.max(axis=0)
            b.parts.append(rp)
            b.assemblies.setdefault(asm_r, []).append(len(b.parts) - 1)
        b.add_hinge(asm_r, "door", (0.0, 1.0, 0.0),
                    (x_f + 0.008, (y_bot + lay.y_belt(x_f)) / 2, -hinge_z),
                    (0.0, 65.0), open_sign=+1.0)

        # --- panel-gap seals on the body around the opening -------------------
        yb_mid = lay.y_belt((x_f + x_r) / 2)
        seal_pts = [
            [(x_f - gap / 2, y_bot - 0.02, surface_z(cache, x_f - gap / 2, y_bot - 0.02, 1.0) + 0.001),
             (x_f + rake / 2 - gap / 2, yb_mid, surface_z(cache, x_f + rake / 2 - gap / 2, yb_mid, 1.0) + 0.001),
             (x_f + rake - gap / 2, lay.y_belt(x_f + rake) + 0.012, surface_z(cache, x_f + rake - gap / 2, lay.y_belt(x_f + rake), 1.0) + 0.001)],
            [(x_r + gap / 2, y_bot - 0.02, surface_z(cache, x_r + gap / 2, y_bot - 0.02, 1.0) + 0.001),
             (x_r + gap / 2, yb_mid, surface_z(cache, x_r + gap / 2, yb_mid, 1.0) + 0.001),
             (x_r + gap / 2, lay.y_belt(x_r) + 0.012, surface_z(cache, x_r + gap / 2, lay.y_belt(x_r), 1.0) + 0.001)],
            [(x_f, y_bot - 0.018, surface_z(cache, x_f, y_bot - 0.018, 1.0) + 0.001),
             ((x_f + x_r) / 2, y_bot - 0.018, surface_z(cache, (x_f + x_r) / 2, y_bot - 0.018, 1.0) + 0.001),
             (x_r, y_bot - 0.018, surface_z(cache, x_r, y_bot - 0.018, 1.0) + 0.001)],
        ]
        for si, pts in enumerate(seal_pts):
            _panel_gap_seal(b, f"{asm}_seal_{si}", pts)
            # mirrored seal on the right
            pts_r = [(px, py, -pz) for px, py, pz in pts]
            _panel_gap_seal(b, f"{asm_r}_seal_{si}", pts_r)


def build_hood(b: _VehicleBuilder, cache: dict) -> None:
    cp, lay = b.cp, b.lay
    rows = max(5, b.lod["panel_grid"] // 2)
    cols = max(6, b.lod["panel_grid"])
    x0, x1 = lay.x_nose + 0.10, cp.x_cowl - 0.075
    grid = np.zeros((rows, cols, 3))
    for ri, xx in enumerate(np.linspace(x0, x1, rows)):
        w = lay.half_width(xx)
        z_edge = 0.79 * w
        for ci, zz in enumerate(np.linspace(-z_edge, z_edge, cols)):
            base = lay.y_hood(xx)
            crown = 0.018 * (1.0 - (zz / max(z_edge, 1e-9)) ** 2)
            edge_drop = 0.004 * (abs(zz) / max(z_edge, 1e-9)) ** 3
            grid[ri, ci] = [xx, base + crown - edge_drop, zz]
    mesh = thin_panel(grid, 0.007)
    colors = b.livery_grid_colors(grid)
    b.add("hood", mesh, "metal", colors, assembly="hood",
          metadata={"zone": "hood"})
    b.add_hinge("hood", "hood", (0.0, 0.0, 1.0),
                (x1 - 0.005, lay.y_hood(x1) + 0.008, 0.0),
                (0.0, 60.0), open_sign=-1.0)
    # parting-line seal around the hood
    zs = 0.79 * lay.half_width(x0)
    _panel_gap_seal(b, "hood_seal_l",
                    [(x0, lay.y_hood(x0) - 0.002, zs),
                     ((x0 + x1) / 2, lay.y_hood((x0 + x1) / 2) + 0.004,
                      0.79 * lay.half_width((x0 + x1) / 2)),
                     (x1, lay.y_hood(x1) + 0.002, 0.79 * lay.half_width(x1))])
    _panel_gap_seal(b, "hood_seal_r",
                    [(x0, lay.y_hood(x0) - 0.002, -zs),
                     ((x0 + x1) / 2, lay.y_hood((x0 + x1) / 2) + 0.004,
                      -0.79 * lay.half_width((x0 + x1) / 2)),
                     (x1, lay.y_hood(x1) + 0.002, -0.79 * lay.half_width(x1))])


def build_rear_closure(b: _VehicleBuilder, cache: dict) -> None:
    """Trunk lid (notchback/wedge), hatch (hatchback/suv/van) or tailgate."""
    cp, lay = b.cp, b.lay
    rows = max(4, b.lod["panel_grid"] // 2)
    cols = max(6, b.lod["panel_grid"])
    style = cp.body_style

    if style == "bed":
        # tailgate: vertical panel closing the bed's rear, bottom-hinged;
        # the tub only carries a low rear valance there (tail_bed region)
        y_floor = lay.y_floor_int + 0.16
        y_top = cp.y_belt_rear + 0.30
        x_g = lay.x_tail - 0.035
        grid = np.zeros((rows, cols, 3))
        for ri, yy in enumerate(np.linspace(y_floor + 0.065, y_top - 0.01, rows)):
            w = 0.90 * lay.half_width(x_g)
            for ci, zz in enumerate(np.linspace(-w, w, cols)):
                grid[ri, ci] = [x_g + 0.004 * (1 - (zz / max(w, 1e-9)) ** 2), yy, zz]
        mesh = thin_panel(grid, 0.007)
        b.add("tailgate", mesh, "metal", b.livery_grid_colors(grid),
              assembly="trunk", metadata={"zone": "tailgate"})
        b.add_hinge("trunk", "trunk", (0.0, 0.0, 1.0),
                    (x_g, y_floor + 0.075, 0.0), (0.0, 70.0), open_sign=-1.0)
        return

    if style in ("hatch", "van"):
        # hatch frame from the roof rear edge down the backlight plane
        grid = np.zeros((rows, cols, 3))
        for ri, t in enumerate(np.linspace(0.0, 1.0, rows)):
            xx = (lay.x_roof_rear - 0.015) - t * (lay.x_roof_rear - lay.x_tail + 0.10)
            yy = (lay.y_roof - 0.010
                  - t * (lay.y_roof - lay.y_deck(lay.x_tail) - 0.022))
            w = (0.90 * lay.half_width(max(xx, lay.x_cabin_rear))
                 if t > 0.55 else _rail_z(b, max(xx, lay.x_roof_rear - 0.03)) + 0.01)
            for ci, zz in enumerate(np.linspace(-w, w, cols)):
                grid[ri, ci] = [xx, yy + 0.006 * (1 - (zz / max(w, 1e-9)) ** 2), zz]
        mesh = thin_panel(grid, 0.007)
        b.add("hatch", mesh, "metal", b.livery_grid_colors(grid),
              assembly="trunk", metadata={"zone": "hatch"})
        # hatch glass inset (upper 55 % of the slope)
        g_rows = max(3, rows // 2)
        ggrid = np.zeros((g_rows, cols, 3))
        for ri, t in enumerate(np.linspace(0.04, 0.55, g_rows)):
            xx = (lay.x_roof_rear - 0.015) - t * (lay.x_roof_rear - lay.x_tail + 0.10)
            yy = (lay.y_roof - 0.010
                  - t * (lay.y_roof - lay.y_deck(lay.x_tail) - 0.022))
            w = _rail_z(b, max(xx, lay.x_roof_rear - 0.03))
            for ci, zz in enumerate(np.linspace(-0.92 * w, 0.92 * w, cols)):
                ggrid[ri, ci] = [xx - 0.004, yy + 0.010
                                 + 0.006 * (1 - (zz / max(w, 1e-9)) ** 2), zz]
        glass = thin_panel(ggrid, 0.004)
        b.add("hatch_glass", glass, "glass", _color_fill(glass, GLASS_TINT),
              assembly="trunk", metadata={"zone": "glass"})
        b.add_hinge("trunk", "hatch", (0.0, 0.0, 1.0),
                    (lay.x_roof_rear - 0.02, lay.y_roof - 0.004, 0.0),
                    (0.0, 70.0), open_sign=+1.0)
        return

    # notchback / wedge trunk lid
    x0, x1 = lay.x_cabin_rear + 0.035, lay.x_tail - 0.055
    grid = np.zeros((rows, cols, 3))
    for ri, xx in enumerate(np.linspace(x0, x1, rows)):
        w = 0.88 * lay.half_width(xx)
        for ci, zz in enumerate(np.linspace(-w, w, cols)):
            crown = 0.015 * (1.0 - (zz / max(w, 1e-9)) ** 2)
            grid[ri, ci] = [xx, lay.y_deck(xx) + 0.006 + crown, zz]
    mesh = thin_panel(grid, 0.007)
    b.add("trunk_lid", mesh, "metal", b.livery_grid_colors(grid),
          assembly="trunk", metadata={"zone": "trunk"})
    b.add_hinge("trunk", "trunk", (0.0, 0.0, 1.0),
                (x0 + 0.005, lay.y_deck(x0) + 0.008, 0.0),
                (0.0, 70.0), open_sign=+1.0)
    _panel_gap_seal(b, "trunk_seal_l",
                    [(x0, lay.y_deck(x0) + 0.004, 0.88 * lay.half_width(x0)),
                     ((x0 + x1) / 2, lay.y_deck((x0 + x1) / 2) + 0.006,
                      0.88 * lay.half_width((x0 + x1) / 2)),
                     (x1, lay.y_deck(x1) + 0.002, 0.88 * lay.half_width(x1))])
    _panel_gap_seal(b, "trunk_seal_r",
                    [(x0, lay.y_deck(x0) + 0.004, -0.88 * lay.half_width(x0)),
                     ((x0 + x1) / 2, lay.y_deck((x0 + x1) / 2) + 0.006,
                      -0.88 * lay.half_width((x0 + x1) / 2)),
                     (x1, lay.y_deck(x1) + 0.002, -0.88 * lay.half_width(x1))])


# ---------------------------------------------------------------------------
# wheels: tire carcass + tread lugs, rim + spokes, brakes
# ---------------------------------------------------------------------------


def _tire_mesh(b: _VehicleBuilder) -> Mesh:
    """Lathed tire carcass: real cross-section (beads, shoulders, tread
    band) — the smooth base under the lug tread, never a plain torus."""
    cp = b.cp
    r_out = cp.wheel_diameter / 2.0
    r_rim = cp.rim_diameter / 2.0
    tw = cp.tire_width
    profile = np.array([
        [r_rim - 0.006, -0.44 * tw],
        [r_out - 0.030, -0.47 * tw],
        [r_out - 0.006, -0.40 * tw],
        [r_out, -0.30 * tw],
        [r_out, 0.30 * tw],
        [r_out - 0.006, 0.40 * tw],
        [r_out - 0.030, 0.47 * tw],
        [r_rim - 0.006, 0.44 * tw],
        [r_rim - 0.006, 0.20 * tw],
        [r_rim - 0.010, 0.00],
        [r_rim - 0.006, -0.20 * tw],
    ], dtype=np.float64)
    return lathe(profile, segments=b.lod["lathe_seg"])


def _tire_colors(mesh: Mesh, r_out: float, tw: float,
                 stripe: bool) -> np.ndarray:
    """Sidewall / tread zones; at low LOD the tread gets angular striping
    (tread-normal vertex striping) instead of geometry lugs."""
    v = mesh[0]
    r = np.hypot(v[:, 0], v[:, 2])
    on_tread = (r > r_out - 0.008) & (np.abs(v[:, 1]) < 0.34 * tw)
    colors = np.zeros((v.shape[0], 3), dtype=np.float32)
    colors[:] = np.asarray((0.075, 0.075, 0.08), dtype=np.float32)   # sidewall
    colors[on_tread] = np.asarray(RUBBER, dtype=np.float32)
    if stripe:
        ang = np.arctan2(v[:, 2], v[:, 0])
        band = np.floor((ang + math.pi) / TAU * 40.0) % 2
        stripe_mask = on_tread & (band > 0.5)
        colors[stripe_mask] *= 1.8
    return colors


def build_wheels(b: _VehicleBuilder) -> None:
    cp, lay = b.cp, b.lay
    r_out = cp.wheel_diameter / 2.0
    r_rim = cp.rim_diameter / 2.0
    tw = cp.tire_width
    seg = b.lod["lathe_seg"]

    # -- tire carcass ----------------------------------------------------------
    tire = _tire_mesh(b)
    tire_cols = _tire_colors(tire, r_out, tw, stripe=(b.lod["lugs"] == 0))

    # -- tread lugs: one shared block, chevron-instanced around the tire -------
    lug_mesh = box_mesh((0.016, 0.075, 0.055))
    lug_cols = _color_fill(lug_mesh, (0.035, 0.035, 0.04))
    lug_instances: list[np.ndarray] = []
    if b.lod["lugs"] > 0:
        per_row = max(6, b.lod["lugs"] // 2)
        for row, (y_row, yaw) in enumerate(((+0.052, -22.0), (-0.052, +22.0))):
            for i in range(per_row):
                m = (_axis_angle_matrix((0, 1, 0), TAU * i / per_row +
                                        row * math.pi / per_row)
                     @ _translation(r_out + 0.001, y_row, 0.0)
                     @ _axis_angle_matrix((1, 0, 0), math.radians(yaw)))
                lug_instances.append(m)

    # -- rim: barrel + face in one closed lathe --------------------------------
    rw = 0.80 * tw
    rim_profile = np.array([
        [r_rim, -0.50 * rw],
        [r_rim, 0.50 * rw],
        [r_rim - 0.012, 0.52 * rw],
        [0.062, 0.46 * rw],
        [0.050, 0.40 * rw],
        [0.050, 0.20 * rw],
        [0.062, 0.10 * rw],
        [r_rim - 0.012, -0.44 * rw],
    ], dtype=np.float64)
    rim = lathe(rim_profile, segments=seg)
    rim_cols = _color_fill(rim, CHROME)

    # -- spokes (alloy) or plain face (steel) ----------------------------------
    spoke_mesh = box_mesh((r_rim - 0.030, 0.020, 0.042))
    spoke_cols = _color_fill(spoke_mesh, CHROME)
    spoke_instances: list[np.ndarray] = []
    if b.lod["spokes"]:
        n_spokes = 5
        for i in range(n_spokes):
            spoke_instances.append(
                _axis_angle_matrix((0, 1, 0), TAU * i / n_spokes)
                @ _translation((r_rim - 0.028) / 2 + 0.028, 0.40 * rw, 0.0))

    # -- hub + centre cap + lug nuts --------------------------------------------
    hub = cylinder_mesh(0.048, 0.55 * rw, segments=max(10, seg // 2))
    hub_cols = _color_fill(hub, (0.55, 0.57, 0.60))
    cap = cylinder_mesh(0.020, 0.57 * rw, segments=10)
    cap_cols = _color_fill(cap, (0.10, 0.10, 0.11))
    nut_mesh = cylinder_mesh(0.008, 0.49 * rw, segments=6)
    nut_cols = _color_fill(nut_mesh, (0.75, 0.77, 0.80))
    nut_instances = []
    if b.lod["lug_nuts"]:
        for i in range(5):
            nut_instances.append(
                _translation(0.033 * math.cos(TAU * i / 5), 0.0,
                             0.033 * math.sin(TAU * i / 5)))

    # -- brakes -------------------------------------------------------------------
    disc = cylinder_mesh(0.36 * cp.rim_diameter, 0.026, segments=max(14, seg // 2))
    disc_cols = _color_fill(disc, (0.42, 0.42, 0.45))
    caliper = rounded_box((0.115, 0.055, 0.135), exponent=2.6, seg_u=8, seg_v=4)
    caliper_cols = _color_fill(caliper, (0.55, 0.08, 0.06))
    mc = np.eye(4)
    mc[:3, 3] = [-0.30 * cp.rim_diameter, -0.02, 0.0]

    # -- place at the four corners -------------------------------------------------
    corners = [
        ("fl", _translation(0.0, lay.y_wc, lay.z_wheel)
         @ _axis_angle_matrix((1, 0, 0), math.pi / 2)),
        ("rl", _translation(cp.wheelbase, lay.y_wc, lay.z_wheel)
         @ _axis_angle_matrix((1, 0, 0), math.pi / 2)),
        ("fr", _translation(0.0, lay.y_wc, -lay.z_wheel)
         @ _axis_angle_matrix((1, 0, 0), -math.pi / 2)),
        ("rr", _translation(cp.wheelbase, lay.y_wc, -lay.z_wheel)
         @ _axis_angle_matrix((1, 0, 0), -math.pi / 2)),
    ]
    for tag, m in corners:
        b.add(f"wheel_tire_{tag}", tire, "plastic", tire_cols, transform=m,
              metadata={"zone": "tire", "corner": tag})
        if lug_instances:
            b.add(f"wheel_tread_{tag}", lug_mesh, "plastic", lug_cols,
                  transform=m, instances=lug_instances,
                  metadata={"zone": "tread", "corner": tag})
        b.add(f"wheel_rim_{tag}", rim, "metal", rim_cols, transform=m,
              metadata={"zone": "rim", "corner": tag})
        if spoke_instances:
            b.add(f"wheel_spoke_{tag}", spoke_mesh, "metal", spoke_cols,
                  transform=m, instances=spoke_instances,
                  metadata={"zone": "rim", "corner": tag})
        b.add(f"wheel_hub_{tag}", hub, "metal", hub_cols, transform=m,
              metadata={"zone": "rim", "corner": tag})
        b.add(f"wheel_cap_{tag}", cap, "plastic", cap_cols, transform=m,
              metadata={"zone": "rim", "corner": tag})
        if nut_instances:
            b.add(f"wheel_nuts_{tag}", nut_mesh, "metal", nut_cols,
                  transform=m, instances=nut_instances,
                  metadata={"zone": "rim", "corner": tag})
        b.add(f"brake_disc_{tag}", disc, "metal", disc_cols, transform=m,
              metadata={"zone": "brake", "corner": tag})
        b.add(f"brake_caliper_{tag}", caliper, "metal", caliper_cols,
              transform=m @ mc, metadata={"zone": "brake", "corner": tag})


# ---------------------------------------------------------------------------
# engine bay
# ---------------------------------------------------------------------------


def build_engine(b: _VehicleBuilder) -> None:
    """Simplified engine visible when the hood opens (bay is a real well)."""
    cp, lay = b.cp, b.lay
    if not b.lod["engine"]:
        return
    x_mid = (lay.x_nose + 0.30 + cp.x_cowl) / 2
    y = lay.y_bay
    block = rounded_box((0.58, 0.34, 0.52), exponent=2.4, seg_u=10, seg_v=5)
    mb = np.eye(4)
    mb[:3, 3] = [x_mid, y + 0.19, 0.0]
    b.add("engine_block", block, "metal", _color_fill(block, (0.16, 0.16, 0.17)),
          transform=mb, metadata={"zone": "engine"})
    cover = rounded_box((0.44, 0.09, 0.40), exponent=2.8, seg_u=8, seg_v=4)
    mc = np.eye(4)
    mc[:3, 3] = [x_mid, y + 0.40, 0.0]
    b.add("engine_cover", cover, "plastic",
          _color_fill(cover, (0.08, 0.10, 0.14)), transform=mc,
          metadata={"zone": "engine"})
    battery = box_mesh((0.20, 0.17, 0.16))
    mbat = np.eye(4)
    mbat[:3, 3] = [x_mid + 0.32, y + 0.10, -0.28]
    b.add("engine_battery", battery, "plastic",
          _color_fill(battery, (0.07, 0.07, 0.08)), transform=mbat,
          metadata={"zone": "engine"})
    intake = tube_along([(x_mid - 0.28, y + 0.32, 0.16),
                         (x_mid - 0.05, y + 0.40, 0.20),
                         (x_mid + 0.20, y + 0.36, 0.18)],
                        0.045, sides=8)
    b.add("engine_intake", intake, "plastic",
          _color_fill(intake, (0.10, 0.10, 0.11)), metadata={"zone": "engine"})


# ---------------------------------------------------------------------------
# interior
# ---------------------------------------------------------------------------


def build_interior(b: _VehicleBuilder, cache: dict) -> None:
    """Seats, steering (LHD default), dashboard, console, door cards, floor.

    Everything is a named part; door cards live in the door assemblies so
    they articulate with the doors. Visible through the thin glass panels.
    """
    cp, lay = b.cp, b.lay
    high = b.interior_detail == "high"
    y_floor = lay.y_floor_int
    x_seat = cp.x_cowl + 0.68
    z_seat = 0.36
    yb0 = cp.y_belt_front

    # -- floor + tunnel ---------------------------------------------------------
    x_f0, x_f1 = cp.x_cowl + 0.05, lay.x_cabin_rear + 0.05
    grid = np.zeros((3, 5, 3))
    for ri, xx in enumerate(np.linspace(x_f0, x_f1, 3)):
        for ci, zz in enumerate(np.linspace(-0.62, 0.62, 5)):
            hump = 0.05 * math.exp(-((zz / 0.16) ** 2))
            grid[ri, ci] = [xx, y_floor + 0.012 + hump, zz]
    mesh = thin_panel(grid, 0.010)
    b.add("interior_floor", mesh, "fabric", _color_fill(mesh, CARPET),
          metadata={"zone": "interior"})

    # -- seats ---------------------------------------------------------------------
    def bucket(name, x, z, sport=False):
        bol = 0.04 if sport else 0.0
        cushion = rounded_box((0.50, 0.13 + bol, 0.50), exponent=2.8,
                              seg_u=10, seg_v=5)
        mc = np.eye(4)
        mc[:3, 3] = [x, y_floor + 0.20, z]
        cols = _color_fill(cushion, SEAT_FABRIC)
        # darker side bolsters (zone shading)
        cols[np.abs(cushion[0][:, 2]) > 0.18] *= 0.75
        b.add(f"{name}_cushion", cushion, "fabric", cols, transform=mc,
              metadata={"zone": "seat"})
        tilt = math.radians(14.0)
        back = rounded_box((0.13, 0.58, 0.48), exponent=2.8, seg_u=10, seg_v=5)
        mb = _translation(x + 0.26, y_floor + 0.52, z) @ \
            _axis_angle_matrix((0, 0, 1), -tilt)
        cols_b = _color_fill(back, SEAT_FABRIC)
        cols_b[np.abs(back[0][:, 2]) > 0.17] *= 0.75
        b.add(f"{name}_backrest", back, "fabric", cols_b, transform=mb,
              metadata={"zone": "seat"})
        if high:
            head = rounded_box((0.11, 0.15, 0.24), exponent=3.0, seg_u=8, seg_v=4)
            mh = _translation(x + 0.33, y_floor + 0.92, z) @ \
                _axis_angle_matrix((0, 0, 1), -tilt)
            b.add(f"{name}_headrest", head, "fabric",
                  _color_fill(head, np.asarray(SEAT_FABRIC) * 0.85),
                  transform=mh, metadata={"zone": "seat"})

    def bench(name, x):
        cushion = rounded_box((0.50, 0.13, 1.30), exponent=2.8, seg_u=10, seg_v=5)
        mc = np.eye(4)
        mc[:3, 3] = [x, y_floor + 0.19, 0.0]
        b.add(f"{name}_cushion", cushion, "fabric",
              _color_fill(cushion, SEAT_FABRIC), transform=mc,
              metadata={"zone": "seat"})
        back = rounded_box((0.13, 0.55, 1.28), exponent=2.8, seg_u=10, seg_v=5)
        mb = _translation(x + 0.26, y_floor + 0.50, 0.0) @ \
            _axis_angle_matrix((0, 0, 1), math.radians(-12.0))
        b.add(f"{name}_backrest", back, "fabric",
              _color_fill(back, SEAT_FABRIC), transform=mb,
              metadata={"zone": "seat"})
        if high:
            for zi in (-0.42, 0.42):
                hr = rounded_box((0.11, 0.13, 0.22), exponent=3.0,
                                 seg_u=8, seg_v=4)
                mh = _translation(x + 0.31, y_floor + 0.86, zi) @ \
                    _axis_angle_matrix((0, 0, 1), math.radians(-12.0))
                b.add(f"{name}_headrest_{'l' if zi > 0 else 'r'}", hr,
                      "fabric", _color_fill(hr, np.asarray(SEAT_FABRIC) * 0.85),
                      transform=mh, metadata={"zone": "seat"})

    sport = cp.body_style == "wedge"
    if high or True:
        bucket("seat_fl", x_seat, z_seat, sport)
        bucket("seat_fr", x_seat, -z_seat, sport)
    if cp.rows >= 2:
        bench("seat_row2", x_seat + 0.80)
    if cp.rows >= 3:
        bench("seat_row3", x_seat + 1.62)

    # -- steering (left-hand drive) ---------------------------------------------
    z_drv = z_seat
    x_col0, y_col0 = cp.x_cowl + 0.16, yb0 - 0.10
    x_col1, y_col1 = cp.x_cowl + 0.44, yb0 - 0.22
    column = tube_along([(x_col0, y_col0, z_drv), (x_col1, y_col1, z_drv)],
                        0.021, sides=8)
    b.add("steering_column", column, "metal",
          _color_fill(column, INTERIOR_DARK), metadata={"zone": "steering"})
    rim = ring_mesh(0.185, 0.017, seg_u=20, seg_v=6)
    d = np.array([x_col0 - x_col1, y_col0 - y_col1, 0.0])   # toward driver
    d = -d / (np.linalg.norm(d) + 1e-12)                     # rim plane normal
    z_ax = -d                                                # face the driver
    x_ax = np.cross(np.array([0.0, 1.0, 0.0]), z_ax)
    x_ax /= np.linalg.norm(x_ax) + 1e-12
    y_ax = np.cross(z_ax, x_ax)
    mw = np.eye(4)
    mw[:3, 0], mw[:3, 1], mw[:3, 2] = x_ax, y_ax, z_ax
    mw[:3, 3] = [x_col1 + 0.03, y_col1 - 0.01, z_drv]
    b.add("steering_wheel", rim, "plastic", _color_fill(rim, INTERIOR_DARK),
          transform=mw, metadata={"zone": "steering"})
    hub = cylinder_mesh(0.045, 0.05, segments=10)
    mh = mw @ _axis_angle_matrix((1, 0, 0), math.pi / 2)
    b.add("steering_hub", hub, "plastic", _color_fill(hub, INTERIOR_TRIM),
          transform=mh, metadata={"zone": "steering"})
    spoke = box_mesh((0.34, 0.018, 0.02))
    msp = np.eye(4)
    msp[:3, 0], msp[:3, 1], msp[:3, 2] = x_ax, y_ax, z_ax
    msp[:3, 3] = [x_col1 + 0.03, y_col1 - 0.01, z_drv]
    b.add("steering_spoke", spoke, "plastic",
          _color_fill(spoke, INTERIOR_TRIM), transform=msp,
          metadata={"zone": "steering"})

    # -- dashboard ------------------------------------------------------------------
    rows, cols_n = 4, max(6, b.lod["panel_grid"] // 2)
    grid = np.zeros((rows, cols_n, 3))
    x_dash = cp.x_cowl + 0.12
    for ri, t in enumerate(np.linspace(0.0, 1.0, rows)):
        yy = yb0 - 0.24 + t * 0.26
        xx = x_dash + 0.06 * math.sin(t * math.pi) - 0.04 * t
        for ci, zz in enumerate(np.linspace(-0.72, 0.72, cols_n)):
            grid[ri, ci] = [xx - 0.05 * (zz / 0.72) ** 2, yy, zz]
    dash = thin_panel(grid, 0.012)
    b.add("dash_main", dash, "plastic", _color_fill(dash, INTERIOR_DARK),
          metadata={"zone": "dashboard"})
    if high:
        binna = rounded_box((0.22, 0.07, 0.26), exponent=2.8, seg_u=8, seg_v=4)
        mbn = np.eye(4)
        mbn[:3, 3] = [x_dash + 0.02, yb0 - 0.005, z_drv]
        b.add("dash_binnacle", binna, "plastic",
              _color_fill(binna, INTERIOR_DARK), transform=mbn,
              metadata={"zone": "dashboard"})
        cluster = box_mesh((0.012, 0.07, 0.20))
        mcl = np.eye(4)
        mcl[:3, 3] = [x_dash + 0.10, yb0 - 0.05, z_drv]
        b.add("dash_cluster", cluster, "glass",
              _color_fill(cluster, (0.04, 0.07, 0.09)), transform=mcl,
              metadata={"zone": "dashboard"})
        stack = rounded_box((0.16, 0.24, 0.22), exponent=2.6, seg_u=8, seg_v=4)
        mst = np.eye(4)
        mst[:3, 3] = [x_dash + 0.03, yb0 - 0.16, 0.0]
        b.add("dash_center_stack", stack, "plastic",
              _color_fill(stack, INTERIOR_TRIM), transform=mst,
              metadata={"zone": "dashboard"})
        screen = box_mesh((0.012, 0.10, 0.16))
        msc = np.eye(4)
        msc[:3, 3] = [x_dash + 0.115, yb0 - 0.10, 0.0]
        b.add("dash_screen", screen, "glass",
              _color_fill(screen, (0.03, 0.05, 0.08)), transform=msc,
              metadata={"zone": "dashboard"})
        for vi, vz in enumerate((-0.52, -0.22, 0.22, 0.52)):
            vent = box_mesh((0.10, 0.035, 0.09))
            mv = np.eye(4)
            mv[:3, 3] = [x_dash + 0.075, yb0 - 0.045, vz]
            b.add(f"dash_vent_{vi}", vent, "plastic",
                  _color_fill(vent, (0.05, 0.05, 0.055)), transform=mv,
                  metadata={"zone": "dashboard"})

    # -- center console ---------------------------------------------------------
    if high:
        console = rounded_box((0.62, 0.16, 0.24), exponent=2.8, seg_u=10, seg_v=5)
        mco = np.eye(4)
        mco[:3, 3] = [x_seat + 0.05, y_floor + 0.14, 0.0]
        b.add("console", console, "plastic",
              _color_fill(console, INTERIOR_TRIM), transform=mco,
              metadata={"zone": "console"})
        shifter = tube_along([(x_seat - 0.18, y_floor + 0.22, 0.0),
                              (x_seat - 0.16, y_floor + 0.30, 0.0)],
                             0.014, sides=8)
        b.add("console_shifter", shifter, "metal",
              _color_fill(shifter, INTERIOR_DARK), metadata={"zone": "console"})

    # -- door cards (articulate with the doors) -----------------------------------
    if high:
        names = ["fl", "rl"][: len(_door_spans(cp, lay))]
        spans = [(f"door_{n}", a, b_) for n, (a, b_) in
                 zip(names, _door_spans(cp, lay))]
        for asm_l, x_f, x_r in spans:
            for sgn, asm in ((1.0, asm_l),
                             (-1.0, asm_l.replace("_fl", "_fr")
                              .replace("_rl", "_rr"))):
                rows_c, cols_c = 4, max(5, b.lod["panel_grid"] // 2)
                grid = np.zeros((rows_c, cols_c, 3))
                for ri, t in enumerate(np.linspace(0.0, 1.0, rows_c)):
                    for ci, xx in enumerate(np.linspace(x_f + 0.04, x_r - 0.04,
                                                        cols_c)):
                        yy = 0.44 + t * (lay.y_belt(xx) - 0.03 - 0.44)
                        zz = surface_z(cache, xx, yy, sgn) - sgn * 0.0105
                        grid[ri, ci] = [xx, yy, zz]
                card = thin_panel(grid, 0.006)
                cols = _color_fill(card, INTERIOR_TRIM)
                # insert band: lighter armrest-height zone
                cols[(card[0][:, 1] > 0.60) & (card[0][:, 1] < 0.68)] *= 1.5
                b.add(f"{asm}_card", card, "fabric", cols, assembly=asm,
                      metadata={"zone": "door_card", "side":
                                "left" if sgn > 0 else "right"})
                arm = rounded_box((0.32, 0.045, 0.048), exponent=3.0,
                                  seg_u=8, seg_v=4)
                xa = (x_f + x_r) / 2
                ma = np.eye(4)
                ma[:3, 3] = [xa, 0.64,
                             surface_z(cache, xa, 0.64, sgn) - sgn * 0.040]
                b.add(f"{asm}_card_armrest", arm, "fabric",
                      _color_fill(arm, np.asarray(SEAT_FABRIC) * 0.9),
                      transform=ma, assembly=asm, metadata={"zone": "door_card"})


# ---------------------------------------------------------------------------
# tub vertex-color zones
# ---------------------------------------------------------------------------


def _tub_colors(b: _VehicleBuilder, mesh: Mesh, cache: dict) -> np.ndarray:
    """Per-vertex paint/dark zones for the body tub.

    Zones (finer than per-part flat colors): paint with a lower-flank
    gradient, dark underbody, dark wheel-arch liners, dark open wells
    (engine bay / cabin tub / trunk / bed), painted closed decks.
    """
    positions = cache["positions"]
    nr = len(positions)
    colors = np.repeat(b.paint[None, :], mesh[0].shape[0], axis=0)
    lay = b.lay
    arch = cache["arch"]
    for i, x in enumerate(positions):
        reg = lay.region(float(x))
        in_arch = (abs(x - arch["x_front"]) < arch["r"]
                   or abs(x - arch["x_rear"]) < arch["r"])
        base = i * N_SECTION_POINTS
        for j in range(N_SECTION_POINTS):
            idx = base + j
            if j in J_UNDER:
                colors[idx] = (0.085, 0.085, 0.09)
            elif in_arch and j in J_ARCH_R + J_ARCH_L:
                colors[idx] = (0.075, 0.075, 0.08)
            elif j in J_TOP:
                if reg in ("bay", "trunk", "bed", "cabin"):
                    colors[idx] = (0.105, 0.105, 0.11)
                else:
                    colors[idx] = b.paint
            elif j in J_FLANK_LO:
                colors[idx] = b.paint * 0.93
    return colors


# ---------------------------------------------------------------------------
# articulation clearance checking
# ---------------------------------------------------------------------------


def _point_polygon_depth(pt: np.ndarray, poly: np.ndarray) -> float:
    """Signed depth of (y, z) point in a simple polygon: > 0 = inside."""
    y, z = float(pt[0]), float(pt[1])
    inside = False
    best = math.inf
    k = len(poly)
    for i in range(k):
        y0, z0 = poly[i]
        y1, z1 = poly[(i + 1) % k]
        # even-odd crossing on a horizontal ray in +z
        if (z0 > z) != (z1 > z):
            y_int = y0 + (y1 - y0) * (z - z0) / (z1 - z0)
            if y < y_int:
                inside = not inside
        # distance to the edge
        ey, ez = y1 - y0, z1 - z0
        t = ((y - y0) * ey + (z - z0) * ez) / (ey * ey + ez * ez + 1e-30)
        t = min(max(t, 0.0), 1.0)
        dy, dz = y - (y0 + t * ey), z - (z0 + t * ez)
        best = min(best, math.hypot(dy, dz))
    return best if inside else -best


def _point_solid_depth(pt: np.ndarray, poly: np.ndarray) -> float:
    """Signed depth in the tub's SOLID MATERIAL at one station.

    The section loop encloses both sheet-metal and the open wells (engine
    bay / cabin / trunk / bed) — a hood or trunk lid closing *over* a well
    is inside the loop but occupies air, not material. The solid excludes
    the well air above the wall polyline (section points 9…13, the top
    band). > 0 = inside solid (a real collision); < 0 = clear.
    """
    depth = _point_polygon_depth(pt, poly)
    if depth <= 0.0:
        return depth
    y, z = float(pt[0]), float(pt[1])
    top = poly[9:14]                       # wall polyline, z ascending
    w_top = max(abs(float(top[0, 1])), abs(float(top[-1, 1])))
    if abs(z) <= w_top:
        y_top = float(np.interp(z, top[:, 1], top[:, 0]))
        if y > y_top:
            return -depth                  # well air, not material
    return depth


def check_swing_clearance(spec: VehicleSpec, assembly: str,
                          samples: int = 14) -> float:
    """Maximum penetration (m) of an articulated assembly into the body tub
    across its full ROM. Negative = always clear.

    The check sweeps the assembly, transforms every part vertex, and tests
    it against the interpolated tub cross-section at its x position
    (`_point_solid_depth` — sheet metal only, well air excluded). For
    **doors** the test is scoped to the fender zone (vertices at or ahead
    of the hinge line): the door opening itself is air by construction —
    what must never collide is the door's front edge swinging through the
    fender, exactly the homologation concern.
    """
    cache = spec.geometry_cache["tub"]
    positions = cache["positions"]
    sections = cache["sections"]
    h = spec.articulations[assembly]
    x_fender = h.origin[0] + 0.005 if h.kind == "door" else None
    worst = -math.inf
    for s in range(samples + 1):
        frac = s / samples
        rot = spec.assembly_transform(assembly, frac)
        for idx in spec.assemblies[assembly]:
            p = spec.parts[idx]
            world = rot @ (np.eye(4) if p.transform is None else p.transform)
            v, _ = _apply_transform(p.vertices, p.normals, world)
            step = max(1, v.shape[0] // 220)
            for pt in v[::step]:
                x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                if x <= positions[0] or x >= positions[-1]:
                    continue
                if x_fender is not None and x > x_fender:
                    continue
                i = int(np.searchsorted(positions, x) - 1)
                i = min(max(i, 0), len(positions) - 2)
                t = (x - positions[i]) / max(positions[i + 1] - positions[i],
                                             1e-12)
                poly = sections[i] * (1.0 - t) + sections[i + 1] * t
                depth = _point_solid_depth(np.array([y, z]), poly)
                worst = max(worst, depth)
    return float(worst)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def build_vehicle(params: dict | None = None) -> VehicleSpec:
    """Build a complete parametric vehicle. See module docstring for params."""
    p = dict(params or {})
    cls_name = str(p.get("class", "sedan")).strip().lower()
    if cls_name not in VEHICLE_CLASSES:
        raise ValueError(
            f"unknown vehicle class {cls_name!r}; pick {sorted(VEHICLE_CLASSES)}")
    cp = VEHICLE_CLASSES[cls_name]
    paint = _rgb(p.get("color", "deep_blue"))
    livery = p.get("livery")
    if livery not in (None, "racing_stripes", "two_tone"):
        raise ValueError("livery must be None | 'racing_stripes' | 'two_tone'")
    lod_name = str(p.get("lod", "high")).lower()
    if lod_name not in LOD_PRESETS:
        raise ValueError(f"lod must be one of {sorted(LOD_PRESETS)}")
    lod = LOD_PRESETS[lod_name]
    interior_detail = str(p.get("interior_detail", "high")).lower()
    if interior_detail not in ("low", "high"):
        raise ValueError("interior_detail must be 'low' or 'high'")

    lay = _Layout.build(cp)
    b = _VehicleBuilder(cp, lay, lod, paint, livery)
    b.interior_detail = interior_detail

    tub_mesh, cache = build_tub(lay, lod)
    b.add("body_tub", tub_mesh, "metal", _tub_colors(b, tub_mesh, cache),
          metadata={"zone": "body"})
    build_bumpers(b)
    build_fascia_details(b, cache)
    build_greenhouse(b, cache)
    build_doors(b, cache)
    build_hood(b, cache)
    build_rear_closure(b, cache)
    build_wheels(b)
    build_engine(b)
    build_interior(b, cache)

    # default articulation state
    doors_open = p.get("doors_open", False)
    default_state: dict[str, float] = {}
    if isinstance(doors_open, dict):
        for k, v in doors_open.items():
            default_state[str(k)] = float(v)
    elif doors_open is True:
        for asm in b.articulations:
            if asm.startswith("door_"):
                default_state[asm] = 1.0
    elif doors_open not in (False, None):
        frac = float(doors_open)
        for asm in b.articulations:
            if asm.startswith("door_"):
                default_state[asm] = frac

    dimensions = {
        "length_m": cp.length,
        "width_m": cp.width,
        "height_m": cp.height,
        "wheelbase_m": cp.wheelbase,
        "track_m": cp.track,
        "wheel_diameter_m": cp.wheel_diameter,
        "ground_clearance_m": cp.clearance,
    }
    return VehicleSpec(
        vehicle_class=cls_name, params=p, parts=b.parts,
        articulations=b.articulations, assemblies=b.assemblies,
        dimensions=dimensions, default_state=default_state,
        geometry_cache={"tub": cache, "layout": lay},
    )


__all__ = [
    "VEHICLE_CLASSES", "LOD_PRESETS", "PAINT_COLORS", "HingeSpec",
    "VehiclePart", "VehicleSpec", "build_vehicle", "check_swing_clearance",
    "loft_sections", "thin_panel", "lathe", "make_section",
    "subtract_arch_from_section",
]
