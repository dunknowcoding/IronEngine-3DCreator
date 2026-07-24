"""Water containers with an editable fluid body (CR_FloraWater).

Vessels / basins / ponds / aquariums / buckets generated as two logical
parts:

1. a **container** (floor + walls + rim) with a well-defined interior
   cavity, and
2. a separate **water body** — a surface disc / body fitted to that
   interior, topped by a meniscus lip — whose ``params["extras"]["fluid"]``
   block carries the fluid properties a simulator consumes.

``fill_level`` (0..1) is editable after generation:
`set_fill_level(spec, fill)` returns a new spec with the water part
rescaled (surface height, meniscus position, and ``volume_m3`` recomputed
from the cavity cross-section). The water surface always stays at or
below the container rim — ``fill_level < 1`` guarantees a visible air gap.

Fluid extras schema (``extras.fluid``)
--------------------------------------
There is NO pre-existing fluid block in either repo (checked
IronEngine-Sim: ``physics/solvers/fluids_sph.py`` and ``rendering/water.py``
are stubs; ``force_fields.py`` only *reserves* BuoyancyField for water;
``assets/material_library.py`` defines a ``water_surface`` physics
material). This module therefore DEFINES the block, mirroring the
``cloth`` passthrough convention (3DCreator ``soft_author`` →
``core.manifest`` extras → Sim ``scene_io`` reader):

.. code-block:: json

    "fluid": {
      "fluid": "water",                  // fluid kind tag
      "solver": "fluids_sph",            // Sim solver registry name (stub)
      "physics_material": "water_surface", // Sim material_library key
      "volume_m3": 0.00062,              // current fill volume
      "fill_level": 0.7,                 // 0..1 of interior depth
      "density_kg_m3": 1000.0,
      "viscosity_pa_s": 0.001,
      "surface_tension_n_m": 0.072,
      "restitution": 0.0,
      "color_tint": [0.08, 0.15, 0.22],  // matches Sim u_water_color
      "meniscus_radius_m": 0.0027,       // capillary length sqrt(σ/ρg)
      "container_part": "basin_wall",    // label of the containing part
      "water_part": "water",             // label of the water body part
      "meniscus_part": "water_meniscus", // label of the meniscus lip part
      "cavity": "round",                 // round | box
      "interior_bottom_m": 0.010,        // cavity floor (spec-local Y)
      "interior_depth_m": 0.051,         // cavity depth below the rim
      "interior_radius_m": 0.134         // round cavities
      // "interior_size_m": [0.232, 0.152]  // box cavities (X, Z)
    }

Consumption contract for the integrator (documented, not wired here):
add ``"fluid"`` to ``_EXTRA_PASSTHROUGH_BLOCKS`` in
``core/manifest.py`` so the block lands verbatim in the .iemodel.json
manifest; Sim side, mirror ``_cloth_extras_from_manifest`` with a
``_fluid_extras_from_manifest`` reader, tag the ``water``-labelled part
with ``"water"`` (the render-world water pass opts in on that tag), bind
``physics_material`` for buoyancy contacts, and feed
``density/viscosity/surface_tension/volume`` to the ``fluids_sph``
solver once it lands. The same block also rides inside the water
primitive's ``params["extras"]["fluid"]``, so it survives
``GenerationSpec.to_json``/``from_json`` round-trips unchanged.

Note on colour: the compositor colours every part from the single
``spec.color``; the water part's blue comes from ``color_tint`` applied
downstream (BonaFide / Sim), exactly like the material-map pipeline.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..alignment.schema import GenerationSpec, Primitive
from .style_families import FamilyContext

WATER_CONTAINERS = ("basin", "bucket", "pond", "aquarium", "vessel")

# Default water albedo — identical to Sim render_world's u_water_color.
DEFAULT_WATER_TINT = (0.08, 0.15, 0.22)

# Gap between the water body and the cavity wall (no z-fighting, and the
# meniscus lip — which climbs the wall by up to its radius — never pokes
# through it).
_WALL_GAP = 0.004


@dataclass
class FluidProperties:
    """Sim-consumable fluid state for one water body."""

    fluid: str = "water"
    solver: str = "fluids_sph"
    physics_material: str = "water_surface"
    volume_m3: float = 0.0
    fill_level: float = 0.7
    density_kg_m3: float = 1000.0
    viscosity_pa_s: float = 0.001
    surface_tension_n_m: float = 0.072
    restitution: float = 0.0
    color_tint: tuple[float, float, float] = DEFAULT_WATER_TINT
    meniscus_radius_m: float = 0.0027
    container_part: str = ""
    water_part: str = "water"
    meniscus_part: str = "water_meniscus"
    cavity: str = "round"                 # round | box
    interior_bottom_m: float = 0.0
    interior_depth_m: float = 0.0
    interior_radius_m: float = 0.0        # round cavities
    interior_size_m: tuple[float, float] | None = None   # box cavities (X, Z)

    def __post_init__(self) -> None:
        self.fill_level = float(np.clip(self.fill_level, 0.0, 1.0))
        if self.cavity not in ("round", "box"):
            raise ValueError(f"unknown cavity {self.cavity!r}")
        if self.cavity == "round" and self.interior_radius_m <= 0.0:
            raise ValueError("round cavity needs interior_radius_m > 0")
        if self.cavity == "box" and not self.interior_size_m:
            raise ValueError("box cavity needs interior_size_m")

    # -- derived quantities ------------------------------------------------
    @property
    def capillary_length_m(self) -> float:
        """sqrt(σ / (ρ g)) — the physical meniscus scale."""
        return math.sqrt(self.surface_tension_n_m / (self.density_kg_m3 * 9.81))

    def cross_section_m2(self) -> float:
        if self.cavity == "round":
            return math.pi * self.interior_radius_m ** 2
        w, d = self.interior_size_m
        return w * d

    def capacity_m3(self) -> float:
        return self.cross_section_m2() * self.interior_depth_m

    def volume_at(self, fill_level: float) -> float:
        return self.capacity_m3() * float(np.clip(fill_level, 0.0, 1.0))

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["color_tint"] = list(self.color_tint)
        if self.interior_size_m is not None:
            d["interior_size_m"] = list(self.interior_size_m)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FluidProperties":
        d = dict(data)
        if d.get("color_tint") is not None:
            d["color_tint"] = tuple(d["color_tint"])
        if d.get("interior_size_m") is not None:
            d["interior_size_m"] = tuple(d["interior_size_m"])
        return cls(**d)


# JSON-schema-flavoured documentation of the extras.fluid block (see the
# module docstring for the full consumption contract).
FLUID_EXTRAS_SCHEMA: dict[str, Any] = {
    "block": "fluid",
    "carriers": ["primitive.params.extras.fluid", "spec.manifest_extras.fluid",
                 "manifest top-level 'fluid' (once wired by integrator)"],
    "required": ["fluid", "volume_m3", "fill_level", "density_kg_m3",
                 "viscosity_pa_s", "surface_tension_n_m", "restitution",
                 "color_tint", "water_part", "cavity", "interior_bottom_m",
                 "interior_depth_m"],
    "properties": {
        "fluid": {"type": "string", "default": "water"},
        "solver": {"type": "string", "default": "fluids_sph",
                   "note": "IronEngine-Sim solver registry name (stub in MVP)"},
        "physics_material": {"type": "string", "default": "water_surface",
                             "note": "IronEngine-Sim material_library key"},
        "volume_m3": {"type": "number", "minimum": 0.0},
        "fill_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "density_kg_m3": {"type": "number", "default": 1000.0},
        "viscosity_pa_s": {"type": "number", "default": 0.001},
        "surface_tension_n_m": {"type": "number", "default": 0.072},
        "restitution": {"type": "number", "default": 0.0},
        "color_tint": {"type": "array", "items": "number", "length": 3,
                       "default": list(DEFAULT_WATER_TINT)},
        "meniscus_radius_m": {"type": "number", "default": 0.0027},
        "cavity": {"enum": ["round", "box"]},
        "container_part": {"type": "string",
                           "note": "label of the containing part"},
        "water_part": {"type": "string", "default": "water"},
        "meniscus_part": {"type": "string", "default": "water_meniscus"},
        "interior_bottom_m": {"type": "number"},
        "interior_depth_m": {"type": "number"},
        "interior_radius_m": {"type": "number", "note": "round cavities"},
        "interior_size_m": {"type": "array", "items": "number", "length": 2,
                            "note": "box cavities (X, Z)"},
    },
}


# ----------------------------------------------------------------------
# water-body emission
# ----------------------------------------------------------------------

def _emit_water(ctx: FamilyContext, fluid: FluidProperties) -> None:
    """Emit the `water` body + `water_meniscus` lip for the given fill."""
    fill = fluid.fill_level
    h_fill = max(fill * fluid.interior_depth_m, 1e-4)
    y0 = fluid.interior_bottom_m
    men = float(np.clip(fluid.capillary_length_m, 0.001, _WALL_GAP - 0.001))
    fluid.meniscus_radius_m = men
    extras = {"fluid": fluid.to_dict()}

    if fluid.cavity == "round":
        r_w = fluid.interior_radius_m - _WALL_GAP
        ctx.add("cylinder", (0.0, y0 + h_fill / 2, 0.0),
                {"radius": r_w, "height": h_fill, "caps": True,
                 "material": "glass", "extras": extras},
                fluid.water_part)
        ctx.add("torus", (0.0, y0 + h_fill, 0.0),
                {"major_radius": r_w, "minor_radius": men,
                 "material": "glass"},
                fluid.meniscus_part)
    else:
        w_i, d_i = fluid.interior_size_m
        w_w, d_w = w_i - 2 * _WALL_GAP, d_i - 2 * _WALL_GAP
        ctx.add("box", (0.0, y0 + h_fill / 2, 0.0),
                {"size": [w_w, h_fill, d_w], "material": "glass",
                 "extras": extras},
                fluid.water_part)
        # Meniscus frame: a hair-thin slab a meniscus-width wider than the
        # water slab, climbing the walls (still inside the cavity).
        ctx.add("box", (0.0, y0 + h_fill, 0.0),
                {"size": [w_w + 2 * men, 0.0015, d_w + 2 * men],
                 "material": "glass"},
                fluid.meniscus_part)


def _finish_spec(ctx: FamilyContext, kind: str, fluid: FluidProperties,
                 seed: int, n_points: int, color) -> GenerationSpec:
    fluid.volume_m3 = fluid.volume_at(fluid.fill_level)
    spec = GenerationSpec(
        shape=f"water_{kind}",
        n_points=int(n_points),
        bbox_size=(1.0, 1.0, 1.0),
        primitives=ctx.primitives,
        features=ctx.features,
        color=color,
        seed=int(seed or 0),
    )
    # Re-embed the final numbers (volume) into both extras carriers.
    block = fluid.to_dict()
    spec.manifest_extras = {"fluid": block,
                            "physics": {"body_type": "rigid"}}
    for prim in spec.primitives:
        if prim.label == fluid.water_part:
            prim.params["extras"] = {"fluid": block}
    return spec


# ----------------------------------------------------------------------
# container grammars (metric scale at size_scale = 1)
# ----------------------------------------------------------------------

def build_basin(ctx: FamilyContext, s: float, fill: float, tint) -> FluidProperties:
    ctx.shape = "water_basin"
    mat = "ceramic"
    outer_r, floor_t, wall_h = 0.14 * s, 0.010 * s, 0.055 * s
    ctx.add("cylinder", (0, floor_t / 2, 0),
            {"radius": outer_r, "height": floor_t, "caps": True},
            "basin_floor", material=mat)
    ctx.add("cylinder", (0, floor_t + wall_h / 2, 0),
            {"radius": outer_r, "height": wall_h, "caps": False},
            "basin_wall", material=mat)
    rim_minor = 0.0045 * s
    ctx.add("torus", (0, floor_t + wall_h, 0),
            {"major_radius": outer_r - 0.002 * s, "minor_radius": rim_minor},
            "basin_rim", material=mat)
    ctx.add_feature("ridges", "all", count=10, depth=0.0015 * s)
    fluid = FluidProperties(fill_level=fill, color_tint=tint,
                            container_part="basin_wall", cavity="round",
                            interior_bottom_m=floor_t,
                            interior_depth_m=wall_h - _WALL_GAP * s,
                            interior_radius_m=outer_r - 0.006 * s)
    _emit_water(ctx, fluid)
    return fluid


def build_bucket(ctx: FamilyContext, s: float, fill: float, tint) -> FluidProperties:
    ctx.shape = "water_bucket"
    mat = "metal"
    r_bot, r_mid = 0.085 * s, 0.095 * s
    floor_t, h1, h2 = 0.010 * s, 0.08 * s, 0.08 * s
    ctx.add("cylinder", (0, floor_t / 2, 0),
            {"radius": r_bot, "height": floor_t, "caps": True},
            "bucket_floor", material=mat)
    # Stepped taper: two open cylinder bands (cone walls aren't hollowable).
    ctx.add("cylinder", (0, floor_t + h1 / 2, 0),
            {"radius": r_bot, "height": h1, "caps": False},
            "bucket_wall_lo", material=mat)
    ctx.add("cylinder", (0, floor_t + h1 + h2 / 2, 0),
            {"radius": r_mid, "height": h2, "caps": False},
            "bucket_wall_hi", material=mat)
    top_y = floor_t + h1 + h2
    ctx.add("torus", (0, top_y, 0),
            {"major_radius": r_mid, "minor_radius": 0.005 * s},
            "bucket_rim", material=mat)
    # Handle: thin tube arcing over the mouth.
    hr = r_mid + 0.01 * s
    ctx.add("tube", (0, 0, 0),
            {"path": [[-r_mid, top_y, 0.0], [0.0, top_y + hr, 0.0],
                      [r_mid, top_y, 0.0]],
             "radius": 0.004 * s, "caps": True, "height": hr * 2},
            "bucket_handle", material=mat)
    fluid = FluidProperties(fill_level=fill, color_tint=tint,
                            container_part="bucket_wall_lo", cavity="round",
                            interior_bottom_m=floor_t,
                            interior_depth_m=top_y - floor_t - _WALL_GAP * s,
                            interior_radius_m=r_bot - 0.006 * s)
    _emit_water(ctx, fluid)
    return fluid


def build_pond(ctx: FamilyContext, s: float, fill: float, tint) -> FluidProperties:
    ctx.shape = "water_pond"
    w, d, t = 0.60 * s, 0.60 * s, 0.040 * s
    ctx.add("box", (0, t / 2, 0), {"size": [w, t, d]}, "pond_bank",
            material="stone")
    ctx.add_feature("relief", "pond_bank", amplitude=0.008 * s,
                    frequency=6.0, octaves=3, pebbles=3)
    # Stone ring rim around the water line. The spill point is the LOWEST
    # stone top — the cavity depth stops a meniscus-gap below it so a full
    # pond never overtops the ring.
    ring_r, n_stones = 0.19 * s, 8
    spill = math.inf
    for i in range(n_stones):
        a = 2 * math.pi * i / n_stones
        r = float(ctx.uniform(0.020, 0.035)) * s
        sx, sz = math.cos(a) * ring_r, math.sin(a) * ring_r
        ry = r * float(ctx.uniform(0.7, 1.0))
        ctx.add("ellipsoid", (sx, t - r * 0.2, sz),
                {"radii": [r, ry, r]}, f"pond_stone_{i}",
                ry=float(ctx.uniform(0, math.pi)), material="stone")
        spill = min(spill, t - r * 0.2 + ry)
    fluid = FluidProperties(fill_level=fill, color_tint=tint,
                            container_part="pond_bank", cavity="round",
                            interior_bottom_m=t - 0.005 * s,
                            interior_depth_m=spill - (t - 0.005 * s) - _WALL_GAP * s,
                            interior_radius_m=0.15 * s)
    _emit_water(ctx, fluid)
    return fluid


def build_aquarium(ctx: FamilyContext, s: float, fill: float, tint) -> FluidProperties:
    ctx.shape = "water_aquarium"
    w, d, h, gt = 0.24 * s, 0.16 * s, 0.14 * s, 0.004 * s
    glass = "glass"
    ctx.add("box", (0, gt / 2, 0), {"size": [w, gt, d]}, "aquarium_floor",
            material=glass)
    for sz_ in (-1, 1):
        ctx.add("box", (0, gt + h / 2, sz_ * (d / 2 - gt / 2)),
                {"size": [w, h, gt]}, f"aquarium_wall_z{sz_}", material=glass)
    for sx_ in (-1, 1):
        ctx.add("box", (sx_ * (w / 2 - gt / 2), gt + h / 2, 0),
                {"size": [gt, h, d - 2 * gt]}, f"aquarium_wall_x{sx_}",
                material=glass)
    fluid = FluidProperties(fill_level=fill, color_tint=tint,
                            container_part="aquarium_wall_z-1", cavity="box",
                            interior_bottom_m=gt,
                            interior_depth_m=h - 0.010 * s,
                            interior_size_m=(w - 2 * gt, d - 2 * gt))
    _emit_water(ctx, fluid)
    return fluid


def build_vessel(ctx: FamilyContext, s: float, fill: float, tint) -> FluidProperties:
    ctx.shape = "water_vessel"
    mat = "ceramic"
    belly_r, belly_h, floor_t = 0.11 * s, 0.10 * s, 0.010 * s
    ctx.add("cylinder", (0, floor_t / 2, 0),
            {"radius": belly_r, "height": floor_t, "caps": True},
            "vessel_floor", material=mat)
    # Straight cylindrical belly = the fillable cavity (open wall surfaces).
    ctx.add("cylinder", (0, floor_t + belly_h / 2, 0),
            {"radius": belly_r, "height": belly_h, "caps": False},
            "vessel_belly", material=mat)
    y = floor_t + belly_h
    # Shoulder steps in, then an open neck and a rim lip.
    for i, (r, hh) in enumerate(((0.085 * s, 0.020 * s), (0.062 * s, 0.020 * s))):
        ctx.add("cylinder", (0, y + hh / 2, 0),
                {"radius": r, "height": hh, "caps": False},
                f"vessel_shoulder_{i}", material=mat)
        y += hh
    neck_r, neck_h = 0.055 * s, 0.050 * s
    ctx.add("cylinder", (0, y + neck_h / 2, 0),
            {"radius": neck_r, "height": neck_h, "caps": False},
            "vessel_neck", material=mat)
    y += neck_h
    ctx.add("torus", (0, y, 0),
            {"major_radius": neck_r, "minor_radius": 0.005 * s},
            "vessel_rim", material=mat)
    ctx.add_feature("ridges", "all", count=12, depth=0.002 * s)
    fluid = FluidProperties(fill_level=fill, color_tint=tint,
                            container_part="vessel_belly", cavity="round",
                            interior_bottom_m=floor_t,
                            interior_depth_m=belly_h - _WALL_GAP * s,
                            interior_radius_m=belly_r - 0.006 * s)
    _emit_water(ctx, fluid)
    return fluid


CONTAINER_BUILDERS = {
    "basin": build_basin,
    "bucket": build_bucket,
    "pond": build_pond,
    "aquarium": build_aquarium,
    "vessel": build_vessel,
}


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------

def water_container_spec(kind: str = "basin", fill_level: float = 0.7,
                         size_scale: float = 1.0, seed: int = 0,
                         n_points: int = 50_000,
                         color_tint: tuple[float, float, float] = DEFAULT_WATER_TINT,
                         color: tuple[float, float, float] = (0.88, 0.86, 0.80),
                         ) -> GenerationSpec:
    """Build one water container at real metric scale (Y-up, floor at y=0).

    The returned spec contains the container parts plus a ``water`` body
    and a ``water_meniscus`` lip; the water primitive carries
    ``params["extras"]["fluid"]`` and the spec carries
    ``manifest_extras["fluid"]`` — both are the same `FluidProperties`
    block. Specs are emitted at true scale on purpose: ``volume_m3`` and
    the interior measurements are metric facts, so no bbox fit is applied.
    """
    if kind not in WATER_CONTAINERS:
        raise ValueError(f"unknown water container {kind!r} {WATER_CONTAINERS}")
    fill_level = float(np.clip(fill_level, 0.0, 1.0))
    size_scale = float(np.clip(size_scale, 0.05, 20.0))
    rng = np.random.default_rng(seed or None)
    ctx = FamilyContext(rng=rng, target_parts=64)
    fluid = CONTAINER_BUILDERS[kind](ctx, size_scale, fill_level, tuple(color_tint))
    return _finish_spec(ctx, kind, fluid, seed, n_points, color)


def find_water_parts(spec: GenerationSpec) -> tuple[Primitive | None, Primitive | None]:
    """(water body, meniscus) primitives of a water-container spec."""
    body = lip = None
    for prim in spec.primitives:
        if prim.label == "water":
            body = prim
        elif prim.label == "water_meniscus":
            lip = prim
    return body, lip


def fluid_of(spec: GenerationSpec) -> FluidProperties:
    """Recover the fluid block from a spec (water-part extras first)."""
    body, _ = find_water_parts(spec)
    block = None
    if body is not None:
        block = (body.params.get("extras") or {}).get("fluid")
    if block is None:
        block = (getattr(spec, "manifest_extras", None) or {}).get("fluid")
    if not isinstance(block, dict):
        raise ValueError("spec carries no extras.fluid block")
    return FluidProperties.from_dict(block)


def set_fill_level(spec: GenerationSpec, fill_level: float) -> GenerationSpec:
    """Return a NEW spec with the water body rescaled to `fill_level`.

    Surface height, meniscus position, and ``volume_m3`` are recomputed
    from the cavity cross-section stored in the fluid extras; container
    parts are carried over unchanged (deep-copied — the input spec is
    never mutated). Fill 1.0 tops out at the cavity depth (rim-safe);
    fill < 1 leaves a visible air gap below the rim.
    """
    fill_level = float(np.clip(fill_level, 0.0, 1.0))
    fluid = fluid_of(spec)
    fluid.fill_level = fill_level
    h_fill = max(fill_level * fluid.interior_depth_m, 1e-4)
    y0 = fluid.interior_bottom_m
    men = fluid.meniscus_radius_m
    fluid.volume_m3 = fluid.volume_at(fill_level)
    block = fluid.to_dict()

    new_prims: list[Primitive] = []
    for prim in spec.primitives:
        params = dict(prim.params or {})
        T = [list(row) for row in prim.transform]
        if prim.label == fluid.water_part:
            if fluid.cavity == "round":
                params["height"] = h_fill
            else:
                params["size"] = [params["size"][0], h_fill, params["size"][2]]
            T[1][3] = y0 + h_fill / 2
            params["extras"] = {"fluid": block}
        elif prim.label == fluid.meniscus_part:
            T[1][3] = y0 + h_fill
        new_prims.append(Primitive(kind=prim.kind, transform=T,
                                   params=params, label=prim.label))
    out = GenerationSpec(
        shape=spec.shape, n_points=spec.n_points, bbox_size=spec.bbox_size,
        primitives=new_prims, features=list(spec.features),
        color=spec.color, seed=spec.seed,
    )
    out.manifest_extras = {"fluid": block,
                           "physics": {"body_type": "rigid"}}
    return out


def set_fill_volume(spec: GenerationSpec, volume_m3: float) -> GenerationSpec:
    """Fill to an absolute volume (inverse of `set_fill_level`)."""
    fluid = fluid_of(spec)
    cap = fluid.capacity_m3()
    fill = 0.0 if cap <= 0 else float(np.clip(volume_m3 / cap, 0.0, 1.0))
    return set_fill_level(spec, fill)
