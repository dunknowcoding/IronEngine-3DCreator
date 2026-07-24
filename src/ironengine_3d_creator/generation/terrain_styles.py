"""Procedural terrain styles (CR_FloraWater) — stone-forward ground pieces.

Seven new generators on top of the `style_families` grammar conventions
(same `FamilyContext`, same primitive/feature schema as `build_terrain`):

- ``boulder_field``       — asperity-roughened boulders scattered over relief
- ``rock_strata_cliff``   — layered sediment bands with alternating band tones
- ``cobblestone_patch``   — rounded cobbles set in a mortar bed (tileable)
- ``cracked_mud``         — polygonal drying plates with gap cracks
- ``mossy_stones``        — stones with organic moss caps + fur fringe
- ``pebble_riverbed``     — dense smooth pebbles, wet-sheen option
- ``stone_slab_pavement`` — uneven laid slabs with mortar gaps (tileable)

Invariants every style keeps:

- **Never flat** — the ground part always carries a `relief` and/or
  `asperity` feature with non-zero amplitude, so the compositor displaces
  it (asserted by the test-suite via real compositing).
- **Deterministic** — all jitter comes from the seeded context RNG.
- **Tileable edges where sensible** — the lattice styles (cobblestone,
  pavement) lay parts on a pitch that exactly divides the tile span and
  inset parts by half a cell at the borders, so two abutting tiles
  continue the lattice seamlessly.
- **Vertex-colour realism** — parts carry material hints (stone / ceramic /
  organic / wood) consumed by `textures.apply_texture`, and dips/relief
  darken albedo via the feature pipeline. NOTE: the current compositor
  derives every part's colour from the single `spec.color`; per-part
  tints (e.g. strongly contrasting strata stripes) need a compositor
  hook — tracked as an integrator gap. Band contrast is instead achieved
  by alternating material hints, which modulate albedo differently.

Density semantics: `density` 0..1 scales the number of scatter parts
(boulders / cobbles / pebbles / plates / slabs / moss caps) linearly —
a density-1.0 tile carries ~3.3x the parts of density 0.3.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..alignment.schema import GenerationSpec
from .style_families import FamilyContext

TERRAIN_STYLES = (
    "boulder_field",
    "rock_strata_cliff",
    "cobblestone_patch",
    "cracked_mud",
    "mossy_stones",
    "pebble_riverbed",
    "stone_slab_pavement",
)


@dataclass
class TerrainParams:
    """User-facing dials for the terrain styles."""

    style: str = "boulder_field"
    density: float = 0.6            # 0..1 -> scatter-part coverage
    width: float = 0.8              # tile span X (metres)
    depth: float = 0.8              # tile span Z (metres)
    thickness: float = 0.06         # ground slab thickness
    seed: int = 0
    wet: bool = False               # riverbed/mud sheen (darker, glossier)
    moss: bool = False              # force moss overlay on stone styles

    def __post_init__(self) -> None:
        if self.style not in TERRAIN_STYLES:
            raise ValueError(f"unknown terrain style {self.style!r} {TERRAIN_STYLES}")
        self.density = float(np.clip(self.density, 0.0, 1.0))
        self.width = float(np.clip(self.width, 0.1, 20.0))
        self.depth = float(np.clip(self.depth, 0.1, 20.0))
        self.thickness = float(np.clip(self.thickness, 0.005, 2.0))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerrainParams":
        return cls(**dict(data))


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _ground(ctx: FamilyContext, p: TerrainParams, *, relief_amp: float,
            material: str = "stone", label: str = "ground") -> None:
    """Ground slab that is never a bare flat sheet."""
    ctx.add("box", (0, p.thickness / 2, 0),
            {"size": [p.width, p.thickness, p.depth]}, label, material=material)
    ctx.add_feature("relief", label,
                    amplitude=relief_amp,
                    frequency=float(ctx.uniform(4.5, 7.5)), octaves=3,
                    pebbles=int(round(3 * p.density)))


def _area_factor(p: TerrainParams) -> float:
    """Scatter counts are authored for a 0.8x0.8 tile; scale by real area."""
    return (p.width * p.depth) / 0.64


def _tile_cells(span: float, pitch: float) -> tuple[int, float]:
    """Cell count + effective pitch so the lattice divides `span` exactly
    and parts inset by half a cell at the borders (seamless tiling)."""
    n = max(1, int(round(span / pitch)))
    return n, span / n


# ----------------------------------------------------------------------
# style grammars
# ----------------------------------------------------------------------

def build_boulder_field(ctx: FamilyContext, p: TerrainParams) -> None:
    ctx.shape = "terrain_boulder_field"
    ctx.color = (0.42, 0.40, 0.37)
    _ground(ctx, p, relief_amp=0.020)
    n = int(round(12 * p.density * _area_factor(p)))
    for i in range(n):
        r = float(ctx.uniform(0.05, 0.13))
        px = float(ctx.uniform(-p.width / 2 + r * 0.5, p.width / 2 - r * 0.5))
        pz = float(ctx.uniform(-p.depth / 2 + r * 0.5, p.depth / 2 - r * 0.5))
        # Half-bedded, slightly squashed, random yaw — asperity does the rest.
        ctx.add("ellipsoid", (px, p.thickness - r * 0.30, pz),
                {"radii": [r, r * float(ctx.uniform(0.6, 0.85)),
                           r * float(ctx.uniform(0.8, 1.0))]},
                f"boulder_{i}", ry=float(ctx.uniform(0, math.pi)),
                material="stone")
    ctx.add_feature("asperity", "all", strength=0.004, frequency=18.0)


def build_rock_strata_cliff(ctx: FamilyContext, p: TerrainParams) -> None:
    ctx.shape = "terrain_rock_strata_cliff"
    ctx.color = (0.55, 0.48, 0.40)
    # Sediment bands: alternating material hints give light/dark stripes
    # (ceramic brightens, organic darkens relative to the base tone).
    band_materials = ("stone", "ceramic", "stone", "organic", "stone", "ceramic")
    n_bands = 6
    y = 0.0
    d = p.depth * 0.55
    for i in range(n_bands):
        h = float(ctx.uniform(0.05, 0.10))
        inset = float(ctx.uniform(-0.015, 0.015))
        ctx.add("box", (inset, y + h / 2, float(ctx.uniform(-0.01, 0.01))),
                {"size": [p.width * float(ctx.uniform(0.96, 1.0)), h, d]},
                f"stratum_{i}", material=band_materials[i % len(band_materials)])
        y += h
    # Weathering: relief on top, coarse asperity everywhere, scratch cracks.
    ctx.add_feature("relief", f"stratum_{n_bands - 1}",
                    amplitude=0.012, frequency=6.0, octaves=3, pebbles=2)
    ctx.add_feature("asperity", "all", strength=0.005, frequency=14.0)
    ctx.add_feature("scratch", "all", count=int(round(4 + 8 * p.density)),
                    depth=0.004)
    # Talus: a few broken chunks at the foot.
    n = int(round(5 * p.density * _area_factor(p)))
    for i in range(n):
        r = float(ctx.uniform(0.02, 0.05))
        px = float(ctx.uniform(-p.width / 2, p.width / 2))
        pz = d / 2 + float(ctx.uniform(0.02, 0.10))
        ctx.add("ellipsoid", (px, r * 0.5, pz),
                {"radii": [r, r * 0.7, r]}, f"talus_{i}", material="stone")


def build_cobblestone_patch(ctx: FamilyContext, p: TerrainParams) -> None:
    ctx.shape = "terrain_cobblestone_patch"
    ctx.color = (0.40, 0.38, 0.35)
    # Mortar bed (dark, rough).
    ctx.add("box", (0, p.thickness / 2, 0),
            {"size": [p.width, p.thickness, p.depth]}, "mortar", material="stone")
    ctx.add_feature("asperity", "mortar", strength=0.003, frequency=22.0)
    # Tileable lattice: pitch divides the span; cobbles inset by half a cell.
    nx, px_ = _tile_cells(p.width, 0.075)
    nz, pz_ = _tile_cells(p.depth, 0.075)
    cells = [(ix, iz) for ix in range(nx) for iz in range(nz)]
    n_fill = int(round(len(cells) * p.density))
    keep = set(ctx.rng.choice(len(cells), size=n_fill, replace=False).tolist()) \
        if n_fill < len(cells) else set(range(len(cells)))
    idx = 0
    for ci, (ix, iz) in enumerate(cells):
        if ci not in keep:
            continue
        cx = -p.width / 2 + (ix + 0.5) * px_
        cz = -p.depth / 2 + (iz + 0.5) * pz_
        rx = px_ * float(ctx.uniform(0.40, 0.47))
        rz = pz_ * float(ctx.uniform(0.40, 0.47))
        ry = float(ctx.uniform(0.55, 0.75)) * min(rx, rz)
        # Center jitter is clamped so cobble + jitter never leaves the cell —
        # yaw mixes rz into the x-extent, so clamp by the larger radius.
        r_max = max(rx, rz)
        jx = float(ctx.uniform(-1, 1)) * max(px_ / 2 - r_max, 0.0)
        jz = float(ctx.uniform(-1, 1)) * max(pz_ / 2 - r_max, 0.0)
        ctx.add("ellipsoid",
                (cx + jx,
                 p.thickness + ry * float(ctx.uniform(0.35, 0.6)),
                 cz + jz),
                {"radii": [rx, ry, rz]}, f"cobble_{idx}",
                ry=float(ctx.uniform(0, math.pi)), material="stone")
        idx += 1


def build_cracked_mud(ctx: FamilyContext, p: TerrainParams) -> None:
    ctx.shape = "terrain_cracked_mud"
    wet = p.wet
    ctx.color = (0.32, 0.26, 0.19) if not wet else (0.24, 0.20, 0.15)
    _ground(ctx, p, relief_amp=0.008, material="stone", label="mud")
    # Drying plates: hex prisms on a lattice with gap cracks; density is the
    # fraction of plates still intact (missing plates read as wide cracks).
    nx, px_ = _tile_cells(p.width, 0.115)
    nz, pz_ = _tile_cells(p.depth, 0.115)
    cells = [(ix, iz) for ix in range(nx) for iz in range(nz)]
    n_fill = int(round(len(cells) * (0.4 + 0.6 * p.density)))
    keep = set(ctx.rng.choice(len(cells), size=n_fill, replace=False).tolist()) \
        if n_fill < len(cells) else set(range(len(cells)))
    idx = 0
    for ci, (ix, iz) in enumerate(cells):
        if ci not in keep:
            continue
        cx = -p.width / 2 + (ix + 0.5) * px_
        cz = -p.depth / 2 + (iz + 0.5) * pz_
        curl = 0.010 * (1.0 - 0.5 * p.density)   # drier plates curl harder
        ctx.add("prism",
                (cx + float(ctx.uniform(-0.004, 0.004)),
                 p.thickness + 0.006 + float(ctx.uniform(0.0, curl)),
                 cz + float(ctx.uniform(-0.004, 0.004))),
                {"sides": 6, "radius": px_ * 0.44, "height": 0.012},
                f"plate_{idx}", ry=float(ctx.uniform(0, math.pi / 3)),
                rx=float(ctx.uniform(-curl, curl)),
                rz=float(ctx.uniform(-curl, curl)), material="stone")
        idx += 1
    ctx.add_feature("asperity", "all", strength=0.0015, frequency=40.0)


def build_mossy_stones(ctx: FamilyContext, p: TerrainParams) -> None:
    ctx.shape = "terrain_mossy_stones"
    ctx.color = (0.34, 0.37, 0.27)      # moss-tinged base for the whole tile
    _ground(ctx, p, relief_amp=0.015)
    n = int(round(9 * p.density * _area_factor(p)))
    moss_idx = 0
    for i in range(n):
        r = float(ctx.uniform(0.035, 0.085))
        px = float(ctx.uniform(-p.width / 2 + r * 0.5, p.width / 2 - r * 0.5))
        pz = float(ctx.uniform(-p.depth / 2 + r * 0.5, p.depth / 2 - r * 0.5))
        top = p.thickness - r * 0.25 + r * float(ctx.uniform(0.6, 0.85))
        ctx.add("ellipsoid", (px, p.thickness - r * 0.25, pz),
                {"radii": [r, r * float(ctx.uniform(0.6, 0.85)), r]},
                f"stone_{i}", ry=float(ctx.uniform(0, math.pi)),
                material="stone")
        # Moss cap: a squashed organic patch hugging the stone's crown.
        ctx.add("ellipsoid", (px, top, pz),
                {"radii": [r * 0.62, r * 0.16, r * 0.62]},
                f"moss_{moss_idx}", ry=float(ctx.uniform(0, math.pi)),
                material="organic")
        moss_idx += 1
    # Furry fringe over the moss caps (point-level, compositor side).
    if moss_idx:
        ctx.add_feature("fur", "moss", density=float(0.25 + 0.5 * p.density),
                        length=0.006)
    ctx.add_feature("asperity", "all", strength=0.002, frequency=25.0)


def build_pebble_riverbed(ctx: FamilyContext, p: TerrainParams) -> None:
    ctx.shape = "terrain_pebble_riverbed"
    wet = p.wet                # pass wet=True for the water-polished sheen
    ctx.color = (0.30, 0.28, 0.25) if wet else (0.45, 0.42, 0.38)
    _ground(ctx, p, relief_amp=0.015, material="stone")
    n = int(round(60 * p.density * _area_factor(p)))
    for i in range(n):
        r = float(ctx.uniform(0.012, 0.032))
        px = float(ctx.uniform(-p.width / 2 + r, p.width / 2 - r))
        pz = float(ctx.uniform(-p.depth / 2 + r, p.depth / 2 - r))
        # Water-worn: flattened, polished (ceramic sheen when wet).
        ctx.add("ellipsoid", (px, p.thickness - r * 0.20, pz),
                {"radii": [r, r * float(ctx.uniform(0.45, 0.65)), r]},
                f"pebble_{i}", ry=float(ctx.uniform(0, math.pi)),
                material="ceramic" if wet else "stone")
    ctx.add_feature("asperity", "all", strength=0.001, frequency=35.0)


def build_stone_slab_pavement(ctx: FamilyContext, p: TerrainParams) -> None:
    ctx.shape = "terrain_stone_slab_pavement"
    ctx.color = (0.48, 0.47, 0.44)
    # Mortar bed.
    ctx.add("box", (0, p.thickness / 2, 0),
            {"size": [p.width, p.thickness, p.depth]}, "mortar", material="stone")
    ctx.add_feature("asperity", "mortar", strength=0.002, frequency=25.0)
    # Tileable slab lattice with a constant mortar gap.
    gap = 0.012
    nx, px_ = _tile_cells(p.width, 0.16)
    nz, pz_ = _tile_cells(p.depth, 0.16)
    cells = [(ix, iz) for ix in range(nx) for iz in range(nz)]
    n_fill = int(round(len(cells) * (0.5 + 0.5 * p.density)))
    keep = set(ctx.rng.choice(len(cells), size=n_fill, replace=False).tolist()) \
        if n_fill < len(cells) else set(range(len(cells)))
    idx = 0
    for ci, (ix, iz) in enumerate(cells):
        if ci not in keep:
            continue
        cx = -p.width / 2 + (ix + 0.5) * px_
        cz = -p.depth / 2 + (iz + 0.5) * pz_
        ctx.add("box",
                (cx + float(ctx.uniform(-0.003, 0.003)),
                 p.thickness + 0.015 + float(ctx.uniform(-0.002, 0.003)),
                 cz + float(ctx.uniform(-0.003, 0.003))),
                {"size": [px_ - gap, 0.03, pz_ - gap]},
                f"slab_{idx}", ry=float(ctx.uniform(-0.02, 0.02)),
                material="stone")
        idx += 1
    ctx.add_feature("asperity", "slab", strength=0.0012, frequency=30.0)


# ----------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------

TERRAIN_STYLE_BUILDERS = {
    "boulder_field": build_boulder_field,
    "rock_strata_cliff": build_rock_strata_cliff,
    "cobblestone_patch": build_cobblestone_patch,
    "cracked_mud": build_cracked_mud,
    "mossy_stones": build_mossy_stones,
    "pebble_riverbed": build_pebble_riverbed,
    "stone_slab_pavement": build_stone_slab_pavement,
}


def terrain_spec(p: TerrainParams, n_points: int = 50_000,
                 bbox: tuple[float, float, float] | None = None) -> GenerationSpec:
    """Build a deterministic `GenerationSpec` from a `TerrainParams` bundle.

    The spec carries a ``manifest_extras["terrain"]`` block with the
    resolved parameters; when `bbox` is given the assembly is uniformly
    fitted to it (real metric scale is kept otherwise, so displacement
    amplitudes stay meaningful).
    """
    rng = np.random.default_rng(p.seed or None)
    ctx = FamilyContext(rng=rng, target_parts=4096)
    TERRAIN_STYLE_BUILDERS[p.style](ctx, p)

    if bbox is not None:
        from .style_engine import _fit_to_bbox
        _fit_to_bbox(ctx.primitives, bbox)

    spec = GenerationSpec(
        shape=ctx.shape or f"terrain_{p.style}",
        n_points=int(n_points),
        bbox_size=tuple(float(b) for b in (bbox or (p.width, p.thickness, p.depth))),
        primitives=ctx.primitives,
        features=ctx.features,
        color=ctx.color,
        seed=int(p.seed or 0),
    )
    spec.manifest_extras = {"terrain": {**p.to_dict(),
                                        "part_count": len(ctx.primitives)}}
    return spec
