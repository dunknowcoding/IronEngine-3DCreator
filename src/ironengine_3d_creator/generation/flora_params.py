"""Unified parameter layer for flora generators (CR_FloraWater).

One `FloraParams` bundle drives every plant archetype — trees with
branch-level control, grass patches, shrubs, and flowers — so a UI or an
LLM tool-call can dial a plant up/down with a single coherent vocabulary:

- ``density``   0..1 — scales leaf / petal / blade / floret COUNTS
  (strictly proportional, no hidden floor: density 1.0 always carries
  ~3.3x the blades of density 0.3). Structure (trunk, branches) is kept
  mostly density-independent so sparse trees still read as trees.
- ``size_scale`` — uniform multiplier on every length (metres at 1.0).
- ``style``     — species preset: oak / maple / pine / palm / fern
  (trees + tree-fern), rose / lavender / sunflower (flowers),
  meadow / lawn (grass), boxwood (shrub).
- ``season``    — spring / summer / autumn / winter: swaps the leaf/blossom
  colour palettes and the leaf-presence factor (deciduous species drop
  nearly all leaves in winter; evergreens keep theirs).
- ``age``       — sapling / mature / ancient: trunk girth, height and
  gnarl (trunk wander + root flare).
- ``branching`` — whorled (conifer rings) / alternate (broadleaf spiral).
  Defaults to the species' natural habit; explicit override wins.
- ``seed``      — full determinism: same params + seed => identical spec.

Instancing-aware emission
-------------------------
Repeated parts (leaves, needles, fronds, petals, florets, grass blades)
are emitted from a small set of *templates*: every instance of a template
shares IDENTICAL geometry params and carries
``params["instance_of"] = "<template name>"``; only the transform varies.
Exporters can therefore bake one mesh per template and stamp N transforms.
`collect_instance_groups(spec)` recovers the grouping.

The builders write into the existing `style_families.FamilyContext`
grammar and return ordinary `GenerationSpec`s, so the compositor,
analytic-mesh exporter, and manifest pipeline work unchanged. The spec's
``manifest_extras`` attribute carries a ``flora`` block with the resolved
parameters for downstream consumers (same convention as soft_author).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..alignment.schema import GenerationSpec
from .style_families import FamilyContext

# ----------------------------------------------------------------------
# Parameter bundle
# ----------------------------------------------------------------------

FLORA_KINDS = ("tree", "grass", "shrub", "flower")
SEASONS = ("spring", "summer", "autumn", "winter")
AGES = ("sapling", "mature", "ancient")
BRANCHING = ("whorled", "alternate")


@dataclass
class FloraParams:
    """User-facing dials for any generated plant."""

    kind: str = "tree"             # tree | grass | shrub | flower
    density: float = 0.7           # 0..1 -> leaf/petal/blade/floret counts
    size_scale: float = 1.0        # uniform length multiplier
    style: str = "oak"             # species preset name (see SPECIES)
    season: str = "summer"         # spring|summer|autumn|winter
    age: str = "mature"            # sapling|mature|ancient
    seed: int = 0
    branching: str | None = None   # whorled|alternate; None = species habit
    # Grass-patch extent (metres at size_scale 1); ignored for other kinds.
    patch: tuple[float, float] = (0.6, 0.6)

    def __post_init__(self) -> None:
        if self.kind not in FLORA_KINDS:
            raise ValueError(f"unknown flora kind {self.kind!r} {FLORA_KINDS}")
        if self.season not in SEASONS:
            raise ValueError(f"unknown season {self.season!r} {SEASONS}")
        if self.age not in AGES:
            raise ValueError(f"unknown age {self.age!r} {AGES}")
        if self.branching is not None and self.branching not in BRANCHING:
            raise ValueError(f"unknown branching {self.branching!r} {BRANCHING}")
        if self.style not in SPECIES:
            raise ValueError(f"unknown flora style {self.style!r} {sorted(SPECIES)}")
        self.density = float(np.clip(self.density, 0.0, 1.0))
        self.size_scale = float(np.clip(self.size_scale, 0.05, 20.0))
        if SPECIES[self.style].kind != self.kind:
            # Friendly auto-correction: the species preset is authoritative.
            self.kind = SPECIES[self.style].kind

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["patch"] = list(self.patch)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FloraParams":
        d = dict(data)
        if "patch" in d:
            d["patch"] = tuple(d["patch"])
        return cls(**d)


# ----------------------------------------------------------------------
# Species presets
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SpeciesPreset:
    name: str
    kind: str                 # tree | grass | shrub | flower
    height: float             # mature height in metres
    trunk_radius: float       # mature trunk radius (0 for trunkless)
    branching: str            # whorled | alternate | none
    leaf_style: str           # broadleaf | needle | frond | blade | petal
    max_leaves: int           # leaf/petal/blade/floret parts at density=1
    max_branches: int
    evergreen: bool = False
    blades_per_m2: int = 0    # grass only: blade count at density=1 per m^2


SPECIES: dict[str, SpeciesPreset] = {
    # trees — broadleaf
    "oak":      SpeciesPreset("oak", "tree", 3.2, 0.16, "alternate", "broadleaf", 48, 7),
    "maple":    SpeciesPreset("maple", "tree", 2.8, 0.14, "alternate", "broadleaf", 60, 6),
    # trees — conifer / palm / fern
    "pine":     SpeciesPreset("pine", "tree", 3.8, 0.15, "whorled", "needle", 84, 18, evergreen=True),
    "palm":     SpeciesPreset("palm", "tree", 3.4, 0.13, "none", "frond", 13, 0, evergreen=True),
    "fern":     SpeciesPreset("fern", "tree", 0.5, 0.008, "none", "frond", 16, 0, evergreen=True),
    # flowers
    "rose":     SpeciesPreset("rose", "flower", 0.50, 0.004, "none", "petal", 14, 0),
    "lavender": SpeciesPreset("lavender", "flower", 0.45, 0.0025, "none", "petal", 26, 0),
    "sunflower": SpeciesPreset("sunflower", "flower", 0.90, 0.006, "none", "petal", 17, 0),
    # grass / shrub
    "meadow":   SpeciesPreset("meadow", "grass", 0.08, 0.0, "none", "blade", 0, 0,
                              blades_per_m2=240),
    "lawn":     SpeciesPreset("lawn", "grass", 0.045, 0.0, "none", "blade", 0, 0,
                              blades_per_m2=420),
    "boxwood":  SpeciesPreset("boxwood", "shrub", 0.6, 0.012, "alternate", "broadleaf", 40, 5,
                              evergreen=True),
}


# ----------------------------------------------------------------------
# Season palettes + age profiles
# ----------------------------------------------------------------------

# Leaf / blade base colours per season (first entry is the dominant one).
SEASON_LEAF_PALETTES: dict[str, tuple[tuple[float, float, float], ...]] = {
    "spring": ((0.52, 0.72, 0.32), (0.60, 0.76, 0.36), (0.46, 0.66, 0.30)),
    "summer": ((0.24, 0.46, 0.22), (0.30, 0.52, 0.26), (0.20, 0.40, 0.24)),
    "autumn": ((0.78, 0.48, 0.16), (0.82, 0.62, 0.20), (0.66, 0.32, 0.14),
               (0.72, 0.38, 0.18)),
    "winter": ((0.20, 0.34, 0.25), (0.24, 0.38, 0.28)),   # evergreen tones
}

# Petal / blossom accent per season (species petal colour is jittered on top).
SEASON_BLOSSOM: dict[str, tuple[float, float, float] | None] = {
    "spring": (0.95, 0.82, 0.86),
    "summer": (0.92, 0.85, 0.30),
    "autumn": None,
    "winter": None,
}

# Species petal hues for the flower kinds.
SPECIES_PETAL_COLOR: dict[str, tuple[float, float, float]] = {
    "rose": (0.78, 0.16, 0.22),
    "lavender": (0.58, 0.42, 0.78),
    "sunflower": (0.92, 0.74, 0.14),
}

AGE_PROFILES: dict[str, dict[str, float]] = {
    "sapling": {"height": 0.45, "girth": 0.50, "gnarl": 0.02,
                "branch_factor": 0.6, "leaf_factor": 0.7},
    "mature":  {"height": 1.00, "girth": 1.00, "gnarl": 0.12,
                "branch_factor": 1.0, "leaf_factor": 1.0},
    "ancient": {"height": 0.85, "girth": 1.90, "gnarl": 0.45,
                "branch_factor": 1.15, "leaf_factor": 0.75},
}


def leaf_presence(preset: SpeciesPreset, season: str) -> float:
    """Fraction of the summer foliage still present in `season`."""
    if season in ("spring", "summer"):
        return 1.0
    if season == "autumn":
        return 0.85
    return 0.90 if preset.evergreen else 0.04     # winter


def leaf_color(p: FloraParams, rng: np.random.Generator) -> tuple[float, float, float]:
    """Season-aware leaf/blade colour, lightly jittered (deterministic)."""
    palette = SEASON_LEAF_PALETTES[p.season]
    base = palette[int(rng.integers(0, len(palette)))]
    j = rng.uniform(-0.04, 0.04, 3)
    return tuple(float(np.clip(c + dj, 0.03, 0.95)) for c, dj in zip(base, j))


# ----------------------------------------------------------------------
# Shared emission helpers
# ----------------------------------------------------------------------

def _leaf_template_params(p: FloraParams, preset: SpeciesPreset,
                          scale: float) -> dict[str, Any]:
    """Identical geometry for every leaf instance of this species (instancing)."""
    if preset.leaf_style == "needle":
        return {"radius": 0.014 * scale, "height": 0.09 * scale,
                "material": "organic", "instance_of": f"needle_{preset.name}"}
    if preset.leaf_style == "frond":
        return {"size": [0.055 * scale, 0.34 * scale], "thickness": 0.002 * scale,
                "bend": 0.55, "material": "organic",
                "instance_of": f"frond_{preset.name}"}
    # broadleaf (oak/maple/shrub) — maple gets a wider blade.
    w = 0.10 if preset.name == "maple" else 0.075
    return {"size": [w * scale, 0.17 * scale], "thickness": 0.0016 * scale,
            "bend": 0.45, "material": "organic",
            "instance_of": f"leaf_{preset.name}"}


def _emit_leaf_cluster(ctx: FamilyContext, centre: tuple[float, float, float],
                       cluster_r: float, n: int, template: dict,
                       rng: np.random.Generator, prefix: str, idx0: int) -> None:
    """`n` instanced leaves arranged in a rough ball around `centre`."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    kind = "cone" if template.get("instance_of", "").startswith("needle") else "panel"
    for i in range(n):
        yaw = i * golden + float(rng.uniform(-0.15, 0.15))
        tilt = float(rng.uniform(0.35, 1.25))
        r = cluster_r * float(rng.uniform(0.35, 1.0))
        cx = centre[0] + math.sin(tilt) * math.cos(yaw) * r
        cy = centre[1] + math.cos(tilt) * r * 0.8
        cz = centre[2] + math.sin(tilt) * math.sin(yaw) * r
        if kind == "cone":
            ctx.add("cone", (cx, cy, cz), dict(template), f"{prefix}_{idx0 + i}",
                    rx=tilt - math.pi / 2, ry=yaw)
        else:
            ctx.add("panel", (cx, cy, cz), dict(template), f"{prefix}_{idx0 + i}",
                    rx=tilt, ry=yaw)


def _trunk_at(path: list[list[float]], t: float) -> tuple[float, float, float]:
    """Point on the (piecewise-linear) trunk path at height fraction `t`."""
    t = float(np.clip(t, 0.0, 1.0))
    n_seg = len(path) - 1
    f = t * n_seg
    i = min(int(f), n_seg - 1)
    u = f - i
    a, b = path[i], path[i + 1]
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u,
            a[2] + (b[2] - a[2]) * u)


def _branch(ctx: FamilyContext, base: tuple[float, float, float],
            direction: tuple[float, float, float], length: float,
            radius: float, label: str, material: str) -> tuple[float, float, float]:
    """One tapered branch; returns the tip position (world)."""
    bx, by, bz = direction
    mid = (base[0] + bx * length / 2, base[1] + by * length / 2,
           base[2] + bz * length / 2)
    ctx.add("cylinder", mid,
            {"radius": radius, "height": length, "caps": True},
            label, rx=-math.atan2(bz, by), rz=math.atan2(bx, by), material=material)
    return (base[0] + bx * length, base[1] + by * length, base[2] + bz * length)


# ----------------------------------------------------------------------
# tree
# ----------------------------------------------------------------------

def build_tree(ctx: FamilyContext, p: FloraParams) -> None:
    preset = SPECIES[p.style]
    age = AGE_PROFILES[p.age]
    rng = ctx.rng
    s = p.size_scale
    H = preset.height * age["height"] * s
    R = preset.trunk_radius * age["girth"] * s
    gnarl = age["gnarl"]
    bark = "wood"
    ctx.shape = f"flora_tree_{preset.name}"
    ctx.color = leaf_color(p, rng)

    branching = p.branching or preset.branching
    if branching == "none":
        branching = preset.branching

    # -- trunk: tapered tube along a gnarled path (wander ∝ age gnarl) ----
    if R > 0.0:
        lean = float(rng.uniform(-0.05, 0.05))
        n_seg = 4
        path: list[list[float]] = []
        for i in range(n_seg + 1):
            t = i / n_seg
            wander = gnarl * H * 0.10 * math.sin(t * math.pi * 2.0 + rng.uniform(0, 1.0))
            path.append([lean * H * t + wander, H * t,
                         gnarl * H * 0.08 * math.cos(t * math.pi * 1.7)])
        seg = H / n_seg
        for i in range(n_seg):
            r0 = R * (1.0 - 0.45 * (i / n_seg))
            r1 = R * (1.0 - 0.45 * ((i + 1) / n_seg))
            ctx.add("tube", (0, 0, 0),
                    {"path": [path[i], path[i + 1]], "radius": r0, "radius2": r1,
                     "caps": i == 0, "height": seg},
                    f"trunk_{i}", material=bark)
        # Root flare + ancient buttress roots.
        ctx.add("cone", (0, R * 1.2, 0),
                {"radius": R * (1.9 if p.age == "ancient" else 1.4),
                 "height": R * 2.4}, "root_flare", material=bark)
        if p.age == "ancient":
            for k in range(4):
                a = 2 * math.pi * k / 4 + float(rng.uniform(-0.3, 0.3))
                ctx.add("tube", (0, 0, 0),
                        {"path": [[math.cos(a) * R * 0.8, R * 1.4, math.sin(a) * R * 0.8],
                                  [math.cos(a) * R * 3.0, 0.0, math.sin(a) * R * 3.0]],
                         "radius": R * 0.35, "radius2": R * 0.12, "caps": True,
                         "height": R * 3.0},
                        f"root_{k}", material=bark)

    # -- foliage budget ----------------------------------------------------
    presence = leaf_presence(preset, p.season)
    n_leaves = int(round(preset.max_leaves * p.density * presence
                         * age["leaf_factor"]))
    template = _leaf_template_params(p, preset, s)

    if preset.branching == "none":
        # Crown of fronds straight from the trunk top (palm / tree-fern).
        top = (path[-1][0] if R > 0 else 0.0, H, path[-1][2] if R > 0 else 0.0)
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(n_leaves):
            yaw = i * golden
            droop = float(rng.uniform(0.7, 1.35))     # arc outward-down
            fl = float(template["size"][1])
            cx = top[0] + math.sin(droop) * math.cos(yaw) * fl * 0.45
            cy = top[1] + math.cos(droop) * fl * 0.45
            cz = top[2] + math.sin(droop) * math.sin(yaw) * fl * 0.45
            ctx.add("panel", (cx, cy, cz), dict(template), f"frond_{i}",
                    rx=droop, ry=yaw)
    else:
        # -- branch-level control ------------------------------------------
        n_branches = max(2, int(round(preset.max_branches * age["branch_factor"]
                                      * (0.6 + 0.4 * p.density))))
        tips: list[tuple[float, float, float]] = []
        if branching == "whorled":
            # Conifer rings: K branches per whorl at stacked heights.
            n_whorls = max(3, n_branches // 3)
            per = max(2, n_branches // n_whorls)
            for w_i in range(n_whorls):
                t = 0.35 + 0.60 * (w_i / max(n_whorls - 1, 1))
                y = H * t
                ring_off = float(rng.uniform(0, math.pi))
                blen = H * 0.30 * (1.15 - t)                    # shorter near top
                base = _trunk_at(path, t) if R > 0 else (0.0, y, 0.0)
                for j in range(per):
                    yaw = 2 * math.pi * j / per + ring_off
                    droop = float(rng.uniform(0.15, 0.35))      # slight sag
                    d = (math.cos(yaw) * math.cos(droop), math.sin(droop),
                         math.sin(yaw) * math.cos(droop))
                    tips.append(_branch(ctx, base,
                                        d, blen, R * 0.30,
                                        f"branch_{w_i}_{j}", bark))
        else:
            # Alternate spiral: golden-angle yaw, staggered heights.
            golden = math.pi * (3.0 - math.sqrt(5.0))
            for i in range(n_branches):
                t = 0.45 + 0.50 * (i / max(n_branches - 1, 1))
                yaw = i * golden + float(rng.uniform(-0.2, 0.2))
                tilt = float(rng.uniform(0.5, 0.95))
                blen = H * 0.28 * (1.2 - t)
                d = (math.cos(yaw) * math.sin(tilt), math.cos(tilt),
                     math.sin(yaw) * math.sin(tilt))
                base = _trunk_at(path, t) if R > 0 else (0.0, H * t, 0.0)
                tips.append(_branch(ctx, base,
                                    d, blen, R * 0.32, f"branch_{i}", bark))

        # Distribute the leaf budget over branch tips + a crown tuft.
        per_tip = n_leaves // max(len(tips), 1)
        rem = n_leaves - per_tip * len(tips)
        cluster_r = H * 0.13
        used = 0
        for i, tip in enumerate(tips):
            n_i = per_tip + (1 if i < rem else 0)
            if n_i <= 0:
                continue
            _emit_leaf_cluster(ctx, tip, cluster_r, n_i, template, rng,
                               "leaf", used)
            used += n_i

    # Spring blossom accents tucked into the crown.
    blossom = SEASON_BLOSSOM[p.season]
    if blossom is not None and preset.kind == "tree" and n_leaves > 0:
        n_bloom = max(1, int(round(n_leaves * 0.15)))
        for i in range(n_bloom):
            a = float(rng.uniform(0, 2 * math.pi))
            rr = H * float(rng.uniform(0.05, 0.16))
            ctx.add("sphere", (math.cos(a) * rr, H * float(rng.uniform(0.75, 1.05)),
                               math.sin(a) * rr),
                    {"radius": 0.012 * s, "material": "ceramic",
                     "instance_of": "blossom"}, f"blossom_{i}")

    ctx.add_feature("asperity", "all", strength=0.0012 * s, frequency=45.0)


# ----------------------------------------------------------------------
# grass patch
# ----------------------------------------------------------------------

# Blade templates (height buckets) — instancing keeps 3 shared meshes.
_BLADE_TEMPLATES = (
    {"radius": 0.0016, "height": 0.045, "instance_of": "blade_short"},
    {"radius": 0.0020, "height": 0.070, "instance_of": "blade_mid"},
    {"radius": 0.0024, "height": 0.100, "instance_of": "blade_tall"},
)


def build_grass(ctx: FamilyContext, p: FloraParams) -> None:
    preset = SPECIES[p.style]
    rng = ctx.rng
    s = p.size_scale
    ctx.shape = f"flora_grass_{preset.name}"
    ctx.color = leaf_color(p, rng)

    w = p.patch[0] * s
    d = p.patch[1] * s
    area_m2 = w * d

    # Soil bed — never flat (relief displaces it at composite time).
    t = 0.02 * s
    ctx.add("box", (0, t / 2, 0), {"size": [w, t, d]}, "soil", material="stone")
    ctx.add_feature("relief", "soil", amplitude=0.006 * s, frequency=9.0,
                    octaves=3, pebbles=int(round(3 * p.density)))

    # STRICTLY density-linear blade count: blades/m^2 preset * area * density.
    n_blades = int(round(preset.blades_per_m2 * area_m2 * p.density))
    for i in range(n_blades):
        tmpl = _BLADE_TEMPLATES[int(rng.integers(0, len(_BLADE_TEMPLATES)))]
        gx = float(rng.uniform(-w / 2, w / 2))
        gz = float(rng.uniform(-d / 2, d / 2))
        params = {"radius": tmpl["radius"] * s, "height": tmpl["height"] * s,
                  "material": "organic", "instance_of": tmpl["instance_of"]}
        ctx.add("cone", (gx, t + params["height"] / 2 - 0.004, gz), params,
                f"blade_{i}", rx=float(rng.uniform(-0.12, 0.12)),
                rz=float(rng.uniform(-0.18, 0.18)),
                ry=float(rng.uniform(0, math.pi)))


# ----------------------------------------------------------------------
# shrub
# ----------------------------------------------------------------------

def build_shrub(ctx: FamilyContext, p: FloraParams) -> None:
    preset = SPECIES[p.style]
    rng = ctx.rng
    s = p.size_scale
    H = preset.height * s
    ctx.shape = f"flora_shrub_{preset.name}"
    ctx.color = leaf_color(p, rng)

    # Multi-stem base fanning from the root crown.
    n_stems = preset.max_branches
    tips = []
    for i in range(n_stems):
        yaw = 2 * math.pi * i / n_stems + float(rng.uniform(-0.3, 0.3))
        tilt = float(rng.uniform(0.25, 0.65))
        slen = H * float(rng.uniform(0.55, 0.85))
        d = (math.cos(yaw) * math.sin(tilt), math.cos(tilt),
             math.sin(yaw) * math.sin(tilt))
        tips.append(_branch(ctx, (0.0, 0.02 * s, 0.0), d, slen,
                            preset.trunk_radius * s, f"stem_{i}", "wood"))

    presence = leaf_presence(preset, p.season)
    n_leaves = int(round(preset.max_leaves * p.density * presence))
    template = _leaf_template_params(p, preset, s * 0.7)
    per_tip = n_leaves // max(len(tips), 1)
    rem = n_leaves - per_tip * len(tips)
    used = 0
    for i, tip in enumerate(tips):
        n_i = per_tip + (1 if i < rem else 0)
        if n_i <= 0:
            continue
        _emit_leaf_cluster(ctx, tip, H * 0.30, n_i, template, rng, "leaf", used)
        used += n_i
    ctx.add_feature("asperity", "all", strength=0.0006 * s, frequency=55.0)


# ----------------------------------------------------------------------
# flowers (rose / lavender / sunflower)
# ----------------------------------------------------------------------

def build_flower_parametric(ctx: FamilyContext, p: FloraParams) -> None:
    preset = SPECIES[p.style]
    rng = ctx.rng
    s = p.size_scale
    H = preset.height * s
    stem_mat = "organic"
    petal = SPECIES_PETAL_COLOR[p.style]
    ctx.shape = f"flora_flower_{preset.name}"
    ctx.color = petal

    # Stem (gentle seeded bend) + a ground leaf pair.
    bend = float(rng.uniform(-0.05, 0.05)) * s
    ctx.add("tube", (0, 0, 0),
            {"path": [[0.0, 0.0, 0.0], [bend * 0.4, H * 0.5, 0.0], [bend, H, 0.0]],
             "radius": preset.trunk_radius * s, "caps": True, "height": H},
            "stem", material=stem_mat)
    top = (bend, H, 0.0)
    n_petals = max(3, int(round(preset.max_leaves * p.density)))

    if p.style == "lavender":
        # Spike inflorescence: tiny instanced florets hugging the upper stem.
        spike_h = H * 0.35
        tmpl = {"radius": 0.006 * s, "material": "organic",
                "instance_of": "floret_lavender"}
        for i in range(n_petals):
            t = i / max(n_petals - 1, 1)
            yaw = i * 2.4
            rr = 0.010 * s * (1.0 - 0.6 * t)
            ctx.add("sphere", (top[0] + math.cos(yaw) * rr,
                               top[1] - spike_h + spike_h * t,
                               top[2] + math.sin(yaw) * rr),
                    dict(tmpl), f"floret_{i}")
        return

    if p.style == "sunflower":
        # Seed disc + single instanced petal ray ring.
        disc_r = 0.055 * s
        ctx.add("ellipsoid", top, {"radii": [disc_r, 0.015 * s, disc_r]},
                "disc", material="wood")
        tmpl = {"size": [0.022 * s, 0.075 * s], "thickness": 0.0014 * s,
                "bend": 0.35, "material": "organic",
                "instance_of": "petal_sunflower"}
        for i in range(n_petals):
            yaw = 2 * math.pi * i / n_petals
            tilt = float(rng.uniform(1.15, 1.45))
            cx = top[0] + math.sin(yaw) * disc_r * 1.05
            cz = top[2] + math.cos(yaw) * disc_r * 1.05
            cy = top[1] + 0.006 * s
            ctx.add("panel", (cx, cy, cz), dict(tmpl), f"petal_{i}",
                    rx=tilt, ry=yaw)
        return

    # rose — two instanced petal rings around a receptacle.
    ctx.add("sphere", top, {"radius": 0.012 * s}, "receptacle", material=stem_mat)
    tmpl = {"size": [0.030 * s, 0.055 * s], "thickness": 0.0014 * s,
            "bend": 0.5, "material": "organic", "instance_of": "petal_rose"}
    n_outer = max(3, int(round(n_petals * 0.6)))
    n_inner = max(2, n_petals - n_outer)
    idx = 0
    for ring, (n_ring, tilt, rr) in enumerate(
            ((n_outer, float(rng.uniform(0.9, 1.2)), 0.030 * s),
             (n_inner, float(rng.uniform(0.4, 0.7)), 0.016 * s))):
        for i in range(n_ring):
            yaw = 2 * math.pi * i / n_ring + ring * 0.45
            ctx.add("panel", (top[0] + math.sin(yaw) * rr,
                              top[1] + math.cos(tilt) * 0.02 * s,
                              top[2] + math.cos(yaw) * rr),
                    dict(tmpl), f"petal_{idx}", rx=tilt, ry=yaw)
            idx += 1


# ----------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------

FLORA_BUILDERS = {
    "tree": build_tree,
    "grass": build_grass,
    "shrub": build_shrub,
    "flower": build_flower_parametric,
}


def flora_spec(p: FloraParams, n_points: int = 50_000,
               bbox: tuple[float, float, float] | None = None) -> GenerationSpec:
    """Build a deterministic `GenerationSpec` from a `FloraParams` bundle.

    The spec carries a ``manifest_extras["flora"]`` block with the resolved
    parameters (same side-channel convention as ``soft_author``); when
    `bbox` is given the assembly is uniformly fitted to it via the style
    engine's bbox fitter.
    """
    rng = np.random.default_rng(p.seed or None)
    # Generous part budget: counts are governed by `density`, never starved
    # by the complexity heuristic.
    ctx = FamilyContext(rng=rng, target_parts=4096)
    FLORA_BUILDERS[p.kind](ctx, p)

    if bbox is not None:
        from .style_engine import _fit_to_bbox
        _fit_to_bbox(ctx.primitives, bbox)

    spec = GenerationSpec(
        shape=ctx.shape or f"flora_{p.kind}",
        n_points=int(n_points),
        bbox_size=tuple(float(b) for b in (bbox or (1.0, 1.0, 1.0))),
        primitives=ctx.primitives,
        features=ctx.features,
        color=ctx.color,
        seed=int(p.seed or 0),
    )
    preset = SPECIES[p.style]
    spec.manifest_extras = {
        "flora": {
            **p.to_dict(),
            "resolved": {
                "branching": p.branching or preset.branching,
                "leaf_presence": leaf_presence(preset, p.season),
                "leaf_style": preset.leaf_style,
                "evergreen": preset.evergreen,
                "part_count": len(ctx.primitives),
            },
        }
    }
    return spec


def collect_instance_groups(spec: GenerationSpec) -> dict[str, list[int]]:
    """Template name -> indices of primitives sharing that instanced mesh.

    Repeated flora parts (leaves / blades / petals / florets) carry
    ``params["instance_of"]``; every primitive in a group has identical
    geometry params, so an exporter can emit one mesh + N transforms.
    """
    groups: dict[str, list[int]] = {}
    for i, prim in enumerate(spec.primitives):
        key = (prim.params or {}).get("instance_of")
        if isinstance(key, str) and key:
            groups.setdefault(key, []).append(i)
    return groups
