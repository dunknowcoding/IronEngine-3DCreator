"""Parametric style families — seeded grammars that assemble coherent objects.

Each family is a function `build_<family>(ctx)` that appends primitives and
features to a `FamilyContext`. The grammars encode proportion rules (seat
height vs. leg length, neck vs. body radius, …), bilateral/radial symmetry,
curved parts (torus/helix), gaps and holes, and per-part materials — all
driven by the context's seeded RNG so the same seed always yields the same
object.

The grammars are written against the *current* primitive schema but look up
kind availability at call time (`ctx.pick_kind`), preferring newer kinds
(superellipsoid / tube / arch / panel) when a future schema provides them and
falling back to the classic 10 kinds otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..alignment.schema import (
    FEATURE_KINDS,
    PRIMITIVE_KINDS,
    Feature,
    Primitive,
)
from .primitives import SAMPLERS

# Materials that exist in generation.textures.MATERIALS — keep in sync.
MATERIAL_PALETTES: dict[str, tuple[str, ...]] = {
    "furniture": ("wood", "fabric", "leather", "metal"),
    "creature": ("organic", "leather"),
    "mechanical": ("metal", "stone"),
    "architecture": ("stone", "ceramic"),
    "plant": ("wood", "organic"),
    "vessel": ("ceramic", "stone", "metal"),
    "abstract": ("metal", "stone", "ceramic", "organic"),
}

# Base-color palettes (RGB in [0, 1]); one is picked per object and jittered.
COLOR_PALETTES: dict[str, tuple[tuple[float, float, float], ...]] = {
    "furniture": ((0.50, 0.34, 0.20), (0.62, 0.50, 0.36), (0.30, 0.22, 0.16),
                  (0.55, 0.55, 0.58), (0.36, 0.42, 0.34)),
    "creature": ((0.62, 0.50, 0.34), (0.45, 0.55, 0.40), (0.70, 0.62, 0.52),
                 (0.50, 0.42, 0.55), (0.72, 0.45, 0.30)),
    "mechanical": ((0.55, 0.56, 0.60), (0.40, 0.42, 0.46), (0.62, 0.52, 0.30),
                   (0.35, 0.45, 0.55), (0.58, 0.30, 0.26)),
    "architecture": ((0.85, 0.82, 0.74), (0.72, 0.70, 0.66), (0.60, 0.55, 0.48),
                     (0.80, 0.76, 0.68)),
    "plant": ((0.30, 0.48, 0.24), (0.24, 0.42, 0.30), (0.45, 0.55, 0.25),
              (0.35, 0.50, 0.35)),
    "vessel": ((0.92, 0.91, 0.85), (0.35, 0.50, 0.62), (0.55, 0.35, 0.28),
               (0.30, 0.55, 0.50), (0.75, 0.70, 0.60)),
    "abstract": ((0.40, 0.70, 1.00), (0.85, 0.55, 0.25), (0.60, 0.60, 0.62),
                 (0.75, 0.35, 0.45), (0.45, 0.75, 0.60)),
}


# ----------------------------------------------------------------------
# Context + helpers
# ----------------------------------------------------------------------

@dataclass
class FamilyContext:
    """Everything a family grammar needs to build one object."""

    rng: np.random.Generator
    target_parts: int            # soft budget from the complexity setting
    shape: str = "abstract"
    primitives: list[Primitive] = field(default_factory=list)
    features: list[Feature] = field(default_factory=list)
    color: tuple[float, float, float] = (0.6, 0.6, 0.6)
    # Manifest extras side-channel (iemodel/3): adapters building through
    # other modules (water/flora/terrain/human/building/vehicle) stash their
    # resolved metadata blocks here; StyleEngine.generate attaches them to
    # the produced spec as `manifest_extras`.
    extras: dict = field(default_factory=dict)

    # -- kind availability -------------------------------------------------
    @property
    def kinds(self) -> frozenset[str]:
        """Primitive kinds that are actually generatable *right now*.

        A kind qualifies only when the whole chain supports it: the schema
        accepts it, a surface sampler exists, and the validator has defaults
        for it (all three looked up at call time). While the new kinds
        (superellipsoid / tube / sweep / arch / panel) are landing piecemeal
        this keeps the grammar on the classic 10; the moment support is
        complete the new kinds are picked up automatically.
        """
        try:
            from ..alignment import validator as _validator
            defaults = getattr(_validator, "_PARAM_DEFAULTS", {}) or {}
        except Exception:  # defensive: never let availability checks crash
            defaults = {}
        return frozenset(
            k.lower() for k in PRIMITIVE_KINDS
            if k.lower() in SAMPLERS and k.lower() in defaults
        )

    def pick_kind(self, *preference: str) -> str:
        """First preferred kind available in the schema; else 'box'."""
        avail = self.kinds
        for k in preference:
            if k.lower() in avail:
                return k.lower()
        return "box"

    def maybe(self, p: float) -> bool:
        return bool(self.rng.random() < p)

    def uniform(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive [lo, hi]."""
        return int(self.rng.integers(lo, hi + 1))

    # -- emission ----------------------------------------------------------
    def add(self, kind: str, translate=(0.0, 0.0, 0.0), params: dict | None = None,
            label: str | None = None, scale=(1.0, 1.0, 1.0),
            ry: float = 0.0, rx: float = 0.0, rz: float = 0.0,
            material: str | None = None) -> Primitive:
        params = dict(params or {})
        if material:
            params["material"] = material
        prim = Primitive(
            kind=kind,
            transform=_T(translate, scale, ry=ry, rx=rx, rz=rz),
            params=params,
            label=label,
        )
        self.primitives.append(prim)
        return prim

    def add_feature(self, kind: str, region="all", **params) -> None:
        """Append a surface feature, guarding against schema drift."""
        if kind.lower() not in {k.lower() for k in FEATURE_KINDS}:
            return
        self.features.append(Feature(kind=kind.lower(), region=region, params=params))

    def room(self) -> int:
        """How many more parts fit the complexity budget."""
        return max(0, self.target_parts - len(self.primitives))

    def pick_palette(self, family: str) -> None:
        palette = COLOR_PALETTES.get(family, COLOR_PALETTES["abstract"])
        base = palette[int(self.rng.integers(0, len(palette)))]
        j = self.rng.uniform(-0.06, 0.06, 3)
        self.color = tuple(float(np.clip(c + dj, 0.05, 0.95)) for c, dj in zip(base, j))

    def material(self, family: str, *prefer: str) -> str:
        avail = MATERIAL_PALETTES.get(family, ("stone",))
        prefs = [m for m in prefer if m in avail]
        pool = prefs or list(avail)
        return pool[int(self.rng.integers(0, len(pool)))]


def _T(translate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
       ry: float = 0.0, rx: float = 0.0, rz: float = 0.0) -> list[list[float]]:
    """4x4 transform = T · Ry · Rx · Rz · S (angles in radians)."""
    cy, sy = math.cos(ry), math.sin(ry)
    cx, sx = math.cos(rx), math.sin(rx)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    m = rot_y @ rot_x @ rot_z @ np.diag(scale)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = m
    out[:3, 3] = translate
    return out.astype(np.float32).tolist()


def _horizontal_slab(ctx: FamilyContext, kind: str, translate,
                     w: float, t: float, d: float,
                     label: str, material: str) -> Primitive:
    """Horizontal slab (seat / desktop / tabletop).

    Panel-native when the schema supports panels: `size` is the 2-element
    in-plane [w, d], `thickness` is a separate param, and the panel is laid
    flat with rx=π/2 (panel local in-plane Y → world Z, thickness → world Y).
    Falls back to box semantics [w, t, d] when only boxes are available.
    """
    if kind == "panel":
        return ctx.add("panel", translate,
                       {"size": [w, d], "thickness": t},
                       label, rx=math.pi / 2, material=material)
    return ctx.add(kind, translate, {"size": [w, t, d]}, label, material=material)


def _upright_slab(ctx: FamilyContext, kind: str, translate,
                  w: float, h: float, t: float,
                  label: str, material: str) -> Primitive:
    """Vertical slab (chair-back slat).

    A panel's native orientation is already upright (in-plane X → world X,
    in-plane Y → world Y, thickness → world Z), so no rotation is needed.
    Falls back to box semantics [w, h, t] otherwise.
    """
    if kind == "panel":
        return ctx.add("panel", translate,
                       {"size": [w, h], "thickness": t},
                       label, material=material)
    return ctx.add(kind, translate, {"size": [w, h, t]}, label, material=material)


# ----------------------------------------------------------------------
# furniture — chairs / tables / stools / benches
# ----------------------------------------------------------------------

def build_furniture(ctx: FamilyContext) -> None:
    ctx.shape = "furniture"
    ctx.pick_palette("furniture")
    archetype = ["chair", "table", "stool", "bench"][ctx.randint(0, 3)]
    wood = ctx.material("furniture", "wood")
    soft = ctx.material("furniture", "fabric", "leather")

    top_kind = ctx.pick_kind("panel", "box")
    w = ctx.uniform(0.7, 1.1) if archetype != "bench" else ctx.uniform(1.2, 1.7)
    d = ctx.uniform(0.6, 1.0)
    # Real-world surface heights (truth table): chair/stool seat ~0.45 m,
    # bench 0.42–0.48 m, table top 0.72–0.76 m. leg_h + top_h = surface.
    if archetype == "table":
        top_h = ctx.uniform(0.04, 0.06)
        leg_h = ctx.uniform(0.68, 0.71)
    elif archetype == "bench":
        top_h = ctx.uniform(0.04, 0.06)
        leg_h = ctx.uniform(0.38, 0.42)
    else:  # chair / stool
        top_h = ctx.uniform(0.04, 0.07)
        leg_h = ctx.uniform(0.40, 0.43)
    top_y = leg_h + top_h / 2

    # Legs: 4 mirrored (or 3 for a quirky stool) — symmetry across X and Z.
    n_leg_pairs = 2
    leg_r = ctx.uniform(0.035, 0.06)
    leg_kind = ctx.pick_kind("tube", "cylinder")
    positions = []
    if archetype == "stool" and ctx.maybe(0.4):
        positions = [(0.0, 0.0)]  # pedestal stool
    else:
        for sx_ in (-1, 1):
            for sz_ in (-1, 1):
                positions.append((sx_ * (w / 2 - leg_r * 1.5), sz_ * (d / 2 - leg_r * 1.5)))
    for i, (lx, lz) in enumerate(positions):
        if lx == 0.0 and lz == 0.0:
            ctx.add("cylinder", (0, leg_h / 2, 0),
                    {"radius": leg_r * 2.2, "height": leg_h, "caps": True},
                    f"pedestal_{i}", material=wood)
            ctx.add("cylinder", (0, 0.02, 0),
                    {"radius": leg_r * 5.0, "height": 0.04, "caps": True},
                    "pedestal_foot", material=wood)
        else:
            ctx.add(leg_kind, (lx, leg_h / 2, lz),
                    {"radius": leg_r, "height": leg_h, "caps": True},
                    f"leg_{i}", material=wood)

    # Top slab (seat / tabletop).
    cushion = archetype in ("chair", "stool", "bench") and ctx.maybe(0.6)
    _horizontal_slab(ctx, top_kind, (0, top_y, 0), w, top_h, d,
                     "seat" if cushion else "top",
                     soft if cushion else wood)

    # Cushion on top of the seat.
    if cushion and ctx.room() > 0:
        ck = ctx.pick_kind("superellipsoid", "ellipsoid", "box")
        ctx.add(ck, (0, top_y + top_h / 2 + 0.035, 0),
                {"radii": [w * 0.42, 0.045, d * 0.42], "size": [w * 0.84, 0.07, d * 0.84]},
                "cushion", material=soft)

    # Cross braces between the legs (adds structure; skipped for pedestal).
    if len(positions) > 2:
        while ctx.room() > (6 if archetype == "chair" else 2) and ctx.maybe(0.75):
            side = ctx.randint(0, 3)
            y = ctx.uniform(0.12, leg_h * 0.6)
            if side < 2:  # braces along Z at ±X
                x = (w / 2 - leg_r * 1.5) * (1 if side == 0 else -1)
                ctx.add("cylinder", (x, y, 0),
                        {"radius": leg_r * 0.6, "height": d - 3 * leg_r, "caps": True},
                        f"brace_z_{side}", rx=math.pi / 2, material=wood)
            else:          # braces along X at ±Z
                z = (d / 2 - leg_r * 1.5) * (1 if side == 2 else -1)
                ctx.add("cylinder", (0, y, z),
                        {"radius": leg_r * 0.6, "height": w - 3 * leg_r, "caps": True},
                        f"brace_x_{side}", rz=math.pi / 2, material=wood)
            break  # at most one brace pair per roll keeps silhouettes clean

    # Chair back: posts + slats with gaps.
    if archetype == "chair":
        back_h = ctx.uniform(0.35, 0.55)
        bz = -(d / 2 - leg_r)
        for sx_ in (-1, 1):
            ctx.add(leg_kind, (sx_ * (w / 2 - leg_r * 1.5), leg_h + back_h / 2, bz),
                    {"radius": leg_r * 0.8, "height": back_h, "caps": True},
                    f"back_post_{sx_}", material=wood)
        n_slats = min(ctx.randint(1, 3), max(1, ctx.room() - 1))
        for i in range(n_slats):
            sy_ = leg_h + back_h * (0.35 + 0.3 * i)
            _upright_slab(ctx, top_kind, (0, sy_, bz),
                          w - 2 * leg_r, ctx.uniform(0.06, 0.12), top_h * 0.7,
                          f"back_slat_{i}", wood)

    # Wear and tear.
    ctx.add_feature("scratch", "all", count=ctx.randint(4, 14), depth=ctx.uniform(0.003, 0.008))
    if ctx.maybe(0.4):
        ctx.add_feature("ridges", "leg_0", count=ctx.randint(8, 18), depth=0.004)


# ----------------------------------------------------------------------
# creature — bilaterally symmetric animals / critters
# ----------------------------------------------------------------------

def build_creature(ctx: FamilyContext) -> None:
    ctx.shape = "creature"
    ctx.pick_palette("creature")
    skin = ctx.material("creature", "organic")

    body_kind = ctx.pick_kind("superellipsoid", "ellipsoid")
    bl = ctx.uniform(0.45, 0.65)   # body half-length (z)
    bh = ctx.uniform(0.26, 0.38)   # body half-height
    bw = ctx.uniform(0.28, 0.42)   # body half-width
    body_y = bh + ctx.uniform(0.18, 0.30)
    ctx.add(body_kind, (0, body_y, 0),
            {"radii": [bw, bh, bl]}, "body", material=skin)

    # Head + snout + eyes (mirrored).
    head_r = ctx.uniform(0.16, 0.24)
    head_z = bl * 0.95 + head_r * 0.4
    head_y = body_y + bh * ctx.uniform(0.3, 0.7)
    ctx.add("sphere", (0, head_y, head_z), {"radius": head_r}, "head", material=skin)
    if ctx.room() > 4 and ctx.maybe(0.7):
        ctx.add("cone", (0, head_y - head_r * 0.15, head_z + head_r * 0.9),
                {"radius": head_r * 0.45, "height": head_r * 0.9},
                "snout", rx=math.pi / 2, material=skin)
    eye_r = head_r * ctx.uniform(0.16, 0.24)
    for sx_ in (-1, 1):
        ctx.add("sphere", (sx_ * head_r * 0.45, head_y + head_r * 0.30, head_z + head_r * 0.75),
                {"radius": eye_r}, f"eye_{sx_}", material="ceramic")

    # Legs: 2 or 4 mirrored pairs of capsules, each WITH a paw (a flattened
    # foot ellipsoid, not a raw column stub). Leg tops embed ~30 % of the
    # body depth so limbs visibly grow out of the torso instead of floating
    # beneath it.
    n_pairs = 2 if ctx.maybe(0.65) else 1
    body_bottom = body_y - bh
    leg_len = body_bottom + bh * 0.3            # floor → 30 % into the belly
    leg_r = ctx.uniform(0.05, 0.08)
    paw_h = leg_r * 0.55
    for pair in range(n_pairs):
        z = (bl * 0.55) * (1 if pair == 0 else -1)
        for sx_ in (-1, 1):
            lx = sx_ * bw * 0.7
            ctx.add("capsule", (lx, leg_len / 2, z),
                    {"radius": leg_r, "height": leg_len},
                    f"leg_{pair}_{sx_}", material=skin)
            ctx.add("ellipsoid", (lx, paw_h, z + bl * 0.06),
                    {"radii": [leg_r * 1.5, paw_h, leg_r * 2.1]},
                    f"paw_{pair}_{sx_}", material=skin)

    # Tail: curved helix or simple cone, pointing backwards.
    if ctx.room() > 1 and ctx.maybe(0.8):
        if ctx.maybe(0.5):
            ctx.add("helix", (0, body_y + 0.05, -bl - 0.12),
                    {"radius": 0.10, "pitch": 0.10, "turns": ctx.uniform(1.0, 2.2),
                     "thickness": ctx.uniform(0.03, 0.05)},
                    "tail", rx=math.pi / 2, material=skin)
        else:
            ctx.add("cone", (0, body_y, -bl - 0.18),
                    {"radius": ctx.uniform(0.06, 0.10), "height": 0.36},
                    "tail", rx=-math.pi / 2, material=skin)

    # Ears / horns: mirrored cones.
    if ctx.room() > 2 and ctx.maybe(0.7):
        ear_h = head_r * ctx.uniform(0.7, 1.2)
        for sx_ in (-1, 1):
            ctx.add("cone", (sx_ * head_r * 0.5, head_y + head_r * 0.95, head_z - head_r * 0.1),
                    {"radius": head_r * 0.22, "height": ear_h},
                    f"ear_{sx_}", rz=-sx_ * 0.35, material=skin)

    # Belly patch / back ridge decoration.
    if ctx.room() > 0 and ctx.maybe(0.5):
        ctx.add("ellipsoid", (0, body_y + bh * 0.9, 0),
                {"radii": [bw * 0.5, bh * 0.25, bl * 0.6]},
                "back_patch", material=ctx.material("creature", "leather"))

    ctx.add_feature("fur", "all",
                    density=ctx.uniform(0.3, 0.7), length=ctx.uniform(0.008, 0.02))
    ctx.add_feature("bump_field", "body", count=ctx.randint(10, 40), radius=ctx.uniform(0.02, 0.04))


# ----------------------------------------------------------------------
# mechanical — machines, gear assemblies, engine blocks
# ----------------------------------------------------------------------

def build_mechanical(ctx: FamilyContext) -> None:
    ctx.shape = "mechanical"
    ctx.pick_palette("mechanical")
    metal = "metal"

    # Chassis.
    cw = ctx.uniform(0.6, 0.9)
    ch = ctx.uniform(0.25, 0.4)
    cd = ctx.uniform(0.45, 0.7)
    ctx.add("box", (0, ch / 2 + 0.06, 0),
            {"size": [cw, ch, cd]}, "chassis", material=metal)
    ctx.add("box", (0, 0.03, 0),
            {"size": [cw * 1.15, 0.06, cd * 1.15]}, "base_plate", material=metal)

    # Flywheel / gear: torus rim + hub + spokes.
    gear_r = ctx.uniform(0.18, 0.28)
    gear_y = ch + 0.06 + gear_r * 0.4
    ctx.add("torus", (0, gear_y, cd * 0.28),
            {"major_radius": gear_r, "minor_radius": gear_r * 0.18},
            "flywheel", rx=math.pi / 2, material=metal)
    ctx.add("cylinder", (0, gear_y, cd * 0.28),
            {"radius": gear_r * 0.22, "height": 0.10, "caps": True},
            "hub", rx=math.pi / 2, material=metal)
    n_spokes = min(ctx.randint(3, 6), max(0, ctx.room() - 6))
    for i in range(n_spokes):
        a = 2 * math.pi * i / n_spokes
        ctx.add("cylinder",
                (math.cos(a) * gear_r * 0.55, gear_y + math.sin(a) * gear_r * 0.55, cd * 0.28),
                {"radius": gear_r * 0.07, "height": gear_r * 1.05, "caps": True},
                f"spoke_{i}", rz=a + math.pi / 2, material=metal)

    # Pistons: mirrored vertical cylinders with rods.
    n_pistons = min(ctx.randint(2, 4), max(0, ctx.room() - 4))
    for i in range(n_pistons):
        x = -cw * 0.3 + (cw * 0.6) * (i / max(1, n_pistons - 1) if n_pistons > 1 else 0.5)
        ph = ctx.uniform(0.22, 0.34)
        ctx.add("cylinder", (x, ch + 0.06 + ph / 2, -cd * 0.22),
                {"radius": ctx.uniform(0.06, 0.09), "height": ph, "caps": True},
                f"piston_{i}", material=metal)
        if ctx.room() > 2:
            ctx.add("cylinder", (x, ch + 0.06 + ph + 0.08, -cd * 0.22),
                    {"radius": 0.025, "height": 0.16, "caps": True},
                    f"rod_{i}", material=metal)

    # Springs (helixes) on the sides.
    for sx_ in (-1, 1):
        if ctx.room() <= 1:
            break
        if ctx.maybe(0.8):
            ctx.add("helix", (sx_ * (cw / 2 + 0.10), ch * 0.7, 0),
                    {"radius": 0.07, "pitch": 0.055,
                     "turns": ctx.uniform(3.0, 5.0), "thickness": 0.02},
                    f"spring_{sx_}", rz=sx_ * 0.25, material=metal)

    # Bolt ring around the chassis top.
    n_bolts = min(ctx.randint(4, 8), ctx.room())
    for i in range(n_bolts):
        a = 2 * math.pi * i / n_bolts
        ctx.add("cylinder",
                (math.cos(a) * cw * 0.38, ch + 0.075, math.sin(a) * cd * 0.38),
                {"radius": 0.02, "height": 0.03, "caps": True},
                f"bolt_{i}", material=metal)

    ctx.add_feature("ridges", "chassis", count=ctx.randint(10, 20), depth=0.004)
    ctx.add_feature("scratch", "all", count=ctx.randint(6, 16), depth=ctx.uniform(0.003, 0.007))
    if ctx.maybe(0.35):
        ctx.add_feature("erosion", "all", strength=ctx.uniform(0.004, 0.01))


# ----------------------------------------------------------------------
# architecture — colonnades, archways, facades, towers
# ----------------------------------------------------------------------

def build_architecture(ctx: FamilyContext) -> None:
    ctx.shape = "architecture"
    ctx.pick_palette("architecture")
    stone = ctx.material("architecture", "stone")

    span = ctx.uniform(1.0, 1.6)
    col_h = ctx.uniform(0.9, 1.4)
    col_r = ctx.uniform(0.09, 0.15)

    # Stepped plinth (2-3 stacked slabs with decreasing footprint).
    n_steps = ctx.randint(1, 3)
    for i in range(n_steps):
        f = 1.0 - 0.12 * i
        ctx.add("box", (0, 0.04 + 0.08 * i, 0),
                {"size": [(span + 0.5) * f, 0.08, 0.55 * f]},
                f"step_{i}", material=stone)
    base_y = 0.08 * n_steps

    # Columns: an even number, mirrored about the center, with gaps between.
    n_pairs = min(ctx.randint(1, 3), max(1, (ctx.room() - 4) // 4))
    col_kind = ctx.pick_kind("tube", "cylinder", "prism")
    for pair in range(n_pairs):
        x = span * (0.5 - pair * 0.35)
        for sx_ in (-1, 1):
            ctx.add(col_kind, (sx_ * x, base_y + col_h / 2, 0),
                    {"radius": col_r, "height": col_h, "sides": 8, "caps": True},
                    f"column_{pair}_{sx_}", material=stone)
            if ctx.room() > 2 and ctx.maybe(0.7):  # capital
                ctx.add("box", (sx_ * x, base_y + col_h + 0.03, 0),
                        {"size": [col_r * 3.2, 0.06, col_r * 3.2]},
                        f"capital_{pair}_{sx_}", material=stone)

    # Entablature beam spanning the columns.
    ctx.add("box", (0, base_y + col_h + 0.10, 0),
            {"size": [span + 0.4, 0.12, 0.30]}, "beam", material=stone)

    # Arch above the beam (uses the dedicated arch kind when available).
    if ctx.room() > 1 and ctx.maybe(0.8):
        arch_kind = ctx.pick_kind("arch", "torus")
        ctx.add(arch_kind, (0, base_y + col_h + 0.16 + span * 0.25, 0),
                {"major_radius": span * 0.28, "minor_radius": 0.07,
                 "radius": span * 0.28, "thickness": 0.14},
                "arch", material=stone)

    # Finials on the outer columns.
    if ctx.room() > 2 and ctx.maybe(0.6):
        for sx_ in (-1, 1):
            ctx.add("sphere", (sx_ * span * 0.5, base_y + col_h + 0.25, 0),
                    {"radius": col_r * 0.9}, f"finial_{sx_}", material=stone)

    # Pediment cone for temple flavor.
    if ctx.room() > 0 and ctx.maybe(0.35):
        ctx.add("cone", (0, base_y + col_h + 0.16 + span * 0.5, 0),
                {"radius": span * 0.3, "height": 0.22},
                "pediment", material=stone)

    ctx.add_feature("erosion", "all", strength=ctx.uniform(0.004, 0.012))
    ctx.add_feature("ridges", f"column_0_-1", count=ctx.randint(10, 18), depth=0.004)
    if ctx.maybe(0.3):
        ctx.add_feature("holes", "beam", count=ctx.randint(1, 3), radius=ctx.uniform(0.03, 0.06))


# ----------------------------------------------------------------------
# plant — trees, shrubs, potted greenery
# ----------------------------------------------------------------------

def build_plant(ctx: FamilyContext) -> None:
    ctx.shape = "plant"
    ctx.pick_palette("plant")
    bark = ctx.material("plant", "wood")
    leaf_mat = ctx.material("plant", "organic")

    trunk_h = ctx.uniform(0.5, 0.9)
    trunk_r = ctx.uniform(0.05, 0.09)
    lean = ctx.uniform(-0.12, 0.12)

    # Trunk: straight cylinder or a curled helix for alien-looking plants.
    if ctx.maybe(0.25):
        ctx.add("helix", (0, trunk_h / 2, 0),
                {"radius": 0.10, "pitch": trunk_h / ctx.uniform(2.0, 3.0),
                 "turns": ctx.uniform(2.0, 3.0), "thickness": trunk_r * 0.7},
                "trunk", rz=lean, material=bark)
    else:
        ctx.add("cylinder", (0, trunk_h / 2, 0),
                {"radius": trunk_r, "height": trunk_h, "caps": True},
                "trunk", rz=lean, material=bark)

    # Branches: radial fan near the top, each tipped with a foliage cluster.
    n_branches = min(ctx.randint(3, 7), max(1, (ctx.room() - 2) // 2))
    crown_r = ctx.uniform(0.30, 0.5)
    for i in range(n_branches):
        a = 2 * math.pi * i / n_branches + ctx.uniform(-0.2, 0.2)
        tilt = ctx.uniform(0.5, 1.0)
        blen = ctx.uniform(0.25, 0.45)
        bx = math.cos(a) * math.sin(tilt)
        by = math.cos(tilt)
        bz = math.sin(a) * math.sin(tilt)
        mid = (bx * blen / 2, trunk_h + by * blen / 2, bz * blen / 2)
        ctx.add("cylinder", mid,
                {"radius": trunk_r * 0.45, "height": blen, "caps": True},
                f"branch_{i}", rx=math.atan2(bz, by) * -1, rz=math.atan2(bx, by), material=bark)
        tip = (bx * blen, trunk_h + by * blen, bz * blen)
        fr = ctx.uniform(0.14, 0.24)
        ctx.add("ellipsoid", tip,
                {"radii": [fr, fr * ctx.uniform(0.7, 1.1), fr]},
                f"foliage_{i}", material=leaf_mat)

    # Top crown cluster.
    ctx.add("ellipsoid", (lean * trunk_h * 0.5, trunk_h + crown_r * 0.35, 0),
            {"radii": [crown_r * 0.7, crown_r * 0.55, crown_r * 0.7]},
            "crown", material=leaf_mat)

    # Fruits / flowers: small bright spheres tucked into the crown.
    n_fruit = min(ctx.randint(0, 6), ctx.room())
    for i in range(n_fruit):
        a = ctx.uniform(0, 2 * math.pi)
        rr = crown_r * ctx.uniform(0.3, 0.65)
        ctx.add("sphere",
                (math.cos(a) * rr, trunk_h + crown_r * ctx.uniform(0.1, 0.5), math.sin(a) * rr),
                {"radius": ctx.uniform(0.025, 0.045)}, f"fruit_{i}", material="ceramic")

    # Roots / ground flare.
    if ctx.room() > 1 and ctx.maybe(0.5):
        ctx.add("cone", (0, 0.08, 0),
                {"radius": trunk_r * 2.2, "height": 0.16}, "root_flare", material=bark)

    ctx.add_feature("bump_field", "crown", count=ctx.randint(20, 60), radius=ctx.uniform(0.02, 0.05))
    if ctx.maybe(0.5):
        ctx.add_feature("fur", "trunk", density=ctx.uniform(0.2, 0.5), length=0.012)  # moss


# ----------------------------------------------------------------------
# vessel — pots, vases, jars, jugs (lathe-style stacked profiles)
# ----------------------------------------------------------------------

def build_vessel(ctx: FamilyContext) -> None:
    ctx.shape = "vessel"
    ctx.pick_palette("vessel")
    ceramic = ctx.material("vessel", "ceramic")

    # A stacked profile: foot → body → shoulder → neck → rim. Radii follow a
    # randomized silhouette curve so each pot has its own character.
    body_r = ctx.uniform(0.24, 0.36)
    foot_r = body_r * ctx.uniform(0.45, 0.65)
    neck_r = body_r * ctx.uniform(0.30, 0.55)
    body_h = ctx.uniform(0.30, 0.45)

    y = 0.0
    # Foot ring.
    ctx.add("torus", (0, y + 0.035, 0),
            {"major_radius": foot_r, "minor_radius": 0.035}, "foot", material=ceramic)
    y += 0.07
    # Base bulge.
    ctx.add("ellipsoid", (0, y + 0.08, 0),
            {"radii": [body_r * 0.8, 0.09, body_r * 0.8]}, "base", material=ceramic)
    y += 0.14
    # Main body — ellipsoid belly or straight cylinder wall.
    if ctx.maybe(0.6):
        ctx.add("ellipsoid", (0, y + body_h / 2, 0),
                {"radii": [body_r, body_h / 2, body_r]}, "body", material=ceramic)
    else:
        ctx.add("cylinder", (0, y + body_h / 2, 0),
                {"radius": body_r * 0.9, "height": body_h, "caps": False},
                "body", material=ceramic)
    y += body_h
    # Shoulder taper.
    ctx.add("ellipsoid", (0, y + 0.07, 0),
            {"radii": [neck_r * 1.5, 0.08, neck_r * 1.5]}, "shoulder", material=ceramic)
    y += 0.13
    # Neck.
    neck_h = ctx.uniform(0.12, 0.22)
    ctx.add("cylinder", (0, y + neck_h / 2, 0),
            {"radius": neck_r, "height": neck_h, "caps": False}, "neck", material=ceramic)
    y += neck_h
    # Rim: flared torus lip.
    ctx.add("torus", (0, y + 0.02, 0),
            {"major_radius": neck_r * 1.15, "minor_radius": 0.022}, "rim", material=ceramic)

    # Optional lid: cone + knob.
    if ctx.room() > 2 and ctx.maybe(0.45):
        ctx.add("cone", (0, y + 0.07, 0),
                {"radius": neck_r * 1.2, "height": 0.10}, "lid", material=ceramic)
        ctx.add("sphere", (0, y + 0.14, 0),
                {"radius": 0.035}, "lid_knob", material=ceramic)

    # Handles: mirrored tori (or tubes when the schema has them) on the body.
    if ctx.room() > 2 and ctx.maybe(0.6):
        handle_kind = ctx.pick_kind("tube", "torus")
        hy = y - body_h * 0.35
        for sx_ in (-1, 1):
            ctx.add(handle_kind, (sx_ * (body_r + 0.06), hy, 0),
                    {"major_radius": 0.09, "minor_radius": 0.022,
                     "radius": 0.022, "height": 0.18},
                    f"handle_{sx_}", rz=math.pi / 2, ry=math.pi / 2, material=ceramic)

    # Surface decoration: painted bands, throwing ridges, or a drilled hole.
    ctx.add_feature("curve_pattern", "body",
                    frequency=ctx.uniform(4.0, 9.0), amplitude=ctx.uniform(0.004, 0.01))
    if ctx.maybe(0.5):
        ctx.add_feature("ridges", "shoulder", count=ctx.randint(10, 20), depth=0.004)
    if ctx.maybe(0.2):
        ctx.add_feature("holes", "body", count=ctx.randint(1, 4), radius=ctx.uniform(0.03, 0.05))


# ----------------------------------------------------------------------
# abstract — balanced / orbiting / stacked sculpture compositions
# ----------------------------------------------------------------------

def build_abstract(ctx: FamilyContext) -> None:
    ctx.shape = "abstract"
    ctx.pick_palette("abstract")
    mat = ctx.material("abstract")
    mat2 = ctx.material("abstract")

    mode = ["stacked", "orbiting", "interlocking"][ctx.randint(0, 2)]
    curved = [k for k in ("torus", "helix", "ellipsoid", "cone", "prism", "sphere")
              if k in ctx.kinds]

    def _rand_params(kind: str) -> dict:
        if kind == "torus":
            return {"major_radius": ctx.uniform(0.15, 0.35),
                    "minor_radius": ctx.uniform(0.04, 0.10)}
        if kind == "helix":
            return {"radius": ctx.uniform(0.12, 0.25), "pitch": ctx.uniform(0.08, 0.2),
                    "turns": ctx.uniform(2.0, 5.0), "thickness": ctx.uniform(0.02, 0.05)}
        if kind == "ellipsoid":
            return {"radii": [ctx.uniform(0.1, 0.3), ctx.uniform(0.1, 0.3), ctx.uniform(0.1, 0.3)]}
        if kind == "cone":
            return {"radius": ctx.uniform(0.1, 0.25), "height": ctx.uniform(0.2, 0.5)}
        if kind == "prism":
            return {"sides": ctx.randint(3, 8), "radius": ctx.uniform(0.12, 0.28),
                    "height": ctx.uniform(0.15, 0.45)}
        return {"radius": ctx.uniform(0.1, 0.25)}

    if mode == "stacked":
        # A totem: parts piled along Y with random twists.
        y = 0.0
        n = min(ctx.target_parts, ctx.randint(4, 8))
        for i in range(n):
            kind = curved[int(ctx.rng.integers(0, len(curved)))]
            h_est = 0.25
            ctx.add(kind, (0, y + h_est / 2, 0), _rand_params(kind),
                    f"piece_{i}", ry=ctx.uniform(0, math.pi),
                    material=mat if i % 2 == 0 else mat2)
            y += h_est * ctx.uniform(0.7, 1.0)
    elif mode == "orbiting":
        # A core with satellites on a ring (radial symmetry + gaps).
        ctx.add("sphere", (0, 0.45, 0), {"radius": ctx.uniform(0.15, 0.22)},
                "core", material=mat)
        n = min(ctx.target_parts - 1, ctx.randint(3, 8))
        orbit = ctx.uniform(0.32, 0.45)
        for i in range(n):
            a = 2 * math.pi * i / n + ctx.uniform(-0.15, 0.15)
            kind = curved[int(ctx.rng.integers(0, len(curved)))]
            ctx.add(kind, (math.cos(a) * orbit, 0.45 + ctx.uniform(-0.1, 0.1), math.sin(a) * orbit),
                    _rand_params(kind), f"satellite_{i}",
                    ry=a, rx=ctx.uniform(-0.5, 0.5), material=mat2)
        # Ring binding the satellites.
        if ctx.room() > 0:
            ctx.add("torus", (0, 0.45, 0),
                    {"major_radius": orbit, "minor_radius": 0.02}, "orbit_ring", material=mat)
    else:
        # Interlocking: 2-3 large curved pieces through each other + accents.
        n_big = min(3, max(2, ctx.target_parts // 3))
        for i in range(n_big):
            kind = "torus" if "torus" in ctx.kinds else "ellipsoid"
            ctx.add(kind, (ctx.uniform(-0.1, 0.1), 0.35 + 0.15 * i, ctx.uniform(-0.1, 0.1)),
                    {"major_radius": 0.28, "minor_radius": 0.06},
                    f"loop_{i}", rx=ctx.uniform(0, math.pi), ry=ctx.uniform(0, math.pi),
                    material=mat if i % 2 == 0 else mat2)
        while ctx.room() > 0 and len(ctx.primitives) < ctx.target_parts:
            kind = curved[int(ctx.rng.integers(0, len(curved)))]
            ctx.add(kind,
                    (ctx.uniform(-0.3, 0.3), ctx.uniform(0.1, 0.7), ctx.uniform(-0.3, 0.3)),
                    _rand_params(kind), f"accent_{len(ctx.primitives)}",
                    ry=ctx.uniform(0, math.pi), material=mat2)

    ctx.add_feature("curve_pattern", "all",
                    frequency=ctx.uniform(5.0, 10.0), amplitude=ctx.uniform(0.005, 0.015))
    if ctx.maybe(0.35):
        ctx.add_feature("holes", "all", count=ctx.randint(1, 3), radius=ctx.uniform(0.04, 0.07))


FAMILY_BUILDERS = {
    "furniture": build_furniture,
    "creature": build_creature,
    "mechanical": build_mechanical,
    "architecture": build_architecture,
    "plant": build_plant,
    "vessel": build_vessel,
    "abstract": build_abstract,
}


# ======================================================================
# CR_ComplexBuilder extension — exquisite style families (real-world scale)
#
# Seven grammar fragments for truly complex objects. Conventions:
# - real-world dimensions (fence post ~1.1 m, column ~3 m, chair seat 0.45 m,
#   spaceship toy-model 2 m) — the engine's bbox fit rescales uniformly, so
#   proportions are what matters here;
# - a fixed core of <= 5 parts so tiny complexity budgets stay valid, with
#   all enrichment gated on ctx.room();
# - every part labelled snake_case; articulation-style naming on the robot.
# ======================================================================

MATERIAL_PALETTES.update({
    "rococo_fence": ("metal", "stone"),
    "neoclassical_column": ("stone", "ceramic"),
    "modern_luxury": ("stone", "metal", "ceramic"),
    "futurist_chair": ("metal", "leather", "fabric", "ceramic"),
    "desktop_computer": ("metal", "ceramic"),
    "spaceship": ("metal", "ceramic"),
    "robot": ("metal", "ceramic"),
})

COLOR_PALETTES.update({
    "rococo_fence": ((0.15, 0.16, 0.18), (0.55, 0.38, 0.18), (0.28, 0.44, 0.38),
                     (0.72, 0.58, 0.28)),
    "neoclassical_column": ((0.88, 0.86, 0.80), (0.78, 0.74, 0.66),
                            (0.72, 0.62, 0.48), (0.55, 0.55, 0.57)),
    "modern_luxury": ((0.85, 0.83, 0.78), (0.20, 0.20, 0.22), (0.80, 0.68, 0.45),
                      (0.62, 0.58, 0.52)),
    "futurist_chair": ((0.22, 0.23, 0.26), (0.85, 0.82, 0.75), (0.85, 0.45, 0.20),
                       (0.25, 0.50, 0.55)),
    "desktop_computer": ((0.75, 0.76, 0.78), (0.08, 0.08, 0.09), (0.90, 0.90, 0.90),
                         (0.30, 0.60, 0.90)),
    "spaceship": ((0.70, 0.72, 0.75), (0.85, 0.86, 0.88), (0.85, 0.45, 0.15),
                  (0.20, 0.50, 0.60), (0.30, 0.32, 0.35)),
    "robot": ((0.88, 0.89, 0.90), (0.30, 0.32, 0.36), (0.90, 0.50, 0.15),
              (0.45, 0.55, 0.65), (0.75, 0.25, 0.20)),
})


def _scroll_path(cx: float, cy: float, r0: float, turns: float,
                 mirror_x: bool = False, n: int = 18) -> list[list[float]]:
    """Planar spiral curl (rococo scrollwork) in the XY plane, z = 0.

    Radius shrinks from r0 to ~0.25 r0 over `turns` revolutions. Kept within
    ±0.35 so the conservative tube half-extent estimate stays sane.
    """
    pts = []
    for i in range(n):
        t = i / (n - 1)
        th = turns * (2.0 * math.pi) * t
        r = r0 * (1.0 - 0.75 * t)
        x = r * math.cos(th)
        pts.append([cx + (-x if mirror_x else x), cy + r * math.sin(th), 0.0])
    return pts


def _tube_height_param(path: list[list[float]]) -> float:
    """Y-extent of a path — feeds the engine's conservative half extents."""
    ys = [p[1] for p in path]
    return max(1e-3, max(ys) - min(ys))


# ----------------------------------------------------------------------
# rococo_fence — lathed posts + scrollwork + arch infill (post ~1.1 m)
# ----------------------------------------------------------------------

def build_rococo_fence(ctx: FamilyContext) -> None:
    ctx.shape = "rococo_fence"
    ctx.pick_palette("rococo_fence")
    iron = "metal"
    stone = "stone"

    span = ctx.uniform(1.3, 1.7)
    post_h = 1.10
    post_r = ctx.uniform(0.065, 0.08)
    shaft_y0, shaft_y1 = 0.12, 1.02
    rail_bot_y, rail_top_y = 0.24, 0.90

    # Core (5): two post shafts, two rails, arch infill.
    for sx_ in (-1, 1):
        ctx.add("cylinder", (sx_ * span / 2, (shaft_y0 + shaft_y1) / 2, 0),
                {"radius": post_r, "height": shaft_y1 - shaft_y0, "caps": True},
                f"post_{'l' if sx_ < 0 else 'r'}", material=iron)
    ctx.add("box", (0, rail_bot_y, 0), {"size": [span, 0.045, 0.03]},
            "rail_bottom", material=iron)
    ctx.add("box", (0, rail_top_y, 0), {"size": [span, 0.05, 0.03]},
            "rail_top", material=iron)
    # Shallow rococo arch: chord across the posts, apex `rise` above the rail.
    chord = span - 2 * post_r
    rise = ctx.uniform(0.12, 0.18)
    arch_R = (rise * rise + (chord / 2) ** 2) / (2 * rise)
    half_ang = math.atan2(chord / 2, arch_R - rise)
    ctx.add("arch", (0, rail_top_y - (arch_R - rise), 0),
            {"major_radius": arch_R, "minor_radius": 0.022,
             "arc": 2 * half_ang, "start_angle": math.pi / 2 - half_ang},
            "arch_infill", material=iron)

    # Lathed post dressing: plinth, base ring, collar, urn finial.
    if ctx.room() >= 4:
        for sx_ in (-1, 1):
            x = sx_ * span / 2
            ctx.add("box", (x, 0.06, 0), {"size": [0.24, 0.12, 0.24]},
                    f"post_plinth_{'l' if sx_ < 0 else 'r'}", material=stone)
            ctx.add("torus", (x, shaft_y0 + 0.05, 0),
                    {"major_radius": post_r + 0.015, "minor_radius": 0.02},
                    f"post_base_ring_{'l' if sx_ < 0 else 'r'}", material=iron)
    if ctx.room() >= 4:
        for sx_ in (-1, 1):
            x = sx_ * span / 2
            ctx.add("torus", (x, shaft_y1 - 0.06, 0),
                    {"major_radius": post_r + 0.012, "minor_radius": 0.016},
                    f"post_collar_{'l' if sx_ < 0 else 'r'}", material=iron)
            ctx.add("sphere", (x, post_h + 0.02, 0), {"radius": post_r * 1.25},
                    f"finial_{'l' if sx_ < 0 else 'r'}", material=iron)

    # Vertical bars between the rails.
    n_bars = min(ctx.randint(5, 9), max(0, ctx.room() - 2))
    bar_xs = []
    for i in range(n_bars):
        t = (i + 1) / (n_bars + 1)
        x = -span / 2 + post_r + t * (span - 2 * post_r)
        bar_xs.append(x)
        ctx.add("cylinder", (x, (rail_bot_y + rail_top_y) / 2, 0),
                {"radius": 0.012, "height": rail_top_y - rail_bot_y, "caps": True},
                f"vbar_{i}", material=iron)

    # Scrollwork curls: mirrored spiral tubes flanking the panel centre.
    if ctx.room() >= 2:
        r0 = ctx.uniform(0.16, 0.24)
        cy = (rail_bot_y + rail_top_y) / 2
        for mx_, side in ((False, "l"), (True, "r")):
            path = _scroll_path((-0.32 if not mx_ else 0.32), cy, r0,
                                ctx.uniform(1.1, 1.6), mirror_x=mx_)
            ctx.add("tube", (0, 0, 0),
                    {"path": path, "radius": 0.014, "caps": True,
                     "height": _tube_height_param(path)},
                    f"scroll_{side}", material=iron)

    # Spear tips on the bars.
    if ctx.room() >= len(bar_xs):
        for i, x in enumerate(bar_xs):
            ctx.add("cone", (x, rail_top_y + 0.045, 0),
                    {"radius": 0.02, "height": 0.09},
                    f"spear_{i}", material=iron)

    ctx.add_feature("scratch", "all", count=ctx.randint(2, 8),
                    depth=ctx.uniform(0.002, 0.005))


# ----------------------------------------------------------------------
# neoclassical_column — sliced entasis shaft + capital (~3 m)
# ----------------------------------------------------------------------

def build_neoclassical_column(ctx: FamilyContext) -> None:
    ctx.shape = "neoclassical_column"
    ctx.pick_palette("neoclassical_column")
    stone = "stone"

    H = ctx.uniform(2.6, 3.2)
    r_base = ctx.uniform(0.15, 0.18)
    r_top = r_base * ctx.uniform(0.74, 0.82)
    plinth_h = 0.12
    capital_h = 0.28
    shaft_h = H - plinth_h - capital_h

    # Core (5): plinth, 2 shaft slices, echinus, abacus.
    n_slices = 2 + min(4, max(0, ctx.room() - 6))
    ctx.add("box", (0, plinth_h / 2, 0), {"size": [r_base * 4.4, plinth_h, r_base * 4.4]},
            "plinth", material=stone)

    def _radius(t: float) -> float:
        # Entasis: linear taper + a slight bulge at one-third height.
        return (r_base + (r_top - r_base) * t
                + 0.035 * r_base * math.sin(math.pi * t))

    slice_h = shaft_h / n_slices
    for i in range(n_slices):
        t0, t1 = i / n_slices, (i + 1) / n_slices
        r_mid = _radius((t0 + t1) / 2)
        ctx.add("cylinder", (0, plinth_h + (i + 0.5) * slice_h, 0),
                {"radius": r_mid, "height": slice_h * 1.001, "caps": True},
                f"shaft_{i}", material=stone)

    shaft_top = plinth_h + shaft_h
    # Echinus: cone flaring upward (apex down) + square abacus.
    ctx.add("cone", (0, shaft_top + capital_h * 0.4, 0),
            {"radius": r_top * 1.45, "height": capital_h * 0.8},
            "echinus", rx=math.pi, material=stone)
    ctx.add("box", (0, shaft_top + capital_h * 0.9, 0),
            {"size": [r_top * 3.4, capital_h * 0.2, r_top * 3.4]},
            "abacus", material=stone)

    # Base mouldings: two tori at the shaft foot.
    if ctx.room() >= 2:
        for i in range(2):
            ctx.add("torus", (0, plinth_h + 0.03 + 0.05 * i, 0),
                    {"major_radius": r_base * (1.12 - 0.06 * i),
                     "minor_radius": 0.028},
                    f"base_mould_{i}", material=stone)
    # Necking ring below the capital.
    if ctx.room() >= 1:
        ctx.add("torus", (0, shaft_top - 0.02, 0),
                {"major_radius": r_top * 1.05, "minor_radius": 0.022},
                "necking", material=stone)
    # Volute scrolls on the capital (ionic flavour).
    if ctx.room() >= 2:
        for sx_ in (-1, 1):
            ctx.add("torus", (sx_ * r_top * 1.15, shaft_top + capital_h * 0.55, 0),
                    {"major_radius": r_top * 0.42, "minor_radius": 0.03},
                    f"volute_{'l' if sx_ < 0 else 'r'}", ry=math.pi / 2, material=stone)
    # Entablature slab.
    if ctx.room() >= 1:
        ctx.add("box", (0, shaft_top + capital_h + 0.08, 0),
                {"size": [r_top * 4.2, 0.12, r_top * 2.2]},
                "entablature", material=stone)

    # Fluting on the shaft + gentle weathering.
    ctx.add_feature("ridges", "shaft_0", count=ctx.randint(16, 24), depth=0.005)
    ctx.add_feature("erosion", "all", strength=ctx.uniform(0.003, 0.008))


# ----------------------------------------------------------------------
# modern_luxury — beveled monoliths + metal accents (~1.2 m console)
# ----------------------------------------------------------------------

def build_modern_luxury(ctx: FamilyContext) -> None:
    ctx.shape = "modern_luxury"
    ctx.pick_palette("modern_luxury")
    stone = "stone"
    gold = "metal"
    glass = "ceramic"

    bevel = [ctx.uniform(0.25, 0.45), ctx.uniform(0.25, 0.45)]

    # Core (4): main monolith, plinth, gold trim bar, accent sculpture.
    ctx.add("superellipsoid", (0, 0.56, 0),
            {"radii": [0.55, 0.44, 0.22], "exponents": bevel},
            "monolith", material=stone)
    ctx.add("superellipsoid", (0, 0.05, 0),
            {"radii": [0.62, 0.05, 0.27], "exponents": [0.3, 0.3]},
            "plinth", material=gold)
    ctx.add("box", (0.30, 0.56, 0.215), {"size": [0.018, 0.80, 0.012]},
            "trim_vertical", material=gold)
    ctx.add("torus", (-0.24, 0.72, 0),
            {"major_radius": 0.15, "minor_radius": 0.045},
            "sculpture_ring", ry=ctx.uniform(0, math.pi), material=gold)

    # Secondary monolith + floating glass shelf.
    if ctx.room() >= 1:
        ctx.add("superellipsoid", (0.44, 0.30, 0),
                {"radii": [0.14, 0.20, 0.14], "exponents": bevel},
                "monolith_side", material=stone)
    if ctx.room() >= 1:
        ctx.add("superellipsoid", (-0.05, 1.04, 0),
                {"radii": [0.34, 0.015, 0.18], "exponents": [0.2, 0.2]},
                "shelf_glass", material=glass)
    # Horizontal gold inlays.
    n_inlay = min(ctx.randint(1, 3), ctx.room())
    for i in range(n_inlay):
        y = 0.30 + 0.22 * i
        ctx.add("box", (-0.12, y, 0.218), {"size": [0.55, 0.012, 0.008]},
                f"inlay_{i}", material=gold)
    # Small vessel accent on the shelf.
    if ctx.room() >= 1 and ctx.maybe(0.7):
        ctx.add("cylinder", (0.12, 1.12, 0.02),
                {"radius": 0.045, "height": 0.15, "caps": True},
                "accent_vessel", material=glass)

    ctx.add_feature("curve_pattern", "monolith",
                    frequency=ctx.uniform(2.0, 4.0), amplitude=0.004)


# ----------------------------------------------------------------------
# futurist_chair — cantilevered shell chair (seat at 0.45 m)
# ----------------------------------------------------------------------

def build_futurist_chair(ctx: FamilyContext) -> None:
    ctx.shape = "futurist_chair"
    ctx.pick_palette("futurist_chair")
    metal = "metal"
    shell_mat = ctx.material("futurist_chair", "ceramic", "leather")
    soft = ctx.material("futurist_chair", "fabric", "leather")

    seat_h = 0.45

    # Core (5): disc foot, column, seat shell, wrap-around back, cushion.
    ctx.add("cylinder", (0, 0.02, 0), {"radius": 0.27, "height": 0.04, "caps": True},
            "foot_disc", material=metal)
    ctx.add("cylinder", (0, seat_h / 2, 0),
            {"radius": 0.032, "height": seat_h - 0.04, "caps": True},
            "column", material=metal)
    ctx.add("superellipsoid", (0, seat_h, 0.01),
            {"radii": [0.27, 0.04, 0.26], "exponents": [0.4, 0.5]},
            "seat_shell", material=shell_mat)
    ctx.add("panel", (0, seat_h + 0.30, -0.235),
            {"size": [0.44, 0.52], "thickness": 0.028,
             "bend": ctx.uniform(0.4, 0.7)},
            "back_shell", material=shell_mat)
    ctx.add("superellipsoid", (0, seat_h + 0.065, 0.02),
            {"radii": [0.22, 0.032, 0.21], "exponents": [0.35, 0.45]},
            "cushion", material=soft)

    # Armrest loops.
    if ctx.room() >= 2:
        for sx_ in (-1, 1):
            path = [[sx_ * 0.27, seat_h + 0.02, -0.16],
                    [sx_ * 0.30, seat_h + 0.16, 0.0],
                    [sx_ * 0.28, seat_h + 0.20, 0.16]]
            ctx.add("tube", (0, 0, 0),
                    {"path": path, "radius": 0.016, "caps": True, "height": 0.2},
                    f"armrest_{'l' if sx_ < 0 else 'r'}", material=metal)
    # Headrest + footrest ring + trim accent.
    if ctx.room() >= 1:
        ctx.add("superellipsoid", (0, seat_h + 0.60, -0.22),
                {"radii": [0.14, 0.05, 0.06], "exponents": [0.4, 0.4]},
                "headrest", material=soft)
    if ctx.room() >= 1:
        ctx.add("torus", (0, 0.16, 0),
                {"major_radius": 0.20, "minor_radius": 0.014},
                "footrest_ring", material=metal)
    if ctx.room() >= 1 and ctx.maybe(0.6):
        ctx.add("box", (0, seat_h + 0.30, -0.27),
                {"size": [0.30, 0.015, 0.01]},
                "back_trim", material=metal)

    ctx.add_feature("scratch", "column", count=ctx.randint(2, 6), depth=0.002)


# ----------------------------------------------------------------------
# desktop_computer — tower + monitor + peripherals (~0.5 m real scale)
# ----------------------------------------------------------------------

def build_desktop_computer(ctx: FamilyContext) -> None:
    ctx.shape = "desktop_computer"
    ctx.pick_palette("desktop_computer")
    alu = "metal"
    plastic = "ceramic"

    tower_x = -0.36
    # Core (5): tower, screen, stand neck, stand foot, keyboard.
    ctx.add("box", (tower_x, 0.23, 0), {"size": [0.19, 0.44, 0.42]},
            "tower", material=alu)
    ctx.add("box", (0.08, 0.40, 0), {"size": [0.54, 0.32, 0.028]},
            "screen", material=plastic)
    ctx.add("box", (0.08, 0.16, -0.02), {"size": [0.05, 0.14, 0.03]},
            "stand_neck", material=alu)
    ctx.add("box", (0.08, 0.012, 0.02), {"size": [0.22, 0.024, 0.17]},
            "stand_foot", material=alu)
    ctx.add("box", (0.08, 0.012, 0.30), {"size": [0.36, 0.02, 0.13]},
            "keyboard", material=plastic)

    # Screen bezel inset + power LED.
    if ctx.room() >= 2:
        ctx.add("box", (0.08, 0.40, 0.016), {"size": [0.50, 0.28, 0.004]},
                "screen_inset", material=plastic)
        ctx.add("box", (tower_x + 0.096, 0.40, 0.10), {"size": [0.004, 0.012, 0.05]},
                "led_strip", material=plastic)
    # Cooling fan ring on the tower front.
    if ctx.room() >= 1:
        ctx.add("torus", (tower_x, 0.20, 0.212),
                {"major_radius": 0.06, "minor_radius": 0.012},
                "fan_ring", rx=math.pi / 2, material=plastic)
    # Key bank + mouse.
    if ctx.room() >= 1:
        ctx.add("box", (0.08, 0.024, 0.30), {"size": [0.33, 0.008, 0.10]},
                "key_bank", material=alu)
    if ctx.room() >= 1:
        ctx.add("ellipsoid", (0.34, 0.02, 0.30), {"radii": [0.035, 0.02, 0.055]},
                "mouse", material=plastic)
    # Curved cable from tower to screen.
    if ctx.room() >= 1 and ctx.maybe(0.8):
        path = [[tower_x + 0.05, 0.30, -0.21],
                [-0.15, 0.26, -0.16],
                [0.02, 0.24, -0.06]]
        ctx.add("tube", (0, 0, 0),
                {"path": path, "radius": 0.008, "caps": True, "height": 0.06},
                "cable", material=plastic)

    ctx.add_feature("scratch", "tower", count=ctx.randint(1, 5), depth=0.002)


# ----------------------------------------------------------------------
# spaceship — sliced hull + greeble arrays (toy-model 2 m)
# ----------------------------------------------------------------------

def build_spaceship(ctx: FamilyContext) -> None:
    ctx.shape = "spaceship"
    ctx.pick_palette("spaceship")
    hull_mat = "metal"
    accent = "ceramic"

    # Core (5): nose, mid hull slice, aft hull slice, nozzle, wing plane.
    ctx.add("cone", (0, 0.30, 0.78), {"radius": 0.20, "height": 0.44},
            "nose", rx=math.pi / 2, material=hull_mat)
    ctx.add("cylinder", (0, 0.30, 0.28), {"radius": 0.25, "height": 0.62, "caps": True},
            "hull_mid", rx=math.pi / 2, material=hull_mat)
    ctx.add("cylinder", (0, 0.30, -0.33), {"radius": 0.21, "height": 0.62, "caps": True},
            "hull_aft", rx=math.pi / 2, material=hull_mat)
    ctx.add("cone", (0, 0.30, -0.80), {"radius": 0.16, "height": 0.34},
            "nozzle", rx=-math.pi / 2, material=hull_mat)
    ctx.add("box", (0, 0.26, -0.10), {"size": [1.35, 0.035, 0.42]},
            "wing", material=hull_mat)

    # Canopy + vertical fin.
    if ctx.room() >= 2:
        ctx.add("ellipsoid", (0, 0.44, 0.30), {"radii": [0.10, 0.09, 0.20]},
                "canopy", material=accent)
        ctx.add("box", (0, 0.52, -0.48), {"size": [0.03, 0.30, 0.26]},
                "fin_vertical", material=hull_mat)
    # Forward hull slice (smooths the nose transition).
    if ctx.room() >= 1:
        ctx.add("cylinder", (0, 0.30, 0.60), {"radius": 0.225, "height": 0.10,
                                              "caps": True},
                "hull_fwd", rx=math.pi / 2, material=hull_mat)
    # Wing-tip pods.
    if ctx.room() >= 2:
        for sx_ in (-1, 1):
            ctx.add("cylinder", (sx_ * 0.66, 0.26, -0.10),
                    {"radius": 0.055, "height": 0.30, "caps": True},
                    f"pod_{'l' if sx_ < 0 else 'r'}", rx=math.pi / 2, material=hull_mat)
    # Greeble array: small surface details on the hull spine + flanks.
    n_greeble = min(ctx.randint(4, 8), max(0, ctx.room() - 1))
    for i in range(n_greeble):
        z = ctx.uniform(-0.55, 0.5)
        side = ctx.randint(0, 2)
        if side == 0:
            pos = (0, 0.30 + 0.25, z)
            size = [ctx.uniform(0.04, 0.10), 0.03, ctx.uniform(0.05, 0.14)]
        else:
            sx_ = 1.0 if side == 1 else -1.0
            pos = (sx_ * 0.245, 0.30, z)
            size = [0.03, ctx.uniform(0.04, 0.09), ctx.uniform(0.05, 0.14)]
        ctx.add("box", pos, {"size": size}, f"greeble_{i}", material=accent)
    # Engine glow disc.
    if ctx.room() >= 1:
        ctx.add("cylinder", (0, 0.30, -0.965), {"radius": 0.115, "height": 0.02,
                                                "caps": True},
                "engine_glow", rx=math.pi / 2, material=accent)

    ctx.add_feature("ridges", "hull_mid", count=ctx.randint(6, 14), depth=0.003)
    ctx.add_feature("scratch", "all", count=ctx.randint(2, 8), depth=0.002)


# ----------------------------------------------------------------------
# robot — jointed primitive assembly, articulation-ready naming (~0.7 m)
# ----------------------------------------------------------------------

def build_robot(ctx: FamilyContext) -> None:
    ctx.shape = "robot"
    ctx.pick_palette("robot")
    metal = "metal"
    shell = "ceramic"

    # Core (3): torso, neck, head. Limbs are staged by priority below.
    ctx.add("superellipsoid", (0, 0.44, 0),
            {"radii": [0.14, 0.12, 0.095], "exponents": [0.5, 0.5]},
            "torso", material=shell)
    ctx.add("cylinder", (0, 0.565, 0), {"radius": 0.03, "height": 0.05, "caps": True},
            "neck", material=metal)
    ctx.add("superellipsoid", (0, 0.645, 0),
            {"radii": [0.085, 0.07, 0.08], "exponents": [0.5, 0.5]},
            "head", material=shell)

    hip_y, knee_y, ankle_y = 0.32, 0.17, 0.045
    shoulder_y, elbow_y, wrist_y = 0.52, 0.40, 0.285

    def _pair(kind, y, x, params, stem, mat, **rot):
        for sx_ in (-1, 1):
            ctx.add(kind, (sx_ * x, y, 0), dict(params),
                    f"{stem}_{'l' if sx_ < 0 else 'r'}", material=mat, **rot)

    # Staged limb construction: every stage costs 2 parts (antenna 1) and is
    # only built when the complexity budget has room. Order is chosen so a
    # mid-size budget still yields a complete humanoid silhouette.
    def _stage_thighs():
        _pair("capsule", (hip_y + knee_y) / 2, 0.075,
              {"radius": 0.036, "height": hip_y - knee_y - 0.03}, "thigh", metal)

    def _stage_shoulders():
        _pair("sphere", shoulder_y, 0.165, {"radius": 0.048}, "shoulder", metal)

    def _stage_upper_arms():
        _pair("capsule", (shoulder_y + elbow_y) / 2 - 0.01, 0.175,
              {"radius": 0.030, "height": shoulder_y - elbow_y - 0.03},
              "upper_arm", metal)

    def _stage_shins():
        _pair("capsule", (knee_y + ankle_y) / 2, 0.075,
              {"radius": 0.030, "height": knee_y - ankle_y - 0.02}, "shin", metal)

    def _stage_feet():
        _pair("box", 0.018, 0.075, {"size": [0.075, 0.036, 0.12]}, "foot", shell)

    def _stage_forearms():
        _pair("capsule", (elbow_y + wrist_y) / 2, 0.175,
              {"radius": 0.026, "height": elbow_y - wrist_y - 0.02},
              "forearm", metal)

    def _stage_hips():
        _pair("sphere", hip_y, 0.085, {"radius": 0.048}, "hip", metal)

    def _stage_knees():
        _pair("sphere", knee_y, 0.075, {"radius": 0.038}, "knee", metal)

    def _stage_elbows():
        _pair("sphere", elbow_y, 0.175, {"radius": 0.033}, "elbow", metal)

    def _stage_hands():
        _pair("box", wrist_y - 0.03, 0.175, {"size": [0.05, 0.07, 0.03]},
              "hand", shell)

    def _stage_eyes():
        for sx_ in (-1, 1):
            ctx.add("sphere", (sx_ * 0.032, 0.655, 0.072),
                    {"radius": 0.016}, f"eye_{'l' if sx_ < 0 else 'r'}",
                    material=metal)

    def _stage_antenna():
        ctx.add("cylinder", (0, 0.74, 0), {"radius": 0.008, "height": 0.06,
                                           "caps": True},
                "antenna_stem", material=metal)
        if ctx.room() >= 1:
            ctx.add("sphere", (0, 0.78, 0), {"radius": 0.018},
                    "antenna_tip", material=shell)

    stages = [
        (2, _stage_thighs), (2, _stage_shoulders), (2, _stage_upper_arms),
        (2, _stage_shins), (2, _stage_feet), (2, _stage_forearms),
        (2, _stage_hips), (2, _stage_knees), (2, _stage_elbows),
        (2, _stage_hands), (2, _stage_eyes), (1, _stage_antenna),
    ]
    for cost, stage in stages:
        if ctx.room() >= cost:
            stage()

    # Chest plate + pelvis accent when the budget allows.
    if ctx.room() >= 1:
        ctx.add("box", (0, 0.46, 0.085), {"size": [0.16, 0.14, 0.02]},
                "chest_plate", material=metal)
    if ctx.room() >= 1:
        ctx.add("box", (0, 0.30, 0), {"size": [0.15, 0.07, 0.09]},
                "pelvis", material=shell)

    ctx.add_feature("scratch", "all", count=ctx.randint(2, 8), depth=0.002)


FAMILY_BUILDERS = {
    **FAMILY_BUILDERS,
    "rococo_fence": build_rococo_fence,
    "neoclassical_column": build_neoclassical_column,
    "modern_luxury": build_modern_luxury,
    "futurist_chair": build_futurist_chair,
    "desktop_computer": build_desktop_computer,
    "spaceship": build_spaceship,
    "robot": build_robot,
}

# ======================================================================
# CR_Quality organic grammars — leaf / flower / insect / terrain and the
# lathe-true vessel. Real-world scale; every part attaches to a parent.
# ======================================================================

MATERIAL_PALETTES.update({
    "insect": ("organic", "ceramic"),
    "flower": ("organic", "ceramic"),
    "leaf": ("organic",),
    "terrain": ("stone", "organic"),
})

COLOR_PALETTES.update({
    "insect": ((0.65, 0.12, 0.08), (0.12, 0.10, 0.10), (0.55, 0.35, 0.12),
               (0.20, 0.30, 0.15)),
    "flower": ((0.85, 0.30, 0.40), (0.90, 0.75, 0.20), (0.75, 0.35, 0.60),
               (0.90, 0.90, 0.88)),
    "leaf": ((0.25, 0.45, 0.18), (0.35, 0.52, 0.22), (0.20, 0.38, 0.25)),
    "terrain": ((0.32, 0.24, 0.16), (0.40, 0.33, 0.24), (0.28, 0.22, 0.15)),
})


# ----------------------------------------------------------------------
# insect — head / thorax / abdomen + 6 jointed legs + wings + antennae
# ----------------------------------------------------------------------

def build_insect(ctx: FamilyContext) -> None:
    ctx.shape = "insect"
    ctx.pick_palette("insect")
    chitin = ctx.material("insect", "organic")
    wing_mat = ctx.material("insect", "ceramic")

    # Real-world beetle scale (~3 cm body); the bbox fit rescales uniformly.
    ab = ctx.uniform(0.013, 0.018)      # abdomen half-width
    body_y = ab * 1.15

    # Core: abdomen, thorax, head — three distinct segments, touching.
    ctx.add("ellipsoid", (0, body_y, -ab * 1.3),
            {"radii": [ab, ab * 0.80, ab * 1.35]}, "abdomen", material=chitin)
    ctx.add("ellipsoid", (0, body_y * 1.05, ab * 0.35),
            {"radii": [ab * 0.70, ab * 0.60, ab * 0.70]}, "thorax", material=chitin)
    ctx.add("sphere", (0, body_y * 1.10, ab * 1.25),
            {"radius": ab * 0.45}, "head", material=chitin)

    # 6 jointed legs: femur angled up-out, tibia down to the floor.
    leg_r = ab * ctx.uniform(0.10, 0.14)
    for i, z in enumerate((ab * 0.75, ab * 0.25, -ab * 0.30)):
        for sx_ in (-1, 1):
            hip = (sx_ * ab * 0.55, body_y * 0.95, z)
            knee = (sx_ * ab * 1.30, body_y * 0.55, z + ab * 0.10)
            foot = (sx_ * ab * 1.75, 0.0, z + ab * 0.25)
            femur_len = math.dist(hip, knee)
            tibia_len = math.dist(knee, foot)
            fmid = tuple((a + b) / 2 for a, b in zip(hip, knee))
            tmid = tuple((a + b) / 2 for a, b in zip(knee, foot))
            # Tilt about Z so the capsule's Y axis runs hip→knee / knee→foot.
            fa = math.atan2(knee[0] - hip[0], hip[1] - knee[1])
            ta = math.atan2(foot[0] - knee[0], knee[1] - foot[1])
            ctx.add("capsule", fmid, {"radius": leg_r, "height": femur_len},
                    f"femur_{i}_{sx_}", rz=fa, material=chitin)
            ctx.add("capsule", tmid, {"radius": leg_r * 0.8, "height": tibia_len},
                    f"tibia_{i}_{sx_}", rz=ta, material=chitin)

    # Wings: two thin bent panels laid over the abdomen (elytra).
    wing_kind = ctx.pick_kind("panel", "box")
    for sx_ in (-1, 1):
        if wing_kind == "panel":
            ctx.add("panel", (sx_ * ab * 0.42, body_y + ab * 0.72, -ab * 1.2),
                    {"size": [ab * 0.85, ab * 2.2], "thickness": ab * 0.06,
                     "bend": ctx.uniform(0.3, 0.55)},
                    f"wing_{sx_}", rx=math.pi / 2, ry=-sx_ * 0.12, material=wing_mat)
        else:
            ctx.add("box", (sx_ * ab * 0.42, body_y + ab * 0.72, -ab * 1.2),
                    {"size": [ab * 0.85, ab * 0.06, ab * 2.2]},
                    f"wing_{sx_}", ry=-sx_ * 0.12, material=wing_mat)

    # Antennae: two thin tubes curving forward off the head.
    for sx_ in (-1, 1):
        path = [[sx_ * ab * 0.15, body_y * 1.15, ab * 1.55],
                [sx_ * ab * 0.45, body_y * 1.35, ab * 1.95],
                [sx_ * ab * 0.75, body_y * 1.25, ab * 2.25]]
        ctx.add("tube", (0, 0, 0),
                {"path": path, "radius": leg_r * 0.45, "caps": True,
                 "height": ab * 0.8},
                f"antenna_{sx_}", material=chitin)

    # Compound eyes.
    for sx_ in (-1, 1):
        ctx.add("sphere", (sx_ * ab * 0.30, body_y * 1.18, ab * 1.45),
                {"radius": ab * 0.14}, f"eye_{sx_}", material=wing_mat)

    ctx.add_feature("ridges", "abdomen", count=ctx.randint(6, 10), depth=ab * 0.03)
    ctx.add_feature("asperity", "all", strength=ab * 0.02, frequency=60.0)


# ----------------------------------------------------------------------
# flower — petal ring + stamen cluster + curved stem (+ a leaf)
# ----------------------------------------------------------------------

def _leaf_parts(ctx: FamilyContext, base_y: float, scale: float,
                stem: str, label_prefix: str, leaf_mat: str,
                lean: float = 0.5, yaw: float = 0.0) -> None:
    """One leaf: curved blade panel + midrib tube + side veins."""
    blade_l = 0.14 * scale
    blade_w = 0.055 * scale
    blade_kind = ctx.pick_kind("panel", "box")
    bx = math.sin(lean) * blade_l * 0.5
    by = base_y + math.cos(lean) * blade_l * 0.5 * 0.25
    if blade_kind == "panel":
        ctx.add("panel", (bx, by, 0.0),
                {"size": [blade_w, blade_l], "thickness": 0.0012 * scale,
                 "bend": ctx.uniform(0.35, 0.6)},
                f"{label_prefix}_blade", rx=lean, ry=yaw, material=leaf_mat)
    else:
        ctx.add("box", (bx, by, 0.0),
                {"size": [blade_w, 0.0012 * scale, blade_l]},
                f"{label_prefix}_blade", rx=lean, ry=yaw, material=leaf_mat)
    # Midrib: thin tube from the blade base to its tip.
    tip = (math.sin(lean) * blade_l, base_y + math.cos(lean) * blade_l * 0.35, 0.0)
    path = [[0.0, base_y, 0.0],
            [tip[0] * 0.5, (base_y + tip[1]) / 2 + 0.004 * scale, 0.0],
            [tip[0], tip[1], 0.0]]
    ctx.add("tube", (0, 0, 0),
            {"path": path, "radius": 0.0018 * scale, "caps": True,
             "height": blade_l},
            f"{label_prefix}_midrib", material=stem)
    # Side veins: 2-3 pairs of hair-thin tubes branching off the midrib.
    n_veins = min(3, max(0, ctx.room() // 2))
    for i in range(n_veins):
        t = 0.30 + 0.20 * i
        vx = tip[0] * t
        vy = base_y + (tip[1] - base_y) * t
        for sx_ in (-1, 1):
            vpath = [[vx, vy, 0.0],
                     [vx + math.sin(lean) * blade_w * 0.35,
                      vy + blade_w * 0.20, sx_ * blade_w * 0.40]]
            ctx.add("tube", (0, 0, 0),
                    {"path": vpath, "radius": 0.0008 * scale, "caps": True,
                     "height": blade_w},
                    f"{label_prefix}_vein_{i}_{sx_}", material=leaf_mat)


def build_flower(ctx: FamilyContext) -> None:
    ctx.shape = "flower"
    ctx.pick_palette("flower")
    stem_mat = ctx.material("flower", "organic")
    petal_mat = ctx.material("flower", "organic", "ceramic")

    # Curved stem rooted at the origin (real-world daisy scale ~25 cm).
    stem_h = ctx.uniform(0.20, 0.35)
    stem_r = ctx.uniform(0.003, 0.006)
    bend = ctx.uniform(-0.06, 0.06)
    path = [[0.0, 0.0, 0.0],
            [bend * 0.4, stem_h * 0.5, 0.0],
            [bend, stem_h, 0.0]]
    ctx.add("tube", (0, 0, 0),
            {"path": path, "radius": stem_r, "caps": True, "height": stem_h},
            "stem", material=stem_mat)

    top = (bend, stem_h, 0.0)
    # Receptacle the petals grow from.
    ctx.add("sphere", top, {"radius": stem_r * 2.2}, "receptacle",
            material=stem_mat)

    # Petal ring: 5–8 curved panels tilted outward around the receptacle.
    n_petals = min(ctx.randint(5, 8), max(3, ctx.room() - 6))
    petal_l = ctx.uniform(0.05, 0.09)
    petal_w = petal_l * ctx.uniform(0.40, 0.55)
    tilt = ctx.uniform(0.85, 1.30)      # outward lean from vertical (open bloom)
    petal_kind = ctx.pick_kind("panel", "box")
    for i in range(n_petals):
        yaw = 2 * math.pi * i / n_petals
        dx, dz = math.sin(yaw), math.cos(yaw)
        cx = top[0] + dx * math.sin(tilt) * petal_l * 0.45
        cy = top[1] + math.cos(tilt) * petal_l * 0.45
        cz = top[2] + dz * math.sin(tilt) * petal_l * 0.45
        if petal_kind == "panel":
            ctx.add("panel", (cx, cy, cz),
                    {"size": [petal_w, petal_l], "thickness": 0.0015,
                     "bend": ctx.uniform(0.3, 0.55)},
                    f"petal_{i}", rx=tilt, ry=yaw, material=petal_mat)
        else:
            ctx.add("box", (cx, cy, cz),
                    {"size": [petal_w, petal_l, 0.0015]},
                    f"petal_{i}", rx=tilt, ry=yaw, material=petal_mat)

    # Stamen cluster: 3–5 filaments with anther tips around the receptacle.
    n_stamen = min(ctx.randint(3, 5), max(0, ctx.room() // 2))
    fil_h = stem_r * 6.0
    for i in range(n_stamen):
        yaw = 2 * math.pi * i / max(n_stamen, 1) + 0.4
        fx = top[0] + math.sin(yaw) * stem_r * 1.2
        fz = top[2] + math.cos(yaw) * stem_r * 1.2
        ctx.add("cylinder", (fx, top[1] + fil_h / 2, fz),
                {"radius": stem_r * 0.35, "height": fil_h, "caps": True},
                f"filament_{i}", material=stem_mat)
        ctx.add("sphere", (fx, top[1] + fil_h, fz),
                {"radius": stem_r * 0.8}, f"anther_{i}", material=petal_mat)

    # Optional single leaf halfway up the stem.
    if ctx.room() >= 4 and ctx.maybe(0.6):
        _leaf_parts(ctx, stem_h * 0.45, 0.7, stem_mat, "leaf", stem_mat,
                    lean=ctx.uniform(0.6, 1.0), yaw=ctx.uniform(0.0, math.pi))

    ctx.add_feature("asperity", "all", strength=0.0004, frequency=50.0)


# ----------------------------------------------------------------------
# leaf — petiole + blade + midrib + veins (standalone ground leaf)
# ----------------------------------------------------------------------

def build_leaf(ctx: FamilyContext) -> None:
    ctx.shape = "leaf"
    ctx.pick_palette("leaf")
    stem_mat = ctx.material("leaf", "organic")
    leaf_mat = ctx.material("leaf", "organic")

    # Petiole: short curved tube from the floor to the blade base.
    petiole_h = ctx.uniform(0.03, 0.06)
    ctx.add("tube", (0, 0, 0),
            {"path": [[0.0, 0.0, 0.0], [0.004, petiole_h * 0.6, 0.0],
                      [0.006, petiole_h, 0.0]],
             "radius": 0.0025, "caps": True, "height": petiole_h},
            "petiole", material=stem_mat)

    scale = ctx.uniform(0.9, 1.4)
    _leaf_parts(ctx, petiole_h, scale, stem_mat, "leaf", leaf_mat,
                lean=ctx.uniform(0.5, 0.9), yaw=ctx.uniform(-0.3, 0.3))

    ctx.add_feature("asperity", "all", strength=0.0003, frequency=70.0)


# ----------------------------------------------------------------------
# terrain — soil slab with relief, pebbles, and sparse grass blades
# ----------------------------------------------------------------------

def build_terrain(ctx: FamilyContext) -> None:
    ctx.shape = "terrain"
    ctx.pick_palette("terrain")
    soil = ctx.material("terrain", "stone")
    grass = ctx.material("terrain", "organic")

    # Ground slab: never a bare flat sheet — the relief feature displaces it.
    w = ctx.uniform(0.5, 0.9)
    d = ctx.uniform(0.5, 0.9)
    t = ctx.uniform(0.04, 0.08)
    ctx.add("box", (0, t / 2, 0), {"size": [w, t, d]}, "ground", material=soil)
    ctx.add_feature("relief", "ground",
                    amplitude=ctx.uniform(0.010, 0.022),
                    frequency=ctx.uniform(4.0, 7.0), octaves=3,
                    pebbles=ctx.randint(2, 5))

    # Scattered pebbles half-bedded in the surface.
    n_pebbles = min(ctx.randint(3, 6), max(0, ctx.room() - 4))
    for i in range(n_pebbles):
        pr = ctx.uniform(0.012, 0.035)
        px = ctx.uniform(-w / 2 + pr, w / 2 - pr)
        pz = ctx.uniform(-d / 2 + pr, d / 2 - pr)
        ctx.add("ellipsoid", (px, t - pr * 0.35, pz),
                {"radii": [pr, pr * ctx.uniform(0.55, 0.8), pr]},
                f"pebble_{i}", material=soil)

    # Sparse grass blades: thin tapered cones poking out of the soil.
    n_grass = min(ctx.randint(4, 8), max(0, ctx.room()))
    for i in range(n_grass):
        gh = ctx.uniform(0.03, 0.08)
        gx = ctx.uniform(-w / 2, w / 2)
        gz = ctx.uniform(-d / 2, d / 2)
        ctx.add("cone", (gx, t + gh / 2 - 0.005, gz),
                {"radius": ctx.uniform(0.002, 0.004), "height": gh},
                f"grass_{i}", rz=ctx.uniform(-0.15, 0.15), material=grass)


# ----------------------------------------------------------------------
# vessel v2 — lathe-true profile: stacked frusta + rim torus
# ----------------------------------------------------------------------

def build_vessel(ctx: FamilyContext) -> None:  # noqa: F811 — lathe-true replacement
    ctx.shape = "vessel"
    ctx.pick_palette("vessel")
    ceramic = ctx.material("vessel", "ceramic")

    # Lathe profile as (radius, height) stations bottom → top: foot, belly,
    # shoulder, neck. The neck is ALWAYS narrower than the belly (truth
    # table: neck_rel 0.30–0.40), so the silhouette reads as a vase, never
    # a bucket.
    belly_r = ctx.uniform(0.20, 0.30)
    foot_r = belly_r * ctx.uniform(0.45, 0.60)
    neck_r = belly_r * ctx.uniform(0.30, 0.40)
    total_h = ctx.uniform(0.35, 0.55)
    # Station heights as fractions of total height (starts ON the floor).
    fracs = (0.0, 0.07, 0.25, 0.45, 0.62, 0.74, 0.82, 0.88, 1.00)
    # Radius at each station: narrow foot, swell to belly, taper to neck,
    # slight flare at the lip.
    radii = (foot_r * 0.9, foot_r, belly_r * 0.85, belly_r,
             belly_r * 0.92, belly_r * 0.62, neck_r * 1.15,
             neck_r, neck_r * 1.08)
    # 8 stacked frusta between the 9 stations; radius → radius2 along +Y.
    y = 0.0
    for i in range(8):
        y0 = fracs[i] * total_h
        y1 = fracs[i + 1] * total_h
        seg_h = y1 - y0
        if seg_h <= 1e-6:
            continue
        ctx.add("tube", (0, 0, 0),
                {"path": [[0.0, y0, 0.0], [0.0, y1, 0.0]],
                 "radius": radii[i], "radius2": radii[i + 1],
                 "caps": i == 0, "height": seg_h},
                f"lathe_{i}", material=ceramic)
        y = y1
    # Rim: torus lip capping the mouth.
    ctx.add("torus", (0, total_h, 0),
            {"major_radius": radii[-1], "minor_radius": belly_r * 0.06},
            "rim", material=ceramic)

    # Optional handles: mirrored tori on the shoulder.
    if ctx.room() > 2 and ctx.maybe(0.5):
        handle_kind = ctx.pick_kind("tube", "torus")
        hy = total_h * 0.70
        for sx_ in (-1, 1):
            ctx.add(handle_kind, (sx_ * (belly_r + 0.04), hy, 0),
                    {"major_radius": belly_r * 0.35,
                     "minor_radius": belly_r * 0.07,
                     "radius": belly_r * 0.07, "height": belly_r * 0.7},
                    f"handle_{sx_}", rz=math.pi / 2, ry=math.pi / 2,
                    material=ceramic)

    # Throwing ridges + micro roughness: a hand-thrown pot is never polished.
    ctx.add_feature("ridges", "all", count=ctx.randint(8, 16), depth=0.003)
    ctx.add_feature("asperity", "all", strength=0.0008, frequency=45.0)


# ----------------------------------------------------------------------
# registry (CR_Quality additions; appended so earlier bindings stay valid)
# ----------------------------------------------------------------------

FAMILY_BUILDERS.update({
    "vessel": build_vessel,
    "insect": build_insect,
    "flower": build_flower,
    "leaf": build_leaf,
    "terrain": build_terrain,
})


# ======================================================================
# CR_Integrator extension — cross-module family adapters
#
# The landed feature modules (human_anatomy / building_arch / flora_params /
# terrain_styles / water / vehicle_design) expose one-call builders with
# their own result types. The adapters below wrap them into the
# FamilyContext grammar so the style engine can route prompts to them:
#
#   human           -> human_anatomy.build_human    (AABB-ellipsoid proxy)
#   building        -> building_arch.build_building (decor-trimmed to budget)
#   flora_param     -> flora_params flora builders  (budget-trimmed)
#   water_container -> water.water_container_spec   (fluid extras preserved)
#   boulder_field / rock_strata_cliff / cobblestone_patch / cracked_mud /
#   mossy_stones / pebble_riverbed / stone_slab_pavement
#                   -> terrain_styles builders      (one family per sub-style)
#   vehicle         -> vehicle_design.build_vehicle (AABB-ellipsoid proxy)
#
# Every adapter honors ctx.room() (the engine's complexity budget): parts
# are emitted in priority order and trimmed so the part-count invariant the
# style-engine tests enforce keeps holding. Human/vehicle are mesh-native
# builders; their style-engine proxies fit one ellipsoid per major part
# AABB (labels + per-part albedo via the compositor's params["color"] tint
# hook). Call the modules' own builders directly for full-fidelity meshes.
# ======================================================================


def _seed_from(ctx: FamilyContext) -> int:
    """Derive a deterministic child seed from the context RNG."""
    return int(ctx.rng.integers(0, 2**31 - 1))


def _trim(prims: list[Primitive], room: int, rank) -> list[Primitive]:
    """Keep at most `room` primitives, chosen by `rank(prim, index)` (low
    wins), preserving original emission order among the survivors."""
    room = max(int(room), 0)
    if len(prims) <= room:
        return list(prims)
    order = sorted(range(len(prims)), key=lambda i: rank(prims[i], i))
    return [prims[i] for i in sorted(order[:room])]


def _fit_aabb_ellipsoid(ctx: FamilyContext, name: str, lo, hi, *,
                        material: str, color=None) -> None:
    """One ellipsoid primitive fitted to a world-space AABB."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    size = np.maximum(hi - lo, 2e-3)
    centre = (lo + hi) / 2.0
    params: dict = {"radii": (size / 2.0).tolist()}
    if color is not None:
        params["color"] = [float(np.clip(c, 0.0, 1.0)) for c in color]
    ctx.add("ellipsoid", tuple(float(c) for c in centre), params, name,
            material=material)


# ----------------------------------------------------------------------
# human — build_human proxy (major bones + garments + hair shell)
# ----------------------------------------------------------------------

_HUMAN_BONE_PRIORITY = (
    "head", "chest", "pelvis", "spine", "neck",
    "upper_leg_l", "upper_leg_r", "lower_leg_l", "lower_leg_r",
    "foot_l", "foot_r",
    "upper_arm_l", "upper_arm_r", "lower_arm_l", "lower_arm_r",
    "hand_l", "hand_r", "clavicle_l", "clavicle_r",
)

_HUMAN_SKIP_TOKENS = (
    "finger", "toe", "nail", "iris", "pupil", "teeth", "nostril", "mouth",
    "eye", "ear", "nose", "jaw", "brow", "cheek", "lip", "tongue",
)


def build_human_family(ctx: FamilyContext) -> None:
    """Style-engine proxy for `human_anatomy.build_human`.

    One ellipsoid per major part AABB (19 Sim bones first, then garments,
    then a merged hair shell), each carrying the part's albedo through the
    compositor's per-part tint hook. Full-fidelity loft meshes remain
    available via `build_human()` directly.
    """
    from .hair import HAIRSTYLES
    from .human_anatomy import build_human

    seed = _seed_from(ctx)
    human = build_human(
        seed=seed,
        gender=float(ctx.rng.uniform(0.0, 1.0)),
        body_type=str(ctx.rng.choice(["slim", "average", "athletic", "heavy"])),
        hair_style=str(ctx.rng.choice(HAIRSTYLES)),
        detail="low",
    )
    aabbs = human.build().aabbs()
    albedos = human.part_albedos()

    # Merge hair sub-parts (scalp, hairline, strands, ties) into one shell.
    hair_lo = hair_hi = None
    for name in [n for n in aabbs if n.startswith("hair")]:
        lo, hi = aabbs.pop(name)
        hair_lo = lo if hair_lo is None else np.minimum(hair_lo, lo)
        hair_hi = hi if hair_hi is None else np.maximum(hair_hi, hi)
    hair_albedo = albedos.get("hair_scalp")

    ordered = [n for n in _HUMAN_BONE_PRIORITY if n in aabbs]
    ordered += sorted(
        n for n in aabbs
        if n not in _HUMAN_BONE_PRIORITY
        and not any(tok in n for tok in _HUMAN_SKIP_TOKENS)
    )
    reserve = 1 if hair_lo is not None else 0
    for name in ordered[: max(0, ctx.room() - reserve)]:
        lo, hi = aabbs[name]
        _fit_aabb_ellipsoid(ctx, name, lo, hi, material="organic",
                            color=albedos.get(name))
    if hair_lo is not None and ctx.room() > 0:
        _fit_aabb_ellipsoid(ctx, "hair", hair_lo, hair_hi, material="organic",
                            color=hair_albedo)
    ctx.shape = "human"
    ctx.extras["human"] = {
        "appearance": human.appearance,
        "proxy": "aabb_ellipsoid",
        "full_fidelity": "generation.human_anatomy.build_human",
    }


# ----------------------------------------------------------------------
# building — build_building proxy (structural first, decor trimmed)
# ----------------------------------------------------------------------


def _building_rank(prim: Primitive, i: int) -> tuple[int, int]:
    label = (prim.label or "").lower()
    if str((prim.params or {}).get("role", "")).lower() == "subtract":
        return (0, i)
    if any(k in label for k in ("slab", "roof")):
        return (1, i)
    if "door" in label:
        return (2, i)
    if any(k in label for k in ("wall", "pier", "partition", "corridor")):
        return (3, i)
    if any(k in label for k in ("lintel", "sill", "bay", "stair")):
        return (4, i)
    if any(k in label for k in ("column", "plinth", "cornice", "balcon")):
        return (5, i)
    return (6, i)   # quoins, downspouts, straps, decor


def _building_massing(ctx: FamilyContext) -> None:
    """Compact massing model for tiny complexity budgets (3-5 parts):
    foundation slab, hollow wall shell, roof slab, and — when the budget
    allows — a door opening (subtract cutter) with an ajar door leaf."""
    w = ctx.uniform(6.0, 10.0)
    d = ctx.uniform(5.0, 8.0)
    h = ctx.uniform(3.0, 4.5)
    ctx.add("box", (0, 0.15, 0), {"size": [w + 0.4, 0.3, d + 0.4]},
            "slab_f0", material="stone")
    ctx.add("box", (0, 0.3 + h / 2, 0), {"size": [w, h, d]},
            "wall_shell", material="stone")
    ctx.add("box", (0, 0.3 + h + 0.125, 0), {"size": [w + 0.3, 0.25, d + 0.3]},
            "roof_slab", material="ceramic")
    if ctx.room() >= 2:
        dh = min(2.1, h - 0.3)
        ctx.add("box", (0, 0.3 + dh / 2, d / 2 - 0.15),
                {"size": [0.9, dh, 0.5], "role": "subtract",
                 "target": "wall_shell"}, "door_opening")
        ctx.add("panel", (0.25, 0.3 + dh / 2, d / 2 + 0.03),
                {"size": [0.9, dh], "thickness": 0.04}, "door",
                ry=float(ctx.uniform(-0.5, 0.5)), material="wood")


def build_building_family(ctx: FamilyContext) -> None:
    """Style-engine proxy for `building_arch.build_building`.

    Uses the full plan→validate→compile pipeline and trims decorative parts
    (quoins / downspouts / cornices / columns) to the complexity budget;
    wall shells, slabs, roofs, doors and their subtract cutters are kept
    first. With a tiny budget a compact massing model is emitted instead.
    Hinge articulation lives on the analytic door parts (not the spec), so
    it is intentionally not re-attached here.
    """
    from . import building_arch as _ba

    room = ctx.room()
    if room < 12:
        _building_massing(ctx)
        ctx.shape = "building"
        return
    seed = _seed_from(ctx)
    style = str(ctx.rng.choice(["neoclassical", "modern", "baroque"]))
    floors = 1 if room < 36 else int(ctx.rng.integers(1, 3))
    res = _ba.build_building({
        "seed": seed, "floors": floors, "style": style,
        "interiors": True, "furniture": False,
        "balcony": bool(ctx.maybe(0.5)),
    })
    spec = res["spec"]
    ctx.primitives.extend(_trim(spec.primitives, room, _building_rank))
    ctx.features.extend(spec.features)
    if spec.color:
        ctx.color = tuple(spec.color)
    ctx.extras["building"] = {
        "style": style, "floors": floors,
        "validation": res["validation"],
        "full_fidelity": "generation.building_arch.build_building",
    }
    ctx.shape = "building"


# ----------------------------------------------------------------------
# flora_param — flora_params builders (budget-trimmed)
# ----------------------------------------------------------------------


def build_flora_param_family(ctx: FamilyContext) -> None:
    """Style-engine proxy for `flora_params.flora_spec`.

    A random species (or a grass patch for tiny budgets) is grown into the
    context and trimmed to the complexity budget — trunk/stem/ground parts
    are emitted first by the flora grammars, so keep-first trimming degrades
    gracefully. The resolved `flora` manifest block is preserved.
    """
    from .flora_params import FloraParams, SPECIES, flora_spec

    room = ctx.room()
    seed = _seed_from(ctx)
    if room < 12:
        style = str(ctx.rng.choice(["meadow", "lawn"]))
    else:
        style = str(ctx.rng.choice(sorted(SPECIES)))
    p = FloraParams(style=style, density=float(ctx.uniform(0.3, 0.8)),
                    season=str(ctx.rng.choice(["spring", "summer", "autumn"])),
                    seed=seed)
    spec = flora_spec(p)
    ctx.primitives.extend(_trim(spec.primitives, room, lambda _p, i: (0, i)))
    ctx.features.extend(spec.features)
    if spec.color:
        ctx.color = tuple(spec.color)
    ctx.extras.update(getattr(spec, "manifest_extras", None) or {})
    ctx.shape = f"flora_{p.kind}"


# ----------------------------------------------------------------------
# water_container — water builders (fluid extras preserved)
# ----------------------------------------------------------------------

# Rough part counts per container kind (floor+wall+rim+water+meniscus …).
_WATER_PART_COUNT = {"basin": 5, "bucket": 7, "aquarium": 7, "vessel": 8,
                     "pond": 11}


def _water_rank(prim: Primitive, i: int) -> tuple[int, int]:
    label = (prim.label or "").lower()
    if label == "water":
        return (0, i)          # the point of the family — never trimmed
    if any(k in label for k in ("floor", "wall", "belly", "tank")):
        return (1, i)
    if "meniscus" in label:
        return (2, i)
    return (3, i)              # rims, handles, decor


def build_water_container_family(ctx: FamilyContext) -> None:
    """Style-engine proxy for `water.water_container_spec`.

    Container + water body + meniscus, with the ``fluid`` extras block
    preserved on ctx.extras (it reaches the iemodel/3 manifest through the
    passthrough). Kind is chosen to fit the complexity budget; optional
    parts (rim, handle, meniscus) are trimmed first."""
    from . import water as _water

    room = ctx.room()
    seed = _seed_from(ctx)
    fits = [k for k in _water.WATER_CONTAINERS
            if _WATER_PART_COUNT.get(k, 8) <= max(room, 3)]
    kind = str(ctx.rng.choice(fits or ["basin"]))
    spec = _water.water_container_spec(
        kind, fill_level=float(ctx.uniform(0.4, 0.9)), seed=seed)
    ctx.primitives.extend(_trim(spec.primitives, room, _water_rank))
    ctx.features.extend(spec.features)
    if spec.color:
        ctx.color = tuple(spec.color)
    ctx.extras.update(getattr(spec, "manifest_extras", None) or {})
    ctx.shape = f"water_{kind}"


# ----------------------------------------------------------------------
# terrain sub-styles — one family per terrain_styles style
# ----------------------------------------------------------------------


def _make_terrain_family(style: str):
    """Style-engine proxy for one `terrain_styles` sub-style."""
    def build(ctx: FamilyContext) -> None:
        from .terrain_styles import TERRAIN_STYLE_BUILDERS, TerrainParams

        seed = _seed_from(ctx)
        p = TerrainParams(style=style, density=float(ctx.uniform(0.3, 0.8)),
                          seed=seed)
        TERRAIN_STYLE_BUILDERS[style](ctx, p)
        # Budget trim: the ground slab is emitted first, scatter parts later.
        del ctx.primitives[max(int(ctx.target_parts), 3):]
        ctx.extras["terrain"] = {**p.to_dict(),
                                 "proxy": "terrain_styles.terrain_spec"}
        ctx.shape = f"terrain_{style}"
    build.__name__ = f"build_terrain_{style}"
    build.__doc__ = f"Style-engine proxy for the {style!r} terrain sub-style."
    return build


build_boulder_field_family = _make_terrain_family("boulder_field")
build_rock_strata_cliff_family = _make_terrain_family("rock_strata_cliff")
build_cobblestone_patch_family = _make_terrain_family("cobblestone_patch")
build_cracked_mud_family = _make_terrain_family("cracked_mud")
build_mossy_stones_family = _make_terrain_family("mossy_stones")
build_pebble_riverbed_family = _make_terrain_family("pebble_riverbed")
build_stone_slab_pavement_family = _make_terrain_family("stone_slab_pavement")


# ----------------------------------------------------------------------
# vehicle — build_vehicle proxy (top-volume parts as AABB ellipsoids)
# ----------------------------------------------------------------------


def build_vehicle_family(ctx: FamilyContext) -> None:
    """Style-engine proxy for `vehicle_design.build_vehicle` (defensive
    registration — CR_Vehicle owns the module). One ellipsoid per major
    part AABB, largest solid volume first, trimmed to the complexity
    budget; the body takes the paint color. Full-fidelity meshes and the
    hinge articulation remain available via `build_vehicle()` directly.
    """
    from .vehicle_design import VEHICLE_CLASSES, build_vehicle

    seed = _seed_from(ctx)
    cls = str(ctx.rng.choice(sorted(VEHICLE_CLASSES)))
    vs = build_vehicle({"seed": seed, "class": cls,
                        "lod": "low", "interior_detail": "low"})

    groups: dict[str, dict] = {}
    for part in vs.parts:
        g = groups.setdefault(part.name, {
            "lo": np.full(3, np.inf), "hi": np.full(3, -np.inf),
            "vol": 0.0, "material": part.material or "metal"})
        g["lo"] = np.minimum(g["lo"], part.aabb_min)
        g["hi"] = np.maximum(g["hi"], part.aabb_max)
        g["vol"] += float(part.solid_volume_m3)
    ordered = sorted(groups.items(), key=lambda kv: -kv[1]["vol"])
    for name, g in ordered[: ctx.room()]:
        _fit_aabb_ellipsoid(ctx, name, g["lo"], g["hi"],
                            material=str(g["material"]))
    ctx.shape = "vehicle"
    ctx.extras["vehicle"] = {
        "class": cls,
        "proxy": "aabb_ellipsoid",
        "full_fidelity": "generation.vehicle_design.build_vehicle",
    }


# ----------------------------------------------------------------------
# registry (CR_Integrator additions; appended so earlier bindings stay
# valid — FAMILY_KEYWORDS / _FAMILY_WEIGHTS live in style_engine.py)
# ----------------------------------------------------------------------

FAMILY_BUILDERS.update({
    "human": build_human_family,
    "building": build_building_family,
    "flora_param": build_flora_param_family,
    "water_container": build_water_container_family,
    "boulder_field": build_boulder_field_family,
    "rock_strata_cliff": build_rock_strata_cliff_family,
    "cobblestone_patch": build_cobblestone_patch_family,
    "cracked_mud": build_cracked_mud_family,
    "mossy_stones": build_mossy_stones_family,
    "pebble_riverbed": build_pebble_riverbed_family,
    "stone_slab_pavement": build_stone_slab_pavement_family,
    "vehicle": build_vehicle_family,
})
