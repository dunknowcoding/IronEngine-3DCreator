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
    top_h = ctx.uniform(0.05, 0.09)
    leg_h = ctx.uniform(0.40, 0.55) if archetype != "table" else ctx.uniform(0.65, 0.80)
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

    # Legs: 2 or 4 mirrored pairs of capsules.
    n_pairs = 2 if ctx.maybe(0.65) else 1
    for pair in range(n_pairs):
        z = (bl * 0.55) * (1 if pair == 0 else -1)
        leg_len = body_y - bh * 0.4
        leg_r = ctx.uniform(0.05, 0.08)
        for sx_ in (-1, 1):
            ctx.add("capsule", (sx_ * bw * 0.7, leg_len / 2, z),
                    {"radius": leg_r, "height": leg_len},
                    f"leg_{pair}_{sx_}", material=skin)

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
