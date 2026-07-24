"""Seeded procedural style engine — frees auto mode from fixed templates.

Public API:
- ``StyleEngine(seed)`` — deterministic generator. ``generate()`` assembles a
  full `GenerationSpec` from one of the style families in
  `generation.style_families`, honoring complexity and bbox budgets.
- ``family_from_prompt(prompt)`` — keyword routing from a free-form prompt
  (or shape hint) to a family, or None when nothing matches.
- ``weighted_random_family(rng)`` — family sampler for 'random' style.
- ``mutate_spec(spec, seed)`` — seeded style mutation applied to an existing
  (e.g. LLM-produced) spec so repeated prompts don't yield identical objects.
- ``diversity_report(n, seed0)`` — headless diversity statistics.

Everything is driven by explicit seeds: same seed → byte-identical spec JSON.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..alignment.schema import GenerationSpec, Primitive
from .style_families import FAMILY_BUILDERS, FamilyContext

_log = logging.getLogger(__name__)

STYLE_FAMILIES: tuple[str, ...] = tuple(FAMILY_BUILDERS.keys())
COMPLEXITY_LEVELS: tuple[str, ...] = ("auto", "simple", "complex")

# Relative family weights for the 'random' style — abstract and creature are
# slightly favored because they showcase the curved primitives best.
_FAMILY_WEIGHTS: dict[str, float] = {
    "furniture": 1.0,
    "creature": 1.2,
    "mechanical": 1.0,
    "architecture": 0.9,
    "plant": 1.0,
    "vessel": 1.0,
    "abstract": 1.2,
    "insect": 0.8,
    "flower": 0.8,
    "leaf": 0.6,
    "terrain": 0.7,
    "rococo_fence": 0.6,
    "neoclassical_column": 0.6,
    "modern_luxury": 0.5,
    "futurist_chair": 0.5,
    "desktop_computer": 0.5,
    "spaceship": 0.5,
    "robot": 0.5,
    # CR_Integrator families — heavy/rare objects get low draw weights.
    "human": 0.5,
    "building": 0.5,
    "water_container": 0.6,
    "vehicle": 0.5,
    "flora_param": 0.7,
    "boulder_field": 0.25,
    "rock_strata_cliff": 0.2,
    "cobblestone_patch": 0.2,
    "cracked_mud": 0.2,
    "mossy_stones": 0.2,
    "pebble_riverbed": 0.2,
    "stone_slab_pavement": 0.2,
}

# Keyword routing: lowercase substrings → family. Checked in order; the first
# family with any hit wins, ties broken by earliest match position (dict
# order breaks same-position ties, so "building" precedes "architecture" and
# "vehicle" precedes "mechanical").
FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "human": ("human", "woman", "man", "person", "people", "girl", "boy",
              "lady", "gentleman", "female", "male", "ponytail", "ponytails",
              "anatomy", "bride"),
    "building": ("building", "house", "cottage", "villa", "mansion",
                 "bungalow", "cabin", "apartment", "townhouse", "duplex",
                 "floor plan", "rooms", "storey"),
    "water_container": ("pond", "aquarium", "basin", "bucket", "birdbath",
                        "water tank", "water"),
    "vehicle": ("vehicle", "car", "sedan", "suv", "hatchback", "notchback",
                "sports car", "automobile", "wheels"),
    "furniture": ("chair", "table", "stool", "bench", "desk", "sofa", "couch",
                  "shelf", "cabinet", "furniture", "seat", "throne"),
    "futurist_chair": ("futuristic chair", "futurist chair", "futuristic",
                       "futurist"),
    "desktop_computer": ("computer", "desktop", "monitor", "keyboard",
                         "laptop", "pc", "workstation"),
    "spaceship": ("spaceship", "spacecraft", "starship", "rocket", "shuttle",
                  "ufo", "satellite", "space station"),
    "robot": ("robot", "robotic", "android", "humanoid", "droid", "automaton",
              "cyborg", "bot"),
    "rococo_fence": ("rococo", "fence", "railing", "balustrade", "lattice",
                     "trellis", "picket", "wrought iron"),
    "neoclassical_column": ("neoclassical", "doric", "ionic", "corinthian",
                            "capital", "pilaster"),
    "modern_luxury": ("luxury", "luxe", "premium", "penthouse"),
    "insect": ("insect", "bug", "ladybug", "ladybird", "beetle", "ant",
               "bee", "wasp", "butterfly", "moth", "dragonfly", "firefly",
               "grasshopper", "cricket", "caterpillar"),
    "flower": ("flower", "blossom", "bloom", "rose", "tulip", "daisy",
               "sunflower", "lily", "orchid", "petal", "bouquet"),
    "leaf": ("leaf", "leaves", "foliage", "frond"),
    "terrain": ("soil", "terrain", "ground", "mud", "dirt", "gravel",
                "lawn", "field", "landscape", "pebble"),
    "creature": ("creature", "animal", "beast", "monster", "cat", "dog", "bird",
                 "rabbit", "bunny", "dragon", "spider", "critter",
                 "pet", "fox", "bear", "turtle", "frog"),
    "mechanical": ("machine", "mechanical", "gear", "engine", "motor",
                   "piston", "vehicle", "car", "tank", "clockwork", "mech",
                   "drone", "turbine"),
    "architecture": ("arch", "archway", "column", "pillar", "temple", "bridge",
                     "tower", "building", "facade", "gate", "monument",
                     "colonnade", "ruin", "castle", "pyramid"),
    "plant": ("tree", "plant", "bush", "shrub", "fern", "bonsai",
              "cactus", "branch", "vine", "mushroom", "forest"),
    "vessel": ("vase", "pot", "jar", "jug", "cup", "mug", "bottle", "bowl",
               "urn", "pitcher", "kettle", "vessel", "container", "amphora",
               "chalice", "goblet"),
    # CR_Integrator families with non-conflicting keywords (ties impossible).
    "flora_param": ("oak", "maple", "pine", "palm", "grass", "meadow",
                    "lavender", "prairie"),
    "boulder_field": ("boulder", "boulders", "boulder field"),
    "rock_strata_cliff": ("strata", "cliff", "rock strata"),
    "cobblestone_patch": ("cobblestone", "cobbles", "cobble"),
    "cracked_mud": ("cracked mud", "mud cracks", "cracked earth"),
    "mossy_stones": ("mossy", "moss", "mossy stones"),
    "pebble_riverbed": ("riverbed", "river bed"),
    "stone_slab_pavement": ("pavement", "slab pavement", "flagstone"),
}

# Part-count targets per complexity level (before budget clamping).
_COMPLEXITY_PARTS: dict[str, tuple[int, int]] = {
    "simple": (3, 8),
    "auto": (6, 18),
    "complex": (15, 40),
}

MAX_PARTS = 40


# ----------------------------------------------------------------------
# routing
# ----------------------------------------------------------------------

def family_from_prompt(prompt: str | None) -> str | None:
    """Map free-form text to a style family via keyword matching.

    Keywords match on word boundaries, with a plural allowance: "gear" hits
    "gears" but "pot" does not hit "potted" (so "potted fern" routes to
    plant). Returns None when nothing matches (caller falls back to weighted
    random).
    """
    if not prompt:
        return None
    text = prompt.lower()

    def _matches(kw: str, pos: int) -> bool:
        left_ok = pos == 0 or not (text[pos - 1].isalnum() or text[pos - 1] == "_")
        end = pos + len(kw)
        right_ok = end >= len(text) or text[end] == "s" or not text[end].isalnum()
        return left_ok and right_ok

    best: tuple[int, str] | None = None  # (earliest match position, family)
    for family, keywords in FAMILY_KEYWORDS.items():
        for kw in keywords:
            pos = 0
            while True:
                pos = text.find(kw, pos)
                if pos < 0:
                    break
                if _matches(kw, pos):
                    if best is None or pos < best[0]:
                        best = (pos, family)
                    break
                pos += 1
    return best[1] if best else None


def weighted_random_family(rng: np.random.Generator) -> str:
    families = list(STYLE_FAMILIES)
    w = np.asarray([_FAMILY_WEIGHTS.get(f, 1.0) for f in families], dtype=np.float64)
    w /= w.sum()
    return families[int(rng.choice(len(families), p=w))]


# ----------------------------------------------------------------------
# geometry fitting
# ----------------------------------------------------------------------

def _local_half_extent(kind: str, params: dict) -> tuple[float, float, float]:
    """Conservative local-space half extents of a primitive kind."""
    g = params.get
    if kind == "box":
        s = g("size", [1, 1, 1])
        s = (s + [1, 1, 1])[:3] if len(s) < 3 else s
        return (s[0] / 2, s[1] / 2, s[2] / 2)
    if kind == "panel":
        # Panel size is the 2-element in-plane [w, l]; thickness is separate.
        s = g("size", [1, 1])
        t = float(g("thickness", 0.02))
        return (s[0] / 2, (s[1] if len(s) > 1 else s[0]) / 2, t / 2)
    if kind == "sphere":
        r = float(g("radius", 0.5))
        return (r, r, r)
    if kind in ("cylinder", "tube"):
        r = float(g("radius", 0.4))
        return (r, float(g("height", 1.0)) / 2, r)
    if kind == "capsule":
        r = float(g("radius", 0.3))
        return (r, float(g("height", 1.0)) / 2 + r, r)
    if kind == "cone":
        r = float(g("radius", 0.5))
        return (r, float(g("height", 1.0)) / 2, r)
    if kind == "torus":
        rr = float(g("major_radius", 0.5)) + float(g("minor_radius", 0.15))
        return (rr, float(g("minor_radius", 0.15)), rr)
    if kind == "arch":
        rr = float(g("major_radius", g("radius", 0.5))) + float(g("minor_radius", g("thickness", 0.15)) / 2)
        return (rr, rr, rr)
    if kind in ("ellipsoid", "superellipsoid"):
        radii = g("radii", [0.5, 0.5, 0.5])
        return (float(radii[0]), float(radii[1]), float(radii[2]))
    if kind == "prism":
        r = float(g("radius", 0.5))
        return (r, float(g("height", 1.0)) / 2, r)
    if kind == "helix":
        rr = float(g("radius", 0.4)) + float(g("thickness", 0.05))
        hh = float(g("pitch", 0.2)) * float(g("turns", 3.0)) / 2 + float(g("thickness", 0.05))
        return (rr, hh, rr)
    if kind == "plane":
        s = g("size", [1, 1])
        return (s[0] / 2, 0.0, (s[1] if len(s) > 1 else s[0]) / 2)
    return (0.5, 0.5, 0.5)  # defensive fallback for unknown future kinds


def _union_aabb(primitives: list[Primitive]) -> tuple[np.ndarray, np.ndarray]:
    """World-space AABB of the whole assembly (exact for transformed boxes,
    conservative for rotated curved parts)."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for p in primitives:
        T = np.asarray(p.transform, dtype=np.float64)
        path = (p.params or {}).get("path")
        if p.kind in ("tube", "sweep") and path:
            # Path-based parts: exact AABB of the swept tube = for each
            # centerline segment, endpoint range inflated by the radius in
            # the directions PERPENDICULAR to the segment (inflating along
            # the axis too would overestimate the extent by r at each end
            # and misground the whole assembly).
            pts = np.asarray(path, dtype=np.float64).reshape(-1, 3)
            r = max(float(p.params.get("radius", 0.05)),
                    float(p.params.get("radius2", 0.0) or 0.0))
            world = pts @ T[:3, :3].T + T[:3, 3]
            for a, b in zip(world[:-1], world[1:]):
                d = b - a
                n = float(np.linalg.norm(d))
                d = d / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])
                perp = r * (1.0 - np.abs(d))
                lo = np.minimum(lo, np.minimum(a, b) - perp)
                hi = np.maximum(hi, np.maximum(a, b) + perp)
            continue
        h = np.asarray(_local_half_extent(p.kind, p.params))
        half = np.abs(T[:3, :3]) @ h
        c = T[:3, 3]
        lo = np.minimum(lo, c - half)
        hi = np.maximum(hi, c + half)
    if not np.isfinite(lo).all():
        return np.zeros(3), np.ones(3)
    return lo, hi


def _fit_to_bbox(primitives: list[Primitive], bbox: tuple[float, float, float],
                 fill: float = 0.92) -> None:
    """Uniformly scale + recenter the assembly (in place) so it fits `bbox`.

    X/Z are centered on the origin; Y rests on the ground plane (y = 0).
    """
    lo, hi = _union_aabb(primitives)
    extent = np.maximum(hi - lo, 1e-3)
    target = np.asarray(bbox, dtype=np.float64) * fill
    s = float((target / extent).min())
    for p in primitives:
        T = np.asarray(p.transform, dtype=np.float64)
        T[:3, :3] *= s
        T[:3, 3] *= s
        p.transform = T.tolist()
    lo2, hi2 = _union_aabb(primitives)
    offset = np.array([-(lo2[0] + hi2[0]) / 2, -lo2[1], -(lo2[2] + hi2[2]) / 2])
    for p in primitives:
        T = np.asarray(p.transform, dtype=np.float64)
        T[:3, 3] += offset
        p.transform = T.astype(np.float32).tolist()


# ----------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------

@dataclass
class StyleEngine:
    """Deterministic seeded generator. One instance per object."""

    seed: int = 0

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed or None)

    def target_parts(self, rng: np.random.Generator, complexity: str,
                     n_points: int) -> int:
        lo, hi = _COMPLEXITY_PARTS.get(complexity, _COMPLEXITY_PARTS["auto"])
        target = int(rng.integers(lo, hi + 1))
        # Budget guard: keep at least ~800 points per part on average so tiny
        # budgets don't starve parts into invisibility.
        affordable = max(3, min(MAX_PARTS, n_points // 800))
        return max(3, min(target, affordable))

    def generate(
        self,
        family: str | None = None,
        complexity: str = "auto",
        n_points: int = 50_000,
        bbox: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> GenerationSpec:
        rng = self._rng()
        if not family or family == "random":
            family = weighted_random_family(rng)
        builder = FAMILY_BUILDERS.get(family)
        if builder is None:
            _log.warning("unknown style family %r — falling back to abstract", family)
            builder = FAMILY_BUILDERS["abstract"]
            family = "abstract"

        ctx = FamilyContext(rng=rng, target_parts=self.target_parts(rng, complexity, n_points))
        builder(ctx)
        _fit_to_bbox(ctx.primitives, bbox)

        spec = GenerationSpec(
            shape=family,
            n_points=int(n_points),
            bbox_size=(float(bbox[0]), float(bbox[1]), float(bbox[2])),
            primitives=ctx.primitives,
            features=ctx.features,
            color=ctx.color,
            seed=int(self.seed or 0),
        )
        if ctx.extras:
            # Manifest extras side-channel (fluid / flora / terrain / proxy
            # metadata from the cross-module family adapters).
            spec.manifest_extras = dict(ctx.extras)
        return spec


def generate_style_spec(
    seed: int,
    family: str | None = None,
    complexity: str = "auto",
    n_points: int = 50_000,
    bbox: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> GenerationSpec:
    """Convenience wrapper around ``StyleEngine(seed).generate(...)``."""
    return StyleEngine(seed=seed).generate(family, complexity, n_points, bbox)


# ----------------------------------------------------------------------
# style mutation (for LLM-produced specs)
# ----------------------------------------------------------------------

def mutate_spec(spec: GenerationSpec, seed: int = 0,
                strength: float = 0.12) -> GenerationSpec:
    """Return a seeded style variation of `spec`.

    Structure is preserved (same kinds, labels, part count, feature kinds);
    numeric params, colors, and small placement jitter vary. With a nonzero
    `seed` the mutation is deterministic; with seed=0 fresh entropy is drawn
    so repeated identical prompts don't yield identical objects.
    """
    rng = np.random.default_rng(seed or None)

    def _jitter(v: float, rel: float = strength, lo: float = 1e-4) -> float:
        return float(max(lo, v * (1.0 + rng.uniform(-rel, rel))))

    def _jitter_value(v: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return _jitter(float(v))
        if isinstance(v, (list, tuple)):
            return [ _jitter_value(x) for x in v ]
        return v

    new_prims: list[Primitive] = []
    for p in spec.primitives:
        params = {k: _jitter_value(v) for k, v in (p.params or {}).items()}
        T = np.asarray(p.transform, dtype=np.float64)
        # Small placement jitter: translation ±strength * 0.1 m, yaw ±strength rad.
        T[:3, 3] += rng.uniform(-strength * 0.1, strength * 0.1, 3)
        yaw = float(rng.uniform(-strength, strength))
        cy, sy = math.cos(yaw), math.sin(yaw)
        rot = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        T[:3, :3] = rot @ T[:3, :3]
        new_prims.append(Primitive(kind=p.kind, transform=T.astype(np.float32).tolist(),
                                   params=params, label=p.label))

    new_feats = []
    for f in spec.features:
        new_feats.append(type(f)(kind=f.kind, region=f.region,
                                 params={k: _jitter_value(v) for k, v in (f.params or {}).items()}))

    color = spec.color
    if color is not None:
        color = tuple(float(np.clip(c + rng.uniform(-strength, strength), 0.0, 1.0))
                      for c in color)

    return GenerationSpec(
        shape=spec.shape,
        n_points=spec.n_points,
        bbox_size=spec.bbox_size,
        primitives=new_prims,
        features=new_feats,
        color=color,
        seed=int(seed or 0),
    )


# ----------------------------------------------------------------------
# diversity report (headless proof)
# ----------------------------------------------------------------------

def diversity_report(n: int = 20, seed0: int = 10_000,
                     n_points: int = 12_000) -> list[dict[str, Any]]:
    """Generate `n` seeded objects across families and return per-object stats.

    Families are cycled round-robin so every family is exercised; seeds differ
    per object. Used by tests and by manual diversity checks.
    """
    from ..alignment.validator import normalize

    stats: list[dict[str, Any]] = []
    for i in range(n):
        family = STYLE_FAMILIES[i % len(STYLE_FAMILIES)]
        seed = seed0 + i
        spec = StyleEngine(seed=seed).generate(family=family, n_points=n_points)
        clean, warns = normalize(spec)
        kinds = sorted({p.kind for p in clean.primitives})
        mats = sorted({(p.params or {}).get("material", "?") for p in clean.primitives})
        lo, hi = _union_aabb(clean.primitives)
        stats.append({
            "i": i,
            "seed": seed,
            "family": family,
            "shape": clean.shape,
            "n_parts": len(clean.primitives),
            "kinds": kinds,
            "n_kinds": len(kinds),
            "materials": mats,
            "features": sorted({f.kind for f in clean.features}),
            "bbox_used": [round(float(hi[j] - lo[j]), 3) for j in range(3)],
            "color": [round(c, 3) for c in (clean.color or (0, 0, 0))],
            "validator_warnings": [w for w in warns if "dropped" in w],
        })
    return stats
