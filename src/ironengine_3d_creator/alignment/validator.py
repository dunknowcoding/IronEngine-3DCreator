"""Validate and normalize a GenerationSpec.

Applies hard caps, drops unknown primitive/feature kinds, fills missing
required params with safe defaults, and ensures `n_points >= 100` so
preview always has something to show.

For the newer complex-geometry kinds (superellipsoid / tube / sweep / arch /
panel) we additionally *clamp* malformed params (negative radii, absurd
exponents, single-point paths, extreme bends) instead of dropping the
primitive — the LLM's intent is usually recoverable.

Finally, per-category *proportion guides* (chair-leg diameter 0.02–0.06 m,
table-top thickness 0.02–0.05 m, …) are applied as soft clamps with
warnings: the LLM's part is kept, only its offending thickness axis is
rescaled into the plausible range.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from .schema import (
    FEATURE_KINDS,
    PRIMITIVE_KINDS,
    Feature,
    GenerationSpec,
    Primitive,
)

_log = logging.getLogger(__name__)

MIN_POINTS = 100
MAX_POINTS = 500_000


_PARAM_DEFAULTS: dict[str, dict] = {
    "box": {"size": [1.0, 1.0, 1.0]},
    "sphere": {"radius": 0.5},
    "cylinder": {"radius": 0.4, "height": 1.0, "caps": True},
    "capsule": {"radius": 0.3, "height": 1.0},
    "cone": {"radius": 0.5, "height": 1.0},
    "torus": {"major_radius": 0.5, "minor_radius": 0.15},
    "ellipsoid": {"radii": [0.6, 0.4, 0.4]},
    "prism": {"sides": 6, "radius": 0.5, "height": 1.0},
    "helix": {"radius": 0.4, "pitch": 0.2, "turns": 3.0, "thickness": 0.05},
    "plane": {"size": [1.0, 1.0]},
    "superellipsoid": {"radii": [0.5, 0.4, 0.45], "exponents": [0.7, 0.7]},
    "tube": {"radius": 0.05, "caps": True},
    "sweep": {"radius": 0.05, "caps": True},
    "arch": {"major_radius": 0.5, "minor_radius": 0.1, "arc": math.pi,
             "start_angle": 0.0, "caps": True},
    "panel": {"size": [1.0, 1.0], "thickness": 0.02, "bend": 0.0},
}


def _as_float(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _as_float_list(value, n: int, default: list[float]) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == n:
        out = [_as_float(v, float("nan")) for v in value]
        if all(math.isfinite(v) for v in out):
            return out
    return list(default)


def _clamp_params(kind: str, params: dict, warnings: list[str]) -> dict:
    """Repair malformed params for the complex-geometry kinds.

    Old kinds are intentionally left untouched (historic behaviour); for the
    new kinds the LLM routinely emits out-of-range values that would blow up
    the samplers, so we clamp into the valid domain with a warning instead
    of dropping the primitive.
    """
    d = _PARAM_DEFAULTS[kind]
    if kind == "superellipsoid":
        radii = _as_float_list(params.get("radii"), 3, d["radii"])
        clamped = [max(1e-4, min(10.0, r)) for r in radii]
        if clamped != radii:
            warnings.append(f"validator: superellipsoid radii {radii} → clamped to {clamped}")
        exps = _as_float_list(params.get("exponents"), 2, d["exponents"])
        clamped_e = [max(0.05, min(4.0, e)) for e in exps]
        if clamped_e != exps:
            warnings.append(f"validator: superellipsoid exponents {exps} → clamped to {clamped_e}")
        params["radii"], params["exponents"] = clamped, clamped_e
    elif kind in ("tube", "sweep"):
        r1 = max(1e-4, min(5.0, _as_float(params.get("radius"), d["radius"])))
        if r1 != params.get("radius"):
            warnings.append(f"validator: {kind} radius → clamped to {r1}")
        params["radius"] = r1
        if params.get("radius2") is not None:
            params["radius2"] = max(1e-4, min(5.0, _as_float(params["radius2"], r1)))
        path = params.get("path")
        if path is not None:
            ok = (
                isinstance(path, (list, tuple)) and len(path) >= 2
                and all(
                    isinstance(pt, (list, tuple)) and len(pt) == 3
                    and all(math.isfinite(_as_float(c, float("nan"))) for c in pt)
                    for pt in path
                )
            )
            if not ok:
                warnings.append(f"validator: {kind} path malformed → using straight bar fallback")
                params.pop("path", None)
                params["height"] = max(1e-3, _as_float(params.get("height"), 1.0))
    elif kind == "arch":
        R = max(1e-3, _as_float(params.get("major_radius"), d["major_radius"]))
        r = max(1e-4, _as_float(params.get("minor_radius"), d["minor_radius"]))
        if r >= R:
            warnings.append(f"validator: arch minor_radius {r} ≥ major_radius {R} → halved")
            r = R * 0.5
        arc = _as_float(params.get("arc"), d["arc"])
        arc = max(0.05, min(2.0 * math.pi, arc))
        if arc != params.get("arc"):
            warnings.append(f"validator: arch arc → clamped to {arc:.3f} rad")
        params.update(major_radius=R, minor_radius=r, arc=arc,
                      start_angle=_as_float(params.get("start_angle"), 0.0))
    elif kind == "panel":
        size = _as_float_list(params.get("size"), 2, d["size"])
        size = [max(1e-3, min(50.0, s)) for s in size]
        t = max(1e-4, min(1.0, _as_float(params.get("thickness"), d["thickness"])))
        bend = _as_float(params.get("bend"), d["bend"])
        clamped_bend = max(-0.95 * math.pi, min(0.95 * math.pi, bend))
        if clamped_bend != bend:
            warnings.append(f"validator: panel bend {bend:.3f} → clamped to {clamped_bend:.3f} rad")
        params.update(size=size, thickness=t, bend=clamped_bend)
    return params


# ---------------------------------------------------------------- proportion guides
#
# Per-category plausible thickness ranges (metres) measured along the part's
# own local axes (so rotated parts are handled correctly):
#   - vertical roles (leg / vbar / stem): the two non-long axes.
#   - thin roles (seat / back / rail / base): the smallest-extent axis.
# These are *soft* clamps: the part is kept and rescaled on the offending
# axis only, with a warning. Ranges follow furniture/industrial norms.

_PROPORTION_RULES: dict[str, dict[str, tuple[float, float]]] = {
    "chair":    {"leg": (0.02, 0.06), "seat": (0.02, 0.08), "back": (0.015, 0.08)},
    "stool":    {"leg": (0.02, 0.06), "seat": (0.02, 0.08)},
    "table":    {"leg": (0.03, 0.09), "seat": (0.02, 0.05)},
    "desk":     {"leg": (0.03, 0.09), "seat": (0.02, 0.05)},
    "lamp":     {"stem": (0.01, 0.04), "base": (0.02, 0.10)},
    "creature": {"leg": (0.03, 0.12)},
    "animal":   {"leg": (0.03, 0.12)},
    "quadruped": {"leg": (0.03, 0.12)},
    "tree":     {"stem": (0.04, 0.30)},
    "fence":    {"vbar": (0.01, 0.05), "rail": (0.02, 0.08)},
    "railing":  {"vbar": (0.01, 0.05), "rail": (0.02, 0.08)},
    "balustrade": {"vbar": (0.01, 0.05), "rail": (0.02, 0.08)},
    "gate":     {"vbar": (0.01, 0.05), "rail": (0.02, 0.08)},
}

_VERTICAL_ROLES = ("leg", "vbar", "stem")
_THIN_ROLES = ("seat", "back", "rail", "base")


def _apply_proportion_rules(spec: GenerationSpec, warnings: list[str]) -> None:
    """Soft-clamp part thicknesses into per-category plausible ranges."""
    from .integrity import _classify_role, _local_aabb  # lazy: no import cycle

    rules = _PROPORTION_RULES.get((spec.shape or "").lower())
    if not rules:
        return
    for prim in spec.primitives:
        if (prim.params or {}).get("role") == "subtract":
            continue  # cutter dimensions are deliberate
        role = _classify_role(prim.label, prim.kind)
        rng = rules.get(role)
        if rng is None:
            continue
        lo, hi = _local_aabb(prim)
        ext = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
        T = np.asarray(prim.transform_matrix(), dtype=np.float64)
        scales = np.linalg.norm(T[:3, :3], axis=0)
        if np.any(scales < 1e-12):
            continue
        world = ext * scales  # thickness of the part along each local axis
        if role in _VERTICAL_ROLES:
            axes = [i for i in np.argsort(world)[::-1][1:]]  # the two non-long axes
        elif role in _THIN_ROLES:
            axes = [int(np.argmin(world))]
        else:
            continue
        changed = False
        for axis in axes:
            w = float(world[axis])
            if w < 1e-9:
                continue
            target = min(max(w, rng[0]), rng[1])
            if abs(target - w) < 1e-6:
                continue
            T[:3, axis] *= target / w
            warnings.append(
                f"validator: {prim.label or prim.kind!r} {role} thickness "
                f"{w:.3f}m → clamped to {target:.3f}m ({spec.shape} proportion guide)"
            )
            changed = True
        if changed:
            prim.transform = T.tolist()


def normalize(spec: GenerationSpec) -> tuple[GenerationSpec, list[str]]:
    """Return (clean_spec, warnings)."""
    warnings: list[str] = []

    n = int(spec.n_points)
    if n < MIN_POINTS:
        warnings.append(f"n_points={n} → clamped to {MIN_POINTS}")
        n = MIN_POINTS
    if n > MAX_POINTS:
        warnings.append(f"n_points={n} → clamped to {MAX_POINTS}")
        n = MAX_POINTS

    bbox = tuple(max(1e-3, min(50.0, float(x))) for x in spec.bbox_size)

    clean_prims: list[Primitive] = []
    for p in spec.primitives:
        kind = str(p.kind).lower()
        if kind not in PRIMITIVE_KINDS:
            warnings.append(f"unknown primitive kind {p.kind!r} dropped")
            continue
        transform = p.transform
        given = {k: v for k, v in (p.params or {}).items() if v is not None}
        if kind == "panel" and isinstance(given.get("size"), (list, tuple)) \
                and len(given["size"]) == 3:
            # Box-style slab shorthand [w, t, d] (shared box/panel call
            # sites): convert to panel-native [w, d] + thickness *before*
            # defaults are merged so the slab thickness wins over the
            # default, and lay the panel flat (local in-plane Y → world Z,
            # thickness → world Y) so it is a true drop-in for a box.
            w3 = _as_float_list(given["size"], 3, [1.0, 0.02, 1.0])
            warnings.append(
                f"validator: panel size got 3 elements {given['size']} — "
                f"interpreted as box slab [w, t, d]"
            )
            given["size"] = [w3[0], w3[2]]
            given.setdefault("thickness", w3[1])
            rot_x90 = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            transform = (np.asarray(p.transform, dtype=np.float64) @ rot_x90).tolist()
        params = dict(_PARAM_DEFAULTS[kind])
        params.update(given)
        params = _clamp_params(kind, params, warnings)
        clean_prims.append(Primitive(
            kind=kind,
            transform=transform,
            params=params,
            label=p.label,
        ))

    if not clean_prims:
        warnings.append("no valid primitives — falling back to a single sphere")
        clean_prims.append(Primitive(
            kind="sphere",
            transform=Primitive.identity_transform(),
            params=dict(_PARAM_DEFAULTS["sphere"]),
            label="fallback",
        ))

    clean_features: list[Feature] = []
    for f in spec.features:
        kind = str(f.kind).lower()
        if kind not in FEATURE_KINDS:
            warnings.append(f"unknown feature kind {f.kind!r} dropped")
            continue
        clean_features.append(Feature(kind=kind, region=f.region, params=dict(f.params or {})))

    color = spec.color
    if color is not None:
        color = tuple(max(0.0, min(1.0, float(c))) for c in color)

    seed = int(spec.seed) if spec.seed else 0

    clean_spec = GenerationSpec(
        shape=spec.shape or "abstract",
        n_points=n,
        bbox_size=bbox,
        primitives=clean_prims,
        features=clean_features,
        color=color,
        seed=seed,
    )
    _apply_proportion_rules(clean_spec, warnings)
    return clean_spec, warnings
