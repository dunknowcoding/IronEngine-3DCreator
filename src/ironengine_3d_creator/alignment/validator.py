"""Validate and normalize a GenerationSpec.

Applies hard caps, drops unknown primitive/feature kinds, fills missing
required params with safe defaults, and ensures `n_points >= 100` so
preview always has something to show.
"""
from __future__ import annotations

import logging

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
}


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
        params = dict(_PARAM_DEFAULTS[kind])
        params.update({k: v for k, v in (p.params or {}).items() if v is not None})
        clean_prims.append(Primitive(
            kind=kind,
            transform=p.transform,
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

    return (
        GenerationSpec(
            shape=spec.shape or "abstract",
            n_points=n,
            bbox_size=bbox,
            primitives=clean_prims,
            features=clean_features,
            color=color,
            seed=seed,
        ),
        warnings,
    )
