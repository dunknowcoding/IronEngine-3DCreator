"""Physical material presets for export manifests.

The LLM emits free-form `params["material"]` hints per primitive (wood,
stone, metal, …). This table maps those hints to physically plausible PBR +
physics values so downstream tools (SceneEditor/Sim) get real roughness,
metallic, density, friction, and restitution instead of untextured defaults.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

# hint -> {roughness, metallic, density_kg_m3, friction, restitution}
MATERIAL_PRESETS: dict[str, dict[str, float]] = {
    "wood":      {"roughness": 0.65, "metallic": 0.0,  "density_kg_m3": 700.0,  "friction": 0.6,  "restitution": 0.25},
    "stone":     {"roughness": 0.85, "metallic": 0.0,  "density_kg_m3": 2700.0, "friction": 0.75, "restitution": 0.15},
    "fabric":    {"roughness": 0.9,  "metallic": 0.0,  "density_kg_m3": 300.0,  "friction": 0.8,  "restitution": 0.05},
    "metal":     {"roughness": 0.35, "metallic": 0.95, "density_kg_m3": 7870.0, "friction": 0.45, "restitution": 0.3},
    "iron":      {"roughness": 0.35, "metallic": 0.95, "density_kg_m3": 7870.0, "friction": 0.5,  "restitution": 0.25},
    "leather":   {"roughness": 0.75, "metallic": 0.0,  "density_kg_m3": 950.0,  "friction": 0.7,  "restitution": 0.2},
    "ceramic":   {"roughness": 0.18, "metallic": 0.0,  "density_kg_m3": 2400.0, "friction": 0.55, "restitution": 0.35},
    "porcelain": {"roughness": 0.18, "metallic": 0.0,  "density_kg_m3": 2400.0, "friction": 0.5,  "restitution": 0.35},
    "plastic":   {"roughness": 0.4,  "metallic": 0.0,  "density_kg_m3": 1050.0, "friction": 0.5,  "restitution": 0.4},
    "glass":     {"roughness": 0.05, "metallic": 0.0,  "density_kg_m3": 2500.0, "friction": 0.35, "restitution": 0.4},
    "brick":     {"roughness": 0.9,  "metallic": 0.0,  "density_kg_m3": 1900.0, "friction": 0.8,  "restitution": 0.1},
    "organic":   {"roughness": 0.8,  "metallic": 0.0,  "density_kg_m3": 1000.0, "friction": 0.6,  "restitution": 0.2},
    "foliage":   {"roughness": 0.9,  "metallic": 0.0,  "density_kg_m3": 700.0,  "friction": 0.7,  "restitution": 0.05},
}

# Shape-level fallbacks when no primitive carries a known material hint.
_SHAPE_FALLBACK: dict[str, str] = {
    "tree": "foliage",
    "rock": "stone",
    "chair": "wood",
    "table": "wood",
    "vase": "ceramic",
    "lamp": "metal",
    "vehicle": "metal",
    "creature": "organic",
}


def default_preset() -> dict[str, float]:
    """Neutral preset for unrecognized materials."""
    return {"roughness": 0.7, "metallic": 0.0, "density_kg_m3": 1000.0,
            "friction": 0.5, "restitution": 0.3}


def resolve_material(spec) -> tuple[str, dict[str, float]]:
    """Pick the most common known material hint across spec primitives.

    Returns (name, preset). Falls back to a shape-based hint, then to the
    neutral default preset (name "default").
    """
    hints: list[str] = []
    for prim in getattr(spec, "primitives", []) or []:
        params: dict[str, Any] = getattr(prim, "params", None) or {}
        hint = params.get("material")
        if isinstance(hint, str):
            key = hint.strip().lower()
            if key in MATERIAL_PRESETS:
                hints.append(key)
    if hints:
        name = Counter(hints).most_common(1)[0][0]
        return name, MATERIAL_PRESETS[name]
    shape = str(getattr(spec, "shape", "") or "").lower()
    name = _SHAPE_FALLBACK.get(shape)
    if name is not None:
        return name, MATERIAL_PRESETS[name]
    return "default", default_preset()
