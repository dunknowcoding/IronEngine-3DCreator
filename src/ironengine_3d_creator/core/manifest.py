"""Rich export manifest (.iemodel.json) written next to PLY/GLB exports.

Downstream tools (IronEngine-SceneEditor / Sim) read this sidecar to recover
units, axis convention, bounds, PBR material, and physics properties that a
raw point cloud or mesh file cannot carry.

Schema ``iemodel/2`` is a superset of ``iemodel/1``: every v1 field is still
emitted (single ``material`` block as the majority fallback, ``mesh`` /
``point_cloud`` / ``spec`` blocks), plus per-part ``materials``/``parts`` and
a physics block with measured solid volume, mass, and a collider hint chosen
from the dominant primitive. Consumers must accept both versions.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from ..generation.materials import MATERIAL_PRESETS, default_preset, resolve_material

_log = logging.getLogger(__name__)

MANIFEST_SCHEMA = "iemodel/2"

# Primitives whose shape a box collider approximates well; everything organic
# or curved gets a convex-hull hint instead.
_BOX_COLLIDER_KINDS = {"box", "prism", "plane"}


def _part_summaries(spec, positions, labels) -> list[dict]:
    """Per-primitive part records: label, kind, material, AABB, solid volume.

    AABBs are measured from the generated points when per-point labels are
    available; otherwise they come from the analytic local AABB pushed
    through the primitive transform. Solid volumes are always analytic
    (exact formulas × |det transform|), never AABB-derived.
    """
    from ..generation.analytic_mesh import (
        local_aabb, part_material_name, primitive_solid_volume,
    )

    parts: list[dict] = []
    prims = getattr(spec, "primitives", []) or []
    for i, prim in enumerate(prims):
        label = prim.label or f"{prim.kind}_{i}"
        T = np.asarray(prim.transform_matrix(), dtype=np.float64)
        det = abs(float(np.linalg.det(T[:3, :3])))
        volume = primitive_solid_volume(prim.kind, prim.params or {}) * (det if det > 1e-12 else 1.0)

        aabb_min = aabb_max = None
        if labels is not None:
            sel = np.asarray(labels) == i
            if sel.any():
                pts = np.asarray(positions, dtype=np.float64)[sel]
                aabb_min, aabb_max = pts.min(axis=0), pts.max(axis=0)
        if aabb_min is None:
            lo, hi = local_aabb(prim.kind, prim.params or {})
            corners = np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            )
            world = (np.concatenate([corners, np.ones((8, 1))], axis=1) @ T.T)[:, :3]
            aabb_min, aabb_max = world.min(axis=0), world.max(axis=0)

        parts.append(
            {
                "label": label,
                "primitive": prim.kind,
                "material": part_material_name(spec, prim),
                "aabb_min": aabb_min.tolist(),
                "aabb_max": aabb_max.tolist(),
                "solid_volume_m3": float(volume),
            }
        )
    return parts


def _choose_collider(parts: list[dict]) -> str:
    """box default, convex for organic/curved kinds, parts when multi-part."""
    if len(parts) > 1:
        return "parts"
    if not parts:
        return "box"
    return "box" if parts[0]["primitive"] in _BOX_COLLIDER_KINDS else "convex"


def build_manifest(
    spec,
    positions: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    mesh_path: str | Path | None = None,
    point_cloud_path: str | Path | None = None,
    mesh_stats: dict | None = None,
    labels: np.ndarray | None = None,
    name: str | None = None,
) -> dict:
    """Assemble the iemodel/2 manifest dict for one exported model.

    `labels` (per-point primitive indices from GenerationResult) enable
    measured per-part AABBs and per-part mean albedo; without them parts fall
    back to analytic AABBs and the spec base color.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if positions.size:
        aabb_min = positions.min(axis=0)
        aabb_max = positions.max(axis=0)
    else:
        aabb_min = aabb_max = np.zeros(3)
    bbox_size = aabb_max - aabb_min

    # ---- v1 majority-fallback material block (kept for old consumers) ------
    mat_name, preset = resolve_material(spec)
    color = getattr(spec, "color", None)
    if color is not None:
        albedo = [float(c) for c in color]
    elif colors is not None and len(colors):
        albedo = np.asarray(colors, dtype=np.float64).reshape(-1, 3).mean(axis=0).tolist()
    else:
        albedo = [0.7, 0.7, 0.7]

    # ---- v2 per-part materials + parts -------------------------------------
    parts = _part_summaries(spec, positions, labels)
    materials: dict[str, dict] = {}
    for i, part in enumerate(parts):
        name_i = part["material"]
        if name_i in materials:
            continue
        p = MATERIAL_PRESETS.get(name_i, default_preset())
        part_albedo = None
        if labels is not None and colors is not None and len(colors):
            sel = np.asarray(labels) == i
            if sel.any():
                part_albedo = (
                    np.asarray(colors, dtype=np.float64).reshape(-1, 3)[sel].mean(axis=0).tolist()
                )
        if part_albedo is None:
            part_albedo = list(albedo)
        materials[name_i] = {
            "albedo": [float(c) for c in part_albedo],
            "roughness": p["roughness"],
            "metallic": p["metallic"],
            "density_kg_m3": p["density_kg_m3"],
            "friction": p["friction"],
            "restitution": p["restitution"],
        }

    # ---- physics ------------------------------------------------------------
    solid_volume = float(sum(p["solid_volume_m3"] for p in parts))
    if solid_volume <= 0.0 and positions.size:
        # Code-mode / freeform clouds have no analytic parts — fall back to
        # the v1 AABB-volume estimate so mass stays sane.
        solid_volume = float(np.prod(np.maximum(bbox_size, 1e-9)))
    mass_kg = solid_volume * preset["density_kg_m3"]

    # ---- mesh / point-cloud blocks ------------------------------------------
    mesh_block = None
    if mesh_path is not None:
        stats = mesh_stats or {}
        mesh_block = {
            "path": Path(mesh_path).name,
            "format": Path(mesh_path).suffix.lstrip(".").lower() or "glb",
            "vertices": int(stats.get("vertices", 0)),
            "faces": int(stats.get("faces", 0)),
            "has_uvs": bool(stats.get("has_uvs", False)),
            "has_vertex_colors": bool(stats.get("has_vertex_colors", True)),
            "analytic": bool(stats.get("analytic", False)),
        }

    cloud_block = None
    if point_cloud_path is not None:
        cloud_block = {
            "path": Path(point_cloud_path).name,
            "format": Path(point_cloud_path).suffix.lstrip(".").lower() or "ply",
            "points": int(positions.shape[0]),
        }

    # ---- name: export file stem first, spec shape as fallback ---------------
    if name is None:
        for candidate in (mesh_path, point_cloud_path):
            if candidate is not None:
                name = Path(candidate).stem
                break
    if not name:
        name = getattr(spec, "shape", None) or "model"

    try:
        spec_block = spec.to_json()
    except Exception:
        prims = getattr(spec, "primitives", []) or []
        spec_block = {
            "shape": getattr(spec, "shape", "abstract"),
            "n_points": int(getattr(spec, "n_points", positions.shape[0])),
            "bbox_size": list(getattr(spec, "bbox_size", bbox_size.tolist())),
            "primitive_count": len(prims),
            "primitive_kinds": sorted({getattr(p, "kind", "?") for p in prims}),
        }

    return {
        "schema": MANIFEST_SCHEMA,
        "name": name,
        "generator": f"ironengine-3d-creator {__version__}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "units": "meters",
        "up_axis": "Y",
        "aabb_min": aabb_min.tolist(),
        "aabb_max": aabb_max.tolist(),
        "bbox_size": bbox_size.tolist(),
        "shape": getattr(spec, "shape", "abstract"),
        "material": {
            "name": mat_name,
            "albedo": albedo,
            "roughness": preset["roughness"],
            "metallic": preset["metallic"],
        },
        "materials": materials,
        "parts": parts,
        "physics": {
            "density_kg_m3": preset["density_kg_m3"],
            "friction": preset["friction"],
            "restitution": preset["restitution"],
            "collider": _choose_collider(parts),
            "dynamic": True,
            "solid_volume_m3": solid_volume,
            "mass_kg": mass_kg,
        },
        "mesh": mesh_block,
        "point_cloud": cloud_block,
        "spec": spec_block,
    }


def write_manifest(path: str | Path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
