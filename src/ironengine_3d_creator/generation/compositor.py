"""Turn a GenerationSpec into (positions, colors) point cloud arrays."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..alignment.schema import GenerationSpec
from .colorize import albedo_colors, base_color
from .features import FEATURE_FUNCS, apply_asperity, apply_fur, apply_holes, region_mask
from .primitives import inside_primitive, sample_primitive
from .sampler import allocate_budget
from .textures import apply_texture, shape_default_material


# Hard materials that get automatic micro-asperity so their surfaces stop
# looking like razor-smooth CG. Strength is derived per part below.
_ASPERITY_MATERIALS = frozenset({"stone", "wood", "concrete", "terracotta"})


@dataclass
class GenerationResult:
    positions: np.ndarray   # (N, 3) float32
    colors: np.ndarray      # (N, 3) float32, in [0, 1]
    labels: np.ndarray      # (N,) int — index into spec.primitives
    label_names: list[str]
    warnings: list[str] = field(default_factory=list)


def _apply_transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts
    h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    return (h @ T.T)[:, :3]


def _is_cutter(prim) -> bool:
    return str((prim.params or {}).get("role", "")).lower() == "subtract"


def generate(spec: GenerationSpec) -> GenerationResult:
    """Procedurally synthesize a point cloud from a validated spec."""
    rng = np.random.default_rng(spec.seed or None)
    counts = allocate_budget(spec.primitives, spec.n_points)
    if any(_is_cutter(p) for p in spec.primitives):
        # Cutters emit no points; renormalise the budget across the visible
        # parts so the cloud still totals ≈ n_points.
        counts = [0 if _is_cutter(p) else int(c)
                  for p, c in zip(spec.primitives, counts)]
        total = sum(counts)
        if total > 0:
            scale = spec.n_points / total
            counts = [int(round(c * scale)) for c in counts]
    warnings: list[str] = []

    chunks_pos: list[np.ndarray] = []
    chunks_lbl: list[np.ndarray] = []
    chunks_col: list[np.ndarray] = []
    label_names: list[str] = []
    base = base_color(spec.shape, spec.color)

    for i, (prim, n) in enumerate(zip(spec.primitives, counts)):
        label_names.append(prim.label or f"{prim.kind}_{i}")
        if _is_cutter(prim):
            # Cutters never emit points of their own; they carve below.
            chunks_pos.append(np.empty((0, 3), dtype=np.float32))
            chunks_lbl.append(np.empty((0,), dtype=np.int32))
            chunks_col.append(np.empty((0, 3), dtype=np.float32))
            continue
        local = sample_primitive(prim.kind, n, prim.params, rng)
        world = _apply_transform(local, prim.transform_matrix())
        chunks_pos.append(world.astype(np.float32, copy=False))
        chunks_lbl.append(np.full(world.shape[0], i, dtype=np.int32))

        # Per-primitive material → either explicit "material" param or
        # heuristic from shape/label.
        material = prim.params.get("material") or shape_default_material(spec.shape, prim.label)
        textured = apply_texture(world, tuple(base.tolist()), material, rng)
        if textured is None:
            # Unbaked albedo (W8): export-ready colors carry no lighting term.
            textured = albedo_colors(world, base, rng)
        # Hard-surface realism: subtle seeded asperity so stone/wood stops
        # looking CG-smooth. Peak displacement ≈ 0.15 % of the part's world
        # extent, clamped to [0.2 mm, 1.5 mm]; opt out with
        # params["asperity"] = 0.
        if material and material.lower() in _ASPERITY_MATERIALS and world.shape[0]:
            opt = prim.params.get("asperity")
            if opt is None or float(opt) > 0.0:
                extent = float(np.max(world.max(axis=0) - world.min(axis=0))) if world.shape[0] else 0.0
                strength = float(opt) if opt is not None else min(0.0015, max(0.0002, extent * 0.0015))
                apply_asperity(world, textured, np.ones(world.shape[0], dtype=bool),
                               {"strength": strength, "frequency": 40.0}, rng)
        chunks_col.append(textured)

    if not chunks_pos:
        positions = np.empty((0, 3), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int32)
        colors = np.empty((0, 3), dtype=np.float32)
    else:
        positions = np.concatenate(chunks_pos, axis=0)
        labels = np.concatenate(chunks_lbl, axis=0)
        colors = np.concatenate(chunks_col, axis=0)

    label_lookup = {name: i for i, name in enumerate(label_names)}

    # CSG-lite subtraction (point-cloud level): drop host points that fall
    # inside a `role: "subtract"` cutter's solid.
    cutters = [(i, p) for i, p in enumerate(spec.primitives) if _is_cutter(p)]
    if cutters and positions.shape[0]:
        keep = np.ones(positions.shape[0], dtype=bool)
        for ci, cutter in cutters:
            target = str((cutter.params or {}).get("target", "") or "")
            T_inv = np.linalg.inv(cutter.transform_matrix().astype(np.float64))
            local = _apply_transform(positions.astype(np.float64), T_inv)
            inside = inside_primitive(cutter.kind, cutter.params or {}, local)
            victims = inside & (labels != ci) & (labels >= 0)
            if target:
                ti = label_lookup.get(target)
                if ti is None:
                    warnings.append(
                        f"subtract: cutter {label_names[ci]!r} target {target!r} not found — ignored"
                    )
                    continue
                victims &= labels == ti
            removed_per_part: dict[int, int] = {}
            for idx in np.unique(labels[victims]):
                removed_per_part[int(idx)] = int((victims & (labels == idx)).sum())
            if not victims.any():
                warnings.append(
                    f"subtract: cutter {label_names[ci]!r} removed no points — no overlap?"
                )
            for part_i, removed in removed_per_part.items():
                total_i = int((labels == part_i).sum())
                if total_i and removed / total_i > 0.95:
                    warnings.append(
                        f"subtract: cutter {label_names[ci]!r} removed {removed}/{total_i} "
                        f"points of {label_names[part_i]!r} — part nearly destroyed (orphan risk)"
                    )
            keep &= ~victims
        if not keep.all():
            positions = positions[keep]
            colors = colors[keep]
            labels = labels[keep]

    # In-place features (deformation / coloring).
    extras_pos: list[np.ndarray] = []
    extras_col: list[np.ndarray] = []
    keep = np.ones(positions.shape[0], dtype=bool)
    for feat in spec.features:
        mask = region_mask(feat.region, labels, label_lookup)
        kind = feat.kind
        if kind in FEATURE_FUNCS:
            FEATURE_FUNCS[kind](positions, colors, mask, feat.params, rng)
        elif kind == "holes":
            keep &= apply_holes(positions, colors, mask, feat.params, rng)
        elif kind == "fur":
            ep, ec = apply_fur(positions, colors, mask, feat.params, rng)
            extras_pos.append(ep); extras_col.append(ec)
        # unknown feature kinds are filtered by validator; tolerate any leftovers.

    if not keep.all():
        positions = positions[keep]
        colors = colors[keep]
        labels = labels[keep]

    if extras_pos:
        positions = np.concatenate([positions, *extras_pos], axis=0)
        colors = np.concatenate([colors, *extras_col], axis=0)
        labels = np.concatenate(
            [labels, *[np.full(p.shape[0], -1, dtype=np.int32) for p in extras_pos]],
            axis=0,
        )

    return GenerationResult(
        positions=positions,
        colors=np.clip(colors, 0.0, 1.0),
        labels=labels,
        label_names=label_names,
        warnings=warnings,
    )
