"""Turn a GenerationSpec into (positions, colors) point cloud arrays."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..alignment.schema import GenerationSpec
from .colorize import base_color, shaded_colors
from .features import FEATURE_FUNCS, apply_fur, apply_holes, region_mask
from .primitives import sample_primitive
from .sampler import allocate_budget
from .textures import apply_texture, shape_default_material


@dataclass
class GenerationResult:
    positions: np.ndarray   # (N, 3) float32
    colors: np.ndarray      # (N, 3) float32, in [0, 1]
    labels: np.ndarray      # (N,) int — index into spec.primitives
    label_names: list[str]


def _apply_transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts
    h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    return (h @ T.T)[:, :3]


def generate(spec: GenerationSpec) -> GenerationResult:
    """Procedurally synthesize a point cloud from a validated spec."""
    rng = np.random.default_rng(spec.seed or None)
    counts = allocate_budget(spec.primitives, spec.n_points)

    chunks_pos: list[np.ndarray] = []
    chunks_lbl: list[np.ndarray] = []
    chunks_col: list[np.ndarray] = []
    label_names: list[str] = []
    base = base_color(spec.shape, spec.color)

    for i, (prim, n) in enumerate(zip(spec.primitives, counts)):
        local = sample_primitive(prim.kind, n, prim.params, rng)
        world = _apply_transform(local, prim.transform_matrix())
        chunks_pos.append(world.astype(np.float32, copy=False))
        chunks_lbl.append(np.full(world.shape[0], i, dtype=np.int32))
        label_names.append(prim.label or f"{prim.kind}_{i}")

        # Per-primitive material → either explicit "material" param or
        # heuristic from shape/label.
        material = prim.params.get("material") or shape_default_material(spec.shape, prim.label)
        textured = apply_texture(world, tuple(base.tolist()), material, rng)
        if textured is None:
            textured = shaded_colors(world, base, rng)
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
    )
