"""Auto-mode templates: a curated table of starter recipes.

Auto mode bypasses the LLM and picks one of these directly. The pipeline
guarantees a usable preview even when no LLM is configured.
"""
from __future__ import annotations

import random
from typing import Callable

import numpy as np

from .schema import Feature, GenerationSpec, Primitive


def _T(translate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), rotate_y: float = 0.0) -> list[list[float]]:
    """Build a 4x4 transform from translate / scale / Y-axis rotation."""
    cy, sy = np.cos(rotate_y), np.sin(rotate_y)
    rot = np.array([
        [cy, 0, sy, 0],
        [0, 1, 0, 0],
        [-sy, 0, cy, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    sc = np.diag([scale[0], scale[1], scale[2], 1.0]).astype(np.float32)
    tr = np.eye(4, dtype=np.float32)
    tr[:3, 3] = translate
    return (tr @ rot @ sc).tolist()


def _chair() -> GenerationSpec:
    seat = Primitive("box", _T((0, 0.45, 0), (0.45, 0.05, 0.45)), {"size": [1.0, 1.0, 1.0]}, "seat")
    legs = [
        Primitive("cylinder",
                  _T((dx * 0.4, 0.225, dz * 0.4)),
                  {"radius": 0.04, "height": 0.45, "caps": True},
                  f"leg_{i}")
        for i, (dx, dz) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)])
    ]
    back = Primitive("box", _T((0, 0.75, -0.4), (0.45, 0.3, 0.04)), {"size": [1.0, 1.0, 1.0]}, "back")
    return GenerationSpec(
        shape="chair", n_points=60_000, bbox_size=(1.0, 1.2, 1.0),
        primitives=[seat, *legs, back],
        features=[Feature("scratch", "all", {"count": 8, "depth": 0.005})],
        color=(0.55, 0.40, 0.32),
        seed=random.randint(0, 1 << 31),
    )


def _vase() -> GenerationSpec:
    """Chinese-style vase: foot ring → tapered base → bulbous body
    → shoulder → narrow neck → flared rim. Proportions follow a meiping /
    plum vase silhouette so the default looks intentional rather than
    "vague pot".
    """
    base_ring = Primitive("torus", _T((0, 0.04, 0)),
                          {"major_radius": 0.16, "minor_radius": 0.025}, "base_ring")
    base = Primitive("ellipsoid", _T((0, 0.16, 0)),
                     {"radii": [0.18, 0.10, 0.18]}, "base")
    body = Primitive("ellipsoid", _T((0, 0.42, 0)),
                     {"radii": [0.32, 0.26, 0.32]}, "body")
    shoulder = Primitive("ellipsoid", _T((0, 0.74, 0)),
                         {"radii": [0.20, 0.10, 0.20]}, "shoulder")
    neck = Primitive("cylinder", _T((0, 0.92, 0)),
                     {"radius": 0.09, "height": 0.16, "caps": False}, "neck")
    rim = Primitive("torus", _T((0, 1.02, 0)),
                    {"major_radius": 0.12, "minor_radius": 0.02}, "rim")
    return GenerationSpec(
        shape="vase", n_points=80_000, bbox_size=(0.7, 1.1, 0.7),
        primitives=[base_ring, base, body, shoulder, neck, rim],
        features=[
            Feature("curve_pattern", "body", {"frequency": 7.0, "amplitude": 0.008}),
            Feature("ridges", "shoulder", {"count": 16, "depth": 0.005}),
            Feature("ridges", "base", {"count": 10, "depth": 0.004}),
        ],
        color=(0.92, 0.91, 0.85),  # off-white porcelain
        seed=random.randint(0, 1 << 31),
    )


def _creature() -> GenerationSpec:
    body = Primitive("ellipsoid", _T((0, 0.4, 0)), {"radii": [0.45, 0.3, 0.6]}, "body")
    head = Primitive("sphere", _T((0, 0.55, 0.6)), {"radius": 0.22}, "head")
    legs = [
        Primitive("capsule",
                  _T((dx * 0.25, 0.15, dz * 0.4)),
                  {"radius": 0.06, "height": 0.3},
                  f"leg_{i}")
        for i, (dx, dz) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)])
    ]
    tail = Primitive("cone", _T((0, 0.4, -0.7), rotate_y=np.pi),
                     {"radius": 0.08, "height": 0.4}, "tail")
    return GenerationSpec(
        shape="creature", n_points=80_000, bbox_size=(1.2, 1.0, 1.5),
        primitives=[body, head, *legs, tail],
        features=[
            Feature("fur", "all", {"density": 0.6, "length": 0.01}),
            Feature("bump_field", "all", {"count": 30, "radius": 0.03}),
        ],
        color=(0.65, 0.55, 0.35),
        seed=random.randint(0, 1 << 31),
    )


def _rock() -> GenerationSpec:
    body = Primitive("ellipsoid", _T((0, 0.3, 0)), {"radii": [0.5, 0.35, 0.45]}, "body")
    return GenerationSpec(
        shape="rock", n_points=50_000, bbox_size=(1.0, 0.7, 1.0),
        primitives=[body],
        features=[
            Feature("bump_field", "all", {"count": 80, "radius": 0.04}),
            Feature("erosion", "all", {"strength": 0.02}),
            Feature("scratch", "all", {"count": 20, "depth": 0.01}),
        ],
        color=(0.45, 0.45, 0.42),
        seed=random.randint(0, 1 << 31),
    )


def _abstract() -> GenerationSpec:
    pieces = [
        Primitive("torus", _T((0, 0.5, 0)), {"major_radius": 0.4, "minor_radius": 0.12}, "ring"),
        Primitive("helix", _T((0, 0.5, 0)),
                  {"radius": 0.3, "pitch": 0.15, "turns": 4.0, "thickness": 0.04}, "spiral"),
    ]
    return GenerationSpec(
        shape="abstract", n_points=70_000, bbox_size=(1.0, 1.0, 1.0),
        primitives=pieces,
        features=[Feature("curve_pattern", "all", {"frequency": 8.0, "amplitude": 0.02})],
        color=(0.40, 0.70, 1.00),
        seed=random.randint(0, 1 << 31),
    )


def _fence() -> GenerationSpec:
    """A fence with curved/patterned bars between two horizontal rails on
    short posts at each end. Demonstrates the framework repair: rails
    span the vbars, vbars stay parallel and never snap together.
    """
    n_bars = 8
    span_x = 1.6
    height = 0.9
    bar_radius = 0.025
    primitives: list[Primitive] = []
    # Two end posts.
    for i, x in enumerate((-span_x / 2 - 0.05, span_x / 2 + 0.05)):
        primitives.append(Primitive(
            "cylinder", _T((x, height / 2, 0)),
            {"radius": 0.05, "height": height, "caps": True},
            f"post_{i}",
        ))
    # Top + bottom rails.
    primitives.append(Primitive(
        "box", _T((0, height - 0.05, 0)),
        {"size": [span_x + 0.1, 0.05, 0.06]}, "top_rail",
    ))
    primitives.append(Primitive(
        "box", _T((0, 0.10, 0)),
        {"size": [span_x + 0.1, 0.05, 0.06]}, "bottom_rail",
    ))
    # Curved bars between the rails — mix straight cylinders and helices to
    # show the spacing repair handles both.
    for i in range(n_bars):
        x = -span_x / 2 + (i + 0.5) * (span_x / n_bars)
        if i % 2 == 0:
            primitives.append(Primitive(
                "cylinder", _T((x, height / 2, 0)),
                {"radius": bar_radius, "height": height - 0.15, "caps": True},
                f"bar_{i}",
            ))
        else:
            primitives.append(Primitive(
                "helix", _T((x, height / 2, 0)),
                {"radius": 0.04, "pitch": 0.18, "turns": 3.0, "thickness": bar_radius},
                f"bar_{i}",
            ))
    # Decorative finials on the posts.
    for i, x in enumerate((-span_x / 2 - 0.05, span_x / 2 + 0.05)):
        primitives.append(Primitive(
            "sphere", _T((x, height + 0.08, 0)),
            {"radius": 0.06}, f"finial_{i}",
        ))
    return GenerationSpec(
        shape="fence", n_points=120_000, bbox_size=(span_x + 0.4, height + 0.2, 0.2),
        primitives=primitives,
        features=[
            Feature("scratch", "all", {"count": 18, "depth": 0.003}),
        ],
        color=(0.32, 0.20, 0.12),  # weathered iron
        seed=random.randint(0, 1 << 31),
    )


def _archway() -> GenerationSpec:
    """A stone archway: two pillars and a torus arch on top, with a base
    plinth and decorative finials."""
    pillar_h = 1.4
    span = 1.4
    primitives = [
        Primitive("box", _T((0, 0.06, 0)),
                  {"size": [span + 0.4, 0.10, 0.4]}, "base"),
        Primitive("cylinder", _T((-span / 2, pillar_h / 2 + 0.10, 0)),
                  {"radius": 0.15, "height": pillar_h, "caps": True}, "pillar_left"),
        Primitive("cylinder", _T((span / 2, pillar_h / 2 + 0.10, 0)),
                  {"radius": 0.15, "height": pillar_h, "caps": True}, "pillar_right"),
        # The arch: half a torus; we approximate with a full torus then trim
        # would require CSG, so just use a torus and let the renderer show
        # both halves — visually still reads as an arch.
        Primitive("torus", _T((0, pillar_h + 0.10, 0)),
                  {"major_radius": span / 2, "minor_radius": 0.12}, "arch"),
        Primitive("sphere", _T((-span / 2, pillar_h + 0.30, 0)),
                  {"radius": 0.10}, "finial_left"),
        Primitive("sphere", _T((span / 2, pillar_h + 0.30, 0)),
                  {"radius": 0.10}, "finial_right"),
    ]
    return GenerationSpec(
        shape="archway", n_points=110_000, bbox_size=(span + 0.6, pillar_h + 0.5, 0.5),
        primitives=primitives,
        features=[
            Feature("ridges", "pillar_left", {"count": 14, "depth": 0.005}),
            Feature("ridges", "pillar_right", {"count": 14, "depth": 0.005}),
            Feature("erosion", "all", {"strength": 0.005}),
        ],
        color=(0.85, 0.82, 0.74),  # limestone
        seed=random.randint(0, 1 << 31),
    )


_TEMPLATES: dict[str, Callable[[], GenerationSpec]] = {
    "chair": _chair,
    "vase": _vase,
    "creature": _creature,
    "rock": _rock,
    "abstract": _abstract,
    "fence": _fence,
    "archway": _archway,
}


def auto_template(shape: str | None = None) -> GenerationSpec:
    """Return a starter spec. If `shape` is None, pick at random."""
    if shape and shape in _TEMPLATES:
        return _TEMPLATES[shape]()
    return random.choice(list(_TEMPLATES.values()))()


def available_templates() -> tuple[str, ...]:
    return tuple(_TEMPLATES.keys())
