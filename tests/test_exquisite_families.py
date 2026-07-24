"""Tests for the exquisite style families (CR_ComplexBuilder extension).

Each new grammar fragment must: validate clean (no dropped kinds), carry
labels on every part, synthesize a point cloud, build analytic meshes within
a sane triangle budget, and — before the engine's bbox fit — use real-world
dimensions (fence post ~1.1 m, column ~3 m, chair seat 0.45 m, spaceship 2 m).
"""
from __future__ import annotations

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec
from ironengine_3d_creator.alignment.validator import normalize
from ironengine_3d_creator.generation.analytic_mesh import build_spec_meshes
from ironengine_3d_creator.generation.compositor import generate
from ironengine_3d_creator.generation.style_engine import StyleEngine
from ironengine_3d_creator.generation.style_families import (
    FAMILY_BUILDERS,
    FamilyContext,
)

EXQUISITE = (
    "rococo_fence",
    "neoclassical_column",
    "modern_luxury",
    "futurist_chair",
    "desktop_computer",
    "spaceship",
    "robot",
)

TRI_BUDGET = 80_000   # per-family analytic mesh budget (all parts combined)
_SMALL = 6_000


def _raw_ctx(family: str, seed: int, target_parts: int = 24) -> FamilyContext:
    """Build the grammar WITHOUT the engine's bbox fit (real-world scale)."""
    rng = np.random.default_rng(seed)
    ctx = FamilyContext(rng=rng, target_parts=target_parts)
    FAMILY_BUILDERS[family](ctx)
    return ctx


def _raw_spec(family: str, seed: int, target_parts: int = 24) -> GenerationSpec:
    ctx = _raw_ctx(family, seed, target_parts)
    return GenerationSpec(
        shape=family, n_points=_SMALL, bbox_size=(50.0, 50.0, 50.0),
        primitives=ctx.primitives, features=ctx.features, color=ctx.color,
        seed=seed,
    )


# ----------------------------------------------------------------------
# validity + budgets (through the seeded engine, like existing families)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("family", EXQUISITE)
@pytest.mark.parametrize("seed", (1, 7, 12345))
def test_family_validates_and_synthesizes(family, seed):
    spec = StyleEngine(seed=seed).generate(family=family, n_points=_SMALL)
    clean, warns = normalize(spec)
    assert not [w for w in warns if "dropped" in w], warns
    assert 3 <= len(clean.primitives) <= 40
    for p in clean.primitives:
        assert p.label, "every part should carry a label"
    res = generate(clean)
    assert res.positions.shape[0] >= _SMALL // 2
    assert np.isfinite(res.positions).all()


@pytest.mark.parametrize("family", EXQUISITE)
def test_family_mesh_build_within_triangle_budget(family):
    spec = StyleEngine(seed=11).generate(family=family, n_points=_SMALL)
    clean, _ = normalize(spec)
    parts = build_spec_meshes(clean)
    assert parts, "family should produce analytic meshes"
    tris = sum(p.faces.shape[0] for p in parts)
    assert tris <= TRI_BUDGET, f"{family}: {tris} tris exceeds {TRI_BUDGET}"
    for p in parts:
        assert np.isfinite(p.vertices).all()
        assert np.isfinite(p.normals).all()


@pytest.mark.parametrize("family", EXQUISITE)
def test_family_part_count_scales_with_complexity(family):
    simple = StyleEngine(seed=3).generate(family=family, complexity="simple",
                                          n_points=_SMALL)
    complex_ = StyleEngine(seed=3).generate(family=family, complexity="complex",
                                            n_points=120_000)
    assert len(complex_.primitives) >= len(simple.primitives)


# ----------------------------------------------------------------------
# real-world dimensions (pre-fit, raw grammar output)
# ----------------------------------------------------------------------

def test_rococo_fence_post_height():
    spec = _raw_spec("rococo_fence", 5)
    parts = build_spec_meshes(normalize(spec)[0])
    hi = max(float(p.aabb_max[1]) for p in parts)
    assert 1.0 <= hi <= 1.25, f"fence post should be ~1.1 m, got {hi}"
    labels = {p.label for p in parts}
    assert any("post" in l for l in labels)
    assert any("scroll" in l or "vbar" in l for l in labels)


def test_neoclassical_column_height_and_slices():
    spec = _raw_spec("neoclassical_column", 5)
    parts = build_spec_meshes(normalize(spec)[0])
    hi = max(float(p.aabb_max[1]) for p in parts)
    assert 2.6 <= hi <= 3.4, f"column should be ~3 m, got {hi}"
    shaft = [p for p in parts if p.label.startswith("shaft_")]
    assert len(shaft) >= 2, "shaft must be sliced"
    assert any(p.label == "abacus" for p in parts)


def test_futurist_chair_seat_height():
    spec = _raw_spec("futurist_chair", 5)
    parts = build_spec_meshes(normalize(spec)[0])
    seat = next(p for p in parts if p.label == "seat_shell")
    seat_y = float((seat.aabb_min[1] + seat.aabb_max[1]) / 2)
    assert seat_y == pytest.approx(0.45, abs=0.02)


def test_spaceship_length_two_metres():
    spec = _raw_spec("spaceship", 5, target_parts=30)
    parts = build_spec_meshes(normalize(spec)[0])
    lo = min(float(p.aabb_min[2]) for p in parts)
    hi = max(float(p.aabb_max[2]) for p in parts)
    assert hi - lo == pytest.approx(2.0, abs=0.15)
    labels = {p.label for p in parts}
    assert any("greeble" in l for l in labels), "greeble array expected"


def test_robot_articulation_naming():
    spec = _raw_spec("robot", 5, target_parts=30)
    labels = {p.label for p in normalize(spec)[0].primitives}
    # Articulation-ready joint names, mirrored _l/_r.
    for stem in ("thigh", "shin", "shoulder", "upper_arm"):
        assert f"{stem}_l" in labels and f"{stem}_r" in labels, stem
    assert "torso" in labels and "head" in labels


def test_modern_luxury_beveled_monolith():
    spec = _raw_spec("modern_luxury", 5)
    prims = {p.label: p for p in normalize(spec)[0].primitives}
    mono = prims["monolith"]
    assert mono.kind == "superellipsoid"
    e = mono.params["exponents"]
    assert all(0.2 <= x <= 0.5 for x in e), "beveled (rounded-box) exponents"
    mats = {p.params.get("material") for p in prims.values()}
    assert "metal" in mats, "metal accents expected"


def test_desktop_computer_has_core_peripherals():
    spec = _raw_spec("desktop_computer", 5)
    labels = {p.label for p in normalize(spec)[0].primitives}
    assert {"tower", "screen", "keyboard"} <= labels
