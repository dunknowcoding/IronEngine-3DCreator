"""Tests for CR_Quality v2: attachment solver, proportion truth tables,
organic grammars (insect / flower / leaf / creature paws), and surface
realism (box bevel, soil relief, cloth drape / thickness / weave)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ironengine_3d_creator.alignment.integrity import (
    assembly_report,
    check_and_fix,
    _world_aabb,
)
from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.alignment.validator import normalize
from ironengine_3d_creator.generation.compositor import generate
from ironengine_3d_creator.generation.features import apply_relief
from ironengine_3d_creator.generation.primitives import sample_box
from ironengine_3d_creator.generation.soft_author import author_cloth
from ironengine_3d_creator.generation.style_engine import StyleEngine


def _P(kind, params, transform, label):
    return Primitive(kind, transform, params, label)


def _T(x, y, z):
    return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]


# ----------------------------------------------------------------------
# 1. attachment solver
# ----------------------------------------------------------------------


def test_broken_chair_floating_seat_and_back_are_welded():
    legs = [
        _P("cylinder", {"radius": 0.03, "height": 0.45}, _T(x, 0.225, z),
           f"leg_{i}")
        for i, (x, z) in enumerate(
            ((-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)))
    ]
    seat = _P("box", {"size": [0.5, 0.05, 0.5]}, _T(0, 1.20, 0), "seat")
    back = _P("box", {"size": [0.5, 0.5, 0.05]}, _T(0, 2.00, -0.22), "back")
    spec = GenerationSpec(shape="chair", primitives=legs + [seat, back])
    spec, warnings = check_and_fix(spec)
    report = assembly_report(spec)
    assert report["floating"] == [], report
    lo, hi = _world_aabb(spec.primitives[4])  # seat
    # Seat sits on the legs (top 0.45 m) with only the weld embed.
    assert lo[1] == pytest.approx(0.45 - 0.002, abs=0.01)
    assert any("integrity" in w for w in warnings)


def test_gate_with_floating_pickets_is_grounded():
    parts = [
        _P("box", {"size": [0.1, 1.2, 0.1]}, _T(-0.9, 0.6, 0), "post_l"),
        _P("box", {"size": [0.1, 1.2, 0.1]}, _T(0.9, 0.6, 0), "post_r"),
        _P("box", {"size": [1.9, 0.08, 0.06]}, _T(0, 0.04, 0), "rail_bottom"),
        _P("box", {"size": [1.9, 0.08, 0.06]}, _T(0, 1.14, 0), "rail_top"),
    ]
    for i in range(5):
        x = -0.6 + 0.3 * i
        parts.append(
            _P("box", {"size": [0.05, 0.9, 0.05]}, _T(x, 2.0, 0),
               f"picket_{i}")
        )
    spec = GenerationSpec(shape="gate", primitives=parts)
    spec, _ = check_and_fix(spec)
    report = assembly_report(spec)
    assert report["floating"] == [], report


def test_table_legs_off_floor_are_rescued():
    legs = [
        _P("cylinder", {"radius": 0.03, "height": 0.72}, _T(x, 0.55, z),
           f"leg_{i}")
        for i, (x, z) in enumerate(
            ((-0.3, -0.2), (0.3, -0.2), (-0.3, 0.2), (0.3, 0.2)))
    ]
    top = _P("box", {"size": [0.9, 0.04, 0.6]}, _T(0, 0.93, 0), "tabletop")
    spec = GenerationSpec(shape="table", primitives=legs + [top])
    spec, _ = check_and_fix(spec)
    for p in spec.primitives:
        if not (p.label or "").startswith("leg"):
            continue
        lo, _ = _world_aabb(p)
        assert lo[1] == pytest.approx(0.0, abs=0.003)
    assert assembly_report(spec)["floating"] == []


def test_deep_intrusion_is_lifted_to_weld_embed():
    slab = _P("box", {"size": [0.5, 0.1, 0.5]}, _T(0, 0.05, 0), "slab")
    # 0.1 m cube, sunk 40 mm into the slab top (y=0.1) — beyond the 25 mm
    # limit but under the 50 % swallowed threshold.
    knob = _P("box", {"size": [0.1, 0.1, 0.1]}, _T(0, 0.11, 0), "knob")
    spec = GenerationSpec(shape="shelf", primitives=[slab, knob])
    spec, warnings = check_and_fix(spec)
    lo, _ = _world_aabb(spec.primitives[1])
    assert lo[1] == pytest.approx(0.1 - 0.0015, abs=1e-3)
    assert any("intrusion" in w for w in warnings)
    assert assembly_report(spec)["deep_intrusions"] == []


# ----------------------------------------------------------------------
# 2. proportion truth tables
# ----------------------------------------------------------------------


def _chair_at(seat_y: float, leg_h: float):
    legs = [
        _P("cylinder", {"radius": 0.03, "height": leg_h}, _T(x, leg_h / 2, z),
           f"leg_{i}")
        for i, (x, z) in enumerate(
            ((-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)))
    ]
    seat = _P("box", {"size": [0.5, 0.05, 0.5]}, _T(0, seat_y, 0), "seat")
    return GenerationSpec(shape="chair", primitives=legs + [seat])


def test_chair_surface_height_corrected_into_range():
    spec = _chair_at(seat_y=0.30, leg_h=0.275)  # surface 0.325 m — too low
    spec, warnings = check_and_fix(spec)
    _, hi = _world_aabb(spec.primitives[4])
    assert 0.43 - 1e-3 <= hi[1] <= 0.50 + 1e-3
    assert any("truth table" in w for w in warnings)
    # Legs were stretched, not detached: still grounded, seat still on top.
    assert assembly_report(spec)["floating"] == []


def test_table_surface_height_corrected_into_range():
    legs = [
        _P("cylinder", {"radius": 0.03, "height": 0.55}, _T(x, 0.275, z),
           f"leg_{i}")
        for i, (x, z) in enumerate(
            ((-0.3, -0.2), (0.3, -0.2), (-0.3, 0.2), (0.3, 0.2)))
    ]
    top = _P("box", {"size": [0.9, 0.04, 0.6]}, _T(0, 0.57, 0), "tabletop")
    spec = GenerationSpec(shape="table", primitives=legs + [top])
    spec, warnings = check_and_fix(spec)
    _, hi = _world_aabb(spec.primitives[4])
    assert 0.72 - 1e-3 <= hi[1] <= 0.76 + 1e-3
    assert any("truth table" in w for w in warnings)


def test_vase_neck_wider_than_belly_is_narrowed():
    body = _P("cylinder", {"radius": 0.10, "height": 0.30}, _T(0, 0.15, 0),
              "body")
    neck = _P("cylinder", {"radius": 0.12, "height": 0.10}, _T(0, 0.35, 0),
              "neck")
    spec = GenerationSpec(shape="vase", primitives=[body, neck])
    spec, warnings = check_and_fix(spec)
    _, hi = _world_aabb(spec.primitives[1])
    neck_r = (hi[0] - (-hi[0])) / 2 if hi[0] > 0 else None
    lo_n, hi_n = _world_aabb(spec.primitives[1])
    neck_r = float(hi_n[0] - lo_n[0]) / 2
    assert neck_r == pytest.approx(0.055, abs=0.01)
    assert any("neck" in w and "narrowed" in w for w in warnings)


def test_reversed_capital_cone_is_flipped():
    shaft = _P("cylinder", {"radius": 0.08, "height": 1.0}, _T(0, 0.5, 0),
               "shaft")
    capital = _P("cone", {"radius": 0.14, "height": 0.12}, _T(0, 1.06, 0),
                 "capital")
    spec = GenerationSpec(shape="pillar", primitives=[shaft, capital])
    spec, warnings = check_and_fix(spec)
    T = np.asarray(spec.primitives[1].transform, dtype=np.float64)
    col_y = T[:3, 1]
    # Local +Y (cone apex) must now point world-down (wide end up).
    assert float(col_y[1]) < 0.0
    assert any("reversed cone" in w for w in warnings)


# ----------------------------------------------------------------------
# 3. organic grammars
# ----------------------------------------------------------------------


def test_insect_family_anatomy_and_connectivity():
    spec = StyleEngine(seed=3).generate(family="insect", n_points=6000)
    labels = [p.label or "" for p in spec.primitives]
    assert any("head" in l for l in labels)
    assert any("thorax" in l for l in labels)
    assert any("abdomen" in l for l in labels)
    assert sum(l.startswith("femur_") for l in labels) == 6
    assert sum(l.startswith("tibia_") for l in labels) == 6
    assert sum("wing" in l for l in labels) == 2
    assert sum("antenna" in l for l in labels) == 2
    clean, _ = normalize(spec)
    fixed, _ = check_and_fix(clean)
    assert assembly_report(fixed)["floating"] == []
    res = generate(clean)
    assert np.isfinite(res.positions).all()


def test_creature_family_has_paws():
    spec = StyleEngine(seed=5).generate(family="creature", n_points=6000)
    labels = [p.label or "" for p in spec.primitives]
    assert sum(l.startswith("paw_") for l in labels) >= 4


def test_flower_and_leaf_families_generate_connected():
    for family in ("flower", "leaf"):
        spec = StyleEngine(seed=7).generate(family=family, n_points=6000)
        clean, warns = normalize(spec)
        assert not [w for w in warns if "dropped" in w], warns
        res = generate(clean)
        assert res.positions.shape[0] >= 3000
        assert np.isfinite(res.positions).all()
        fixed, _ = check_and_fix(clean)
        assert assembly_report(fixed)["floating"] == []


# ----------------------------------------------------------------------
# 4. surface realism
# ----------------------------------------------------------------------


def test_box_bevel_chamfers_edges():
    rng_plain = np.random.default_rng(42)
    rng_bevel = np.random.default_rng(42)
    params = {"size": [1.0, 1.0, 1.0]}
    plain = sample_box(20000, dict(params, bevel=0.0), rng_plain)
    beveled = sample_box(20000, dict(params), rng_bevel)  # default auto bevel
    # Chamfer plane: for every axis pair, |xi| + |xj| <= hi + hj - bevel.
    # Default bevel = 6 % of the min half-extent, clamped to [0.5 mm, 4 mm].
    bevel = min(0.004, max(0.0005, 0.06 * 0.5))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert float((np.abs(beveled[:, i]) + np.abs(beveled[:, j])).max()) \
            <= 1.0 - bevel + 1e-5
    # The unbeveled box really does have near-sharp edges to remove.
    assert float((np.abs(plain[:, 0]) + np.abs(plain[:, 1])).max()) \
        > 1.0 - bevel


def test_relief_keeps_point_count_and_is_deterministic():
    xs = np.linspace(-0.5, 0.5, 60)
    zz, xx = np.meshgrid(xs, xs, indexing="ij")
    base = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()],
                    axis=-1).astype(np.float32)
    colors = np.full((base.shape[0], 3), 0.5, dtype=np.float32)
    mask = np.ones(base.shape[0], dtype=bool)
    params = {"amplitude": 0.02, "frequency": 6.0, "pebbles": 4}

    p1, c1 = base.copy(), colors.copy()
    apply_relief(p1, c1, mask, params, np.random.default_rng(9))
    p2, c2 = base.copy(), colors.copy()
    apply_relief(p2, c2, mask, params, np.random.default_rng(9))

    assert p1.shape == base.shape
    np.testing.assert_allclose(p1, p2, atol=0, rtol=0)
    np.testing.assert_allclose(c1, c2, atol=0, rtol=0)
    # Terrain actually undulates (not a flat sheet anymore).
    assert float(p1[:, 1].max() - p1[:, 1].min()) > 0.005


def test_cloth_defaults_keep_legacy_extras():
    res = author_cloth()
    assert res.extras["cloth"] == {
        "width_m": 0.6,
        "height_m": 0.4,
        "resolution": [24, 16],
    }


def test_cloth_cylinder_drape_hangs_down():
    res = author_cloth(drape="cylinder", drape_radius=0.15)
    ys = res.positions[:, 1]
    assert float(ys.max()) == pytest.approx(0.0, abs=1e-3)
    assert float(ys.min()) == pytest.approx(-0.4, abs=1e-3)
    r = np.hypot(res.positions[:, 0], res.positions[:, 2] + 0.15)
    assert float(r.min()) > 0.10
    assert float(r.max()) < 0.20
    assert res.extras["cloth"]["drape"] == "cylinder"


def test_cloth_thickness_adds_back_layer():
    thin = author_cloth(n_points=2000, seed=1)
    thick = author_cloth(n_points=2000, seed=1, thickness=0.004)
    assert thick.positions.shape[0] == 2 * thin.positions.shape[0]
    assert thick.extras["cloth"]["thickness_m"] == pytest.approx(0.004)
    # Mesh is solidified: more vertices than the bare grid (24*16).
    assert thick.parts[0].vertices.shape[0] == 2 * 24 * 16


def test_cloth_weave_modulates_albedo():
    plain = author_cloth(n_points=4000, seed=2)
    woven = author_cloth(n_points=4000, seed=2, weave=True)
    assert not np.allclose(plain.colors, woven.colors)
    # Modulation is subtle: ±4 % around the same base colours.
    ratio = woven.colors / np.maximum(plain.colors, 1e-6)
    assert float(np.abs(ratio - 1.0).max()) <= 0.06
    assert woven.extras["cloth"]["weave"] is True
