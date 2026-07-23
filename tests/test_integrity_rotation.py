"""Tests for integrity rotation repair, interpenetration flagging, and
validator proportion guides (CR_ComplexGeometry)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ironengine_3d_creator.alignment.defaults import auto_template
from ironengine_3d_creator.alignment.integrity import check_and_fix, _world_aabb
from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.alignment.validator import normalize


def _P(kind, params, transform, label):
    return Primitive(kind, transform, params, label)


def _rot_z(a: float) -> list[list[float]]:
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]


def _chair(leg_transforms, leg_params=None):
    leg_params = leg_params or {"radius": 0.03, "height": 0.45}
    legs = [
        _P("cylinder", dict(leg_params), T, f"leg_{i}")
        for i, T in enumerate(leg_transforms)
    ]
    seat = _P("box", {"size": [0.5, 0.05, 0.5]},
              [[1, 0, 0, 0], [0, 1, 0, 0.475], [0, 0, 1, 0], [0, 0, 0, 1]],
              "seat")
    return GenerationSpec(shape="chair", primitives=[seat] + legs)


def _upright_legs():
    return [
        [[1, 0, 0, x], [0, 1, 0, 0.225], [0, 0, 1, z], [0, 0, 0, 1]]
        for x, z in ((-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2))
    ]


def test_rotated_leg_is_uprighted_and_grounded():
    Ts = _upright_legs()
    fallen = _rot_z(math.pi / 2)  # long axis now points along +X
    fallen[0][3], fallen[1][3], fallen[2][3] = -0.2, 0.03, -0.2
    Ts[0] = fallen
    spec, warnings = check_and_fix(_chair(Ts))
    assert any("uprighted" in w and "leg_0" in w for w in warnings)
    for p in spec.primitives:
        if not (p.label or "").startswith("leg"):
            continue
        lo, hi = _world_aabb(p)
        # Vertical again: y span is the long dimension, bottom on the floor.
        assert hi[1] - lo[1] == pytest.approx(0.45, abs=1e-4)
        assert lo[1] == pytest.approx(0.0, abs=1e-4)
        assert hi[0] - lo[0] == pytest.approx(0.06, abs=1e-4)


def test_upright_chair_untouched_by_rotation_repair():
    spec, warnings = check_and_fix(_chair(_upright_legs()))
    assert not any("uprighted" in w for w in warnings)


def test_fence_picket_spacing_preserved():
    spec = auto_template("fence")
    before = {
        p.label: (float(np.asarray(p.transform)[0, 3]),
                  float(np.asarray(p.transform)[2, 3]))
        for p in spec.primitives
    }
    spec, warnings = check_and_fix(spec)
    for p in spec.primitives:
        bx, bz = before[p.label]
        assert float(np.asarray(p.transform)[0, 3]) == pytest.approx(bx, abs=1e-6)
        assert float(np.asarray(p.transform)[2, 3]) == pytest.approx(bz, abs=1e-6)
    # No picket was dragged toward a neighbour.
    assert not any("pulled" in w for w in warnings)


def test_swallowed_part_flagged():
    big = _P("box", {"size": [1, 1, 1]},
             [[1, 0, 0, 0], [0, 1, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]], "body")
    small = _P("box", {"size": [0.2, 0.2, 0.2]},
               [[1, 0, 0, 0], [0, 1, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]],
               "knob")
    spec = GenerationSpec(shape="other", primitives=[big, small])
    _, warnings = check_and_fix(spec)
    assert any("knob" in w and "inside" in w for w in warnings)


def test_overlap_joint_not_flagged():
    # A leg top overlapping a seat bottom is an idiomatic join, not a
    # swallowed part — must NOT trigger the interpenetration flag.
    spec, warnings = check_and_fix(_chair(_upright_legs()))
    assert not any("inside" in w for w in warnings)


def test_proportion_clamp_fat_chair_leg():
    # diameter 0.12 > chair-leg guide max 0.06 → soft clamp with warning.
    spec = _chair(_upright_legs(), leg_params={"radius": 0.06, "height": 0.45})
    clean, warnings = normalize(spec)
    assert any("proportion guide" in w for w in warnings)
    lo, hi = _world_aabb(clean.primitives[1])
    assert hi[0] - lo[0] == pytest.approx(0.06, abs=1e-5)


def test_proportion_guide_accepts_good_chair():
    # diameter 0.06 is exactly the guide max (inclusive) → no clamp.
    spec = _chair(_upright_legs(), leg_params={"radius": 0.03, "height": 0.45})
    clean, warnings = normalize(spec)
    assert not any("proportion guide" in w for w in warnings)
    lo, hi = _world_aabb(clean.primitives[1])
    assert hi[0] - lo[0] == pytest.approx(0.06, abs=1e-5)


def test_proportion_clamp_thin_tabletop():
    # table seat (tabletop) guide is 0.02–0.05 m; 0.10 → clamped to 0.05.
    top = _P("box", {"size": [1.2, 0.10, 0.8]},
             [[1, 0, 0, 0], [0, 1, 0, 0.75], [0, 0, 1, 0], [0, 0, 0, 1]],
             "tabletop")
    leg = _P("cylinder", {"radius": 0.04, "height": 0.72},
             [[1, 0, 0, 0.5], [0, 1, 0, 0.36], [0, 0, 1, 0.3], [0, 0, 0, 1]],
             "leg_0")
    spec = GenerationSpec(shape="table", primitives=[top, leg])
    clean, warnings = normalize(spec)
    assert any("tabletop" in w and "proportion guide" in w for w in warnings)
    lo, hi = _world_aabb(clean.primitives[0])
    assert hi[1] - lo[1] == pytest.approx(0.05, abs=1e-5)
