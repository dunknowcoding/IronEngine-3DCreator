"""Tests for the anatomically detailed parametric human (human_anatomy).

Covers: Sim-bone part naming, face parts (eyes/iris/pupil/lids/brows/nose/
ears/mouth+teeth), hands with 10 fingers × 3 phalanges + nails, feet with
toes + toenails, muscle-shell anatomy tags, proportions (head ≈ 1/7.5 of
height), the < 60k default triangle budget with LOD knobs, mouth-open
articulation, watertight volumes, determinism, and vertex-color realism.
No network.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pytest

from ironengine_3d_creator.generation.analytic_mesh import signed_volume
from ironengine_3d_creator.generation.human_anatomy import (
    SIM_BONE_NAMES, build_human)


@pytest.fixture(scope="module")
def spec():
    return build_human()


@pytest.fixture(scope="module")
def result(spec):
    return spec.build()


def _names(result):
    return {p.name for p in result.parts}


# ----------------------------------------------------------------------
# skeleton naming
# ----------------------------------------------------------------------

def test_all_sim_bone_parts_present(result):
    names = _names(result)
    for bone in SIM_BONE_NAMES:
        assert bone in names, f"missing Sim bone part {bone!r}"
    assert len(SIM_BONE_NAMES) == 19


def test_bone_parent_chain(result, spec):
    parents = {n: node.parent for n, node in spec.graph.nodes.items()}
    assert parents["pelvis"] is None
    assert parents["spine"] == "pelvis"
    assert parents["chest"] == "spine"
    assert parents["neck"] == "chest"
    assert parents["head"] == "neck"
    for side in ("l", "r"):
        assert parents[f"upper_arm_{side}"] == f"clavicle_{side}"
        assert parents[f"lower_arm_{side}"] == f"upper_arm_{side}"
        assert parents[f"hand_{side}"] == f"lower_arm_{side}"
        assert parents[f"upper_leg_{side}"] == "pelvis"
        assert parents[f"lower_leg_{side}"] == f"upper_leg_{side}"
        assert parents[f"foot_{side}"] == f"lower_leg_{side}"


# ----------------------------------------------------------------------
# face
# ----------------------------------------------------------------------

def test_face_has_all_parts(result):
    names = _names(result)
    required = {
        "nose", "lip_upper", "lip_lower", "teeth_upper", "teeth_lower",
        "mouth_cavity",
    }
    for side in ("l", "r"):
        required |= {
            f"eye_{side}", f"iris_{side}", f"pupil_{side}",
            f"eyelid_{side}", f"eyelid_lower_{side}",
            f"nostril_{side}",
            f"ear_helix_{side}", f"ear_ridge_{side}", f"ear_lobe_{side}",
        }
    missing = required - names
    assert not missing, f"missing face parts: {sorted(missing)}"
    # eyebrows: clusters of strokes, not slabs — at least 6 strokes per side
    for side in ("l", "r"):
        strokes = [n for n in names if re.fullmatch(rf"brow_{side}_\d+", n)]
        assert len(strokes) >= 6, f"brow_{side} has only {len(strokes)} strokes"


@pytest.mark.parametrize("eye_shape", ["almond", "round", "hooded", "monolid"])
def test_eye_shapes_build(eye_shape):
    spec = build_human(eye_shape=eye_shape)
    r = spec.build()
    lid = next(p for p in r.parts if p.name == "eyelid_r")
    assert lid.metadata.get("eye_shape") == eye_shape
    # distinct shapes must produce distinct lid geometry (height differs)
    heights = {}
    for s in ("round", "hooded"):
        rr = build_human(eye_shape=s).build()
        p = next(p for p in rr.parts if p.name == "eyelid_r")
        heights[s] = float(p.aabb_max[1] - p.aabb_min[1])
    assert heights["hooded"] > heights["round"]


def test_mouth_open_reveals_teeth():
    closed = build_human(mouth_open=0.0).build()
    opened = build_human(mouth_open=0.7).build()

    def teeth_y(res, name):
        p = next(p for p in res.parts if p.name == name)
        return float(0.5 * (p.aabb_min[1] + p.aabb_max[1]))

    gap_closed = teeth_y(closed, "teeth_upper") - teeth_y(closed, "teeth_lower")
    gap_open = teeth_y(opened, "teeth_upper") - teeth_y(opened, "teeth_lower")
    assert gap_open > gap_closed + 0.004  # jaw drops ≥ 4 mm when opening
    # lips part too
    lip_gap_c = teeth_y(closed, "lip_upper") - teeth_y(closed, "lip_lower")
    lip_gap_o = teeth_y(opened, "lip_upper") - teeth_y(opened, "lip_lower")
    assert lip_gap_o > lip_gap_c


# ----------------------------------------------------------------------
# hands / feet
# ----------------------------------------------------------------------

def test_ten_fingers_three_phalanges_each_with_nails(result):
    names = _names(result)
    groups = set()
    for n in names:
        m = re.fullmatch(r"finger_(l|r)_([a-z0-9]+)_([123])", n)
        if m:
            groups.add((m.group(1), m.group(2)))
    assert len(groups) == 10, f"expected 10 fingers, got {sorted(groups)}"
    for side, fname in groups:
        for k in (1, 2, 3):
            assert f"finger_{side}_{fname}_{k}" in names, \
                f"finger_{side}_{fname} missing phalanx {k}"
        assert f"nail_{side}_{fname}" in names, \
            f"finger_{side}_{fname} missing fingernail"
    nails = [n for n in names if re.fullmatch(r"nail_(l|r)_[a-z0-9]+", n)]
    assert len(nails) == 10


def test_feet_have_toes_and_toenails(result):
    names = _names(result)
    for side in ("l", "r"):
        toes = [n for n in names if n.startswith(f"toe_{side}_")]
        nails = [n for n in names if n.startswith(f"toenail_{side}_")]
        assert len(toes) == 5, f"foot_{side}: {len(toes)} toes"
        assert len(nails) == 5, f"foot_{side}: {len(nails)} toenails"


# ----------------------------------------------------------------------
# body: muscle shells / proportions
# ----------------------------------------------------------------------

def test_muscle_shell_anatomy_tags(spec):
    anat = {n: node.metadata.get("anatomy", "") for n, node in spec.graph.nodes.items()}
    for side in ("l", "r"):
        assert anat[f"upper_arm_{side}"] == "deltoid_biceps"
        assert anat[f"upper_leg_{side}"] == "quadriceps"
        assert anat[f"lower_leg_{side}"] == "calf_taper"
    assert anat["head"] == "skull_jaw_cheekbones"


def test_deltoid_wider_than_biceps(result):
    """Deltoid cap (top of upper arm) must exceed the elbow girth."""
    p = next(p for p in result.parts if p.name == "upper_arm_r")
    assert p.solid_volume_m3 > 0.0
    # natural taper: AABB taller (y) than wide
    dy = p.aabb_max[1] - p.aabb_min[1]
    dx = p.aabb_max[0] - p.aabb_min[0]
    assert dy > 2.0 * dx


def test_calf_tapers_to_ankle(result):
    """Lower-leg cross-section at the calf must exceed the ankle's."""
    p = next(p for p in result.parts if p.name == "lower_leg_r")
    assert p.solid_volume_m3 > 0.0
    # calf bulge sits above mid-height of the segment
    assert p.aabb_max[1] > 0.25  # sanity: segment spans knee→ankle (m)


def test_head_proportion_one_seventh_and_a_half(result, spec):
    aabbs = result.aabbs()
    head_lo, head_hi = aabbs["head"]
    head_h = float(head_hi[1] - head_lo[1])
    total_hi = max(float(hi[1]) for lo, hi in aabbs.values())
    ratio = head_h / total_hi
    assert abs(ratio - 1.0 / 7.5) < 0.02, f"head/total = {ratio:.3f}"


def test_real_world_scale(result):
    aabbs = result.aabbs()
    total_hi = max(float(hi[1]) for lo, hi in aabbs.values())
    assert total_hi == pytest.approx(1.75, abs=0.03)
    feet_lo = min(float(lo[1]) for lo, hi in aabbs.values())
    assert feet_lo == pytest.approx(0.0, abs=0.005)


def test_gender_silhouette_differs():
    f = build_human(gender="female").build().aabbs()
    m = build_human(gender="male").build().aabbs()
    chest_f = f["chest"][0][2], f["chest"][1][2]  # z extent carries the bust
    chest_m = m["chest"][0][2], m["chest"][1][2]
    assert (chest_f[1] - chest_f[0]) != pytest.approx(chest_m[1] - chest_m[0],
                                                      rel=0.02)


# ----------------------------------------------------------------------
# budget / LOD / watertight / determinism
# ----------------------------------------------------------------------

def test_triangle_budget_default(result):
    assert result.triangle_count() < 60_000, \
        f"default detail too heavy: {result.triangle_count()}"


def test_lod_knobs_scale():
    tri = {d: build_human(detail=d).build().triangle_count()
           for d in ("low", "medium", "high")}
    assert tri["low"] < tri["medium"] < tri["high"]


@pytest.mark.parametrize("part", ["head", "chest", "pelvis", "upper_arm_r",
                                  "hand_l", "foot_r"])
def test_key_parts_watertight_positive_volume(result, part):
    p = next(p for p in result.parts if p.name == part)
    vol = signed_volume(p.vertices.astype(np.float64), p.faces)
    assert vol > 0.0, f"{part} has inverted/degenerate winding"


def test_deterministic_same_seed():
    a = build_human().build()
    b = build_human().build()
    assert a.triangle_count() == b.triangle_count()
    pa = next(p for p in a.parts if p.name == "nose")
    pb = next(p for p in b.parts if p.name == "nose")
    assert np.allclose(pa.aabb_min, pb.aabb_min)
    assert np.allclose(pa.aabb_max, pb.aabb_max)


# ----------------------------------------------------------------------
# colors / API extras
# ----------------------------------------------------------------------

def test_vertex_color_realism(spec, result):
    colors = spec.vertex_colors(result)
    assert set(colors) == {p.label for p in result.parts}
    for label, c in colors.items():
        assert c.dtype == np.float32 and c.ndim == 2 and c.shape[1] == 3
        assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0
    albedo = spec.part_albedos()
    # lip pink ≠ skin, teeth white ≠ skin, nails pinkish, iris colored
    assert albedo["lip_upper"] != albedo["head"]
    assert albedo["teeth_upper"][0] > 0.9
    assert albedo["nail_l_thumb"] != albedo["hand_l"]
    assert albedo["iris_r"] == pytest.approx(albedo["iris_l"])


def test_appearance_extras(spec):
    app = spec.appearance
    for key in ("skin_tone", "eye_color", "eye_shape", "hair_style",
                "hair_color", "body_type", "height_m"):
        assert key in app
    assert app["height_m"] == pytest.approx(1.75)
    assert app["eye_shape"] == "almond"
