"""Tests for the hairstyle library (generation.hair).

Covers: all 8 styles build, scalp shell + visible hairline strokes, strand
clusters with part-line/crown-whorl support, per-strand wind_response
spring-chain physics metadata (+ documented wind-speed mapping), style-
specific structure (horseshoe bare crown, twin ponytail clusters), and the
LOD knob's effect on strand counts. No network.
"""
from __future__ import annotations

import re

import pytest

from ironengine_3d_creator.generation.hair import HAIRSTYLES, WIND_SPEED_MAPPING
from ironengine_3d_creator.generation.human_anatomy import build_human

REQUIRED_WIND_KEYS = {
    "model", "segments", "rest_lengths_m", "point_masses_g",
    "angular_stiffness_n_m_rad", "damping_ratio", "drag_area_m2",
    "anchor_part", "root_offset_head_local", "length_m",
}


def _names(spec):
    return set(spec.graph.nodes)


@pytest.mark.parametrize("style", HAIRSTYLES)
def test_every_hairstyle_builds(style):
    spec = build_human(hair_style=style)
    r = spec.build()
    assert r.triangle_count() > 0
    assert spec.extras["hair"]["style"] == style


def test_hair_style_registry_complete():
    assert set(HAIRSTYLES) == {
        "bald", "buzz", "curly", "twin_ponytails", "slicked",
        "horseshoe", "long_straight", "bob"}


def test_bald_has_no_hair_geometry():
    spec = build_human(hair_style="bald")
    names = _names(spec)
    hair_parts = [n for n in names if n.startswith(("hair", "hairline"))]
    assert hair_parts == []
    assert spec.extras["hair"]["strand_count"] == 0


@pytest.mark.parametrize("style", ["buzz", "curly", "twin_ponytails",
                                   "slicked", "long_straight", "bob"])
def test_scalp_shell_and_visible_hairline(style):
    spec = build_human(hair_style=style)
    names = _names(spec)
    assert "hair_scalp" in names, f"{style}: no scalp shell"
    strokes = [n for n in names if re.fullmatch(r"hairline_\d+", n)]
    assert len(strokes) >= 6, f"{style}: only {len(strokes)} hairline strokes"


@pytest.mark.parametrize("style", ["curly", "twin_ponytails", "slicked",
                                   "long_straight", "bob", "horseshoe"])
def test_strand_wind_physics_metadata(style):
    spec = build_human(hair_style=style)
    strands = [(n, node) for n, node in spec.graph.nodes.items()
               if node.metadata.get("wind_response")]
    assert strands, f"{style}: no strands carry wind_response"
    for name, node in strands:
        wr = node.metadata["wind_response"]
        missing = REQUIRED_WIND_KEYS - set(wr)
        assert not missing, f"{name}: wind_response missing {missing}"
        assert wr["model"] == "spring_chain"
        assert wr["anchor_part"] == "head"
        assert wr["drag_area_m2"] > 0.0
        assert wr["angular_stiffness_n_m_rad"] > 0.0
        assert wr["segments"] == len(wr["rest_lengths_m"])
        assert len(wr["point_masses_g"]) == wr["segments"] + 1


def test_wind_speed_mapping_documented():
    for key in ("indoor_still", "desk_fan", "outdoor_gust"):
        assert key in WIND_SPEED_MAPPING
    assert WIND_SPEED_MAPPING["indoor_still"]["v_m_s"][1] <= 0.2
    assert WIND_SPEED_MAPPING["desk_fan"]["v_m_s"] == (1.0, 3.0)
    assert WIND_SPEED_MAPPING["outdoor_gust"]["v_m_s"] == (5.0, 15.0)
    assert "deflection" in WIND_SPEED_MAPPING["model"]


def test_horseshoe_bares_the_crown():
    spec = build_human(hair_style="horseshoe")
    names = _names(spec)
    # patches cover sides + nape, never a full scalp shell
    assert "hair_scalp" not in names
    assert {"hair_patch_l", "hair_patch_r", "hair_patch_nape"} <= names
    # every strand roots BELOW the crown zone (male-pattern bare top)
    H = spec.params.height_m
    for n, node in spec.graph.nodes.items():
        wr = node.metadata.get("wind_response")
        if wr:
            root_y = wr["root_offset_head_local"][1]
            assert root_y < (0.945 - 0.935) * H + 0.002, \
                f"{n} roots on the crown: y_local={root_y:.4f}"


def test_twin_ponytails_have_two_clusters_and_ties():
    spec = build_human(hair_style="twin_ponytails")
    names = _names(spec)
    tails_l = [n for n in names if n.startswith("hair_tail_l_")]
    tails_r = [n for n in names if n.startswith("hair_tail_r_")]
    assert len(tails_l) >= 5 and len(tails_r) >= 5
    assert {"hair_tie_l", "hair_tie_r"} <= names


def test_strand_count_scales_with_detail():
    low = build_human(hair_style="long_straight", detail="low")
    high = build_human(hair_style="long_straight", detail="high")
    assert low.extras["hair"]["strand_count"] < high.extras["hair"]["strand_count"]


def test_hair_color_applied_to_strands():
    spec = build_human(hair_style="long_straight", hair_color="red")
    node = spec.graph.nodes["hair_strand_000"]
    assert node.metadata["albedo"][0] > node.metadata["albedo"][1]  # reddish
