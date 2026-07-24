"""Door & window style library tests: open methods, articulation blocks,
visible hardware, decorations, and building-integration of every door type."""
from __future__ import annotations

import numpy as np
import pytest

from ironengine_3d_creator.generation import doors
from ironengine_3d_creator.generation import building_arch as ba

VALID_JOINT_KINDS = {"revolute", "prismatic", "continuous"}


def _assert_articulation_valid(res: doors.DoorResult, method: str):
    art = res.extras.get("articulation")
    assert art is not None, f"{method}: no articulation block"
    assert res.extras["physics"]["body_type"] == "articulated"
    assert art["open_method"] == method
    labels = {p.label for p in res.parts}
    for j in art["joints"]:
        assert j["kind"] in VALID_JOINT_KINDS or j.get("continuous"), \
            f"{method}: bad joint kind {j['kind']}"
        assert j["parent"] in labels, f"{method}: parent {j['parent']} not a part"
        assert j["child"] in labels, f"{method}: child {j['child']} not a part"
        assert len(j["axis"]) == 3
        if j["kind"] == "prismatic":
            assert "limits_m" in j and j["limits_m"][1] > j["limits_m"][0]
    return art


# ---------------------------------------------------------------------------
# every open method produces a valid articulation block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["hinged_single", "hinged_double", "sliding",
                                    "french", "revolving", "garage"])
def test_door_articulation_blocks_valid(method):
    res = doors.build_door(method)
    assert res.parts, f"{method}: no parts"
    art = _assert_articulation_valid(res, method)
    if method.startswith("hinged") or method == "french":
        # ROM 0–110° on at least one hinge joint
        roms = [j["limits_deg"] for j in art["joints"] if j["kind"] == "revolute"]
        assert any(abs(abs(lim[1] - lim[0]) - 110.0) < 1e-6 for lim in roms), \
            f"{method}: no 0–110° hinge ROM"
    if method == "revolving":
        assert any(j.get("continuous") for j in art["joints"])
    if method == "garage":
        # section hinges between stacked panels
        assert len([j for j in art["joints"] if "section_hinge" in j["name"]]) >= 2


def test_hinged_door_has_visible_hardware():
    res = doors.hinged_door()
    labels = [p.label for p in res.parts]
    # 3 visible barrel hinges (barrel + pin caps + 2 plates each)
    assert sum(1 for l in labels if "hinge" in l and "barrel" in l) == 3
    assert any("plate_leaf" in l for l in labels)
    # lever handle + rosette
    assert any("lever" in l for l in labels)
    assert any("rosette" in l for l in labels)
    # frame: 2 jambs + head
    assert {"door_jamb_l", "door_jamb_r", "door_head"} <= set(labels)
    # swing metadata present with 0–110°
    sw = res.metadata["swing"]
    assert sw["type"] == "arc" and sw["angle_deg"] == [0.0, 110.0]
    assert sw["radius"] > 0.5


def test_door_decorations():
    res = doors.hinged_door(decorations=("moldings", "transom"), house_number="12A",
                            style="panel_wood")
    labels = [p.label for p in res.parts]
    assert any("mold_" in l for l in labels), "no molding parts"
    assert any("transom_glass" in l for l in labels), "no transom"
    assert any("number_plaque" in l for l in labels), "no house number plaque"
    sec = doors.hinged_door(style="metal_security")
    assert any("kickplate" in p.label for p in sec.parts), "no kick plate"


@pytest.mark.parametrize("style", ["panel_wood", "glass", "metal_security"])
def test_door_visual_styles(style):
    res = doors.hinged_door(style=style)
    assert res.parts
    mats = {p.material for p in res.parts}
    if style == "glass":
        assert "glass" in mats
    if style == "metal_security":
        assert "metal" in mats


def test_sliding_door_track_and_rollers():
    res = doors.sliding_door()
    labels = [p.label for p in res.parts]
    assert any("track" in l for l in labels)
    assert sum(1 for l in labels if "roller" in l) == 2
    sw = res.metadata["swing"]
    assert sw["type"] == "slide" and sw["travel"] > 1.0


def test_revolving_door_wings_and_pivot():
    res = doors.revolving_door(wings=4)
    labels = [p.label for p in res.parts]
    assert sum(1 for l in labels if l.startswith("door_wing") and l.endswith(tuple("0123"))) == 4
    assert "door_pivot" in labels


def test_garage_door_sections():
    res = doors.garage_door(sections=4)
    labels = [p.label for p in res.parts]
    assert sum(1 for l in labels if l.startswith("door_section") and "roller" not in l
               and "rib" not in l) == 4
    assert any("track_l" in l or "track_r" in l for l in labels)


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["casement", "sash", "fixed"])
def test_window_builders_valid(kind):
    res = doors.build_window(kind)
    assert res.parts
    labels = [p.label for p in res.parts]
    assert any("sill" in l for l in labels), f"{kind}: no sill"
    assert any("muntin" in l for l in labels), f"{kind}: no muntin grid"
    _assert_articulation_valid(res, kind)
    if kind == "casement":
        # visible hinges on the sashes
        assert any("hinge" in l and "barrel" in l for l in labels)
        assert any("crank" in l for l in labels)
    if kind == "sash":
        joints = res.extras["articulation"]["joints"]
        assert all(j["kind"] == "prismatic" for j in joints)


# ---------------------------------------------------------------------------
# integration: the compiled building carries valid joint metadata end-to-end
# ---------------------------------------------------------------------------


def test_building_extras_joints_reference_real_parts():
    res = ba.build_building({"seed": 8, "floors": 2})
    joints = res["extras"]["articulation"]["joints"]
    assert joints, "building carries no articulation joints"
    labels = {p.label for p in res["parts"]}
    for j in joints:
        assert j["parent"] in labels, f"joint parent {j['parent']} missing from parts"
        assert j["child"] in labels, f"joint child {j['child']} missing from parts"


def test_place_transform_moves_assembly():
    from ironengine_3d_creator.generation.complex_builder import T
    res = doors.hinged_door()
    moved = doors.place(res.parts, T(translate=(5.0, 1.0, -2.0)))
    assert moved and len(moved) == len(res.parts)
    leaf = next(p for p in moved if p.label == "door_leaf0")
    assert abs(leaf.aabb_min[0] - 5.0 - (res.parts[0].aabb_min[0] - res.parts[0].aabb_min[0])) < 10.0
    # all parts shifted by the translation
    orig_c = np.mean([p.aabb_min for p in res.parts], axis=0)
    new_c = np.mean([p.aabb_min for p in moved], axis=0)
    assert np.allclose(new_c - orig_c, [5.0, 1.0, -2.0], atol=0.2)
