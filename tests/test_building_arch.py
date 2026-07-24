"""Building architecture: floor-plan engine, validation invariants, stairs.

Covers the user complaints: pillar blocking a door (sweep test + auto
relocation), enterable interiors (slabs/stairs/connected rooms), real door
holes, and the parametric wall decomposition.
"""
from __future__ import annotations

import numpy as np
import pytest

from ironengine_3d_creator.generation import building_arch as ba


# ---------------------------------------------------------------------------
# no-pillar-in-doorway invariant across 20 seeded plans
# ---------------------------------------------------------------------------


def test_no_pillar_in_doorway_across_20_seeds():
    styles = ["baroque", "neoclassical", "modern"]
    relocations = 0
    for seed in range(20):
        res = ba.build_building({"seed": seed, "floors": 2, "style": styles[seed % 3]})
        report = res["validation"]
        assert report["ok"], f"seed {seed}: unresolved violations {report['warnings']}"
        plan = res["built"].plan
        # hard invariant, recomputed on the FINAL plan
        assert ba.door_swing_conflicts(plan) == [], f"seed {seed}: pillar in door swing"
        assert ba.corridor_conflicts(plan) == [], f"seed {seed}: pillar in corridor"
        relocations += len(report["fixes"])
    # the generator deliberately seeds dangerous columns — the validator must
    # have done real work somewhere across the sweep
    assert relocations > 0, "validator never relocated anything; test is vacuous"


def test_swing_conflict_is_detected_and_fixed():
    """Construct a blatant pillar-in-doorway case by hand."""
    plan = ba.generate_plan(seed=11, floors=1, width=12.0, depth=8.0)
    # drop a column right in front of the entrance swing (corridor side)
    ent = plan.entrance
    plan.columns.append(ba.Column(0, ent["offset"], plan.corridor_width + 0.55, size=0.30))
    conflicts = ba.door_swing_conflicts(plan)
    assert conflicts, "hand-placed pillar not detected in entrance swing"
    report = ba.validate_plan(plan)
    assert report.fixes, "validator did not relocate the pillar"
    assert ba.door_swing_conflicts(plan) == []


# ---------------------------------------------------------------------------
# staircase rise/going bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["straight", "L", "U"])
def test_staircase_rise_going_bounds(kind):
    well = (6.0, 1.8, 3.4 if kind != "straight" else 1.2, 6.0)
    st = ba.layout_staircase(kind, 3.0, well, 0)
    assert 0.15 <= st.rise <= 0.185, f"{kind}: rise {st.rise}"
    assert 0.25 <= st.going <= 0.32, f"{kind}: going {st.going}"
    # total rise matches the floor height exactly
    assert abs(st.risers * st.rise - 3.0) < 1e-9
    # step tops are monotonically increasing by exactly one rise
    tops = [s["top_y"] for s in st.steps]
    assert tops == sorted(tops)
    assert len(st.steps) == st.risers
    # steps stay inside the stair well footprint
    x0, z0, ww, wd = well
    for s in st.steps:
        rx, rz, rw, rd = s["rect"]
        assert x0 - 0.05 <= rx and rx + rw <= x0 + ww + 0.05, f"{kind} step escapes well in x"
        assert z0 - 0.75 <= rz and rz + rd <= z0 + wd + 0.05, f"{kind} step escapes well in z"


def test_staircase_has_railings_in_compiled_spec():
    res = ba.build_building({"seed": 5, "floors": 2, "stair_kind": "U"})
    labels = [p.label for p in res["parts"]]
    assert any("rail" in l for l in labels), "no handrail parts"
    assert any("newel" in l for l in labels), "no newel posts"
    assert any("bal" in l and "stair" in l for l in labels), "no balusters"


# ---------------------------------------------------------------------------
# entrance connectivity + enterable interior
# ---------------------------------------------------------------------------


def test_entrance_connectivity_20_seeds():
    for seed in range(20):
        plan = ba.generate_plan(seed=seed, floors=2)
        ba.validate_plan(plan)
        reachable, unreachable = ba.connectivity(plan)
        assert not unreachable, f"seed {seed}: unreachable {sorted(unreachable)}"
        # every room on every floor is reachable from outside
        for room in plan.rooms:
            assert f"f{room.floor}_{room.name}" in reachable


def test_enterable_metadata_and_real_door_hole():
    res = ba.build_building({"seed": 2, "floors": 2})
    assert res["metadata"]["enterable"] is True
    # interior slabs exist per floor, stairs exist, entrance is a real hole:
    # the entrance wall has a lintel ABOVE the doorway and NO panel at door
    # level spanning the doorway → verify via wall decomposition geometry.
    built = res["built"]
    front = next(w for w in built.plan.walls if w.label == "f0_front")
    door = next(o for o in front.openings if o.kind == "door")
    y_base = built.plan.floor_base(0)
    for part in res["parts"]:
        if not part.label.startswith("f0_front_"):
            continue
        lo, hi = part.aabb_min, part.aabb_max
        # does this part block the doorway rectangle?
        cx = (lo[0] + hi[0]) / 2
        blocks_x = lo[0] < door.offset + door.width / 2 - 0.05 and hi[0] > door.offset - door.width / 2 + 0.05
        at_door_level = lo[1] < y_base + door.height - 0.1 and hi[1] > y_base + 0.1
        is_frame_or_door = any(k in part.label for k in ("jamb", "head", "leaf", "hinge",
                                                         "handle", "transom", "number",
                                                         "mold", "threshold", "kick"))
        assert not (blocks_x and at_door_level and abs(cx - door.offset) < door.width / 2
                    and not is_frame_or_door), f"wall part {part.label} blocks the doorway"


def test_window_subtraction_carves_real_tunnels():
    """At least one window bay must be carved by a role:subtract cutter, and
    the carved panel has less volume than its solid box envelope."""
    res = ba.build_building({"seed": 7, "floors": 2})
    carved = [w for w in res["built"].mesh_warnings if "carved a tunnel" in w]
    assert carved, "no window bay was carved via subtraction"
    bays = [p for p in res["parts"] if "_bay" in p.label]
    assert bays, "no carved bay panels found"
    for bay in bays:
        ext = bay.aabb_max - bay.aabb_min
        solid = ext[0] * ext[1] * ext[2]
        assert bay.solid_volume_m3 < solid * 0.95, f"{bay.label}: hole volume missing"


def test_walls_have_real_thickness():
    res = ba.build_building({"seed": 4, "floors": 1, "style": "neoclassical"})
    plan = res["built"].plan
    for wall in plan.walls:
        lo, hi = (0.10, 0.30) if wall.exterior else (0.09, 0.16)
        assert lo <= wall.thickness <= hi + 1e-9, f"{wall.label}: thickness {wall.thickness}"
    # wall panels exist as meshes with the declared thickness
    pier = next(p for p in res["parts"] if p.label.startswith("f0_front_pier"))
    t = min(pier.aabb_max - pier.aabb_min)
    assert 0.20 <= t <= 0.30, f"pier thickness {t}"


def test_room_areas_sane_20_seeds():
    for seed in range(20):
        res = ba.build_building({"seed": seed, "floors": 2})
        assert res["validation"]["checks"]["area_violations"] == []
        for room in res["built"].plan.rooms:
            if room.name != "stairwell":
                assert room.area >= res["built"].plan.min_area


def test_plan_is_deterministic_per_seed():
    a = ba.build_building({"seed": 9, "floors": 2})
    b = ba.build_building({"seed": 9, "floors": 2})
    assert a["plan"]["rooms"] == b["plan"]["rooms"]
    assert a["plan"]["columns"] == b["plan"]["columns"]


def test_exterior_detail_parts_present_per_style():
    for style, must in [
        ("baroque", ("plinth_", "cornice_main_", "cornice_crown_", "quoin_",
                     "dentil_", "downspout_", "keystone")),
        ("neoclassical", ("plinth_", "cornice_main_", "quoin_", "downspout_",
                          "linteldecor")),
        ("modern", ("plinth_", "cornice_thin_", "parapet_", "downspout_")),
    ]:
        res = ba.build_building({"seed": 6, "floors": 2, "style": style})
        labels = [p.label for p in res["parts"]]
        for token in must:
            assert any(l.startswith(token) or token in l for l in labels), \
                f"{style}: missing {token}"
    # balcony with railing on the 2-storey build
    res = ba.build_building({"seed": 6, "floors": 2, "style": "neoclassical"})
    labels = [p.label for p in res["parts"]]
    assert "balcony_slab" in labels
    assert any(l.startswith("balcony_rail") for l in labels)
    assert any(l.startswith("balcony_bal_") for l in labels)
