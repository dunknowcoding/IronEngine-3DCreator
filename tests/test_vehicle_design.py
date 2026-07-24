"""Tests for the parametric vehicle designer (generation.vehicle_design).

Covers: per-class proportions (real-world scale), wheel-arch subtraction
leaving watertight geometry with no orphans, thin-panel glass layout,
interior part presence at high detail, distinct fascia/wheel parts,
triangle budgets and LOD knobs, vertex-color zones and liveries, and the
bake/articulation API surface. No network, no renderer.
"""
from __future__ import annotations

import math
import math
from collections import Counter

import numpy as np
import pytest

from ironengine_3d_creator.generation import vehicle_design as vd

ALL_CLASSES = ("sedan", "hatchback", "suv", "sports", "pickup", "van")


def _edge_histogram(part: vd.VehiclePart) -> Counter:
    edges = Counter()
    f = part.faces
    for a, b, c in f:
        for e in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((int(e[0]), int(e[1]))))] += 1
    return Counter(edges.values())


# ----------------------------------------------------------------------
# API surface
# ----------------------------------------------------------------------

def test_build_vehicle_returns_spec_with_expected_api():
    spec = vd.build_vehicle({"class": "sedan", "color": "candy_red"})
    assert isinstance(spec, vd.VehicleSpec)
    assert spec.vehicle_class == "sedan"
    assert spec.triangle_count() > 10_000
    assert len(spec.parts) > 100
    s = spec.summary()
    assert s["class"] == "sedan" and s["triangles"] == spec.triangle_count()
    # exporter adapter
    parts = spec.to_analytic_parts()
    assert len(parts) == len(spec.bake())
    assert all(p.vertices.shape[1] == 3 and p.faces.shape[1] == 3 for p in parts)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        vd.build_vehicle({"class": "hovercraft"})
    with pytest.raises(ValueError):
        vd.build_vehicle({"class": "sedan", "color": "invisible_pink"})
    with pytest.raises(ValueError):
        vd.build_vehicle({"class": "sedan", "livery": "flames"})
    with pytest.raises(ValueError):
        vd.build_vehicle({"class": "sedan", "lod": "ultra"})
    with pytest.raises(ValueError):
        vd.build_vehicle({"class": "sedan", "interior_detail": "medium"})


def test_deterministic_build():
    a = vd.build_vehicle({"class": "sedan"})
    b = vd.build_vehicle({"class": "sedan"})
    assert a.triangle_count() == b.triangle_count()
    ta = a.part("body_tub")
    tb = b.part("body_tub")
    assert np.array_equal(ta.vertices, tb.vertices)


# ----------------------------------------------------------------------
# proportions per class (real-world scale)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_class_proportions_match_real_world(cls):
    spec = vd.build_vehicle({"class": cls})
    cp = vd.VEHICLE_CLASSES[cls]
    d = spec.dimensions
    assert d["length_m"] == pytest.approx(cp.length)
    assert d["wheelbase_m"] == pytest.approx(cp.wheelbase)
    # overall AABB (closed state) sanity: spans the nominal footprint
    lo = np.min([p.aabb_min for p in spec.parts], axis=0)
    hi = np.max([p.aabb_max for p in spec.parts], axis=0)
    span = hi - lo
    assert span[0] == pytest.approx(cp.length, abs=0.35)
    # body shell width is exact; mirrors stick out beyond it (as in
    # real-world "width without mirrors" homologation figures)
    tub = spec.part("body_tub")
    assert tub.aabb_max[2] == pytest.approx(cp.width / 2, abs=0.03)
    assert span[2] >= cp.width
    assert span[2] <= cp.width + 0.40
    assert span[1] == pytest.approx(cp.height, abs=0.10)
    # wheels touch the ground plane y=0
    tire = spec.part("wheel_tire_fl")
    assert tire.aabb_min[1] == pytest.approx(0.0, abs=0.01)
    assert tire.aabb_max[1] == pytest.approx(cp.wheel_diameter, abs=0.02)


def test_sedan_specific_proportions():
    spec = vd.build_vehicle({"class": "sedan"})
    d = spec.dimensions
    assert 4.6 <= d["length_m"] <= 5.1
    assert 0.60 <= d["wheel_diameter_m"] <= 0.75


# ----------------------------------------------------------------------
# wheel-arch subtraction: watertight, no orphans, real openings
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_arch_subtraction_leaves_watertight_tub(cls):
    spec = vd.build_vehicle({"class": cls})
    tub = spec.part("body_tub")
    hist = _edge_histogram(tub)
    # every edge shared by exactly 2 faces → 2-manifold, zero orphans
    assert set(hist) == {2}, f"non-manifold edges: {hist}"
    assert vd._signed_volume(tub.vertices, tub.faces) > 0.0


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_arch_openings_exist_at_both_axles(cls):
    spec = vd.build_vehicle({"class": cls})
    cp = vd.VEHICLE_CLASSES[cls]
    cache = spec.geometry_cache["tub"]
    positions = cache["positions"]
    sections = cache["sections"]
    for xw in (0.0, cp.wheelbase):
        i = int(np.argmin(np.abs(positions - xw)))
        sec = sections[i]
        # flank bottom points (rocker zone) lifted far above the floor
        flank_bottom = sec[[3, 4, 5, 17, 18, 19], 0].min()
        assert flank_bottom > cp.clearance + 0.10, (
            f"{cls} axle {xw}: arch bottom {flank_bottom:.3f} m too low")
        # centre floor stays low (arch opens on the flanks only)
        assert abs(sec[0, 0] - cp.clearance) < 0.01


def test_arch_opening_clears_tire():
    spec = vd.build_vehicle({"class": "sedan"})
    lay = spec.geometry_cache["layout"]
    cp = vd.VEHICLE_CLASSES["sedan"]
    arch_top = lay.y_wc + lay.r_arch
    assert arch_top > cp.wheel_diameter + 0.02      # tire top fits the arch
    tire = spec.part("wheel_tire_fl")
    assert tire.aabb_max[1] == pytest.approx(cp.wheel_diameter, abs=0.02)
    # tire outer face stays inboard of the body side
    assert tire.aabb_max[2] < cp.width / 2 + 0.005


# ----------------------------------------------------------------------
# wheels: tread / rim / brakes as distinct detailed parts
# ----------------------------------------------------------------------

def test_wheel_parts_and_tread_detail():
    spec = vd.build_vehicle({"class": "sedan", "lod": "high"})
    for corner in ("fl", "fr", "rl", "rr"):
        for stem in ("wheel_tire_", "wheel_rim_", "wheel_spoke_",
                     "wheel_hub_", "brake_disc_", "brake_caliper_"):
            assert spec.part(stem + corner)
    tread = spec.part("wheel_tread_fl")
    assert len(tread.instances) >= 30          # instanced chevron lugs
    # one lug mesh shared by all instances (zero-copy instancing)
    assert tread.tri_count == tread.faces.shape[0] * (1 + len(tread.instances))
    # tire carcass is a closed lathe solid, not a smooth torus: its local
    # cross-section has a flat tread band (verts at max radius across width)
    tire = spec.part("wheel_tire_fl")
    v = tire.vertices
    r = np.hypot(v[:, 0], v[:, 2])
    tread_band = v[(r > r.max() - 1e-3)]
    assert tread_band.shape[0] >= 2 * 28       # two tread shoulders around
    hist = _edge_histogram(tire)
    assert set(hist) == {2}                    # carcass watertight


def test_wheel_proportions_per_class():
    for cls in ALL_CLASSES:
        spec = vd.build_vehicle({"class": cls})
        cp = vd.VEHICLE_CLASSES[cls]
        tire = spec.part("wheel_tire_fl")
        d = tire.aabb_max[1] - tire.aabb_min[1]
        assert d == pytest.approx(cp.wheel_diameter, abs=0.02)
        w = tire.aabb_max[2] - tire.aabb_min[2]
        assert w == pytest.approx(cp.tire_width, abs=0.03)


# ----------------------------------------------------------------------
# glass: thin panels, see-through layout
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_glass_parts_are_thin_panels_not_blocks(cls):
    spec = vd.build_vehicle({"class": cls})
    glass = [p for p in spec.parts if p.metadata.get("zone") == "glass"]
    assert len(glass) >= 4                      # windshield + backlight + sides
    for p in glass:
        # PCA frame of the vertices: one direction must be millimetres
        # thin (the panel normal) while the other two span real area —
        # robust for raked/curved sheets where an AABB is always fat.
        pts = p.vertices.astype(np.float64) - p.vertices.mean(axis=0)
        svals = np.linalg.svd(pts, compute_uv=False) / math.sqrt(len(pts) - 1)
        svals = np.sort(svals)
        assert svals[0] <= 0.02, (
            f"{p.name} too thick: {svals[0]*1000:.1f} mm rms")
        assert svals[1] >= 0.04, f"{p.name} not a panel: {svals}"


def test_window_band_is_glass_above_beltline():
    spec = vd.build_vehicle({"class": "sedan"})
    cp = vd.VEHICLE_CLASSES["sedan"]
    side_glass = [p for p in spec.parts if p.name.endswith("_glass")
                  and "door" in p.name]
    assert len(side_glass) == 4                 # all four door windows
    for p in side_glass:
        assert p.aabb_min[1] > cp.y_belt_front - 0.05
        assert p.aabb_max[1] > cp.y_belt_front + 0.25


# ----------------------------------------------------------------------
# interior
# ----------------------------------------------------------------------

def test_interior_parts_present_high_detail():
    spec = vd.build_vehicle({"class": "sedan", "interior_detail": "high"})
    names = {p.name for p in spec.parts}
    required = {
        "interior_floor",
        "seat_fl_cushion", "seat_fl_backrest", "seat_fl_headrest",
        "seat_fr_cushion", "seat_fr_backrest", "seat_fr_headrest",
        "seat_row2_cushion", "seat_row2_backrest",
        "steering_wheel", "steering_column", "steering_hub",
        "dash_main", "dash_binnacle", "dash_center_stack", "dash_screen",
        "console", "door_fl_card", "door_fr_card",
        "door_rl_card", "door_rr_card",
    }
    missing = required - names
    assert not missing, f"missing interior parts: {missing}"


def test_interior_low_detail_drops_trim():
    spec = vd.build_vehicle({"class": "sedan", "interior_detail": "low"})
    names = {p.name for p in spec.parts}
    assert {"interior_floor", "seat_fl_cushion", "seat_fl_backrest",
            "steering_wheel", "dash_main"} <= names
    assert "console" not in names
    assert "door_fl_card" not in names
    assert "seat_fl_headrest" not in names


def test_steering_is_left_hand_drive():
    spec = vd.build_vehicle({"class": "sedan"})
    wheel = spec.part("steering_wheel")
    assert wheel.aabb_min[2] > 0.10             # +Z = vehicle left


# ----------------------------------------------------------------------
# fascia: distinct named parts
# ----------------------------------------------------------------------

def test_fascia_parts_distinct():
    spec = vd.build_vehicle({"class": "sedan"})
    names = {p.name for p in spec.parts}
    required = {"grille", "headlamp_l", "headlamp_r", "indicator_l",
                "taillamp_l", "taillamp_r", "bumper_front", "bumper_rear",
                "mirror_l", "mirror_r", "mirror_glass_l"}
    assert required <= names
    # shut-line seals around the doors
    assert "door_fl_seal_0" in names and "door_rr_seal_1" in names


# ----------------------------------------------------------------------
# closures: distinct hinged parts
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_closures_are_separate_named_parts(cls):
    spec = vd.build_vehicle({"class": cls})
    names = {p.name for p in spec.parts}
    assert "hood" in names
    assert "door_fl_shell" in names and "door_fr_shell" in names
    assert "door_fl_frame_top" in names and "door_fl_handle" in names
    style = vd.VEHICLE_CLASSES[cls].body_style
    rear = {"notchback": "trunk_lid", "wedge": "trunk_lid",
            "hatch": "hatch", "van": "hatch", "bed": "tailgate"}[style]
    assert rear in names
    if cls == "sedan":
        assert "door_rl_shell" in names and "door_rr_shell" in names


# ----------------------------------------------------------------------
# triangle budget + LOD
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_triangle_budget_high_detail(cls):
    spec = vd.build_vehicle({"class": cls, "lod": "high",
                             "interior_detail": "high"})
    assert spec.triangle_count() <= 45_000, (
        f"{cls}: {spec.triangle_count()} tris exceeds the 45k budget")


def test_lod_knobs_reduce_triangles():
    counts = []
    for lod in ("high", "mid", "low"):
        spec = vd.build_vehicle({"class": "sedan", "lod": lod,
                                 "interior_detail": "low"})
        counts.append(spec.triangle_count())
    assert counts[0] > counts[1] > counts[2]
    low = vd.build_vehicle({"class": "sedan", "lod": "low"})
    names = {p.name for p in low.parts}
    assert "wheel_tread_fl" not in names        # lugs gated at low LOD


# ----------------------------------------------------------------------
# vertex-color zones
# ----------------------------------------------------------------------

def test_tub_vertex_color_zones():
    spec = vd.build_vehicle({"class": "sedan", "color": "candy_red"})
    tub = spec.part("body_tub")
    assert tub.vertex_colors.shape == tub.vertices.shape
    q = np.round(tub.vertex_colors, 2)
    distinct = np.unique(q, axis=0)
    assert len(distinct) >= 4                    # paint/underbody/arch/wells
    # arch liner zone darker than the paint
    paint = np.asarray(vd.PAINT_COLORS["candy_red"])
    lum = distinct @ np.array([0.2126, 0.7152, 0.0722])
    assert lum.min() < 0.5 * (paint @ np.array([0.2126, 0.7152, 0.0722]))


def test_windshield_tint_band():
    spec = vd.build_vehicle({"class": "sedan"})
    ws = spec.part("windshield")
    cols = ws.vertex_colors
    ys = ws.vertices[:, 1]
    top = cols[ys > np.percentile(ys, 75)].mean(axis=0)
    bot = cols[ys < np.percentile(ys, 25)].mean(axis=0)
    assert top.mean() < 0.8 * bot.mean()        # factory sun strip


def test_livery_changes_colors():
    plain = vd.build_vehicle({"class": "sedan", "color": "deep_blue"})
    striped = vd.build_vehicle({"class": "sedan", "color": "deep_blue",
                                "livery": "racing_stripes"})
    hp = plain.part("hood").vertex_colors
    hs = striped.part("hood").vertex_colors
    assert not np.allclose(hp, hs)
    # stripes introduce a second strong color on the hood
    assert len(np.unique(np.round(hs, 2), axis=0)) >= 2
    tt = vd.build_vehicle({"class": "sedan", "color": "deep_blue",
                           "livery": "two_tone"})
    roof = tt.part("roof").vertex_colors
    body = tt.part("body_tub").vertex_colors
    assert roof.mean() < body.mean()


def test_vertex_colors_match_mesh_sizes_everywhere():
    spec = vd.build_vehicle({"class": "suv", "interior_detail": "high"})
    for p in spec.parts:
        assert p.vertex_colors.shape == p.vertices.shape, p.name
        assert np.all(p.vertex_colors >= 0.0) and np.all(p.vertex_colors <= 1.0)


# ----------------------------------------------------------------------
# bake / articulation state
# ----------------------------------------------------------------------

def test_bake_applies_door_open_state():
    closed = vd.build_vehicle({"class": "sedan"})
    opened = vd.build_vehicle({"class": "sedan", "doors_open": True})
    assert opened.default_state.get("door_fl") == 1.0
    zb_closed = closed.part("door_fl_shell").aabb_max[2]
    baked = opened.bake()
    shell = next(p for p in baked if p.name == "door_fl_shell")
    assert shell.aabb_max[2] > zb_closed + 0.30     # swung outward
    # baked parts keep outward winding (robust check: geometric face
    # normals agree with vertex normals — translation-invariant, unlike
    # divergence volumes of tiny far-from-origin parts)
    for p in baked:
        v0 = p.vertices[p.faces[:, 0]].astype(np.float64)
        v1 = p.vertices[p.faces[:, 1]].astype(np.float64)
        v2 = p.vertices[p.faces[:, 2]].astype(np.float64)
        fn = np.cross(v1 - v0, v2 - v0)
        vn = p.normals[p.faces].astype(np.float64).mean(axis=1)
        agree = np.einsum("ij,ij->i", fn, vn) >= 0.0
        assert agree.mean() > 0.98, p.name


def test_bake_per_assembly_state_dict():
    spec = vd.build_vehicle({"class": "sedan"})
    baked = spec.bake({"door_fl": 1.0})
    fl = next(p for p in baked if p.name == "door_fl_shell")
    rl = next(p for p in baked if p.name == "door_rl_shell")
    assert fl.aabb_max[2] > 1.2                  # open
    assert rl.aabb_max[2] < 1.1                  # still closed
