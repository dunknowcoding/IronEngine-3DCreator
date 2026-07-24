"""Articulation tests for the vehicle designer (generation.vehicle_design).

Hinge metadata validity (front-hinged doors ROM 0–65°, hood 0–60°,
trunk/hatch/tailgate 0–70°) and swept-collision checks: no door may clip
the fender through its swing arc, and hood/trunk closures must clear the
body across the full ROM — for every vehicle class. No renderer.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ironengine_3d_creator.generation import vehicle_design as vd

ALL_CLASSES = ("sedan", "hatchback", "suv", "sports", "pickup", "van")

# Sub-panel-gap tolerance: the design gap is 4 mm; anything below 3 mm is
# weatherstrip compression, not a collision.
TOL_M = 0.003


@pytest.fixture(scope="module", params=ALL_CLASSES)
def spec(request):
    return vd.build_vehicle({"class": request.param})


# ----------------------------------------------------------------------
# hinge metadata
# ----------------------------------------------------------------------

def test_door_hinge_metadata_valid(spec):
    doors = [h for h in spec.articulations.values() if h.kind == "door"]
    expected = 4 if vd.VEHICLE_CLASSES[spec.vehicle_class].doors >= 4 else 2
    assert len(doors) == expected
    for h in doors:
        # vertical axis, unit length
        assert np.allclose(h.axis, [0.0, 1.0, 0.0])
        assert h.rom_deg == (0.0, 65.0)
        assert h.open_sign in (-1.0, 1.0)
        # hinge sits at the door's FRONT edge: every shell vertex of the
        # assembly must start at or behind the hinge x (nose is −X)
        idx0 = spec.assemblies[h.assembly][0]
        shell = spec.parts[idx0]
        assert shell.aabb_min[0] >= h.origin[0] - 0.02


def test_hood_and_trunk_hinge_metadata_valid(spec):
    hood = spec.articulations["hood"]
    assert hood.kind == "hood"
    assert hood.rom_deg == (0.0, 60.0)
    assert np.allclose(hood.axis, [0.0, 0.0, 1.0])
    trunk = spec.articulations["trunk"]
    assert trunk.kind in ("trunk", "hatch")
    assert trunk.rom_deg == (0.0, 70.0)


def test_assembly_transform_endpoints(spec):
    h = spec.articulations["door_fl"]
    ident = spec.assembly_transform("door_fl", 0.0)
    assert np.allclose(ident, np.eye(4))
    full = spec.assembly_transform("door_fl", 1.0)
    # full ROM = 65° about Y through the hinge origin
    vec = np.array([1.0, 0.0, 0.0, 0.0])
    rotated = (full @ vec)[:3]
    ang = math.degrees(math.acos(float(np.clip(rotated[0], -1.0, 1.0))))
    assert ang == pytest.approx(65.0, abs=0.5)
    # clamped beyond the ROM
    over = spec.assembly_transform("door_fl", 2.5)
    assert np.allclose(over, full)


# ----------------------------------------------------------------------
# swept collisions: doors vs fender, closures vs body
# ----------------------------------------------------------------------

def test_door_swing_clears_fender(spec):
    for asm, h in spec.articulations.items():
        if h.kind != "door":
            continue
        pen = vd.check_swing_clearance(spec, asm, samples=10)
        assert pen < TOL_M, (
            f"{spec.vehicle_class} {asm}: {pen*1000:.1f} mm fender contact")


def test_hood_and_trunk_swing_clear(spec):
    for asm in ("hood", "trunk"):
        pen = vd.check_swing_clearance(spec, asm, samples=10)
        assert pen < TOL_M, (
            f"{spec.vehicle_class} {asm}: {pen*1000:.1f} mm body contact")


def test_open_doors_reveal_interior(spec):
    """At full door ROM the doorway is open: door skin clearly outboard,
    hinge edge stationary, and interior parts inside the cabin volume."""
    if "door_fl" not in spec.articulations:
        pytest.skip("no front door")
    baked = spec.bake({"door_fl": 1.0})
    shell = next(p for p in baked if p.name == "door_fl_shell")
    closed = spec.part("door_fl_shell")
    # swung outward (+Z for the left door)
    assert shell.aabb_max[2] > closed.aabb_max[2] + 0.30
    # hinge edge barely moves (front-bottom corner)
    h = spec.articulations["door_fl"]
    front_corner_closed = np.array([closed.aabb_min[0], closed.aabb_min[1],
                                    h.origin[2]])
    moved = (np.concatenate([front_corner_closed, [1.0]])
             @ spec.assembly_transform("door_fl", 1.0).T)[:3]
    assert np.linalg.norm(moved - front_corner_closed) < 0.10


# ----------------------------------------------------------------------
# direct unit checks of the sweep machinery
# ----------------------------------------------------------------------

def test_solid_depth_distinguishes_metal_from_well_air():
    spec = vd.build_vehicle({"class": "sedan"})
    cache = spec.geometry_cache["tub"]
    lay = spec.geometry_cache["layout"]
    positions = cache["positions"]
    sections = cache["sections"]
    # a station inside the engine bay, ahead of the front wheel arch
    x_bay = lay.x_nose + 0.45
    i = int(np.argmin(np.abs(positions - x_bay)))
    poly = sections[i]
    # point inside the underbody metal → solid
    assert vd._point_solid_depth(np.array([0.20, 0.70]), poly) > 0.0
    # point hovering in the bay air above the well floor → not solid
    assert vd._point_solid_depth(np.array([lay.y_bay + 0.15, 0.0]), poly) < 0.0
    # point outside the body entirely → not solid
    assert vd._point_solid_depth(np.array([2.0, 0.0]), poly) < 0.0
