"""Tests for the slicing builder (generation.slicer) — CR_ComplexBuilder.

Covers: profile library validity, loft watertightness (2-manifold edge use),
solid volumes vs. analytic references, outward normals, UV ranges, per-slice
scale/rotation/offset (taper/twist/drift), all three loft axes, and open
(no-cap) lofts. No network.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest

from ironengine_3d_creator.generation import slicer


def _edge_uses(faces: np.ndarray) -> dict[tuple[int, int], int]:
    uses: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in faces.tolist():
        for e in ((a, b), (b, c), (c, a)):
            uses[tuple(sorted(e))] += 1
    return uses


def _normal_agreement(v, n, f) -> float:
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = n[f].mean(axis=1)
    return float((np.einsum("ij,ij->i", fn, vn) > 0).mean())


# ----------------------------------------------------------------------
# profile library
# ----------------------------------------------------------------------

def test_profile_circle_is_ccw_and_sized():
    p = slicer.profile_circle(0.5, 32)
    assert p.shape == (32, 2)
    assert slicer.profile_area(p) == pytest.approx(math.pi * 0.25, rel=0.01)


def test_profile_rounded_rect_area_and_ccw():
    p = slicer.profile_rounded_rect(0.8, 0.5, 0.12, 6)
    expected = 0.8 * 0.5 - (4.0 - math.pi) * 0.12 ** 2
    assert slicer.profile_area(p) == pytest.approx(expected, rel=0.02)


def test_profile_superellipse_circle_limit():
    # exponent 2.0 → ellipse; a == b → circle of radius a.
    p = slicer.profile_superellipse(0.4, 0.4, 2.0, 64)
    assert slicer.profile_area(p) == pytest.approx(math.pi * 0.16, rel=0.01)


def test_profile_polygon_reorders_clockwise_input():
    cw = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    p = slicer.profile_polygon(cw)
    assert slicer.profile_area(p) > 0.0


def test_profile_polygon_rejects_degenerate():
    with pytest.raises(ValueError):
        slicer.profile_polygon([[0.0, 0.0], [1.0, 1.0]])


# ----------------------------------------------------------------------
# watertightness + normals
# ----------------------------------------------------------------------

@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("profile", [
    slicer.profile_circle(0.4, 32),
    slicer.profile_rounded_rect(0.7, 0.4, 0.1, 5),
    slicer.profile_superellipse(0.35, 0.25, 3.0, 32),
])
def test_loft_is_watertight_with_consistent_normals(axis, profile):
    slices = slicer.radius_slices([0.0, 0.5, 1.0], [0.8, 1.1, 0.6])
    v, n, uv, f = slicer.loft(profile, slices, axis=axis)
    uses = _edge_uses(f)
    assert all(c == 2 for c in uses.values()), "closed loft must be 2-manifold"
    assert _normal_agreement(v, n, f) == 1.0
    assert slicer.signed_mesh_volume(v, f) > 0.0
    # UVs stay in [0, 1] with u wrapping around the profile.
    assert uv.min() >= 0.0 and uv.max() <= 1.0


def test_open_loft_has_boundary_but_consistent_normals():
    v, n, uv, f = slicer.loft(
        slicer.profile_circle(0.3, 24), slicer.radius_slices([0.0, 1.0], [1.0, 1.0]),
        caps=False,
    )
    uses = _edge_uses(f)
    boundary = sum(1 for c in uses.values() if c == 1)
    assert boundary == 2 * 24          # two open rings
    assert _normal_agreement(v, n, f) == 1.0


# ----------------------------------------------------------------------
# volumes
# ----------------------------------------------------------------------

def test_loft_volume_cylinder_matches_analytic():
    r, h = 0.5, 1.2
    prof = slicer.profile_circle(r, 64)
    v, n, uv, f = slicer.loft(prof, slicer.radius_slices([0.0, h], [1.0, 1.0]))
    assert slicer.signed_mesh_volume(v, f) == pytest.approx(math.pi * r * r * h, rel=0.01)


def test_loft_volume_frustum_matches_analytic():
    r1, r2, h = 0.5, 0.1, 1.0
    prof = slicer.profile_circle(0.5, 64)
    v, n, uv, f = slicer.loft(prof, slicer.radius_slices([0.0, h], [1.0, 0.2]))
    expected = math.pi * h * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0
    assert slicer.signed_mesh_volume(v, f) == pytest.approx(expected, rel=0.01)


def test_loft_volume_twisted_tapered_close_to_trapezoid():
    prof = slicer.profile_superellipse(0.4, 0.4, 3.0, 40)
    slices = [
        slicer.Slice(p, scale=(r, r), rotation=p * 1.2, offset=(0.1 * p, 0.0))
        for p, r in zip([0.0, 0.3, 0.7, 1.0], [0.3, 0.5, 0.35, 0.15])
    ]
    v, n, uv, f = slicer.loft(prof, slices)
    mesh_vol = slicer.signed_mesh_volume(v, f)
    trap = slicer.loft_volume(prof, slices)
    # Trapezoid overestimates quadratically-varying area; stay within 10%.
    assert mesh_vol == pytest.approx(trap, rel=0.10)
    uses = _edge_uses(f)
    assert all(c == 2 for c in uses.values())


def test_loft_rejects_single_slice():
    with pytest.raises(ValueError):
        slicer.loft(slicer.profile_circle(0.5, 8), [slicer.Slice(0.0)])


# ----------------------------------------------------------------------
# real-world-scale demo shapes (vase / lamp shade)
# ----------------------------------------------------------------------

def test_vase_loft_real_world_scale():
    """A 0.32 m vase: foot → belly → shoulder → neck in 6 slices."""
    prof = slicer.profile_circle(0.05, 32)  # unit profile scaled per slice
    positions = [0.00, 0.02, 0.08, 0.18, 0.26, 0.32]
    radii = [0.55, 0.75, 1.00, 0.80, 0.42, 0.46]
    v, n, uv, f = slicer.loft(prof, slicer.radius_slices(positions, radii))
    lo, hi = v.min(axis=0), v.max(axis=0)
    assert hi[1] - lo[1] == pytest.approx(0.32, abs=1e-6)
    assert hi[0] <= 0.051 and hi[2] <= 0.051
    uses = _edge_uses(f)
    assert all(c == 2 for c in uses.values())
