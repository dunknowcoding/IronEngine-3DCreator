"""Tests for the complex-geometry primitive kinds (CR_ComplexGeometry).

Covers superellipsoid / tube / sweep / arch / panel: surface samplers,
analytic meshes (watertightness + volume vs. analytic), solid-volume
formulas, and the sweep→tube alias.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ironengine_3d_creator.generation.analytic_mesh import (
    MESH_BUILDERS, local_aabb, primitive_solid_volume, signed_volume,
)
from ironengine_3d_creator.generation.primitives import SAMPLERS, primitive_area

_KIND_PARAMS = {
    "superellipsoid": {"radii": [0.5, 0.4, 0.45], "exponents": [0.7, 0.7]},
    "tube": {"path": [[0.0, 0.0, 0.0], [0.2, 0.3, 0.0], [0.2, 0.6, 0.1]],
             "radius": 0.05, "caps": True},
    "sweep": {"path": [[0.0, 0.0, 0.0], [0.0, 0.4, 0.0], [0.3, 0.6, 0.0]],
              "radius": 0.04, "caps": True},
    "arch": {"major_radius": 0.5, "minor_radius": 0.1},
    "panel": {"size": [1.0, 0.6], "thickness": 0.03, "bend": 0.7},
}


def _edge_counts(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Geometric edge-incidence counts (UV-seam duplication allowed)."""
    edge_counts: dict[tuple, int] = {}
    for a, b, c in f:
        pa, pb, pc = (v[i].astype(np.float64) for i in (a, b, c))
        if np.linalg.norm(np.cross(pb - pa, pc - pa)) < 1e-12:
            continue  # degenerate pole triangle
        for p, q in ((pa, pb), (pb, pc), (pc, pa)):
            key = tuple(sorted((tuple(np.round(p, 6)), tuple(np.round(q, 6)))))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    return np.fromiter(edge_counts.values(), dtype=np.int64)


@pytest.mark.parametrize("kind", sorted(_KIND_PARAMS))
def test_sampler_counts_and_bounds(kind):
    rng = np.random.default_rng(7)
    pts = SAMPLERS[kind](4000, _KIND_PARAMS[kind], rng)
    assert pts.shape == (4000, 3)
    assert np.all(np.isfinite(pts))
    lo, hi = local_aabb(kind, _KIND_PARAMS[kind])
    tol = 1e-5
    assert np.all(pts.min(0) >= lo - tol)
    assert np.all(pts.max(0) <= hi + tol)


@pytest.mark.parametrize("kind", sorted(_KIND_PARAMS))
def test_mesh_watertight_and_volume(kind):
    params = _KIND_PARAMS[kind]
    v, n, uv, f = MESH_BUILDERS[kind](params)
    assert v.shape[0] > 0 and f.shape[0] > 0
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-4)
    counts = _edge_counts(v, f)
    assert counts.min() == 2 and counts.max() == 2
    sv = signed_volume(v, f)
    av = primitive_solid_volume(kind, params)
    assert sv > 0.0
    assert sv / av == pytest.approx(1.0, rel=0.15)


def test_superellipsoid_volume_formula_spot_checks():
    # e = 1 → ellipsoid: 4π/3 · rx ry rz
    v = primitive_solid_volume(
        "superellipsoid", {"radii": [1.0, 2.0, 3.0], "exponents": [1.0, 1.0]})
    assert v == pytest.approx(4 * math.pi / 3 * 6.0, rel=1e-9)
    # e = 2 → octahedron: 4/3 · rx ry rz
    v = primitive_solid_volume(
        "superellipsoid", {"radii": [1.0, 2.0, 3.0], "exponents": [2.0, 2.0]})
    assert v == pytest.approx(4.0 / 3.0 * 6.0, rel=1e-9)


def test_arch_volume_is_half_torus_at_default_arc():
    # arc = π (default) → exactly half a torus: π r² · R · arc
    v = primitive_solid_volume("arch", {"major_radius": 0.5, "minor_radius": 0.1})
    assert v == pytest.approx(math.pi * 0.1 ** 2 * 0.5 * math.pi, rel=1e-9)
    full = primitive_solid_volume(
        "arch", {"major_radius": 0.5, "minor_radius": 0.1, "arc": 2 * math.pi})
    assert full == pytest.approx(2 * v, rel=1e-9)


def test_sweep_is_tube_alias():
    params = {"path": [[0.0, 0.0, 0.0], [0.1, 0.3, 0.0], [0.1, 0.5, 0.2]],
              "radius": 0.05, "caps": True}
    for tube_v, sweep_v in zip(MESH_BUILDERS["tube"](params),
                               MESH_BUILDERS["sweep"](params)):
        np.testing.assert_allclose(tube_v, sweep_v, atol=1e-7)
    assert primitive_solid_volume("sweep", params) == \
        pytest.approx(primitive_solid_volume("tube", params))


def test_tube_height_fallback_matches_straight_bar():
    # No `path` → straight vertical bar of `height`; volume = π r² h.
    v = primitive_solid_volume("tube", {"radius": 0.05, "height": 0.8})
    assert v == pytest.approx(math.pi * 0.05 ** 2 * 0.8, rel=1e-9)
    lo, hi = local_aabb("tube", {"radius": 0.05, "height": 0.8})
    assert hi[1] == pytest.approx(0.4 + 0.05, abs=1e-6)
    assert lo[1] == pytest.approx(-0.4 - 0.05, abs=1e-6)


@pytest.mark.parametrize("kind", sorted(_KIND_PARAMS))
def test_primitive_area_positive(kind):
    assert primitive_area(kind, _KIND_PARAMS[kind]) > 0.0


def test_panel_bend_sign_mirrors_z():
    v_pos, _, _, f_pos = MESH_BUILDERS["panel"](
        {"size": [1.0, 0.6], "thickness": 0.03, "bend": 0.7})
    v_neg, _, _, f_neg = MESH_BUILDERS["panel"](
        {"size": [1.0, 0.6], "thickness": 0.03, "bend": -0.7})
    np.testing.assert_allclose(v_pos[:, 0], v_neg[:, 0], atol=1e-6)
    np.testing.assert_allclose(v_pos[:, 1], v_neg[:, 1], atol=1e-6)
    np.testing.assert_allclose(v_pos[:, 2], -v_neg[:, 2], atol=1e-6)
    # Both windings stay outward → positive signed volume of equal magnitude.
    assert signed_volume(v_pos, f_pos) == pytest.approx(
        signed_volume(v_neg, f_neg), rel=1e-6)
