"""Tests for generation.analytic_mesh (F5) — exact per-primitive meshes."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.generation.analytic_mesh import (
    MESH_BUILDERS, AnalyticPart, build_part_mesh, build_spec_meshes,
    primitive_solid_volume, signed_volume,
)

_KIND_PARAMS = {
    "box": {"size": [1.0, 1.0, 1.0]},
    "sphere": {"radius": 0.5},
    "cylinder": {"radius": 0.4, "height": 1.0},
    "capsule": {"radius": 0.3, "height": 1.0},
    "cone": {"radius": 0.5, "height": 1.0},
    "torus": {"major_radius": 0.5, "minor_radius": 0.15},
    "ellipsoid": {"radii": [0.5, 0.3, 0.2]},
    "prism": {"sides": 6, "radius": 0.5, "height": 1.0},
    "helix": {"radius": 0.4, "pitch": 0.2, "turns": 3.0, "thickness": 0.05},
    "plane": {"size": [1.0, 1.0]},
}

# Closed primitives: tessellated volume should approximate the analytic volume
# (inscribed meshes are slightly smaller; helix tube coarsest).
_CLOSED = {"box", "sphere", "cylinder", "capsule", "cone", "torus", "ellipsoid", "prism", "helix"}


@pytest.mark.parametrize("kind", sorted(_KIND_PARAMS))
def test_every_kind_builds_valid_mesh(kind):
    v, n, uv, f = MESH_BUILDERS[kind](_KIND_PARAMS[kind])
    assert v.shape[0] > 0 and f.shape[0] > 0
    assert v.shape[1] == 3 and n.shape == v.shape and uv.shape == (v.shape[0], 2)
    assert f.min() >= 0 and f.max() < v.shape[0]
    # Analytic smooth normals are unit length.
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-4)
    # UVs stay inside the unit square.
    assert uv.min() >= -1e-6 and uv.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("kind", sorted(_CLOSED))
def test_closed_primitives_are_watertightish(kind):
    v, _, _, f = MESH_BUILDERS[kind](_KIND_PARAMS[kind])
    # Geometric watertightness: every non-degenerate edge segment must be
    # matched by exactly one other triangle edge at the same location
    # (UV-seam vertex duplication is allowed; holes are not).
    edge_counts: dict[tuple, int] = {}
    for a, b, c in f:
        pa, pb, pc = (v[i].astype(np.float64) for i in (a, b, c))
        if np.linalg.norm(np.cross(pb - pa, pc - pa)) < 1e-12:
            continue  # degenerate (zero-area) pole triangle
        for p, q in ((pa, pb), (pb, pc), (pc, pa)):
            key = tuple(
                sorted((tuple(np.round(p, 6)), tuple(np.round(q, 6))))
            )
            edge_counts[key] = edge_counts.get(key, 0) + 1
    counts = np.fromiter(edge_counts.values(), dtype=np.int64)
    assert counts.min() == 2 and counts.max() == 2
    # Signed volume is positive (outward winding) and near the analytic value.
    sv = signed_volume(v, f)
    av = primitive_solid_volume(kind, _KIND_PARAMS[kind])
    assert sv > 0.0
    assert sv / av == pytest.approx(1.0, rel=0.15)


def test_solid_volume_formulas():
    assert primitive_solid_volume("box", {"size": [2, 1, 0.5]}) == pytest.approx(1.0)
    assert primitive_solid_volume("sphere", {"radius": 1.0}) == pytest.approx(4 * math.pi / 3)
    assert primitive_solid_volume("cylinder", {"radius": 1, "height": 2}) == pytest.approx(2 * math.pi)
    assert primitive_solid_volume("capsule", {"radius": 1, "height": 2}) == pytest.approx(
        2 * math.pi + 4 * math.pi / 3)
    assert primitive_solid_volume("cone", {"radius": 1, "height": 3}) == pytest.approx(math.pi)
    assert primitive_solid_volume("torus", {"major_radius": 2, "minor_radius": 0.5}) == pytest.approx(
        2 * math.pi ** 2 * 2 * 0.25)
    assert primitive_solid_volume("ellipsoid", {"radii": [1, 2, 3]}) == pytest.approx(8 * math.pi)
    assert primitive_solid_volume("plane", {"size": [5, 5]}) == 0.0


def test_transform_applies_to_vertices_normals_and_volume():
    # Scale x2 in X, translate by (1, 2, 3).
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = 2.0
    T[:3, 3] = [1.0, 2.0, 3.0]
    part = build_part_mesh("sphere", {"radius": 0.5}, T, "ball", "metal")
    assert isinstance(part, AnalyticPart)
    np.testing.assert_allclose(part.aabb_min, [0.0, 1.5, 2.5], atol=1e-5)
    np.testing.assert_allclose(part.aabb_max, [2.0, 2.5, 3.5], atol=1e-5)
    # Volume scales by |det| = 2.
    assert part.solid_volume_m3 == pytest.approx(4 / 3 * math.pi * 0.125 * 2.0, rel=1e-6)
    # Normals stay unit-length after inverse-transpose handling.
    np.testing.assert_allclose(np.linalg.norm(part.normals, axis=1), 1.0, atol=1e-4)


def test_build_spec_meshes_chair_template():
    from ironengine_3d_creator.alignment.defaults import auto_template

    spec = auto_template("chair")
    parts = build_spec_meshes(spec)
    assert len(parts) == len(spec.primitives) == 6
    labels = [p.label for p in parts]
    assert labels == ["seat", "leg_0", "leg_1", "leg_2", "leg_3", "back"]
    # Legs are cylinders standing on the ground plane.
    for part in parts[1:5]:
        assert part.kind == "cylinder"
        assert part.aabb_min[1] == pytest.approx(0.0, abs=1e-5)
        assert part.solid_volume_m3 == pytest.approx(math.pi * 0.04 ** 2 * 0.45, rel=1e-5)
    # World AABB of the seat matches the template transform.
    seat = parts[0]
    np.testing.assert_allclose(seat.aabb_min, [-0.225, 0.425, -0.225], atol=1e-5)
    np.testing.assert_allclose(seat.aabb_max, [0.225, 0.475, 0.225], atol=1e-5)


def test_build_spec_meshes_skips_unknown_kinds():
    spec = GenerationSpec(
        shape="abstract", n_points=10, bbox_size=(1, 1, 1),
        primitives=[
            Primitive("box", Primitive.identity_transform(), {"size": [1, 1, 1]}, "a"),
            Primitive("fractal", Primitive.identity_transform(), {}, "b"),
        ],
        features=[], color=None, seed=0,
    )
    parts = build_spec_meshes(spec)
    assert [p.label for p in parts] == ["a"]
