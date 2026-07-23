"""Tests for CSG-lite subtraction (CR_ComplexGeometry).

Mesh-level carving (`role: "subtract"` cutters through box / panel / prism
hosts) and point-cloud-level filtering in the compositor for unsupported
hosts, plus orphan / sever / target-routing validation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.generation.analytic_mesh import (
    build_spec_meshes_with_report, primitive_solid_volume, signed_volume,
)
from ironengine_3d_creator.generation.compositor import generate


def _P(kind, params, label, transform=None):
    return Primitive(kind, transform or Primitive.identity_transform(),
                     params, label)


def _rot_x90():
    """Map local +Y onto world +Z (cylinder axis → panel thickness axis)."""
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]


def _edge_counts(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    edge_counts: dict[tuple, int] = {}
    for a, b, c in f:
        pa, pb, pc = (v[i].astype(np.float64) for i in (a, b, c))
        if np.linalg.norm(np.cross(pb - pa, pc - pa)) < 1e-12:
            continue
        for p, q in ((pa, pb), (pb, pc), (pc, pa)):
            key = tuple(sorted((tuple(np.round(p, 6)), tuple(np.round(q, 6)))))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    return np.fromiter(edge_counts.values(), dtype=np.int64)


def test_cylinder_through_panel_handle_hole():
    t = 0.04
    host = _P("panel", {"size": [1.0, 0.6], "thickness": t}, "grip")
    cutter = _P("cylinder",
                {"radius": 0.05, "height": 0.2, "role": "subtract"},
                "hole", _rot_x90())
    spec = GenerationSpec(shape="other", primitives=[host, cutter])
    parts, warnings = build_spec_meshes_with_report(spec)
    # Cutter is consumed, host carved: exactly one mesh part.
    assert [p.label for p in parts] == ["grip"]
    part = parts[0]
    # Watertight.
    counts = _edge_counts(part.vertices, part.faces)
    assert counts.min() == 2 and counts.max() == 2
    # Volume = panel − through-hole.
    expected = 1.0 * 0.6 * t - math.pi * 0.05 ** 2 * t
    assert part.solid_volume_m3 == pytest.approx(expected, rel=0.15)
    # Tunnel-wall normals point INWARD (toward the bore axis). Wall verts
    # sit on the bore cylinder with normals lying in the mid-plane
    # (|n_z| ≈ 0), unlike cap verts whose normals are ±z.
    v, n = part.vertices, part.normals
    radial = np.linalg.norm(v[:, :2], axis=1)
    wall = (np.abs(radial - 0.05) < 2e-3) & (np.abs(n[:, 2]) < 0.5)
    assert wall.sum() > 0
    outward = v[wall][:, :2] / radial[wall][:, None]
    dots = (n[wall][:, :2] * outward).sum(axis=1)
    assert np.all(dots < 0.0)


def test_box_through_box_doorway():
    host = _P("box", {"size": [1.0, 1.0, 0.3]}, "wall")
    cutter = _P("box", {"size": [0.3, 0.6, 0.5], "role": "subtract"}, "door")
    spec = GenerationSpec(shape="other", primitives=[host, cutter])
    parts, warnings = build_spec_meshes_with_report(spec)
    assert len(parts) == 1
    counts = _edge_counts(parts[0].vertices, parts[0].faces)
    assert counts.min() == 2 and counts.max() == 2
    expected = 1.0 * 1.0 * 0.3 - 0.3 * 0.6 * 0.3
    assert parts[0].solid_volume_m3 == pytest.approx(expected, rel=0.15)


def test_cylinder_through_prism_cap():
    host = _P("prism", {"sides": 6, "radius": 0.4, "height": 0.8}, "nut")
    cutter = _P("cylinder",
                {"radius": 0.1, "height": 2.0, "role": "subtract"}, "bore")
    spec = GenerationSpec(shape="other", primitives=[host, cutter])
    parts, warnings = build_spec_meshes_with_report(spec)
    assert len(parts) == 1
    counts = _edge_counts(parts[0].vertices, parts[0].faces)
    assert counts.min() == 2 and counts.max() == 2
    expected = (primitive_solid_volume("prism", {"sides": 6, "radius": 0.4, "height": 0.8})
                - math.pi * 0.1 ** 2 * 0.8)
    assert parts[0].solid_volume_m3 == pytest.approx(expected, rel=0.15)


def test_point_level_hollow_mug():
    """Cylinder hosts aren't mesh-carvable → warning + compositor filtering."""
    host = _P("cylinder", {"radius": 0.05, "height": 0.12}, "mug_body")
    cutter = _P("cylinder",
                {"radius": 0.04, "height": 0.20, "role": "subtract"},
                "hollow")
    # Cutter is taller than the host and centred slightly high, so the mug
    # keeps a bottom wall (y ∈ [-0.06, -0.06 + 0.02]) and an open top.
    T = Primitive.identity_transform()
    T[1][3] = 0.04
    cutter.transform = T
    spec = GenerationSpec(shape="vase", n_points=20_000, primitives=[host, cutter])
    parts, warnings = build_spec_meshes_with_report(spec)
    assert any("point-cloud" in w for w in warnings)
    res = generate(spec)
    assert len(res.positions) < 20_000  # interior points were filtered out
    # No surviving point deep inside the hollow region.
    hollow = ((np.linalg.norm(res.positions[:, [0, 2]], axis=1) < 0.038)
              & (res.positions[:, 1] > -0.05))
    assert hollow.sum() == 0


def test_orphan_cutter_warns_and_is_dropped():
    host = _P("box", {"size": [1.0, 1.0, 1.0]}, "body")
    T = Primitive.identity_transform()
    T[0][3] = 5.0
    cutter = _P("cylinder", {"radius": 0.2, "height": 2.0, "role": "subtract"},
                "stray", T)
    spec = GenerationSpec(shape="other", primitives=[host, cutter])
    parts, warnings = build_spec_meshes_with_report(spec)
    assert [p.label for p in parts] == ["body"]
    assert any("overlaps no host" in w for w in warnings)
    assert parts[0].solid_volume_m3 == pytest.approx(1.0, rel=1e-6)


def test_oversized_cutter_never_severs_host():
    host = _P("panel", {"size": [0.5, 0.5], "thickness": 0.04}, "sheet")
    cutter = _P("cylinder",
                {"radius": 0.5, "height": 0.2, "role": "subtract"}, "huge",
                _rot_x90())
    spec = GenerationSpec(shape="other", primitives=[host, cutter])
    parts, warnings = build_spec_meshes_with_report(spec)
    # Mesh carve refused (containment margin) → host keeps full volume.
    assert len(parts) == 1
    assert parts[0].solid_volume_m3 == pytest.approx(0.5 * 0.5 * 0.04, rel=1e-6)
    assert any("point-cloud" in w for w in warnings)


def test_target_param_routes_carve():
    host_a = _P("box", {"size": [0.4, 0.4, 0.4]}, "host_a")
    Tb = Primitive.identity_transform()
    Tb[0][3] = 1.0
    host_b = _P("box", {"size": [0.4, 0.4, 0.4]}, "host_b", Tb)
    cutter = _P("cylinder",
                {"radius": 0.05, "height": 1.0, "role": "subtract",
                 "target": "host_b"}, "drill", Tb)
    spec = GenerationSpec(shape="other", primitives=[host_a, host_b, cutter])
    parts, warnings = build_spec_meshes_with_report(spec)
    by_label = {p.label: p for p in parts}
    assert by_label["host_a"].solid_volume_m3 == pytest.approx(0.4 ** 3, rel=1e-6)
    expected_b = 0.4 ** 3 - math.pi * 0.05 ** 2 * 0.4
    assert by_label["host_b"].solid_volume_m3 == pytest.approx(expected_b, rel=0.15)


def test_unknown_target_warns_and_skips():
    host = _P("box", {"size": [0.4, 0.4, 0.4]}, "real_host")
    cutter = _P("cylinder",
                {"radius": 0.05, "height": 1.0, "role": "subtract",
                 "target": "no_such_part"}, "drill")
    spec = GenerationSpec(shape="other", n_points=5_000, primitives=[host, cutter])
    res = generate(spec)
    assert any("no_such_part" in w for w in res.warnings)
