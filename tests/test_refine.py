"""Tests for generation.refine (CR_TexReal post-refinement).

Covers: crease-aware Loop-lite subdivision (hard edges + volume preserved),
triangle-budget clamping, procedural displacement determinism, degenerate-face
cleanup, shell solidification (real cloth thickness) and the refine_garment
hook (weave attachment + non-cloth passthrough).
"""
from __future__ import annotations

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.generation import refine
from ironengine_3d_creator.generation.analytic_mesh import (
    build_spec_meshes,
    count_degenerate_faces,
    signed_volume,
)


def _spec(*prims):
    return GenerationSpec(shape="test", primitives=list(prims))


def _box() :
    return build_spec_meshes(_spec(Primitive(
        "box", Primitive.identity_transform(),
        {"size": [0.4, 0.4, 0.4], "material": "wood"}, "crate")))[0]


def _sphere():
    return build_spec_meshes(_spec(Primitive(
        "sphere", Primitive.identity_transform(), {"radius": 0.3}, "ball")))[0]


def _cylinder():
    return build_spec_meshes(_spec(Primitive(
        "cylinder", Primitive.identity_transform(),
        {"radius": 0.2, "height": 0.5}, "rod")))[0]


def _boundary_edge_count(faces: np.ndarray) -> int:
    f = np.asarray(faces, dtype=np.int64)
    fe = np.sort(
        np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1).reshape(-1, 2),
        axis=1,
    )
    _, counts = np.unique(fe, axis=0, return_counts=True)
    return int((counts == 1).sum())


# ---------------------------------------------------------------------------
# subdivision contracts
# ---------------------------------------------------------------------------


def test_subdivision_multiplies_triangles_by_four():
    box = _box()
    r1 = refine.refine_mesh(box, levels=1)
    assert r1.triangles_out == box.faces.shape[0] * 4
    r2 = refine.refine_mesh(box, levels=2)
    assert r2.triangles_out == box.faces.shape[0] * 16
    assert r2.levels_applied == 2 == r2.levels_requested


def test_box_hard_edges_and_volume_preserved():
    """All box edges are 90 deg creases: refinement must densify without
    rounding the crate — extents exact, volume exact, corners pinned."""
    box = _box()
    r = refine.refine_mesh(box, levels=2, crease_deg=30.0)
    v0 = abs(signed_volume(box.vertices, box.faces))
    v1 = abs(signed_volume(r.vertices, r.faces))
    assert v1 == pytest.approx(v0, rel=1e-3)
    np.testing.assert_allclose(
        r.vertices.min(axis=0), box.vertices.min(axis=0), atol=1e-6
    )
    np.testing.assert_allclose(
        r.vertices.max(axis=0), box.vertices.max(axis=0), atol=1e-6
    )
    # The 8 original corner positions still exist in the refined mesh.
    corners = np.array(
        [[x, y, z] for x in (-0.2, 0.2) for y in (-0.2, 0.2) for z in (-0.2, 0.2)]
    )
    for c in corners:
        assert np.abs(r.vertices - c).max(axis=1).min() < 1e-6
    assert r.crease_edges > 0


def test_sphere_refines_with_volume_ish_preserved():
    ball = _sphere()
    r = refine.refine_mesh(ball, levels=2, crease_deg=30.0)
    v0 = abs(signed_volume(ball.vertices, ball.faces))
    v1 = abs(signed_volume(r.vertices, r.faces))
    assert 0.90 < v1 / v0 < 1.05
    # Rounded out: refined vertices stay near the analytic radius.
    rad = np.linalg.norm(r.vertices, axis=1)
    assert rad.max() <= 0.3 + 1e-3
    assert rad.min() > 0.25


def test_cylinder_refine_keeps_watertight_and_smooth():
    rod = _cylinder()
    r = refine.refine_mesh(rod, levels=2, crease_deg=30.0)
    v0 = abs(signed_volume(rod.vertices, rod.faces))
    v1 = abs(signed_volume(r.vertices, r.faces))
    assert v1 / v0 == pytest.approx(1.0, abs=0.02)
    # Rims stay crisp: the side wall stays on the analytic circle.
    side = np.linalg.norm(r.vertices[:, [0, 2]], axis=1)
    mid_height = np.abs(r.vertices[:, 1]) < 0.2
    np.testing.assert_allclose(side[mid_height], 0.2, atol=2e-3)


def test_no_degenerate_faces_after_refine():
    for part in (_box(), _sphere(), _cylinder()):
        r = refine.refine_mesh(part, levels=2, crease_deg=25.0)
        assert count_degenerate_faces(r.vertices, r.faces) == 0


def test_uvs_and_normals_carried_through():
    ball = _sphere()
    r = refine.refine_mesh(ball, levels=1)
    assert r.uvs is not None and r.uvs.shape == (r.vertices.shape[0], 2)
    assert r.normals.shape == r.vertices.shape
    norms = np.linalg.norm(r.normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_refine_part_preserves_metadata_and_recomputes_volume():
    box = _box()
    out = refine.refine_part(box, levels=1)
    assert type(out) is type(box)
    assert out.label == box.label and out.material == box.material
    assert out.faces.shape[0] == box.faces.shape[0] * 4
    assert out.solid_volume_m3 == pytest.approx(box.solid_volume_m3, rel=1e-3)


def test_tuple_mesh_input_accepted():
    ball = _sphere()
    r = refine.refine_mesh((ball.vertices, ball.faces, ball.uvs), levels=1)
    # x4 minus the handful of zero-area pole fans the weld cleanup drops.
    assert r.triangles_out >= ball.faces.shape[0] * 4 - 16
    assert r.triangles_out <= ball.faces.shape[0] * 4


def test_empty_mesh_rejected():
    with pytest.raises(ValueError):
        refine.refine_mesh((np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)))


# ---------------------------------------------------------------------------
# triangle budget guardrail
# ---------------------------------------------------------------------------


def test_tri_budget_clamps_levels():
    box = _box()  # 12 faces
    # 12 * 4 = 48 fits; 12 * 16 = 192 must not.
    r = refine.refine_mesh(box, levels=3, tri_budget=100)
    assert r.levels_applied == 1
    assert r.levels_requested == 3
    assert r.triangles_out <= 100


def test_tri_budget_zero_when_even_first_level_exceeds():
    ball = _sphere()  # 964 faces
    r = refine.refine_mesh(ball, levels=2, tri_budget=1000)
    assert r.levels_applied == 0
    assert r.triangles_out == ball.faces.shape[0]


# ---------------------------------------------------------------------------
# displacement
# ---------------------------------------------------------------------------


def test_bump_displacement_deterministic():
    ball = _sphere()
    spec_a = refine.bump_displacement("knit_wool", 0.01, seed=5, uv_scale=(4, 4))
    r1 = refine.refine_mesh(ball, levels=1, displacement=spec_a, seed=1)
    r2 = refine.refine_mesh(ball, levels=1, displacement=spec_a, seed=1)
    np.testing.assert_array_equal(r1.vertices, r2.vertices)
    other = refine.bump_displacement("mud", 0.01, seed=6)
    r3 = refine.refine_mesh(ball, levels=1, displacement=other, seed=1)
    assert not np.array_equal(r1.vertices, r3.vertices)


def test_bump_displacement_offsets_along_normals():
    ball = _sphere()
    plain = refine.refine_mesh(ball, levels=1)
    disp = refine.refine_mesh(
        ball, levels=1, displacement=refine.bump_displacement("skin", 0.02, seed=3)
    )
    delta = disp.vertices - plain.vertices
    moved = np.linalg.norm(delta, axis=1)
    assert moved.max() > 1e-4          # relief actually applied
    assert moved.max() <= 0.02 + 1e-6  # bounded by the scale


def test_array_and_callable_displacement():
    ball = _sphere()
    r0 = refine.refine_mesh(ball, levels=0)
    arr = np.full(r0.vertices.shape[0], 0.005)
    r1 = refine.refine_mesh(ball, levels=0, displacement=arr)
    rad0 = np.linalg.norm(r0.vertices, axis=1)
    rad1 = np.linalg.norm(r1.vertices, axis=1)
    # Computed vertex normals are only approximately radial, so the radius
    # gain matches the offset up to the normal's radial component.
    np.testing.assert_allclose(rad1, rad0 + 0.005, atol=1e-4)

    def loud(pos, nrm, uvs, rng):
        return rng.uniform(0.0, 0.01, pos.shape[0])

    ra = refine.refine_mesh(ball, levels=0, displacement=loud, seed=9)
    rb = refine.refine_mesh(ball, levels=0, displacement=loud, seed=9)
    np.testing.assert_array_equal(ra.vertices, rb.vertices)
    with pytest.raises(ValueError):
        refine.refine_mesh(ball, levels=0, displacement=np.zeros(3))


# ---------------------------------------------------------------------------
# solidify + refine_garment
# ---------------------------------------------------------------------------


def _garment_parts(detail="low"):
    from ironengine_3d_creator.generation.human_anatomy import build_human

    spec = build_human({"clothes": ("tshirt",), "detail": detail, "seed": 11})
    return spec.build().parts


def test_solidify_shell_closes_open_mesh():
    parts = _garment_parts()
    cloth = next(p for p in parts if p.label == "tshirt")
    assert _boundary_edge_count(cloth.faces) > 0  # open cloth tube
    v, n, uv, f = refine.solidify_shell(
        cloth.vertices, cloth.normals, cloth.uvs, cloth.faces, 0.002
    )
    assert _boundary_edge_count(f) == 0           # closed two-sided garment
    assert count_degenerate_faces(v, f) == 0
    assert abs(signed_volume(v, f)) > 0.0
    assert uv is not None and uv.shape[0] == v.shape[0]


def test_refine_garment_thickness_and_weave_attachment():
    parts = _garment_parts()
    out = refine.refine_garment(parts, thickness=0.002, weave="knit_wool")
    assert len(out) == len(parts)
    cloth_in = {p.label for p in parts if refine.is_cloth_part(p)}
    assert cloth_in == {"tshirt", "tshirt_sleeve_l", "tshirt_sleeve_r"}
    by_label = {p.label: p for p in out}
    for label in cloth_in:
        g = by_label[label]
        assert _boundary_edge_count(g.faces) == 0
        assert g.solid_volume_m3 > 0.0
        # Weave maps attached for the image-map GLB export path.
        assert hasattr(g, "maps") and "albedo" in g.maps
        assert g.maps["albedo"].shape == (512, 512, 3)
        assert g.uv_scale == (8.0, 8.0)


def test_refine_garment_non_cloth_passthrough():
    parts = _garment_parts()
    out = refine.refine_garment(parts)
    non_cloth = [p for p in parts if not refine.is_cloth_part(p)]
    assert non_cloth, "expected some non-cloth body parts"
    for p in non_cloth:
        assert any(p is q for q in out)  # same object, untouched


def test_refine_garment_with_subdivision_and_displacement():
    parts = _garment_parts()
    cloth = [p for p in parts if refine.is_cloth_part(p)][:1]
    out = refine.refine_garment(
        cloth, levels=1, displacement_scale=0.0005, weave="linen",
        weave_uv_scale=(4, 4), tri_budget=50_000,
    )
    g = out[0]
    base = cloth[0]
    assert g.faces.shape[0] > base.faces.shape[0]
    assert g.faces.shape[0] <= 50_000
    assert count_degenerate_faces(g.vertices, g.faces) == 0
    assert "albedo" in g.maps


def test_is_cloth_part_detection():
    parts = _garment_parts()
    by_label = {p.label: p for p in parts}
    assert refine.is_cloth_part(by_label["tshirt"])
    assert not refine.is_cloth_part(by_label["pelvis"])
