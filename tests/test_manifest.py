"""Tests for the material table, export manifest, and GLB mesh export."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.core import exporter
from ironengine_3d_creator.core.manifest import (
    MANIFEST_SCHEMA, build_manifest, write_manifest,
)
from ironengine_3d_creator.generation.compositor import generate
from ironengine_3d_creator.generation.materials import (
    MATERIAL_PRESETS, default_preset, resolve_material,
)


def _spec(material: str | None = "wood", shape: str = "chair", n_points: int = 2000) -> GenerationSpec:
    params = {"size": [1.0, 1.0, 1.0]}
    if material is not None:
        params["material"] = material
    seat = Primitive("box", Primitive.identity_transform(), params, "seat")
    return GenerationSpec(
        shape=shape, n_points=n_points, bbox_size=(1.0, 1.0, 1.0),
        primitives=[seat], features=[], color=(0.5, 0.4, 0.3), seed=42,
    )


# ------------------------------------------------------------- materials

def test_resolve_material_explicit_hint():
    name, preset = resolve_material(_spec(material="metal"))
    assert name == "metal"
    assert preset == MATERIAL_PRESETS["metal"]
    assert preset["metallic"] > 0.5
    assert preset["density_kg_m3"] == pytest.approx(7870.0)


def test_resolve_material_unknown_hint_falls_back():
    # Unknown hint + abstract shape → neutral default preset.
    name, preset = resolve_material(_spec(material="unobtanium", shape="abstract"))
    assert name == "default"
    assert preset == default_preset()


def test_resolve_material_shape_fallback():
    name, preset = resolve_material(_spec(material=None, shape="rock"))
    assert name == "stone"
    assert preset == MATERIAL_PRESETS["stone"]


# ------------------------------------------------------------- manifest

def test_manifest_round_trip(tmp_path):
    spec = _spec(material="ceramic", shape="vase")
    result = generate(spec)
    manifest = build_manifest(
        spec, result.positions, result.colors,
        mesh_path="model.glb", point_cloud_path="model.ply",
        mesh_stats={"vertices": 123, "faces": 456},
    )
    out = tmp_path / "model.iemodel.json"
    write_manifest(out, manifest)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema"] == MANIFEST_SCHEMA == "iemodel/3"
    assert loaded["units"] == "meters"
    assert loaded["up_axis"] == "Y"
    # name comes from the export file stem (W24), not the spec shape.
    assert loaded["name"] == "model"

    aabb_min = np.asarray(loaded["aabb_min"])
    aabb_max = np.asarray(loaded["aabb_max"])
    np.testing.assert_allclose(aabb_min, result.positions.min(axis=0), atol=1e-5)
    np.testing.assert_allclose(aabb_max, result.positions.max(axis=0), atol=1e-5)
    np.testing.assert_allclose(loaded["bbox_size"], aabb_max - aabb_min, atol=1e-6)

    mat = loaded["material"]
    assert mat["name"] == "ceramic"
    assert 0.0 <= mat["roughness"] <= 1.0
    assert 0.0 <= mat["metallic"] <= 1.0
    assert len(mat["albedo"]) == 3
    # spec.color is set → albedo comes from the spec, not the mean color.
    np.testing.assert_allclose(mat["albedo"], list(spec.color), atol=1e-6)

    phys = loaded["physics"]
    assert phys["density_kg_m3"] == pytest.approx(2400.0)
    assert 0.0 <= phys["friction"] <= 1.0
    assert 0.0 <= phys["restitution"] <= 1.0

    assert loaded["mesh"]["format"] == "glb"
    assert loaded["mesh"]["vertices"] == 123
    assert loaded["point_cloud"]["points"] == result.positions.shape[0]
    assert loaded["spec"]["shape"] == "vase"


def test_manifest_v2_parts_materials_physics(tmp_path):
    """iemodel/2: per-part materials, measured volumes, collider, mass."""
    seat = Primitive("box", Primitive.identity_transform(),
                     {"size": [0.4, 0.05, 0.4], "material": "fabric"}, "seat")
    leg = Primitive("cylinder", Primitive.identity_transform(),
                    {"radius": 0.02, "height": 0.4, "material": "metal"}, "leg_0")
    ball = Primitive("ellipsoid", Primitive.identity_transform(),
                     {"radii": [0.1, 0.08, 0.1]}, "knob")
    spec = GenerationSpec(
        shape="chair", n_points=3000, bbox_size=(0.5, 0.5, 0.5),
        primitives=[seat, leg, ball], features=[], color=(0.5, 0.4, 0.3), seed=7,
    )
    result = generate(spec)
    manifest = build_manifest(
        spec, result.positions, result.colors,
        mesh_path="chair_x.glb", mesh_stats={"vertices": 10, "faces": 20},
        labels=result.labels,
    )

    # v2 part records: one per primitive, analytic solid volumes.
    parts = {p["label"]: p for p in manifest["parts"]}
    assert set(parts) == {"seat", "leg_0", "knob"}
    assert parts["seat"]["primitive"] == "box"
    assert parts["seat"]["material"] == "fabric"
    assert parts["seat"]["solid_volume_m3"] == pytest.approx(0.4 * 0.05 * 0.4)
    assert parts["leg_0"]["solid_volume_m3"] == pytest.approx(
        np.pi * 0.02 ** 2 * 0.4, rel=1e-6)
    assert parts["knob"]["solid_volume_m3"] == pytest.approx(
        4 / 3 * np.pi * 0.1 * 0.08 * 0.1, rel=1e-6)

    # Per-part materials carry PBR + physics presets.
    mats = manifest["materials"]
    assert set(mats) >= {"fabric", "metal"}
    assert mats["metal"]["metallic"] > 0.5
    assert mats["metal"]["density_kg_m3"] == pytest.approx(7870.0)
    assert mats["fabric"]["roughness"] > 0.5
    assert len(mats["fabric"]["albedo"]) == 3

    # Multi-part object → compound collider; mass = Σ volume × density.
    phys = manifest["physics"]
    assert phys["collider"] == "parts"
    expected_vol = sum(p["solid_volume_m3"] for p in manifest["parts"])
    assert phys["solid_volume_m3"] == pytest.approx(expected_vol)
    # majority material across primitives is fabric+metal+default tie →
    # whatever resolve_material picked; mass must equal volume × density.
    assert phys["mass_kg"] == pytest.approx(expected_vol * phys["density_kg_m3"])


def test_manifest_collider_selection():
    single_box = _spec()
    m = build_manifest(single_box, generate(single_box).positions)
    assert m["physics"]["collider"] == "box"

    ball = GenerationSpec(
        shape="abstract", n_points=500, bbox_size=(1, 1, 1),
        primitives=[Primitive("ellipsoid", Primitive.identity_transform(),
                              {"radii": [0.4, 0.2, 0.4]}, "blob")],
        features=[], color=None, seed=1,
    )
    m = build_manifest(ball, generate(ball).positions)
    assert m["physics"]["collider"] == "convex"


def test_manifest_name_fallbacks():
    spec = _spec()
    result = generate(spec)
    # point cloud path stem wins when no mesh path / explicit name is given.
    m = build_manifest(spec, result.positions, point_cloud_path="foo_bar.ply")
    assert m["name"] == "foo_bar"
    # explicit name wins over everything.
    m = build_manifest(spec, result.positions, mesh_path="a.glb", name="explicit")
    assert m["name"] == "explicit"
    # no file at all → spec shape.
    m = build_manifest(spec, result.positions)
    assert m["name"] == "chair"


def test_manifest_mesh_null_without_mesh():
    spec = _spec()
    result = generate(spec)
    manifest = build_manifest(spec, result.positions, result.colors,
                              point_cloud_path="model.ply")
    assert manifest["mesh"] is None
    assert manifest["point_cloud"]["points"] == result.positions.shape[0]


# ------------------------------------------------------------- GLB export

def test_glb_export_has_geometry_colors_and_normals(tmp_path):
    o3d = pytest.importorskip("open3d")
    spec = _spec(material="wood")
    result = generate(spec)
    assert result.positions.shape[0] >= 2000

    out = exporter.write_glb(tmp_path / "cloud.glb", result.positions, result.colors)
    assert out.exists()

    mesh = o3d.io.read_triangle_mesh(str(out))
    assert len(mesh.triangles) > 0
    assert mesh.has_vertex_colors()
    assert mesh.has_vertex_normals() or mesh.has_triangle_normals()

    aabb = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(aabb.get_extent())
    assert float(np.prod(extent)) > 0.0  # not degenerate
