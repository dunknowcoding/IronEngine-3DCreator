"""Tests for the export formats (W2, W8, W15, W17, F5 GLB PBR)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.core import exporter
from ironengine_3d_creator.generation.compositor import generate


def _cloud(n: int = 500, seed: int = 3):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32)
    colors = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
    return positions, colors


def _chairish_spec(material: str | None = None) -> GenerationSpec:
    params = {"size": [0.4, 0.05, 0.4]}
    if material:
        params["material"] = material
    seat = Primitive("box", Primitive.identity_transform(), params, "part_a")
    leg_params = {"radius": 0.02, "height": 0.4}
    if material:
        leg_params["material"] = material
    leg = Primitive("cylinder", Primitive.identity_transform(), leg_params, "part_b")
    return GenerationSpec(
        shape="abstract", n_points=2000, bbox_size=(0.5, 0.5, 0.5),
        primitives=[seat, leg], features=[], color=(0.5, 0.4, 0.3), seed=11,
    )


# ------------------------------------------------------------- W2: PCD rgb

def test_pcd_rgb_round_trips_through_int_float(tmp_path):
    positions, colors = _cloud()
    out = exporter.write_pcd(tmp_path / "c.pcd", positions, colors)
    lines = out.read_text(encoding="utf-8").splitlines()
    data_start = lines.index("DATA ascii") + 1
    rows = [ln.split() for ln in lines[data_start:]]
    assert len(rows) == positions.shape[0]

    expected_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint32)
    expected_packed = (
        (expected_u8[:, 0] << 16) | (expected_u8[:, 1] << 8) | expected_u8[:, 2]
    )
    for i in (0, len(rows) // 2, len(rows) - 1):
        # Sim's reader contract: rgb recovered with int(float(tok)).
        assert int(float(rows[i][3])) == int(expected_packed[i])
        np.testing.assert_allclose(
            [float(rows[i][0]), float(rows[i][1]), float(rows[i][2])],
            positions[i], atol=1e-6,
        )


# ------------------------------------------------------------- W15: PLY

def test_ply_ascii_round_trip(tmp_path):
    positions, colors = _cloud()
    out = exporter.write_ply(tmp_path / "c.ply", positions, colors)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "format ascii 1.0"
    data_start = lines.index("end_header") + 1
    rows = np.loadtxt(lines[data_start:])
    np.testing.assert_allclose(rows[:, :3], positions, atol=1e-6)
    expected_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(rows[:, 3:6].astype(np.uint8), expected_u8)


def test_ply_binary_flag(tmp_path):
    positions, colors = _cloud(64)
    out = exporter.write_ply(tmp_path / "b.ply", positions, colors, binary=True)
    raw = out.read_bytes()
    assert b"format binary_little_endian 1.0" in raw
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    body = np.frombuffer(
        raw[header_end:],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    assert body.shape[0] == positions.shape[0]
    np.testing.assert_allclose(body["x"], positions[:, 0], atol=1e-7)


# ------------------------------------------------------------- W8: unbaked colors

def test_generated_colors_are_unbaked_albedo():
    """No Lambert term in exported colors (W8): the per-channel ratio to the
    base color stays inside the pure noise band [0.94, 1.06]; the old shaded
    path dipped to 0.65 * 0.92 ≈ 0.60."""
    spec = _chairish_spec(material=None)  # no texture material → albedo path
    result = generate(spec)
    base = np.asarray(spec.color, dtype=np.float64)
    ratio = result.colors.astype(np.float64) / base[None, :]
    assert ratio.min() >= 0.93
    assert ratio.max() <= 1.07


# ------------------------------------------------------------- F5: GLB PBR scene

def test_glb_analytic_scene_has_pbr_nodes_and_color0(tmp_path):
    pytest.importorskip("trimesh")
    pygltflib = pytest.importorskip("pygltflib")
    spec = _chairish_spec(material="wood")
    result = generate(spec)

    out = exporter.write_glb(tmp_path / "chair.glb", result.positions, result.colors, spec=spec)
    assert out.exists()

    g = pygltflib.GLTF2().load(str(out))
    node_names = [n.name for n in g.nodes]
    assert "part_a" in node_names and "part_b" in node_names
    assert len(g.images) >= 1  # baked baseColorTexture present

    # Every mesh primitive carries geometry, UVs, normals, and COLOR_0.
    for mesh in g.meshes:
        for prim in mesh.primitives:
            attrs = {k: v for k, v in prim.attributes.__dict__.items() if v is not None}
            for required in ("POSITION", "NORMAL", "TEXCOORD_0", "COLOR_0"):
                assert required in attrs, f"missing {required}"

    # PBR factors come from the wood preset.
    pbr = g.materials[0].pbrMetallicRoughness
    assert pbr.metallicFactor == pytest.approx(0.0)
    assert pbr.roughnessFactor == pytest.approx(0.65)
    assert pbr.baseColorTexture is not None

    # Analytic AABB matches the spec geometry (box seat of size 0.4 x 0.05 x 0.4
    # at identity + a 0.4-tall cylinder leg) — no reconstruction noise.
    import trimesh
    scene = trimesh.load(str(out), force="scene")
    bounds = scene.bounds
    np.testing.assert_allclose(bounds[0], [-0.2, -0.2, -0.2], atol=1e-4)
    np.testing.assert_allclose(bounds[1], [0.2, 0.2, 0.2], atol=1e-4)


def test_glb_baked_texture_contains_color_variation(tmp_path):
    """The baked baseColorTexture is not a flat image — per-point color
    variation survives as real texels (W7)."""
    pytest.importorskip("trimesh")
    spec = _chairish_spec(material=None)
    result = generate(spec)
    # Strong spatial color gradient to make the check deterministic.
    colors = np.clip(
        (result.positions - result.positions.min(axis=0))
        / (np.ptp(result.positions, axis=0) + 1e-9),
        0, 1,
    ).astype(np.float32)
    out = exporter.write_glb(tmp_path / "g.glb", result.positions, colors, spec=spec)

    import trimesh
    scene = trimesh.load(str(out), force="scene")
    saw_rich_texture = False
    for geom in scene.geometry.values():
        tex = getattr(geom.visual.material, "baseColorTexture", None)
        if tex is not None:
            arr = np.asarray(tex)
            if arr.std(axis=(0, 1)).max() > 5.0:
                saw_rich_texture = True
    assert saw_rich_texture


# ------------------------------------------------------------- W17: OBJ + mtl

def test_obj_writes_mtl_with_kd(tmp_path):
    spec = _chairish_spec(material="wood")
    result = generate(spec)
    out = exporter.write_obj(tmp_path / "m.obj", result.positions, result.colors, spec=spec)
    assert out.exists()
    mtl = out.with_suffix(".mtl")
    assert mtl.exists()

    obj_text = out.read_text(encoding="utf-8")
    assert f"mtllib {mtl.name}" in obj_text
    assert "o part_a" in obj_text and "o part_b" in obj_text
    assert "usemtl wood" in obj_text

    mtl_text = mtl.read_text(encoding="utf-8")
    assert "newmtl wood" in mtl_text
    kd_line = next(ln for ln in mtl_text.splitlines() if ln.startswith("Kd "))
    kd = [float(v) for v in kd_line.split()[1:4]]
    # Kd is in display color space and near the generated mean albedo.
    mean_albedo = result.colors.mean(axis=0)
    np.testing.assert_allclose(kd, mean_albedo, atol=0.15)
