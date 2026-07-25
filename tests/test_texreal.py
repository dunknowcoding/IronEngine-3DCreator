"""Tests for CR_TexReal: new real-world texture kinds + the image-map GLB
export path (real albedo/normal maps embedded as baseColorTexture) verified
end-to-end through the IronEngine-BonaFide loader.

The generic per-kind contracts (channels, determinism, tileability, the
< 200 ms budget at 512 px) run automatically in test_texture_maps.py because
it parametrizes over ``texture_maps.list_texture_kinds()``; this suite pins
the *semantics* of the new kinds and the exporter wiring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.generation import texture_apply as ta
from ironengine_3d_creator.generation import texture_maps as tm
from ironengine_3d_creator.generation import textures as point_textures
from ironengine_3d_creator.generation.analytic_mesh import build_spec_meshes
from ironengine_3d_creator.core import exporter

REPO = Path(__file__).resolve().parent.parent
BONAFIDE_SRC = REPO.parent / "IronEngine-BonaFide" / "src"

NEW_KINDS = (
    "woodland_camo", "desert_camo", "skin", "knit_wool",
    "plaster_wall", "snow", "mud", "chainmail",
)

SEED = 7


# ---------------------------------------------------------------------------
# kind registration + semantics
# ---------------------------------------------------------------------------


def test_new_kinds_registered():
    kinds = tm.list_texture_kinds()
    for kind in NEW_KINDS:
        assert kind in kinds
    # Aliases resolve to the same generators.
    assert tm.TEXTURE_GENERATORS["knit"] is tm.TEXTURE_GENERATORS["knit_wool"]
    assert tm.TEXTURE_GENERATORS["rusted_metal"] is tm.TEXTURE_GENERATORS["rust"]


def _cluster_shares(albedo: np.ndarray, palette: list[tuple]) -> list[float]:
    px = albedo.reshape(-1, 3).astype(np.float32)
    pal = np.asarray(palette, dtype=np.float32)
    d = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    assign = d.argmin(axis=1)
    return [float((assign == i).mean()) for i in range(len(pal))]


def test_woodland_camo_has_multi_color_blotches():
    palette = [(0.30, 0.34, 0.16), (0.15, 0.19, 0.09), (0.29, 0.21, 0.11), (0.09, 0.09, 0.07)]
    maps = tm.generate_maps("woodland_camo", size=512, seed=SEED)
    shares = _cluster_shares(maps["albedo"] / 255.0, palette)
    # At least three palette colours each cover a meaningful area.
    assert sum(s > 0.08 for s in shares) >= 3


def test_desert_camo_is_sand_dominant():
    maps = tm.generate_maps("desert_camo", size=512, seed=SEED)["albedo"] / 255.0
    # Arid palette: red channel leads, blue is lowest.
    assert maps[..., 0].mean() > maps[..., 1].mean() > maps[..., 2].mean()
    assert maps[..., 0].mean() > 0.5


def test_skin_tone_and_microdetail():
    maps = tm.generate_maps("skin", size=512, seed=SEED)
    alb = maps["albedo"] / 255.0
    # Warm skin ordering R > G > B, mid-tone friendly for tinting.
    assert alb[..., 0].mean() > alb[..., 1].mean() > alb[..., 2].mean()
    assert 0.6 < alb[..., 0].mean() < 0.95
    # Pores/freckles: fine-scale luminance variation exists.
    assert maps["bump"].std() > 3.0


def test_snow_is_bright_with_sparkle():
    maps = tm.generate_maps("snow", size=512, seed=SEED)
    alb = maps["albedo"]
    assert alb.mean() > 200  # snow is bright
    # Sparkle glints: a sparse set of near-white pixels.
    assert (alb.mean(axis=2) > 250).mean() > 1e-4
    # Sparkle lowers roughness (glints).
    assert maps["roughness"].min() < 100


def test_chainmail_has_rings_and_gaps():
    maps = tm.generate_maps("chainmail", size=512, seed=SEED)
    alb = maps["albedo"].mean(axis=2)
    assert alb.max() > 120   # metallic ring band
    assert alb.min() < 45    # deep shadow gaps between rings
    assert maps["bump"].std() > 40  # strong ring relief


def test_mud_has_wet_and_dry_regions():
    maps = tm.generate_maps("mud", size=512, seed=SEED)
    alb = maps["albedo"].mean(axis=2) / 255.0
    assert alb.mean() < 0.35                      # overall dark earth
    assert alb.max() > 0.35                       # drying crust / grit
    assert maps["roughness"].min() < 160          # wet patches are glossy-ish


def test_knit_wool_chevron_relief():
    maps = tm.generate_maps("knit_wool", size=512, seed=SEED)
    assert maps["bump"].std() > 20
    alb = maps["albedo"] / 255.0
    assert alb[..., 0].mean() >= alb[..., 2].mean()  # warm wool tint


def test_plaster_wall_is_matte_offwhite():
    maps = tm.generate_maps("plaster_wall", size=512, seed=SEED)
    alb = maps["albedo"] / 255.0
    assert 0.75 < alb.mean() < 0.95
    assert maps["roughness"].mean() > 220


def test_material_bridge_covers_new_kinds():
    assert point_textures.map_kind_for_material("camo") == "woodland_camo"
    assert point_textures.map_kind_for_material("skin") == "skin"
    assert point_textures.map_kind_for_material("wool") == "knit_wool"
    assert point_textures.map_kind_for_material("snow") == "snow"
    for kind in point_textures.MAP_KIND_BY_MATERIAL.values():
        assert kind in tm.TEXTURE_GENERATORS


# ---------------------------------------------------------------------------
# attach conventions
# ---------------------------------------------------------------------------


def _crate_part():
    spec = GenerationSpec(shape="crate", primitives=[
        Primitive("box", Primitive.identity_transform(),
                  {"size": [0.4, 0.4, 0.4], "material": "wood"}, "crate"),
    ])
    return build_spec_meshes(spec)[0]


def test_attach_maps_to_part_contract():
    part = _crate_part()
    maps = tm.generate_maps("wood_walnut", size=256, seed=SEED)
    out = ta.attach_maps_to_part(part, maps, uv_scale=(2, 3), tint=(1.0, 0.5, 0.25))
    assert out is part
    assert out.maps is maps
    assert out.uv_scale == (2.0, 3.0)
    assert out.tint == (1.0, 0.5, 0.25)
    with pytest.raises(KeyError):
        ta.attach_maps_to_part(_crate_part(), {"bump": maps["bump"]})


def test_attach_maps_to_parts_by_label_and_wildcard():
    spec = GenerationSpec(shape="t", primitives=[
        Primitive("box", Primitive.identity_transform(),
                  {"size": [0.2, 0.2, 0.2]}, "a"),
        Primitive("box", Primitive.identity_transform(),
                  {"size": [0.2, 0.2, 0.2]}, "b"),
    ])
    parts = build_spec_meshes(spec)
    generated = ta.attach_maps_to_parts(
        parts, {"a": "brick", "*": "concrete"}, size=256, seed=SEED,
        uv_scale={"a": (3, 3)}, tints={"a": (0.9, 0.2, 0.2)},
    )
    assert set(generated) == {"brick", "concrete"}
    a, b = parts
    assert a.uv_scale == (3.0, 3.0)
    assert a.tint == pytest.approx((0.9, 0.2, 0.2), abs=1e-6)
    assert not hasattr(b, "tint") and b.uv_scale == (1.0, 1.0)


# ---------------------------------------------------------------------------
# GLB image-map export path
# ---------------------------------------------------------------------------


def _write_textured_crate(tmp_path: Path, *, tint=None) -> Path:
    part = _crate_part()
    maps = tm.generate_maps("wood_walnut", size=512, seed=SEED)
    ta.attach_maps_to_part(part, maps, uv_scale=(2, 2), tint=tint)
    out = tmp_path / "crate_tex.glb"
    exporter.write_glb_parts(out, [part])
    return out


def test_glb_embeds_real_maps(tmp_path):
    pygltflib = pytest.importorskip("pygltflib")
    out = _write_textured_crate(tmp_path, tint=(1.0, 0.9, 0.8))
    g = pygltflib.GLTF2().load(str(out))
    # Real albedo + bump-derived normal maps embedded.
    assert len(g.images) == 2
    assert all(i.mimeType == "image/png" for i in g.images)
    assert g.materials[0].pbrMetallicRoughness.baseColorTexture is not None
    assert g.materials[0].normalTexture is not None
    # Factor stays white: the tint rides on COLOR_0 so the texture is not
    # double-modulated (renderers multiply vertex colour x texture).
    assert g.materials[0].pbrMetallicRoughness.baseColorFactor == [1.0, 1.0, 1.0, 1.0]


def test_glb_map_path_keeps_tint_on_color0(tmp_path):
    out = _write_textured_crate(tmp_path, tint=(1.0, 0.5, 0.25))
    trimesh = pytest.importorskip("trimesh")
    loaded = trimesh.load(str(out), process=False)
    geom = next(iter(loaded.geometry.values()))
    rgba = geom.visual.vertex_attributes["color"]
    np.testing.assert_allclose(
        rgba[:, :3].mean(axis=0) / 255.0, [1.0, 0.5, 0.25], atol=2 / 255
    )


def test_glb_vertex_color_fallback_path_intact(tmp_path):
    """Parts WITHOUT attached maps keep the stock baked-vertex-colour path."""
    spec = GenerationSpec(shape="crate", primitives=[
        Primitive("box", Primitive.identity_transform(),
                  {"size": [0.4, 0.4, 0.4], "material": "wood"}, "crate"),
    ])
    parts = build_spec_meshes(spec)
    positions = np.concatenate([p.vertices for p in parts])
    colors = np.tile(np.array([[0.8, 0.2, 0.2]]), (positions.shape[0], 1))
    out = tmp_path / "crate_baked.glb"
    exporter.write_glb_parts(out, parts, positions, colors)
    g = pytest.importorskip("pygltflib").GLTF2().load(str(out))
    assert g.materials[0].pbrMetallicRoughness.baseColorTexture is not None
    # Baked path: factor = mean vertex colour (0..1 floats, not white).
    factor = g.materials[0].pbrMetallicRoughness.baseColorFactor
    assert factor[0] > 0.6 and factor[1] < 0.4


def test_bump_to_normal_u8_mapping():
    from ironengine_3d_creator.core.exporter import _bump_to_normal_u8

    flat = np.full((64, 64), 128, dtype=np.uint8)
    n = _bump_to_normal_u8(flat)
    assert n.shape == (64, 64, 3) and n.dtype == np.uint8
    np.testing.assert_allclose(n.mean(axis=(0, 1)), [128, 128, 255], atol=2)
    ramp = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    n2 = _bump_to_normal_u8(ramp)
    assert n2[..., 0].mean() < 128  # tilted towards -x in tangent space


# ---------------------------------------------------------------------------
# end-to-end: GLB texture round-trip through the BonaFide loader
# ---------------------------------------------------------------------------


@pytest.fixture()
def bonafide():
    if BONAFIDE_SRC.is_dir():
        sys.path.insert(0, str(BONAFIDE_SRC))
    return pytest.importorskip("ironengine_bonafide")


def test_bonafide_loads_embedded_texture(tmp_path, bonafide):
    from ironengine_bonafide.assets.loaders.gltf import load_primitives

    out = _write_textured_crate(tmp_path, tint=(1.0, 0.9, 0.8))
    prims = load_primitives(out)
    assert len(prims) == 1
    mesh = prims[0].mesh
    assert mesh.uvs is not None and mesh.uvs.shape[1] == 2
    # The embedded albedo map is resolved to a real file with visible grain.
    assert mesh.material.albedo_map is not None
    from PIL import Image

    img = np.asarray(Image.open(mesh.material.albedo_map))
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.std() > 5.0  # real wood grain, not a flat fill
    # uv_scale=(2, 2) reached TEXCOORD_0 (tile repeats).
    assert float(mesh.uvs.max()) > 1.5
    # The tint rides on COLOR_0 (not the base colour factor).
    assert mesh.colors is not None
    np.testing.assert_allclose(
        mesh.colors.mean(dim=0).numpy(), [1.0, 0.9, 0.8], atol=0.02
    )


def test_bonafide_roundtrip_refined_and_textured(tmp_path, bonafide):
    """Full CR_TexReal pipeline: refine (densify) + attach maps -> GLB -> load."""
    from ironengine_3d_creator.generation import refine

    part = _crate_part()
    r = refine.refine_part(part, levels=1, crease_deg=30.0)
    ta.attach_maps_to_part(r, tm.generate_maps("woodland_camo", size=512, seed=SEED))
    out = tmp_path / "crate_refined.glb"
    exporter.write_glb_parts(out, [r])

    from ironengine_bonafide.assets.loaders.gltf import load_primitives

    mesh = load_primitives(out)[0].mesh
    assert mesh.num_triangles == part.faces.shape[0] * 4  # refinement survived
    assert mesh.material.albedo_map is not None
    assert mesh.colors is not None
    np.testing.assert_allclose(mesh.colors.mean(dim=0).numpy(), [1.0, 1.0, 1.0],
                               atol=0.02)  # untinted -> white
