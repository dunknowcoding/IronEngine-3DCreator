"""Tests for generation.texture_maps / texture_apply / textures bridge (CR_Textures).

Covers: channel + dtype contract, determinism, tileability (seam continuity),
size/format contracts, UV application to analytic meshes, the ietexture/1
manifest block round-trip, the bake_detail_to_texture doctrine helper, and the
< 200 ms per map at 512 px performance budget.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.generation import textures as point_textures
from ironengine_3d_creator.generation import texture_apply as ta
from ironengine_3d_creator.generation import texture_maps as tm
from ironengine_3d_creator.generation.analytic_mesh import build_spec_meshes

SEED = 3
KINDS = tm.list_texture_kinds()


# ---------------------------------------------------------------------------
# generation contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_generates_valid_channels(kind):
    maps = tm.generate_maps(kind, size=256, seed=SEED)
    assert "albedo" in maps
    alb = maps["albedo"]
    assert alb.shape == (256, 256, 3)
    assert alb.dtype == np.uint8
    # Not a flat solid colour — the map must actually carry surface detail.
    assert alb.std() > 1.0
    for ch, arr in maps.items():
        if ch == "albedo":
            continue
        assert arr.shape == (256, 256), f"{kind}/{ch} shape {arr.shape}"
        assert arr.dtype == np.uint8, f"{kind}/{ch} dtype {arr.dtype}"
        # A uniformly matte/glossy roughness channel is legitimate; relief
        # and coverage channels must actually vary.
        if ch != "roughness":
            assert arr.std() > 0.5, f"{kind}/{ch} is flat"


@pytest.mark.parametrize("kind", KINDS)
def test_determinism_same_seed_identical(kind):
    a = tm.generate_maps(kind, size=256, seed=42)
    b = tm.generate_maps(kind, size=256, seed=42)
    for ch in a:
        np.testing.assert_array_equal(a[ch], b[ch], err_msg=f"{kind}/{ch}")


def test_different_seed_differs():
    a = tm.generate_maps("wood_oak", size=256, seed=1)["albedo"]
    b = tm.generate_maps("wood_oak", size=256, seed=2)["albedo"]
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("kind", KINDS)
def test_tileability_seam_within_interior_variation(kind):
    """Seam pixels may differ only as much as interior adjacent pixels do.

    High-frequency patterns (denim twill, linen weave) legitimately change a
    lot between neighbouring pixels; a seam is a defect only when the wrap
    edge jumps *more* than the pattern's own maximum single-pixel step.
    """
    maps = tm.generate_maps(kind, size=512, seed=SEED)
    for ch, arr in maps.items():
        a = arr.astype(np.int16)
        if a.ndim == 3:
            a = a.mean(axis=2).astype(np.int16)
        seam = max(
            int(np.abs(a[:, 0] - a[:, -1]).max()),
            int(np.abs(a[0, :] - a[-1, :]).max()),
        )
        interior = max(
            int(np.abs(a[:, 1:] - a[:, :-1]).max()),
            int(np.abs(a[1:, :] - a[:-1, :]).max()),
        )
        assert seam <= max(24, interior), f"{kind}/{ch}: seam {seam} > interior {interior}"


@pytest.mark.parametrize("size", [64, 256, 512, 1024])
def test_supported_sizes(size):
    maps = tm.generate_maps("marble", size=size, seed=SEED)
    assert maps["albedo"].shape == (size, size, 3)


@pytest.mark.parametrize("size", [0, 32, 63, 1025, 2048])
def test_unsupported_sizes_raise(size):
    with pytest.raises(ValueError):
        tm.generate_maps("marble", size=size, seed=SEED)


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        tm.generate_maps("unobtainium", size=256, seed=SEED)


@pytest.mark.parametrize("kind", KINDS)
def test_performance_budget_200ms_at_512(kind):
    tm.generate_maps(kind, size=512, seed=SEED)  # warm-up
    best = min(
        _timed(lambda: tm.generate_maps(kind, size=512, seed=SEED)) for _ in range(3)
    )
    assert best < 0.200, f"{kind} took {best * 1000:.0f} ms for one map set at 512px"


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def test_save_maps_roundtrip(tmp_path):
    maps = tm.generate_maps("wood_oak", size=256, seed=SEED)
    paths = tm.save_maps(maps, tmp_path, kind="wood_oak", size=256, seed=SEED)
    assert set(paths) == {"albedo", "bump"}
    for ch, p in paths.items():
        loaded = tm.load_map(p)
        np.testing.assert_array_equal(loaded, maps[ch])


def test_save_maps_packs_rgba_for_alpha_kinds(tmp_path):
    maps = tm.generate_maps("rococo_ornament", size=256, seed=SEED)
    paths = tm.save_maps(maps, tmp_path, kind="rococo_ornament", size=256, seed=SEED)
    assert "rgba" in paths
    rgba = tm.load_map(paths["rgba"])
    assert rgba.shape == (256, 256, 4)
    np.testing.assert_array_equal(rgba[..., :3], maps["albedo"])
    np.testing.assert_array_equal(rgba[..., 3], maps["alpha"])


# ---------------------------------------------------------------------------
# UV application to analytic meshes
# ---------------------------------------------------------------------------


def _table_spec() -> GenerationSpec:
    return GenerationSpec(
        shape="table",
        primitives=[
            Primitive(kind="box", transform=Primitive.identity_transform(),
                      params={"size": [1.0, 0.05, 0.6], "material": "wood"}, label="tabletop"),
            Primitive(kind="cylinder", transform=Primitive.identity_transform(),
                      params={"radius": 0.05, "height": 0.7, "material": "metal"}, label="leg"),
        ],
    )


def test_apply_maps_to_part_reuses_part_uvs():
    part = build_spec_meshes(_table_spec())[0]
    maps = tm.generate_maps("wood_oak", size=256, seed=SEED)
    cols = ta.apply_maps_to_part(part, maps)
    assert cols.shape == (part.vertices.shape[0], 3)
    assert cols.dtype == np.float32
    assert cols.min() >= 0.0 and cols.max() <= 1.0
    assert cols.std() > 0.01  # the grain must actually modulate the surface


def test_apply_maps_uv_scale_changes_output():
    # Cylinder: u varies continuously around the side, so a non-integer
    # horizontal scale re-phases the wrap and must change the sampling.
    part = build_spec_meshes(_table_spec())[1]
    maps = tm.generate_maps("marble", size=256, seed=SEED)
    a = ta.apply_maps_to_part(part, maps, uv_scale=(1.0, 1.0))
    b = ta.apply_maps_to_part(part, maps, uv_scale=(2.5, 1.0))
    assert not np.allclose(a, b)


def test_apply_maps_to_parts_by_label_and_wildcard():
    parts = build_spec_meshes(_table_spec())
    colors, generated = ta.apply_maps_to_parts(
        parts, {"tabletop": "wood_walnut", "*": "brushed_metal"}, size=256, seed=SEED
    )
    assert len(colors) == len(parts)
    assert set(generated) == {"wood_walnut", "brushed_metal"}
    for part, cols in zip(parts, colors):
        assert cols.shape == (part.vertices.shape[0], 3)
    # Same kind assigned twice must share one generated map set (cache).
    colors2, generated2 = ta.apply_maps_to_parts(
        parts, {"tabletop": "brick", "leg": "brick"}, size=256, seed=SEED
    )
    assert set(generated2) == {"brick"}


def test_sample_map_wrap_vs_clamp():
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4) * 16
    wrapped = ta.sample_map(arr, np.array([[1.25, 0.5]]), wrap=True)
    clamped = ta.sample_map(arr, np.array([[1.25, 0.5]]), wrap=False)
    # Wrap reads x=0.25 of the tile; clamp reads the right edge.
    assert not np.isclose(wrapped[0], clamped[0])


# ---------------------------------------------------------------------------
# textures manifest block (ietexture/1)
# ---------------------------------------------------------------------------


def _block() -> dict:
    return ta.textures_manifest_block(
        [
            {"part": "tabletop", "material": "wood", "kind": "wood_oak",
             "channels": ["albedo", "bump"], "uv": {"wrap": "repeat", "scale": [2, 1]}},
            {"part": "leg", "material": "metal", "kind": "brushed_metal",
             "channels": ["albedo", "roughness"]},
        ],
        size=512,
        seed=SEED,
    )


def test_manifest_block_roundtrip():
    block = _block()
    assert ta.validate_textures_block(block) == []
    # JSON round-trip: the block must survive serialization unchanged.
    rt = json.loads(json.dumps(block))
    assert rt == block
    assert ta.validate_textures_block(rt) == []
    # texture -> part -> channel mapping is fully resolvable.
    by_part = {a["part"]: a for a in rt["assignments"]}
    top_albedo = rt["maps"][by_part["tabletop"]["maps"]["albedo"]]
    assert top_albedo["kind"] == "wood_oak"
    assert top_albedo["channel"] == "albedo"
    assert top_albedo["tileable"] is True
    assert top_albedo["file"].endswith(".png")
    assert by_part["tabletop"]["uv"]["scale"] == [2, 1]


def test_manifest_block_embeds_in_iemodel_manifest(tmp_path):
    """The block composes with core.manifest.build_manifest untouched."""
    from ironengine_3d_creator.core.manifest import build_manifest, write_manifest

    spec = _table_spec()
    parts = build_spec_meshes(spec)
    positions = np.concatenate([p.vertices for p in parts], axis=0)
    manifest = build_manifest(spec, positions)
    manifest["textures"] = _block()
    out = tmp_path / "model.iemodel.json"
    write_manifest(out, manifest)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert ta.validate_textures_block(loaded["textures"]) == []
    assert loaded["schema"] == manifest["schema"]  # iemodel fields untouched


def test_validate_textures_block_catches_errors():
    assert ta.validate_textures_block({"schema": "wrong"}) != []
    bad = _block()
    bad["assignments"][0]["maps"]["albedo"] = "missing_map_id"
    errors = ta.validate_textures_block(bad)
    assert any("missing_map_id" in e for e in errors)
    bad2 = _block()
    bad2["maps"]["wood_oak_albedo"]["file"] = "not_a_png.jpg"
    assert any(".png" in e for e in ta.validate_textures_block(bad2))


# ---------------------------------------------------------------------------
# bake doctrine
# ---------------------------------------------------------------------------


def test_bake_detail_to_texture_from_callable():
    maps = ta.bake_detail_to_texture(ta.weave_detail(threads=32), size=256, seed=SEED)
    assert set(maps) == {"albedo", "bump", "normal"}
    assert maps["albedo"].shape == (256, 256, 3)
    assert maps["normal"].shape == (256, 256, 3)
    assert maps["bump"].shape == (256, 256)
    # Deterministic for the same detail + seed.
    again = ta.bake_detail_to_texture(ta.weave_detail(threads=32), size=256, seed=SEED)
    np.testing.assert_array_equal(maps["bump"], again["bump"])
    # The baked height is tileable (integer thread count): the wrap seam
    # never jumps more than the pattern's own maximum single-pixel step.
    b = maps["bump"].astype(np.int16)
    seam = max(
        int(np.abs(b[:, 0] - b[:, -1]).max()),
        int(np.abs(b[0, :] - b[-1, :]).max()),
    )
    interior = max(
        int(np.abs(b[:, 1:] - b[:, :-1]).max()),
        int(np.abs(b[1:, :] - b[:-1, :]).max()),
    )
    assert seam <= max(24, interior)
    # AO must modulate the base colour (crevices darker than ridges).
    assert maps["albedo"].std() > 1.0


def test_bake_detail_flutes_and_pores():
    fluted = ta.bake_detail_to_texture(ta.flute_detail(flutes=16), size=256)
    assert fluted["bump"].std() > 10  # real relief, not flat
    pores = ta.bake_detail_to_texture(ta.pore_detail(cells=8), size=256, seed=SEED)
    np.testing.assert_array_equal(
        pores["bump"],
        ta.bake_detail_to_texture(ta.pore_detail(cells=8), size=256, seed=SEED)["bump"],
    )


def test_bake_detail_from_array_and_shape_check():
    h = np.tile(np.linspace(0, 1, 256, dtype=np.float32), (256, 1))
    maps = ta.bake_detail_to_texture(h, size=256)
    np.testing.assert_allclose(maps["bump"] / 255.0, h, atol=1 / 255)
    with pytest.raises(ValueError):
        ta.bake_detail_to_texture(np.zeros((8, 8), dtype=np.float32), size=256)


def test_baked_maps_feed_apply_maps_to_part():
    """Doctrine end-to-end: baked detail -> maps -> analytic part UVs."""
    part = build_spec_meshes(_table_spec())[1]  # cylinder leg
    maps = ta.bake_detail_to_texture(
        ta.flute_detail(flutes=12), size=256, base_color=(0.6, 0.55, 0.5)
    )
    cols = ta.apply_maps_to_part(part, maps, uv_scale=(1.0, 1.0))
    assert cols.shape == (part.vertices.shape[0], 3)
    assert cols.std() > 0.01  # flutes visible around the cylinder


# ---------------------------------------------------------------------------
# textures.py bridge
# ---------------------------------------------------------------------------


def test_material_to_kind_bridge():
    assert point_textures.map_kind_for_material("wood") == "wood_oak"
    assert point_textures.map_kind_for_material("wood", wood="wood_walnut") == "wood_walnut"
    assert point_textures.map_kind_for_material(" Metal ") == "brushed_metal"
    assert point_textures.map_kind_for_material(None) is None
    assert point_textures.map_kind_for_material("hologram") is None
    for kind in point_textures.MAP_KIND_BY_MATERIAL.values():
        assert kind in tm.TEXTURE_GENERATORS


def test_maps_for_material_generates():
    maps = point_textures.maps_for_material("brick", size=256, seed=SEED)
    assert maps is not None and "albedo" in maps
    assert point_textures.maps_for_material("hologram") is None


def test_existing_point_cloud_api_untouched():
    """The pre-existing per-point texturing contract must not regress."""
    rng = np.random.default_rng(0)
    pos = rng.random((100, 3)).astype(np.float32)
    cols = point_textures.apply_texture(pos, (0.5, 0.5, 0.5), "wood", rng)
    assert cols.shape == (100, 3)
    assert point_textures.apply_texture(pos, (0.5, 0.5, 0.5), "nope", rng) is None
    assert point_textures.shape_default_material("chair", None) == "wood"
