"""Tests for the CR_FloraWater modules: flora_params / terrain_styles / water.

Covers: flora density monotonicity (density 1.0 carries > 3x the grass
blades of density 0.3), per-seed determinism, season/age/branching
behaviour, instancing group integrity, terrain displacement (composited,
never flat), water-container interior fit (no overflow beyond the rim,
surface below rim at fill < 1), extras.fluid round-trips, and fill_level
editing.
"""
from __future__ import annotations

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec
from ironengine_3d_creator.generation import compositor
from ironengine_3d_creator.generation.analytic_mesh import local_aabb
from ironengine_3d_creator.generation.flora_params import (
    FloraParams,
    SPECIES,
    collect_instance_groups,
    flora_spec,
    leaf_presence,
)
from ironengine_3d_creator.generation.terrain_styles import (
    TERRAIN_STYLES,
    TerrainParams,
    terrain_spec,
)
from ironengine_3d_creator.generation.water import (
    FLUID_EXTRAS_SCHEMA,
    WATER_CONTAINERS,
    FluidProperties,
    find_water_parts,
    fluid_of,
    set_fill_level,
    set_fill_volume,
    water_container_spec,
)

_SMALL = 6_000


def _labels(spec: GenerationSpec, prefix: str):
    return [p for p in spec.primitives if (p.label or "").startswith(prefix)]


def _part_aabb(prim) -> tuple[np.ndarray, np.ndarray]:
    """Exact world-space AABB of one primitive.

    Ellipsoids under rotation need the support-function form
    (half_i = sqrt(Σ_j M_ij² e_j²)); boxes/prisms are zonotopes, whose
    exact AABB is |M| @ e. (Pushing local-AABB corners through the
    transform overestimates both.)
    """
    lo, hi = local_aabb(prim.kind, prim.params or {})
    e = (hi - lo) / 2.0
    c = (hi + lo) / 2.0
    T = np.asarray(prim.transform, dtype=np.float64)
    M = T[:3, :3]
    wc = M @ c + T[:3, 3]
    if prim.kind in ("ellipsoid", "sphere"):
        half = np.sqrt((M ** 2) @ (e ** 2))
    else:
        half = np.abs(M) @ e
    return wc - half, wc + half


# ----------------------------------------------------------------------
# flora — density
# ----------------------------------------------------------------------

class TestFloraDensity:
    def test_grass_density_strictly_proportional(self):
        p_dense = FloraParams(kind="grass", style="meadow", density=1.0, seed=7)
        p_sparse = FloraParams(kind="grass", style="meadow", density=0.3, seed=7)
        n_dense = len(_labels(flora_spec(p_dense), "blade"))
        n_sparse = len(_labels(flora_spec(p_sparse), "blade"))
        assert n_dense > 3.0 * n_sparse, (n_dense, n_sparse)

    def test_grass_density_zero_is_bare(self):
        spec = flora_spec(FloraParams(kind="grass", density=0.0, seed=3))
        assert _labels(spec, "blade") == []

    def test_tree_leaf_count_monotonic(self):
        counts = []
        for density in (0.1, 0.4, 0.7, 1.0):
            spec = flora_spec(FloraParams(style="oak", density=density, seed=11))
            counts.append(len(_labels(spec, "leaf")))
        assert counts == sorted(counts) and counts[-1] > counts[0], counts

    def test_tree_keeps_structure_at_low_density(self):
        spec = flora_spec(FloraParams(style="oak", density=0.1, seed=5))
        assert _labels(spec, "trunk"), "trunk must survive density 0.1"
        assert _labels(spec, "branch"), "branches must survive density 0.1"


# ----------------------------------------------------------------------
# flora — determinism / seasons / age / branching / instancing
# ----------------------------------------------------------------------

class TestFloraBehaviour:
    def test_determinism_per_seed(self):
        p = FloraParams(style="maple", density=0.8, seed=42)
        assert flora_spec(p).to_json() == flora_spec(p).to_json()

    def test_different_seeds_differ(self):
        a = flora_spec(FloraParams(style="oak", seed=1)).to_json()
        b = flora_spec(FloraParams(style="oak", seed=2)).to_json()
        assert a != b

    def test_winter_deciduous_drops_leaves(self):
        summer = flora_spec(FloraParams(style="oak", season="summer",
                                        density=1.0, seed=9))
        winter = flora_spec(FloraParams(style="oak", season="winter",
                                        density=1.0, seed=9))
        n_summer = len(_labels(summer, "leaf"))
        n_winter = len(_labels(winter, "leaf"))
        assert n_winter <= 0.15 * n_summer, (n_winter, n_summer)

    def test_winter_evergreen_keeps_needles(self):
        summer = flora_spec(FloraParams(style="pine", season="summer",
                                        density=1.0, seed=9))
        winter = flora_spec(FloraParams(style="pine", season="winter",
                                        density=1.0, seed=9))
        n_summer = len(_labels(summer, "leaf"))
        n_winter = len(_labels(winter, "leaf"))
        assert n_winter >= 0.8 * n_summer, (n_winter, n_summer)

    def test_season_palettes_change_colour(self):
        autumn = flora_spec(FloraParams(style="maple", season="autumn", seed=4))
        summer = flora_spec(FloraParams(style="maple", season="summer", seed=4))
        assert autumn.color != summer.color

    def test_age_changes_girth_and_gnarl(self):
        sapling = flora_spec(FloraParams(style="oak", age="sapling", seed=6))
        ancient = flora_spec(FloraParams(style="oak", age="ancient", seed=6))
        r_sap = sapling.primitives[0].params["radius"]
        r_anc = ancient.primitives[0].params["radius"]
        assert r_anc > 2.5 * r_sap
        assert _labels(ancient, "root_0"), "ancient trees grow buttress roots"
        assert not _labels(sapling, "root_0"), "saplings have no buttress roots"

    def test_branching_override(self):
        whorled = flora_spec(FloraParams(style="oak", branching="whorled",
                                         seed=8))
        labels = {p.label for p in whorled.primitives}
        assert any(lb.startswith("branch_") and lb.count("_") == 2
                   for lb in labels), "whorled branches are labelled branch_<whorl>_<j>"

    @pytest.mark.parametrize("style", ["oak", "maple", "pine", "palm", "fern",
                                       "rose", "lavender", "sunflower",
                                       "meadow", "boxwood"])
    def test_all_species_build(self, style):
        spec = flora_spec(FloraParams(style=style, seed=13))
        assert len(spec.primitives) > 0
        assert spec.manifest_extras["flora"]["style"] == style

    def test_params_round_trip(self):
        p = FloraParams(style="rose", density=0.55, season="spring",
                        age="sapling", seed=21, branching="alternate")
        assert FloraParams.from_dict(p.to_dict()) == p

    def test_instancing_groups_share_geometry(self):
        spec = flora_spec(FloraParams(style="oak", density=1.0, seed=2))
        groups = collect_instance_groups(spec)
        assert "leaf_oak" in groups and len(groups["leaf_oak"]) > 4
        geoms = set()
        for i in groups["leaf_oak"]:
            params = dict(spec.primitives[i].params)
            params.pop("instance_of", None)
            geoms.add(repr(sorted(params.items())))
        assert len(geoms) == 1, "every instanced leaf must share geometry params"

    def test_leaf_presence_table(self):
        oak = SPECIES["oak"]
        pine = SPECIES["pine"]
        assert leaf_presence(oak, "winter") < 0.1
        assert leaf_presence(pine, "winter") > 0.5


# ----------------------------------------------------------------------
# terrain styles
# ----------------------------------------------------------------------

# Ground-ish part that must be displaced per style (never flat).
_DISPLACED_LABEL = {
    "boulder_field": "ground",
    "rock_strata_cliff": "stratum_5",
    "cobblestone_patch": "mortar",
    "cracked_mud": "mud",
    "mossy_stones": "ground",
    "pebble_riverbed": "ground",
    "stone_slab_pavement": "mortar",
}


class TestTerrainStyles:
    @pytest.mark.parametrize("style", TERRAIN_STYLES)
    def test_displacement_non_zero(self, style):
        """Composited ground points must not lie on a flat sheet."""
        spec = terrain_spec(TerrainParams(style=style, density=0.8, seed=17),
                            n_points=_SMALL)
        target = _DISPLACED_LABEL[style]
        idx = next(i for i, p in enumerate(spec.primitives)
                   if (p.label or "") == target)
        out = compositor.generate(spec)
        sel = out.labels == idx
        assert sel.sum() > 50, f"{style}: too few points on {target}"
        y = out.positions[sel, 1]
        assert float(y.std()) > 1e-4, (
            f"{style}: {target} is flat (y std {float(y.std()):.2e})")

    @pytest.mark.parametrize("style", TERRAIN_STYLES)
    def test_determinism(self, style):
        p = TerrainParams(style=style, seed=23)
        assert terrain_spec(p).to_json() == terrain_spec(p).to_json()

    def test_boulder_density_monotonic(self):
        dense = terrain_spec(TerrainParams(style="boulder_field",
                                           density=1.0, seed=5))
        sparse = terrain_spec(TerrainParams(style="boulder_field",
                                            density=0.3, seed=5))
        n_dense = len(_labels(dense, "boulder"))
        n_sparse = len(_labels(sparse, "boulder"))
        assert n_dense >= 3 * n_sparse, (n_dense, n_sparse)

    def test_strata_has_layered_bands(self):
        spec = terrain_spec(TerrainParams(style="rock_strata_cliff", seed=3))
        bands = _labels(spec, "stratum")
        assert len(bands) >= 5
        tops = [_part_aabb(b)[1][1] for b in bands]
        assert tops == sorted(tops), "bands must stack upward"
        # Alternating material hints -> light/dark band stripes.
        mats = [b.params.get("material") for b in bands]
        assert len(set(mats)) >= 2

    @pytest.mark.parametrize("style", ["cobblestone_patch",
                                       "stone_slab_pavement"])
    def test_tileable_lattice_stays_in_bounds(self, style):
        """Lattice parts must keep a half-cell inset so tiles abut."""
        p = TerrainParams(style=style, density=1.0, seed=8)
        spec = terrain_spec(p)
        for prim in spec.primitives:
            if (prim.label or "") in ("mortar", "ground"):
                continue
            lo, hi = _part_aabb(prim)
            assert lo[0] >= -p.width / 2 - 1e-6 and hi[0] <= p.width / 2 + 1e-6
            assert lo[2] >= -p.depth / 2 - 1e-6 and hi[2] <= p.depth / 2 + 1e-6

    def test_moss_overlay_present(self):
        spec = terrain_spec(TerrainParams(style="mossy_stones",
                                          density=0.9, seed=4))
        assert _labels(spec, "stone"), "stones missing"
        assert _labels(spec, "moss"), "moss caps missing"
        kinds = {f.kind for f in spec.features}
        assert "fur" in kinds, "moss fringe uses the fur feature"

    def test_params_round_trip(self):
        p = TerrainParams(style="pebble_riverbed", density=0.4, wet=True, seed=9)
        assert TerrainParams.from_dict(p.to_dict()) == p


# ----------------------------------------------------------------------
# water containers
# ----------------------------------------------------------------------

class TestWaterContainers:
    @pytest.mark.parametrize("kind", WATER_CONTAINERS)
    @pytest.mark.parametrize("fill", [0.3, 0.7, 1.0])
    def test_water_fits_interior(self, kind, fill):
        spec = water_container_spec(kind, fill_level=fill, seed=12)
        fluid = fluid_of(spec)
        body, lip = find_water_parts(spec)
        assert body is not None and lip is not None

        blo, bhi = _part_aabb(body)
        llo, lhi = _part_aabb(lip)

        # 1) water body top is at/below the cavity ceiling, strictly below
        #    the rim when fill < 1.
        cavity_top = fluid.interior_bottom_m + fluid.interior_depth_m
        assert bhi[1] <= cavity_top + 1e-6
        container = [p for p in spec.primitives
                     if (p.label or "") not in ("water", "water_meniscus")]
        rim_top = max(_part_aabb(p)[1][1] for p in container)
        if fill < 1.0:
            assert bhi[1] < rim_top, f"{kind}@{fill}: surface must sit below rim"
        assert bhi[1] <= rim_top + 1e-6, f"{kind}@{fill}: overflow beyond rim!"
        # Meniscus lip never overtakes the rim either.
        assert lhi[1] <= rim_top + 1e-6

        # 2) horizontal fit inside the cavity (with meniscus allowance).
        if fluid.cavity == "round":
            r_max = fluid.interior_radius_m + 1e-6
            assert max(abs(bhi[0]), abs(blo[0])) <= r_max
            assert max(abs(bhi[2]), abs(blo[2])) <= r_max
            assert max(abs(lhi[0]), abs(llo[0])) <= r_max + fluid.meniscus_radius_m
        else:
            w_i, d_i = fluid.interior_size_m
            assert bhi[0] <= w_i / 2 + 1e-6 and blo[0] >= -w_i / 2 - 1e-6
            assert bhi[2] <= d_i / 2 + 1e-6 and blo[2] >= -d_i / 2 - 1e-6

        # 3) fluid bookkeeping is consistent.
        assert fluid.fill_level == pytest.approx(fill)
        expected = fluid.volume_at(fill)
        assert fluid.volume_m3 == pytest.approx(expected)
        if fill > 0:
            assert fluid.volume_m3 > 0

    def test_fill_level_editing_rescales_water(self):
        spec = water_container_spec("basin", fill_level=1.0, seed=2)
        half = set_fill_level(spec, 0.5)
        full_fluid = fluid_of(spec)
        half_fluid = fluid_of(half)
        assert half_fluid.volume_m3 == pytest.approx(full_fluid.volume_m3 * 0.5)
        _, bhi_full = _part_aabb(find_water_parts(spec)[0])
        _, bhi_half = _part_aabb(find_water_parts(half)[0])
        assert bhi_half[1] < bhi_full[1]
        # Input spec is never mutated by the edit.
        assert fluid_of(spec).fill_level == pytest.approx(1.0)

    def test_fill_volume_inverse(self):
        spec = water_container_spec("bucket", fill_level=0.0, seed=2)
        target = 0.0004
        filled = set_fill_volume(spec, target)
        assert fluid_of(filled).volume_m3 == pytest.approx(target, rel=0.02)

    def test_fluid_extras_round_trip_via_spec_json(self):
        """params['extras']['fluid'] survives to_json/from_json verbatim
        (manifest_extras is a build-time attribute, not serialised)."""
        spec = water_container_spec("aquarium", fill_level=0.6, seed=31)
        spec2 = GenerationSpec.from_json(spec.to_json())
        assert fluid_of(spec2).to_dict() == fluid_of(spec).to_dict()

    def test_fluid_properties_dict_round_trip(self):
        f = FluidProperties(fill_level=0.45, cavity="box",
                            interior_bottom_m=0.004, interior_depth_m=0.13,
                            interior_size_m=(0.23, 0.15))
        assert FluidProperties.from_dict(f.to_dict()) == f

    def test_manifest_extras_carries_fluid_block(self):
        spec = water_container_spec("vessel", fill_level=0.8, seed=7)
        block = spec.manifest_extras["fluid"]
        for key in FLUID_EXTRAS_SCHEMA["required"]:
            assert key in block, f"missing fluid key {key!r}"
        assert block["physics_material"] == "water_surface"
        assert block["solver"] == "fluids_sph"

    def test_schema_documents_every_field(self):
        documented = set(FLUID_EXTRAS_SCHEMA["properties"]) | set(
            FLUID_EXTRAS_SCHEMA["required"])
        for f_name in FluidProperties.__dataclass_fields__:
            assert f_name in documented, f"{f_name} undocumented"

    def test_determinism(self):
        a = water_container_spec("pond", fill_level=0.6, seed=19).to_json()
        b = water_container_spec("pond", fill_level=0.6, seed=19).to_json()
        assert a == b

    def test_meniscus_matches_capillary_length(self):
        fluid = fluid_of(water_container_spec("basin", fill_level=0.5, seed=1))
        # sqrt(σ / (ρ g)) ≈ 2.7 mm for pure water.
        assert fluid.meniscus_radius_m == pytest.approx(0.0027, abs=5e-4)


class TestFluidManifestPassthrough:
    """CR_Integrator: 'fluid' rides _EXTRA_PASSTHROUGH_BLOCKS into the
    iemodel/3 manifest, and normalize() preserves spec.manifest_extras so the
    block survives the standard pipeline (normalize → manifest)."""

    def test_fluid_block_lands_in_manifest_verbatim(self):
        from ironengine_3d_creator.core.manifest import (
            _EXTRA_PASSTHROUGH_BLOCKS,
            build_manifest,
        )

        assert "fluid" in _EXTRA_PASSTHROUGH_BLOCKS
        spec = water_container_spec("basin", fill_level=0.6, seed=5)
        res = compositor.generate(spec)
        manifest = build_manifest(spec, res.positions, res.colors,
                                  labels=res.labels, name="basin")
        assert manifest["fluid"] == spec.manifest_extras["fluid"]
        assert manifest["fluid"]["fill_level"] == pytest.approx(0.6)
        assert manifest["fluid"]["physics_material"] == "water_surface"

    def test_normalize_preserves_manifest_extras(self):
        from ironengine_3d_creator.alignment.validator import normalize

        spec = water_container_spec("pond", fill_level=0.5, seed=3)
        assert spec.manifest_extras["fluid"]
        clean, _ = normalize(spec)
        assert getattr(clean, "manifest_extras", None), (
            "normalize() stripped spec.manifest_extras — fluid block lost")
        assert clean.manifest_extras["fluid"] == spec.manifest_extras["fluid"]

    def test_normalize_preserves_flora_and_terrain_extras(self):
        from ironengine_3d_creator.alignment.validator import normalize

        f_spec = flora_spec(FloraParams(style="oak", seed=2))
        t_spec = terrain_spec(TerrainParams(style="boulder_field", seed=2))
        assert "flora" in normalize(f_spec)[0].manifest_extras
        assert "terrain" in normalize(t_spec)[0].manifest_extras
