"""Tests for the CR_Integrator family registrations.

The cross-module adapters (human / building / flora_param / water_container /
7 terrain sub-styles / vehicle) plug the landed feature modules into the
style engine: keyword routing, per-family budgets, determinism, manifest
extras propagation, and the compositor per-part tint hook synergy.
"""
from __future__ import annotations

import numpy as np
import pytest

from ironengine_3d_creator.alignment.validator import normalize
from ironengine_3d_creator.core.manifest import build_manifest
from ironengine_3d_creator.generation.compositor import generate
from ironengine_3d_creator.generation.style_engine import (
    MAX_PARTS,
    STYLE_FAMILIES,
    StyleEngine,
    family_from_prompt,
)
from ironengine_3d_creator.generation.style_families import FAMILY_BUILDERS

NEW_FAMILIES = (
    "human", "building", "flora_param", "water_container", "vehicle",
    "boulder_field", "rock_strata_cliff", "cobblestone_patch", "cracked_mud",
    "mossy_stones", "pebble_riverbed", "stone_slab_pavement",
)

_SMALL = 6_000


# ---------------------------------------------------------------------------
# registration + routing
# ---------------------------------------------------------------------------

def test_new_families_are_registered():
    for fam in NEW_FAMILIES:
        assert fam in FAMILY_BUILDERS, fam
        assert fam in STYLE_FAMILIES, fam


@pytest.mark.parametrize("text,family", [
    ("a building with rooms", "building"),
    ("a woman with ponytails", "human"),
    ("a pond with water", "water_container"),
    ("a two-storey house", "building"),
    ("an aquarium for my fish", "water_container"),
    ("a sedan with racing stripes", "vehicle"),
    ("an oak tree", "flora_param"),
    ("a pine forest edge", "flora_param"),
    ("a boulder field", "boulder_field"),
    ("mossy stones by the path", "mossy_stones"),
    ("a cobblestone patch", "cobblestone_patch"),
    ("the riverbed in summer", "pebble_riverbed"),
])
def test_new_keyword_routing(text, family):
    assert family_from_prompt(text) == family


def test_existing_routes_not_hijacked():
    """Pre-registration routes must keep resolving to their families."""
    assert family_from_prompt("potted fern") == "plant"
    assert family_from_prompt("stone temple with columns") == "architecture"
    assert family_from_prompt("a vase next to a tree") == "vessel"
    assert family_from_prompt("clockwork automaton with gears") == "mechanical"
    assert family_from_prompt("a wooden chair with four legs") == "furniture"
    assert family_from_prompt("a humanoid robot") == "robot"


# ---------------------------------------------------------------------------
# budgets + validity through the engine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", NEW_FAMILIES)
@pytest.mark.parametrize("seed", (1, 7, 12345))
def test_new_family_validates_and_synthesizes(family, seed):
    spec = StyleEngine(seed=seed).generate(family=family, n_points=_SMALL)
    clean, warns = normalize(spec)
    assert not [w for w in warns if "dropped" in w], warns
    assert 3 <= len(clean.primitives) <= MAX_PARTS
    for p in clean.primitives:
        assert p.label, "every part should carry a label"
    res = generate(clean)
    assert res.positions.shape[0] >= _SMALL // 2
    assert np.isfinite(res.positions).all()
    assert np.isfinite(res.colors).all()


@pytest.mark.parametrize("family", NEW_FAMILIES)
def test_new_family_deterministic(family):
    a = StyleEngine(seed=42).generate(family=family, n_points=_SMALL)
    b = StyleEngine(seed=42).generate(family=family, n_points=_SMALL)
    assert a.to_json() == b.to_json()


def test_human_family_carries_bones_and_tinted_parts():
    spec = StyleEngine(seed=5).generate(family="human", complexity="complex",
                                        n_points=120_000)
    labels = {p.label for p in spec.primitives}
    # Sim-bone naming on the proxy parts.
    for stem in ("head", "chest", "pelvis", "upper_leg_l", "upper_leg_r"):
        assert stem in labels, stem
    # Per-part albedo rides the compositor tint hook (params["color"]).
    tinted = [p for p in spec.primitives if (p.params or {}).get("color")]
    assert tinted, "expected per-part color overrides on human parts"


def test_building_family_has_rooms_shell_and_openings():
    spec = StyleEngine(seed=8).generate(family="building", complexity="complex",
                                        n_points=200_000)
    labels = [p.label or "" for p in spec.primitives]
    assert any("slab" in lbl or "wall" in lbl for lbl in labels)
    cutters = [p for p in spec.primitives
               if str((p.params or {}).get("role", "")).lower() == "subtract"]
    # full pipeline buildings carve door/window openings
    assert cutters or len(spec.primitives) <= 5


def test_water_family_fluid_extras_reach_manifest():
    spec = StyleEngine(seed=9).generate(family="water_container", n_points=_SMALL)
    extras = getattr(spec, "manifest_extras", None) or {}
    assert "fluid" in extras, "water family must carry the fluid block"
    # through normalize (pipeline step) the block must survive
    clean, _ = normalize(spec)
    assert "fluid" in (getattr(clean, "manifest_extras", None) or {})
    res = generate(clean)
    manifest = build_manifest(clean, res.positions, res.colors,
                              labels=res.labels, name="pond")
    assert manifest["fluid"]["solver"] == "fluids_sph"
    assert manifest["fluid"]["physics_material"] == "water_surface"


def test_vehicle_family_proxy_parts():
    spec = StyleEngine(seed=3).generate(family="vehicle", complexity="complex",
                                        n_points=120_000)
    labels = [p.label or "" for p in spec.primitives]
    assert any("body" in lbl or "tub" in lbl for lbl in labels)
    assert any("wheel" in lbl or "tire" in lbl for lbl in labels)


def test_terrain_substyles_have_ground_and_scatter():
    for fam, marker in (("boulder_field", "boulder"),
                        ("stone_slab_pavement", "slab")):
        spec = StyleEngine(seed=4).generate(family=fam, complexity="complex",
                                            n_points=120_000)
        labels = [p.label or "" for p in spec.primitives]
        assert any("ground" in lbl or "base" in lbl or marker in lbl for lbl in labels), fam
        extras = getattr(spec, "manifest_extras", None) or {}
        assert "terrain" in extras, fam
