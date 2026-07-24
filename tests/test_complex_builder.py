"""Tests for the part-graph complex builder (generation.complex_builder).

Covers: primitive + loft nodes, attachment transform composition, mirror and
radial-array instancing with shared mesh memory (one definition, many
instances), per-named-part AABB metadata, stats, and world-space baking
(including winding repair for mirrored instances). No network.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ironengine_3d_creator.generation import slicer
from ironengine_3d_creator.generation.analytic_mesh import signed_volume
from ironengine_3d_creator.generation.complex_builder import PartGraph


def _vase_graph() -> PartGraph:
    g = PartGraph("vase")
    g.add_loft(
        "body",
        slicer.profile_circle(0.05, 24),
        slicer.radius_slices([0.0, 0.08, 0.18, 0.26, 0.32],
                             [0.55, 1.0, 0.8, 0.42, 0.46]),
        material="ceramic",
    )
    return g


# ----------------------------------------------------------------------
# nodes: primitives + lofts
# ----------------------------------------------------------------------

def test_loft_node_builds_watertight_part():
    r = _vase_graph().build()
    assert len(r.parts) == 1
    p = r.parts[0]
    assert p.kind == "loft" and p.material == "ceramic"
    assert p.solid_volume_m3 > 0.0
    hi = p.aabb_max
    assert hi[1] == pytest.approx(0.32, abs=1e-6)


def test_unknown_primitive_kind_rejected():
    g = PartGraph()
    with pytest.raises(ValueError):
        g.add_primitive("x", "klein_bottle")


def test_duplicate_name_rejected():
    g = _vase_graph()
    with pytest.raises(ValueError):
        g.add_primitive("body", "box")


# ----------------------------------------------------------------------
# attachments
# ----------------------------------------------------------------------

def test_attach_composes_transforms():
    g = PartGraph()
    g.add_primitive("arm", "box", {"size": [0.1, 0.5, 0.1]}, translate=(0, 0.25, 0))
    g.add_primitive("hand", "sphere", {"radius": 0.05}, parent="arm",
                    translate=(0, 0.30, 0))
    r = g.build()
    hand = next(p for p in r.parts if p.name == "hand")
    # hand local (0, 0.3, 0) under arm (0, 0.25, 0) → world y ≈ 0.55.
    assert hand.aabb_min[1] == pytest.approx(0.50, abs=1e-6)
    assert hand.aabb_max[1] == pytest.approx(0.60, abs=1e-6)


def test_reparent_via_attach():
    g = PartGraph()
    g.add_primitive("base", "box", {"size": [0.2, 0.1, 0.2]}, translate=(1.0, 0, 0))
    g.add_primitive("top", "box", {"size": [0.1, 0.1, 0.1]})
    g.attach("top", "base")
    r = g.build()
    top = next(p for p in r.parts if p.name == "top")
    assert top.aabb_min[0] == pytest.approx(0.95, abs=1e-6)


# ----------------------------------------------------------------------
# instancing: shared mesh memory
# ----------------------------------------------------------------------

def test_mirror_instance_shares_mesh_memory():
    g = PartGraph()
    g.add_primitive("wing_l", "box", {"size": [0.7, 0.03, 0.25]},
                    translate=(0.5, 0.55, 0))
    g.mirror("wing_l", axis="x")
    r = g.build()
    wings = [p for p in r.parts if p.name == "wing_l"]
    assert len(wings) == 2
    assert wings[0].vertices is wings[1].vertices
    assert wings[0].faces is wings[1].faces
    assert np.shares_memory(wings[0].vertices, wings[1].vertices)
    # Mirror AABB: x range reflected.
    assert wings[1].aabb_min[0] == pytest.approx(-wings[0].aabb_max[0], abs=1e-6)
    assert wings[1].aabb_max[0] == pytest.approx(-wings[0].aabb_min[0], abs=1e-6)
    # Volumes equal (|det| = 1 for a reflection).
    assert wings[0].solid_volume_m3 == pytest.approx(wings[1].solid_volume_m3)


def test_radial_array_count_and_memory():
    g = PartGraph()
    g.add_loft("petal", slicer.profile_superellipse(0.06, 0.03, 2.5, 16),
               slicer.radius_slices([0.0, 0.12], [1.0, 0.3]),
               translate=(0.12, 0, 0), rz=-0.4)
    g.array_radial("petal", 6, axis="y")
    r = g.build()
    petals = [p for p in r.parts if p.name == "petal"]
    assert len(petals) == 6
    first = petals[0]
    assert all(p.vertices is first.vertices for p in petals)
    # Six instances arranged around Y: distinct world transforms.
    transforms = {tuple(np.round(p.transform[:3, 3], 6)) for p in petals}
    assert len(transforms) == 6
    # Unique geometry is built once: stats report unique vs total tris.
    st = r.stats()["petal"]
    assert st["instances"] == 6
    assert st["tris_total"] == 6 * st["tris_unique"]


def test_mirror_of_loft_shares_memory():
    g = _vase_graph()
    g.mirror("body", axis="x", coord=0.5)
    r = g.build()
    bodies = [p for p in r.parts if p.name == "body"]
    assert len(bodies) == 2
    assert bodies[0].vertices is bodies[1].vertices


# ----------------------------------------------------------------------
# metadata: AABBs / stats / bake
# ----------------------------------------------------------------------

def test_per_name_aabb_merges_instances():
    g = PartGraph()
    g.add_primitive("post", "cylinder", {"radius": 0.05, "height": 1.1},
                    translate=(0.4, 0.55, 0))
    g.mirror("post", axis="x")
    r = g.build()
    lo, hi = r.aabbs()["post"]
    assert lo[0] == pytest.approx(-0.45, abs=1e-6)
    assert hi[0] == pytest.approx(0.45, abs=1e-6)
    assert hi[1] == pytest.approx(1.1, abs=1e-6)


def test_bake_produces_worldspace_parts_with_positive_volume():
    g = PartGraph()
    g.add_primitive("wing_l", "box", {"size": [0.7, 0.03, 0.25]},
                    translate=(0.5, 0.55, 0))
    g.mirror("wing_l", axis="x")
    g.add_loft("body", slicer.profile_circle(0.05, 16),
               slicer.radius_slices([0.0, 0.3], [1.0, 0.6]))
    r = g.build()
    baked = r.bake()
    assert len(baked) == 3
    for bp in baked:
        assert signed_volume(bp.vertices, bp.faces) > 0.0, bp.label
    wing = next(p for p in baked if p.label == "wing_l")
    assert wing.vertices[:, 0].min() == pytest.approx(0.15, abs=1e-5)
    mwing = next(p for p in baked if p.label == "wing_l#1")
    assert mwing.vertices[:, 0].max() == pytest.approx(-0.15, abs=1e-5)


def test_triangle_count_and_stats():
    g = _vase_graph()
    g.add_primitive("ring", "torus", {"major_radius": 0.05, "minor_radius": 0.01},
                    translate=(0, 0.32, 0))
    r = g.build()
    st = r.stats()
    assert set(st) == {"body", "ring"}
    assert r.triangle_count() == sum(s["tris_total"] for s in st.values())
    assert r.triangle_count() > 0
