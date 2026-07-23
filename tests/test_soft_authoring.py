"""Tests for iemodel/3 non-rigid manifests and generation.soft_author."""
from __future__ import annotations

import json
import math
from collections import Counter, deque

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.core import exporter
from ironengine_3d_creator.core.manifest import (
    BODY_TYPES, MANIFEST_SCHEMA, build_manifest, write_manifest,
)
from ironengine_3d_creator.generation import soft_author as sa


def _box_spec() -> GenerationSpec:
    seat = Primitive("box", Primitive.identity_transform(),
                     {"size": [0.4, 0.2, 0.4], "material": "wood"}, "seat")
    return GenerationSpec(
        shape="chair", n_points=1000, bbox_size=(0.4, 0.2, 0.4),
        primitives=[seat], features=[], color=(0.5, 0.4, 0.3), seed=3,
    )


def _edge_counts(vertices: np.ndarray, faces: np.ndarray) -> Counter:
    counts: Counter = Counter()
    for a, b, c in np.asarray(faces):
        for e in ((a, b), (b, c), (c, a)):
            counts[tuple(sorted((int(e[0]), int(e[1]))))] += 1
    return counts


# ------------------------------------------------------------- manifest v3

def test_manifest_v3_all_blocks_round_trip(tmp_path):
    """All iemodel/3 blocks survive a write/read round-trip; v2 fields intact."""
    extras = {
        "physics": {"body_type": "soft", "mass_kg": 0.123},
        "soft_body": {
            "kind": "cloth", "resolution": [24, 16], "mass_kg": 0.123,
            "stretch_stiffness": 0.8, "bend_stiffness": 0.3, "damping": 0.06,
            "pin_indices": [0, 23, 345, 383],
        },
        "fracture": {
            "threshold_impulse": 7.5, "fragment_count": 40,
            "pattern": "voronoi", "debris_material": "ceramic",
        },
        "articulation": {
            "ragdoll": True,
            "joints": [
                {"name": "knee_l", "kind": "hinge", "parent": "thigh_l",
                 "child": "shin_l", "axis": [1.0, 0.0, 0.0],
                 "limits_deg": [0.0, 135.0]},
            ],
        },
    }
    spec = _box_spec()
    manifest = build_manifest(
        spec, np.array([[0.0, 0.0, 0.0], [0.4, 0.2, 0.4]]),
        mesh_path="m.glb", extras=extras,
    )
    out = tmp_path / "m.iemodel.json"
    write_manifest(out, manifest)
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["schema"] == MANIFEST_SCHEMA == "iemodel/3"
    assert loaded["physics"]["body_type"] == "soft"
    assert loaded["physics"]["mass_kg"] == pytest.approx(0.123)  # extras override
    assert loaded["soft_body"] == extras["soft_body"]
    assert loaded["fracture"] == extras["fracture"]
    assert loaded["articulation"] == extras["articulation"]
    # v2 / v1 fields still present.
    assert set(loaded["materials"]) >= {"wood"}
    assert loaded["parts"][0]["label"] == "seat"
    assert loaded["physics"]["density_kg_m3"] > 0
    assert loaded["mesh"]["format"] == "glb"
    assert loaded["units"] == "meters"


def test_manifest_rigid_default_has_no_nonrigid_blocks():
    """Without extras the manifest is a plain rigid body (backward compatible)."""
    spec = _box_spec()
    manifest = build_manifest(spec, np.array([[0.0, 0.0, 0.0], [0.4, 0.2, 0.4]]))
    assert manifest["schema"] == "iemodel/3"
    assert manifest["physics"]["body_type"] == "rigid"
    for key in ("soft_body", "fracture", "articulation"):
        assert key not in manifest


def test_manifest_extras_from_spec_attribute():
    """A spec carrying `manifest_extras` is honored without an explicit arg."""
    spec = _box_spec()
    spec.manifest_extras = {
        "physics": {"body_type": "frangible"},
        "fracture": sa.fracture_block("ceramic"),
    }
    manifest = build_manifest(spec, np.array([[0.0, 0.0, 0.0], [0.4, 0.2, 0.4]]))
    assert manifest["physics"]["body_type"] == "frangible"
    assert manifest["fracture"]["debris_material"] == "ceramic"


def test_manifest_unknown_body_type_falls_back_to_rigid():
    spec = _box_spec()
    manifest = build_manifest(
        spec, np.array([[0.0, 0.0, 0.0], [0.4, 0.2, 0.4]]),
        extras={"physics": {"body_type": "gaseous"}},
    )
    assert manifest["physics"]["body_type"] == "rigid"
    assert set(BODY_TYPES) == {"rigid", "soft", "frangible", "articulated"}


# ------------------------------------------------------------- cloth

def test_cloth_grid_topology_and_pins():
    w, h = 24, 16
    result = sa.author_cloth(material="cotton", resolution=(w, h), seed=1)
    part = result.parts[0]
    assert part.vertices.shape == (w * h, 3)
    assert part.faces.shape == (2 * (w - 1) * (h - 1), 3)
    # Flat sheet in the XZ plane, up-facing normals.
    np.testing.assert_allclose(part.vertices[:, 1], 0.0, atol=1e-7)
    np.testing.assert_allclose(
        part.normals, np.tile(np.array([[0.0, 1.0, 0.0]]), (w * h, 1)), atol=1e-6)
    # Pins at the four grid corners.
    assert result.extras["soft_body"]["pin_indices"] == [0, w - 1, (h - 1) * w, h * w - 1]

    sb = result.extras["soft_body"]
    assert sb["kind"] == "cloth"
    assert sb["resolution"] == [w, h]
    # cotton towel: 0.35 kg/m^2 over 0.6 x 0.4 m.
    assert sb["mass_kg"] == pytest.approx(0.35 * 0.6 * 0.4)
    assert 0.0 <= sb["stretch_stiffness"] <= 1.0
    assert 0.0 <= sb["bend_stiffness"] <= 1.0
    assert 0.0 <= sb["damping"] <= 1.0

    manifest = result.build_manifest()
    assert manifest["physics"]["body_type"] == "soft"
    assert manifest["physics"]["mass_kg"] == pytest.approx(sb["mass_kg"])
    assert manifest["soft_body"] == sb


def test_cloth_pin_modes_and_fabric_table():
    r = sa.author_cloth(resolution=(8, 6), pins="top_edge", material="denim")
    assert r.extras["soft_body"]["pin_indices"] == list(range(8))
    r = sa.author_cloth(resolution=(8, 6), pins="none")
    assert r.extras["soft_body"]["pin_indices"] == []
    r = sa.author_cloth(resolution=(8, 6), pins=[3, 4])
    assert r.extras["soft_body"]["pin_indices"] == [3, 4]
    # Denim is heavier and stiffer than silk.
    assert (sa.CLOTH_FABRICS["denim"]["area_density_kg_m2"]
            > sa.CLOTH_FABRICS["silk"]["area_density_kg_m2"])
    with pytest.raises(ValueError):
        sa.author_cloth(material="unobtanium")


# ------------------------------------------------------------- rope

def test_rope_soft_body_block_and_mesh():
    result = sa.author_rope(material="hemp", length=1.2, segment_count=24, seed=2)
    sb = result.extras["soft_body"]
    assert sb["kind"] == "rope"
    assert sb["segment_count"] == 24
    assert sb["pin_indices"] == [0, 23]  # both ends pinned by default
    assert sb["mass_kg"] > 0.0
    assert result.extras["physics"]["body_type"] == "soft"

    part = result.parts[0]
    assert part.vertices.shape[0] > 0 and part.faces.shape[0] > 0
    # Capped tube is a closed manifold with positive enclosed volume.
    counts = _edge_counts(part.vertices, part.faces)
    assert set(counts.values()) == {2}
    analytic = math.pi * 0.012 ** 2
    arc = part.solid_volume_m3 / analytic
    assert sa.signed_volume(part.vertices, part.faces) == pytest.approx(
        part.solid_volume_m3, rel=0.05)
    # Polyline arc of a sagging parabola undershoots the nominal length.
    assert arc == pytest.approx(1.2, rel=0.15)

    manifest = result.build_manifest()
    assert manifest["physics"]["body_type"] == "soft"
    assert manifest["soft_body"]["kind"] == "rope"


def test_rope_single_pin_hangs_down():
    result = sa.author_rope(length=1.0, segment_count=12, pins=(0,), seed=4)
    sb = result.extras["soft_body"]
    assert sb["pin_indices"] == [0]
    ys = result.positions[:, 1]
    assert ys.max() <= 0.05 and ys.min() <= -0.9  # hangs below the anchor


# ------------------------------------------------------------- frangible vessel

def test_vessel_wall_thickness_and_watertight():
    R, H, t = 0.09, 0.16, 0.006
    result = sa.author_frangible_vessel(
        material="ceramic", radius=R, height=H, wall_thickness=t, seed=5)
    part = result.parts[0]

    # Wall thickness is real: inner radius = R - t > 0.
    radii = np.linalg.norm(part.vertices[:, [0, 2]], axis=1)
    assert radii.max() == pytest.approx(R, abs=1e-6)
    inner = radii[(radii > 1e-6) & (radii < R - 1e-6)]
    assert inner.min() == pytest.approx(R - t, abs=1e-6)  # R - t > 0
    assert part.solid_volume_m3 == pytest.approx(sa.vessel_solid_volume(R, H, t))
    assert part.solid_volume_m3 > 0.0

    # Watertight: closed manifold (every edge shared by exactly 2 triangles)
    # and signed volume matching the analytic wall volume up to lathe faceting
    # (the 32-gon prism inscribed in the circular profile is ~0.8 % smaller).
    counts = _edge_counts(part.vertices, part.faces)
    assert set(counts.values()) == {2}
    sv = sa.signed_volume(part.vertices, part.faces)
    assert sv == pytest.approx(part.solid_volume_m3, rel=0.02)

    # Point cloud respects the shells: between the rim and the bottom, nothing
    # lands inside the wall band (Ri, R). Rim / bottom / floor bands legitimately
    # span those radii at y = H / 0 / t.
    pr = np.linalg.norm(result.positions[:, [0, 2]], axis=1)
    py = result.positions[:, 1]
    in_wall = (pr > R - t + 1e-4) & (pr < R - 1e-4) & (py > 1e-4) & (py < H - 1e-4)
    assert not in_wall.any()


def test_vessel_fracture_block_scales_with_brittleness():
    glass = sa.fracture_block("glass")
    ceramic = sa.fracture_block("ceramic")
    plastic = sa.fracture_block("plastic")
    # More brittle → lower shatter threshold, more fragments.
    assert glass["threshold_impulse"] < ceramic["threshold_impulse"] < plastic["threshold_impulse"]
    assert glass["fragment_count"] > ceramic["fragment_count"] > plastic["fragment_count"]
    assert glass["pattern"] == "shatter"
    assert ceramic["pattern"] == "voronoi"
    assert ceramic["debris_material"] == "ceramic"

    result = sa.author_frangible_vessel(material="porcelain", seed=6)
    manifest = result.build_manifest()
    assert manifest["physics"]["body_type"] == "frangible"
    assert manifest["fracture"]["debris_material"] == "porcelain"
    assert manifest["physics"]["mass_kg"] > 0.0


# ------------------------------------------------------------- ragdoll

# Independent upper bounds on human range of motion (degrees) used to verify
# the emitted limits are plausible.
_HUMAN_ROM_BOUNDS = {
    "waist": (-45.0, 60.0), "spine": (-35.0, 45.0), "neck": (-70.0, 70.0),
    "shoulder_l": (-70.0, 190.0), "shoulder_r": (-70.0, 190.0),
    "elbow_l": (-5.0, 150.0), "elbow_r": (-5.0, 150.0),
    "wrist_l": (-90.0, 90.0), "wrist_r": (-90.0, 90.0),
    "hip_l": (-40.0, 130.0), "hip_r": (-40.0, 130.0),
    "knee_l": (-5.0, 150.0), "knee_r": (-5.0, 150.0),
    "ankle_l": (-60.0, 30.0), "ankle_r": (-60.0, 30.0),
}


def test_ragdoll_joint_graph_and_rom():
    result = sa.author_ragdoll(n_points=4000, seed=7)
    art = result.extras["articulation"]
    assert art["ragdoll"] is True
    joints = art["joints"]
    assert len(joints) == 15
    assert len(result.parts) == 16

    part_labels = {p.label for p in result.parts}
    assert len(part_labels) == 16

    # Joint graph is a connected tree over the 16 body parts.
    adj: dict[str, set[str]] = {label: set() for label in part_labels}
    for j in joints:
        assert j["parent"] in part_labels and j["child"] in part_labels
        assert j["parent"] != j["child"]
        adj[j["parent"]].add(j["child"])
        adj[j["child"]].add(j["parent"])
    seen = {"pelvis"}
    queue: deque[str] = deque(["pelvis"])
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    assert seen == part_labels  # connected (16 nodes, 15 edges ⇒ tree)

    for j in joints:
        assert j["kind"] in ("ball", "hinge")
        lo, hi = j["limits_deg"]
        assert lo < hi
        # Unit axis.
        assert np.linalg.norm(j["axis"]) == pytest.approx(1.0)
        # Within human ROM bounds.
        blo, bhi = _HUMAN_ROM_BOUNDS[j["name"]]
        assert blo <= lo and hi <= bhi, j["name"]

    # Spot-check canonical values.
    by_name = {j["name"]: j for j in joints}
    assert by_name["knee_l"]["limits_deg"] == [0.0, 135.0]
    assert by_name["elbow_r"]["kind"] == "hinge"
    assert by_name["neck"]["kind"] == "ball"

    manifest = result.build_manifest()
    assert manifest["physics"]["body_type"] == "articulated"
    assert manifest["articulation"] == art
    assert len(manifest["parts"]) == 16
    # Capsule parts at human scale.
    assert manifest["aabb_max"][1] == pytest.approx(1.75 * (1.55 + 0.03 + 0.105) / 1.75, abs=0.05)
    assert manifest["physics"]["mass_kg"] > 0.0


def test_ragdoll_labels_cover_all_parts():
    result = sa.author_ragdoll(n_points=8000, seed=8)
    assert set(np.unique(result.labels).tolist()) == set(range(16))
    assert result.label_names == [p.label for p in result.parts]


# ------------------------------------------------------------- GLB export

@pytest.mark.parametrize("kind,kwargs,expected_nodes", [
    ("cloth", {"resolution": (12, 9), "n_points": 800}, {"cloth"}),
    ("rope", {"segment_count": 10, "n_points": 800}, {"rope"}),
    ("frangible_vessel", {"n_points": 800}, {"vessel"}),
    ("ragdoll", {"n_points": 2000}, {b[0] for b in sa._RAGDOLL_BODY}),
])
def test_glb_export_for_each_object_type(tmp_path, kind, kwargs, expected_nodes):
    trimesh = pytest.importorskip("trimesh")
    result = sa.SOFT_GENERATORS[kind](seed=11, **kwargs)

    out = result.write_glb(tmp_path / f"{kind}.glb")
    assert out.exists() and out.stat().st_size > 0

    scene = trimesh.load(str(out))
    node_names = set(scene.graph.nodes_geometry)
    assert expected_nodes <= node_names
    # Geometry is present and non-degenerate.
    assert len(scene.geometry) >= len(expected_nodes)
    extents = np.asarray(scene.bounds[1] - scene.bounds[0])
    assert float(np.prod(np.maximum(extents, 1e-9))) > 0.0


def test_write_glb_parts_direct(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    result = sa.author_frangible_vessel(seed=12)
    out = exporter.write_glb_parts(
        tmp_path / "vessel_direct.glb", result.parts, result.positions, result.colors)
    assert out.exists()
    scene = trimesh.load(str(out))
    assert "vessel" in set(scene.graph.nodes_geometry)
