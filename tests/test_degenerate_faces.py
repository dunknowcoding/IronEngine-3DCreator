"""Tests for the analytic-mesh degenerate-face cleanup (CR_Integrator).

UV-sphere poles, cone apices and bent-panel seams used to emit exactly
zero-area faces; `weld_mesh` welds coincident vertices and drops them at
the build path so showcase models and the multiview QA report stay clean.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.generation import multiview as mv
from ironengine_3d_creator.generation.analytic_mesh import (
    MESH_BUILDERS,
    build_part_mesh,
    build_spec_meshes,
    count_degenerate_faces,
    weld_mesh,
)


def _t(x=0.0, y=0.0, z=0.0, rx=0.0):
    c, s = math.cos(rx), math.sin(rx)
    return [[1, 0, 0, x], [0, c, -s, y], [0, s, c, z], [0, 0, 0, 1]]


_KIND_PARAMS = {
    "box": {"size": [1.0, 1.0, 1.0]},
    "sphere": {"radius": 0.5},
    "cylinder": {"radius": 0.4, "height": 1.0, "caps": True},
    "capsule": {"radius": 0.3, "height": 1.0},
    "cone": {"radius": 0.5, "height": 1.0},
    "torus": {"major_radius": 0.5, "minor_radius": 0.15},
    "ellipsoid": {"radii": [0.5, 0.3, 0.2]},
    "prism": {"sides": 6, "radius": 0.5, "height": 1.0},
    "helix": {"radius": 0.4, "pitch": 0.2, "turns": 3.0, "thickness": 0.05},
    "plane": {"size": [1.0, 1.0]},
    "superellipsoid": {"radii": [0.5, 0.4, 0.3], "exponents": [0.5, 0.5]},
    "tube": {"path": [[0, 0, 0], [0.2, 0.1, 0], [0.4, 0.3, 0.1]],
             "radius": 0.05, "caps": True},
    "arch": {"major_radius": 0.5, "minor_radius": 0.1},
    "panel": {"size": [1.0, 1.0], "thickness": 0.02},
    "panel_bent": {"size": [1.0, 1.0], "thickness": 0.02, "bend": 0.9},
    "sweep": {"path": [[0, 0, 0], [0.3, 0.2, 0], [0.5, 0.5, 0]],
              "radius": 0.05},
}


# ---------------------------------------------------------------------------
# weld_mesh unit behaviour
# ---------------------------------------------------------------------------

def test_raw_sphere_has_pole_degenerates_and_weld_removes_them():
    v, n, uv, f = MESH_BUILDERS["sphere"](_KIND_PARAMS["sphere"])
    assert count_degenerate_faces(v, f) > 0, "raw sphere must show the pole issue"
    vw, nw, uvw, fw = weld_mesh(v, n, uv, f)
    assert count_degenerate_faces(vw, fw) == 0
    assert fw.max() < vw.shape[0] and fw.min() >= 0
    # degenerate pole fans are dropped; attributes stayed aligned
    assert fw.shape[0] < f.shape[0]
    assert nw.shape == vw.shape and uvw.shape[0] == vw.shape[0]


def test_weld_is_noop_for_already_clean_mesh():
    v, n, uv, f = MESH_BUILDERS["box"](_KIND_PARAMS["box"])
    vw, nw, uvw, fw = weld_mesh(v, n, uv, f)
    # box corners share position but carry split normals/uvs: hard edges
    # must survive — no welding, no dropped faces.
    assert vw.shape[0] == v.shape[0] and fw.shape[0] == f.shape[0]


def test_weld_preserves_winding_and_volume_sign():
    from ironengine_3d_creator.generation.analytic_mesh import signed_volume
    v, n, uv, f = MESH_BUILDERS["capsule"](_KIND_PARAMS["capsule"])
    vw, _, _, fw = weld_mesh(v, n, uv, f)
    assert signed_volume(np.asarray(vw, float), fw) > 0.0


# ---------------------------------------------------------------------------
# build path: every kind comes out clean
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(_KIND_PARAMS))
def test_build_part_mesh_has_no_degenerate_faces(kind):
    params = dict(_KIND_PARAMS[kind])
    real_kind = "panel" if kind == "panel_bent" else kind
    part = build_part_mesh(real_kind, params, np.eye(4), kind, "stone")
    assert count_degenerate_faces(part.vertices, part.faces) == 0, kind
    assert part.faces.shape[0] > 0, kind


# ---------------------------------------------------------------------------
# showcase-style specs + multiview QA theme
# ---------------------------------------------------------------------------

def teapot_like_spec() -> GenerationSpec:
    prims = [
        Primitive("ellipsoid", _t(0, 0.085, 0),
                  {"radii": [0.095, 0.075, 0.095], "material": "porcelain"}, "body"),
        Primitive("ellipsoid", _t(0, 0.090, 0),
                  {"radii": [0.080, 0.062, 0.080], "role": "subtract",
                   "target": "body"}, "hollow"),
        Primitive("cylinder", _t(0, 0.007, 0),
                  {"radius": 0.052, "height": 0.014, "caps": True}, "base"),
        Primitive("torus", _t(0, 0.148, 0),
                  {"major_radius": 0.046, "minor_radius": 0.008}, "rim"),
        Primitive("sphere", _t(0, 0.178, 0), {"radius": 0.012}, "knob"),
        Primitive("cone", _t(0.12, 0.1, 0), {"radius": 0.02, "height": 0.08},
                  "spout_tip"),
    ]
    return GenerationSpec(shape="vase", n_points=20_000,
                          bbox_size=(0.35, 0.2, 0.2), primitives=prims, seed=13)


def creature_spec() -> GenerationSpec:
    return GenerationSpec(
        shape="creature", n_points=20_000, bbox_size=(0.5, 0.6, 1.2), seed=5,
        primitives=[
            Primitive("ellipsoid", _t(), {"radii": [0.18, 0.16, 0.35]}, "body"),
            Primitive("sphere", _t(0, 0.14, 0.44), {"radius": 0.12}, "head"),
            Primitive("cone", _t(-0.06, 0.28, 0.46),
                      {"radius": 0.04, "height": 0.09}, "ear_left"),
            Primitive("cone", _t(0.06, 0.28, 0.46),
                      {"radius": 0.04, "height": 0.09}, "ear_right"),
            Primitive("capsule", _t(-0.09, -0.24, 0.26),
                      {"radius": 0.035, "height": 0.22}, "leg_fl"),
            Primitive("capsule", _t(0.09, -0.24, -0.26),
                      {"radius": 0.045, "height": 0.24}, "leg_hr"),
        ])


def test_showcase_specs_have_zero_degenerate_faces():
    for spec in (teapot_like_spec(), creature_spec()):
        parts = build_spec_meshes(spec)
        total = sum(count_degenerate_faces(p.vertices, p.faces) for p in parts)
        assert total == 0, f"{spec.shape}: {total} degenerate faces"


def test_subtraction_carved_part_is_also_clean():
    parts = build_spec_meshes(teapot_like_spec())
    body = next(p for p in parts if p.label == "body")
    assert count_degenerate_faces(body.vertices, body.faces) == 0


def test_multiview_qa_has_no_degenerate_theme():
    report = mv.qa_report(creature_spec(), reference_compare=False)
    assert report["geometry"]["degenerate_faces"] == 0
    assert not [i for i in report["issues"] if i["code"] == "degenerate_faces"]
