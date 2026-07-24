"""Showcase example 2 — handled teapot with CSG-lite subtraction.

Builds the README's teapot: an ellipsoid body hollowed by a
``role: "subtract"`` cutter, a tapered tube spout, a curved tube handle,
torus rim, superellipsoid lid and a sphere knob. The run prints exactly how
the current pipeline handles the cutter (mesh-level carve when a straight
hole through a supported host is possible, point-cloud subtraction
otherwise).

Run:  python examples/showcase_teapot.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _showcase_common import OUT_ROOT, P, T, run_pipeline, try_render

from ironengine_3d_creator.alignment.schema import GenerationSpec
from ironengine_3d_creator.core.exporter import write_glb, write_ply


def _arc_xy(cx, cy, r, deg0, deg1, n=9, bulge=1):
    """Smooth arc in the XY plane; bulge=+1/-1 chooses the +x/-x side."""
    return [[cx + bulge * r * math.sin(math.radians(t)),
             cy + r * math.cos(math.radians(t)), 0.0]
            for t in np.linspace(deg0, deg1, n)]


def teapot() -> GenerationSpec:
    body = P("ellipsoid", {"radii": [0.095, 0.075, 0.095], "material": "porcelain"},
             "body", T(0, 0.085, 0))
    # the cutter: hollows the body (point-cloud CSG-lite; see printed warnings)
    hollow = P("ellipsoid", {"radii": [0.080, 0.062, 0.080], "role": "subtract",
                             "target": "body"}, "hollow", T(0, 0.090, 0))
    base = P("cylinder", {"radius": 0.052, "height": 0.014, "caps": True,
                          "material": "porcelain"}, "base", T(0, 0.007, 0))
    rim = P("torus", {"major_radius": 0.046, "minor_radius": 0.008,
                      "material": "porcelain"}, "rim", T(0, 0.148, 0))
    lid = P("superellipsoid", {"radii": [0.048, 0.018, 0.048],
                               "exponents": [0.55, 0.55], "material": "porcelain"},
            "lid", T(0, 0.156, 0))
    knob = P("sphere", {"radius": 0.012, "material": "porcelain"}, "knob",
             T(0, 0.178, 0))
    spout = P("tube", {"path": [[0.082, 0.072, 0.0], [0.108, 0.078, 0.0],
                                [0.130, 0.094, 0.0], [0.146, 0.116, 0.0],
                                [0.152, 0.140, 0.0]],
                       "radius": 0.017, "radius2": 0.010, "caps": True,
                       "material": "porcelain"}, "spout")
    handle = P("tube", {"path": _arc_xy(-0.084, 0.082, 0.036, 0, 180, bulge=-1),
                        "radius": 0.009, "caps": True,
                        "material": "porcelain"}, "handle")
    return GenerationSpec(shape="vase", n_points=40_000, bbox_size=(0.35, 0.2, 0.2),
                          primitives=[body, hollow, base, rim, lid, knob,
                                      spout, handle], seed=13)


def main() -> int:
    out_dir = OUT_ROOT / "teapot"
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed, cloud, parts, warnings = run_pipeline(teapot())

    write_ply(out_dir / "teapot.ply", cloud.positions, cloud.colors)
    write_glb(out_dir / "teapot.glb", cloud.positions, cloud.colors, spec=fixed)
    (out_dir / "teapot.spec.json").write_text(
        json.dumps(fixed.to_json(), indent=2), encoding="utf-8")

    print(f"parts: {len(parts)}  points: {len(cloud.positions)}")
    for w in warnings:
        print(f"  warning: {w}")
    overrides = {p.label: (0.93, 0.94, 0.96) for p in parts}
    overrides.update({"lid": (0.24, 0.38, 0.62), "knob": (0.24, 0.38, 0.62)})
    try_render(parts, out_dir / "teapot.png", azimuth=290, elevation=16, fov=40,
               albedo_overrides=overrides)
    print(f"outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
