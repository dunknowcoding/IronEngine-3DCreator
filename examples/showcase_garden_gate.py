"""Showcase example 1 — wrought-iron garden gate (arch + tube + framework).

Builds the README's garden-gate spec: box posts with cone caps, a true
``arch`` top rail, three horizontal rails, seven bars with finial spheres,
two curved ``tube`` scrolls and a torus rosette — then runs the offline
pipeline (validate → integrity repair → point cloud + analytic meshes) and
exports GLB / PLY / spec JSON to ``examples/out/garden_gate/``.

Run:  python examples/showcase_garden_gate.py
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


def garden_gate() -> GenerationSpec:
    prims = []
    for sx in (-1, 1):
        prims.append(P("box", {"size": [0.09, 1.12, 0.09], "material": "iron"},
                       f"post_{'L' if sx < 0 else 'R'}", T(0.56 * sx, 0.56, 0)))
        prims.append(P("cone", {"radius": 0.075, "height": 0.13, "material": "iron"},
                       f"cap_{'L' if sx < 0 else 'R'}", T(0.56 * sx, 1.185, 0)))
    prims.append(P("arch", {"major_radius": 0.56, "minor_radius": 0.028,
                            "material": "iron"}, "arch", T(0, 1.12, 0)))
    for y, name in ((0.10, "rail_bottom"), (0.56, "rail_mid"), (1.00, "rail_top")):
        prims.append(P("cylinder", {"radius": 0.016, "height": 1.02, "caps": True,
                                    "material": "iron"}, name, T(0, y, 0, rz=math.pi / 2)))
    for i, x in enumerate(np.linspace(-0.44, 0.44, 7)):
        prims.append(P("cylinder", {"radius": 0.011, "height": 0.92, "caps": True,
                                    "material": "iron"}, f"bar_{i}", T(x, 0.55, 0)))
        prims.append(P("sphere", {"radius": 0.021, "material": "iron"},
                       f"ball_{i}", T(x, 1.015, 0)))
    left_scroll = [[-0.02, 0.60, 0.0], [-0.16, 0.62, 0.0], [-0.24, 0.72, 0.0],
                   [-0.22, 0.84, 0.0], [-0.12, 0.90, 0.0], [-0.02, 0.87, 0.0]]
    right_scroll = [[-x, y, z] for x, y, z in left_scroll]
    prims.append(P("tube", {"path": left_scroll, "radius": 0.009, "caps": True,
                            "material": "iron"}, "scroll_L"))
    prims.append(P("tube", {"path": right_scroll, "radius": 0.009, "caps": True,
                            "material": "iron"}, "scroll_R"))
    prims.append(P("torus", {"major_radius": 0.045, "minor_radius": 0.010,
                             "material": "iron"}, "rosette", T(0, 0.74, 0)))
    return GenerationSpec(shape="abstract", n_points=40_000,
                          bbox_size=(1.2, 1.6, 0.2), primitives=prims, seed=21)


def main() -> int:
    out_dir = OUT_ROOT / "garden_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed, cloud, parts, warnings = run_pipeline(garden_gate())

    write_ply(out_dir / "garden_gate.ply", cloud.positions, cloud.colors)
    write_glb(out_dir / "garden_gate.glb", cloud.positions, cloud.colors,
              spec=fixed)
    (out_dir / "garden_gate.spec.json").write_text(
        json.dumps(fixed.to_json(), indent=2), encoding="utf-8")

    print(f"parts: {len(parts)}  points: {len(cloud.positions)}")
    for w in warnings:
        print(f"  warning: {w}")
    try_render(parts, out_dir / "garden_gate.png", azimuth=28, elevation=14,
               fov=36, albedo_overrides={p.label: (0.10, 0.10, 0.11) for p in parts})
    print(f"outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
