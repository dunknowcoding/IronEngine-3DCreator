"""Headless complex-geometry demo (CR_ComplexGeometry).

Builds two specs end-to-end — an arched chair and a handled mug — through
the real alignment + generation pipeline:

  normalize → check_and_fix → generate (point cloud) + build_spec_meshes_with_report

Showcases every new capability:
  * new primitive kinds: superellipsoid cushion, bent panel lumbar support,
    arch top rail, curved tube handle
  * CSG-lite subtraction: mesh-level elliptical hole carved through the
    chair back panel (cylinder cutter), point-cloud-level hollowing of the
    mug body (unsupported host → graceful fallback with warning)

Prints a JSON stats block and writes one OBJ per spec into
``tools/out/demo_complex/``. No LLM, no Open3D, no GUI required.

Usage:
  python tools/demo_complex.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ironengine_3d_creator.alignment.integrity import check_and_fix
from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.alignment.validator import normalize
from ironengine_3d_creator.generation.analytic_mesh import (
    build_spec_meshes_with_report, signed_volume,
)
from ironengine_3d_creator.generation.compositor import generate

OUT_DIR = Path(__file__).resolve().parent / "out" / "demo_complex"


def _T(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0):
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Ry @ Rx
    M[:3, 3] = [x, y, z]
    return M.tolist()


def _P(kind, params, label, transform=None):
    return Primitive(kind, transform or Primitive.identity_transform(),
                     params, label)


# ---------------------------------------------------------------- specs


def arched_chair() -> GenerationSpec:
    legs = [
        _P("cylinder", {"radius": 0.025, "height": 0.45, "caps": True},
           f"leg_{i}", _T(x, 0.225, z))
        for i, (x, z) in enumerate(
            [(-0.19, -0.19), (0.19, -0.19), (-0.19, 0.19), (0.19, 0.19)])
    ]
    cushion = _P(
        "superellipsoid",
        {"radii": [0.26, 0.04, 0.24], "exponents": [0.5, 0.5]},
        "seat_cushion", _T(0, 0.51, 0))
    # Back panel sits where the integrity pass attaches it: bottom of the
    # cushion is snapped to leg top (0.45), so the seat centre lands at
    # 0.49 and the back at y = 0.49 + 0.04 + 0.275 = 0.805, z = −0.225.
    back = _P(
        "panel", {"size": [0.5, 0.55], "thickness": 0.03, "bend": 0.0},
        "back", _T(0, 0.805, -0.225))
    back_hole = _P(
        "cylinder",
        {"radius": 0.06, "height": 0.2, "caps": True, "role": "subtract"},
        "back_hole", _T(0, 0.85, -0.225, rx=math.pi / 2))
    lumbar = _P(
        "panel", {"size": [0.4, 0.18], "thickness": 0.02, "bend": 0.6},
        "lumbar_support", _T(0, 0.62, -0.17))
    rail = _P(
        "arch", {"major_radius": 0.25, "minor_radius": 0.03},
        "top_rail", _T(0, 1.08, -0.225))
    return GenerationSpec(
        shape="chair", n_points=30_000, bbox_size=(0.6, 1.4, 0.6),
        primitives=legs + [cushion, back, back_hole, lumbar, rail], seed=7)


def handled_mug() -> GenerationSpec:
    body = _P("cylinder", {"radius": 0.05, "height": 0.12, "caps": True},
              "mug_body", _T(0, 0.06, 0))
    hollow = _P(
        "cylinder",
        {"radius": 0.04, "height": 0.2, "caps": True, "role": "subtract"},
        "hollow", _T(0, 0.10, 0))
    handle = _P(
        "tube",
        {"path": [[0.05, 0.095, 0.0], [0.095, 0.06, 0.0], [0.05, 0.025, 0.0]],
         "radius": 0.008, "caps": True},
        "handle")
    return GenerationSpec(
        shape="vase", n_points=30_000, bbox_size=(0.2, 0.15, 0.15),
        primitives=[body, hollow, handle], seed=11)


# ---------------------------------------------------------------- stats / OBJ


def _edge_stats(v: np.ndarray, f: np.ndarray) -> dict:
    edge_counts: dict[tuple, int] = {}
    for a, b, c in f:
        pa, pb, pc = (v[i].astype(np.float64) for i in (a, b, c))
        if np.linalg.norm(np.cross(pb - pa, pc - pa)) < 1e-12:
            continue
        for p, q in ((pa, pb), (pb, pc), (pc, pa)):
            key = tuple(sorted((tuple(np.round(p, 6)), tuple(np.round(q, 6)))))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    counts = np.fromiter(edge_counts.values(), dtype=np.int64)
    return {"watertight": bool(counts.min() == 2 and counts.max() == 2),
            "edges": int(counts.size)}


def _write_obj(path: Path, parts) -> None:
    lines = ["# demo_complex export"]
    offset = 1
    for part in parts:
        lines.append(f"o {part.label or part.kind}")
        for v in part.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for n in part.normals:
            lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
        for tri in part.faces:
            a, b, c = (int(i) + offset for i in tri)
            lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
        offset += len(part.vertices)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(name: str, spec: GenerationSpec) -> dict:
    clean, w_norm = normalize(spec)
    fixed, w_int = check_and_fix(clean)
    res = generate(fixed)
    parts, w_mesh = build_spec_meshes_with_report(fixed)
    stats = {
        "name": name,
        "shape": fixed.shape,
        "primitives": len(fixed.primitives),
        "mesh_parts": len(parts),
        "parts": [{
            "label": p.label, "kind": p.kind,
            "vertices": int(len(p.vertices)), "faces": int(len(p.faces)),
            "volume_m3": round(float(p.solid_volume_m3), 6),
            **_edge_stats(p.vertices, p.faces),
        } for p in parts],
        "point_cloud": {
            "requested": int(fixed.n_points),
            "returned": int(len(res.positions)),
            "filtered_by_cutters": int(fixed.n_points - len(res.positions)),
        },
        "warnings": w_norm + w_int + list(res.warnings) + w_mesh,
    }
    _write_obj(OUT_DIR / f"{name}.obj", parts)
    return stats


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = [run_one("arched_chair", arched_chair()),
              run_one("handled_mug", handled_mug())]
    (OUT_DIR / "stats.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nOBJ + stats written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
