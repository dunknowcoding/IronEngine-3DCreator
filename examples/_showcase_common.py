"""Shared helpers for the showcase examples: spec authoring + optional render.

Every example is offline and seeded — no LLM, no GUI, no network. Outputs go
to ``examples/out/<example_name>/``. If IronEngine-BonaFide is importable,
each example also renders a polished PNG preview next to the geometry.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ironengine_3d_creator.alignment.integrity import check_and_fix
from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.alignment.validator import normalize
from ironengine_3d_creator.generation.analytic_mesh import (
    build_spec_meshes_with_report,
)
from ironengine_3d_creator.generation.compositor import generate
from ironengine_3d_creator.generation.materials import MATERIAL_PRESETS, default_preset

OUT_ROOT = Path(__file__).resolve().parent / "out"


def T(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0):
    """4x4 TRS transform (rotation only, unit scale)."""
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Ry @ Rx
    M[:3, 3] = [x, y, z]
    return M.tolist()


def P(kind, params, label, transform=None):
    return Primitive(kind, transform or Primitive.identity_transform(),
                     params, label)


def run_pipeline(spec: GenerationSpec):
    """The exact offline pipeline the app uses without an LLM:
    validate/normalize → structural integrity repair → point cloud + meshes."""
    clean, w_norm = normalize(spec)
    fixed, w_int = check_and_fix(clean)
    cloud = generate(fixed)
    parts, w_mesh = build_spec_meshes_with_report(fixed)
    return fixed, cloud, parts, w_norm + w_int + list(cloud.warnings) + w_mesh


def try_render(meshes_parts, out_png: Path, *, azimuth=35.0, elevation=20.0,
               fov=40.0, albedo_overrides=None):
    """Render with IronEngine-BonaFide when it is installed; skip otherwise."""
    try:
        from ironengine_bonafide.api import (
            Background, DirectionalLight, Engine, IBL, Mesh, PBRMaterial,
            PerspectiveCamera, RenderConfig, Scene, render,
        )
    except ImportError:
        print("  (ironengine_bonafide not installed — skipping PNG render)")
        return None

    albedo_overrides = albedo_overrides or {}
    scene = Scene(name=out_png.stem)
    lo_all, hi_all = [], []
    for part in meshes_parts:
        preset = MATERIAL_PRESETS.get(part.material, default_preset())
        mat = PBRMaterial(name=part.material,
                          albedo=albedo_overrides.get(part.label, (0.7, 0.68, 0.65)),
                          roughness=float(preset["roughness"]),
                          metallic=float(preset["metallic"]))
        scene.add(Mesh.from_arrays(part.vertices, part.faces,
                                   normals=part.normals, material=mat,
                                   name=part.label or part.kind))
        lo_all.append(part.aabb_min)
        hi_all.append(part.aabb_max)
    # simple ground disc
    r = 2.0
    gv = np.array([[-r, -0.002, -r], [r, -0.002, -r], [r, -0.002, r], [-r, -0.002, r]],
                  dtype=np.float32)
    gi = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    scene.add(Mesh.from_arrays(gv, gi, material=PBRMaterial(
        name="ground", albedo=(0.25, 0.24, 0.22), roughness=0.92), name="ground"))
    sun = np.array([0.5, 0.72, 0.48])
    scene.add(DirectionalLight(direction=tuple(-sun / np.linalg.norm(sun)),
                               intensity=3.2, color=(1.0, 0.94, 0.85)))
    scene.add(Background(mode="gradient"))

    lo = np.min(lo_all, axis=0)
    hi = np.max(hi_all, axis=0)
    center = (lo + hi) / 2
    dist = float(np.linalg.norm(hi - lo) / 2) / math.tan(math.radians(fov) / 2) * 1.35
    az, el = math.radians(azimuth), math.radians(elevation)
    pos = center + dist * np.array([math.cos(el) * math.cos(az), math.sin(el),
                                    math.cos(el) * math.sin(az)])
    cam = PerspectiveCamera(position=tuple(pos), look_at=tuple(center), fov_deg=fov)
    cfg = RenderConfig(width=1280, height=720, output_color_space="sRGB",
                       shadows="csm", shadow_map_resolution=1024,
                       shadow_bias_constant=0.8, shadow_bias_slope=2.0,
                       bloom=False, exposure=0.95)
    out = render(Engine.auto(), scene, cam, cfg)
    arr = out.rgb.detach().clamp(0.0, 1.0).cpu().numpy()
    from PIL import Image
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(arr * 255.0).astype(np.uint8)).save(out_png,
                                                                 optimize=True)
    print(f"  rendered {out_png}")
    return out_png
