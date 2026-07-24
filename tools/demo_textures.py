"""Demo: tileable procedural texture maps (CR_Textures).

Renders every registered texture kind at 512 px (timed, budget < 200 ms/map),
saves the channel PNGs plus a contact-sheet gallery, then textures an
analytic-mesh spec and exports a GLB whose per-part baseColorTextures are the
sampled maps — plus an iemodel manifest carrying the ietexture/1 `textures`
block. See docs/TEXTURES.md.

Usage:
    python tools/demo_textures.py [--out DIR] [--size 512] [--seed 7]
                                  [--samples DIR] [--skip-glb]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive  # noqa: E402
from ironengine_3d_creator.generation import texture_apply as ta  # noqa: E402
from ironengine_3d_creator.generation import texture_maps as tm  # noqa: E402

# 4 curated sample maps for proof galleries.
SAMPLE_MAPS = [
    ("wood_oak", "albedo"),
    ("marble", "albedo"),
    ("brick", "albedo"),
    ("rococo_ornament", "bump"),
]


def render_gallery(out_dir: Path, size: int, seed: int) -> dict[str, float]:
    """Generate every kind, save channel PNGs + contact sheet; return timings."""
    from PIL import Image

    timings: dict[str, float] = {}
    thumbs = []
    for kind in tm.list_texture_kinds():
        tm.generate_maps(kind, size=size, seed=seed)  # warm-up
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            maps = tm.generate_maps(kind, size=size, seed=seed)
            best = min(best, time.perf_counter() - t0)
        timings[kind] = best * 1000.0
        tm.save_maps(maps, out_dir, kind=kind, size=size, seed=seed)
        thumbs.append((kind, maps["albedo"]))

    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    cell = 192
    sheet = Image.new("RGB", (cols * cell, rows * cell), (24, 24, 26))
    for i, (kind, alb) in enumerate(thumbs):
        tile = Image.fromarray(alb).resize((cell, cell), Image.BILINEAR)
        sheet.paste(tile, ((i % cols) * cell, (i // cols) * cell))
    sheet_path = out_dir / f"gallery_{size}px_s{seed}.png"
    sheet.save(sheet_path)
    return timings


def textured_glb_demo(out_dir: Path, size: int, seed: int) -> Path | None:
    """Spec -> analytic parts -> sampled maps -> stock GLB export + manifest."""
    from ironengine_3d_creator.core import exporter
    from ironengine_3d_creator.core.manifest import build_manifest, write_manifest
    from ironengine_3d_creator.generation.analytic_mesh import build_spec_meshes

    def moved(dx, dy, dz):
        t = np.eye(4, dtype=np.float32)
        t[0, 3], t[1, 3], t[2, 3] = dx, dy, dz
        return t.tolist()

    spec = GenerationSpec(
        shape="table",
        n_points=20_000,
        primitives=[
            Primitive(kind="box", transform=moved(0, 0.55, 0),
                      params={"size": [1.2, 0.06, 0.7], "material": "wood"}, label="tabletop"),
            Primitive(kind="cylinder", transform=moved(-0.5, 0.25, -0.25),
                      params={"radius": 0.04, "height": 0.5, "material": "metal"}, label="leg_a"),
            Primitive(kind="cylinder", transform=moved(0.5, 0.25, 0.25),
                      params={"radius": 0.04, "height": 0.5, "material": "metal"}, label="leg_b"),
            Primitive(kind="box", transform=moved(0, 0.02, 0),
                      params={"size": [0.9, 0.04, 0.5], "material": "stone"}, label="base"),
        ],
    )
    parts = build_spec_meshes(spec)
    assignments = {"tabletop": "wood_walnut", "leg_a": "brushed_metal",
                   "leg_b": "brushed_metal", "base": "marble"}
    uv_scale = {"tabletop": (2.0, 1.0), "leg_a": (3.0, 1.0), "leg_b": (3.0, 1.0)}
    colors, generated = ta.apply_maps_to_parts(
        parts, assignments, size=size, seed=seed, uv_scale=uv_scale
    )

    maps_dir = out_dir / "textures"
    map_files: dict[str, str] = {}
    for kind, maps in generated.items():
        for ch, p in tm.save_maps(maps, maps_dir, kind=kind, size=size, seed=seed).items():
            map_files[f"{kind}_{ch}"] = str(Path(p).relative_to(out_dir))

    positions = np.concatenate([p.vertices for p in parts], axis=0)
    colors_v = np.concatenate(colors, axis=0)
    glb_path = out_dir / "textured_table.glb"
    exporter.write_glb(glb_path, positions, colors_v, spec=spec, texture_size=size)

    manifest = build_manifest(spec, positions, colors_v, mesh_path=glb_path,
                              mesh_stats={"vertices": int(positions.shape[0]),
                                          "has_uvs": True, "analytic": True})
    manifest["textures"] = ta.textures_manifest_block(
        [
            {"part": label, "material": mat, "kind": kind,
             "channels": [c for c in generated[kind] if c != "alpha"],
             "uv": {"wrap": "repeat", "scale": list(uv_scale.get(label, (1, 1)))}}
            for (label, mat, kind) in (
                ("tabletop", "wood", "wood_walnut"),
                ("leg_a", "metal", "brushed_metal"),
                ("leg_b", "metal", "brushed_metal"),
                ("base", "stone", "marble"),
            )
        ],
        map_files=map_files, size=size, seed=seed,
    )
    errors = ta.validate_textures_block(manifest["textures"])
    if errors:
        raise RuntimeError(f"textures block invalid: {errors}")
    manifest_path = out_dir / "textured_table.iemodel.json"
    write_manifest(manifest_path, manifest)
    print(f"  GLB      : {glb_path}")
    print(f"  manifest : {manifest_path} (textures block valid, "
          f"{len(manifest['textures']['maps'])} maps, "
          f"{len(manifest['textures']['assignments'])} assignments)")
    return glb_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "out" / "textures"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--samples", default=None,
                    help="also save the 4 curated sample maps here")
    ap.add_argument("--skip-glb", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(tm.list_texture_kinds())} texture kinds at {args.size}px (seed {args.seed})")
    timings = render_gallery(out_dir, args.size, args.seed)
    worst = 0.0
    for kind, ms in sorted(timings.items(), key=lambda kv: -kv[1]):
        flag = "  <-- OVER 200 ms BUDGET" if ms > 200 else ""
        print(f"  {kind:18s} {ms:7.1f} ms{flag}")
        worst = max(worst, ms)
    print(f"  worst: {worst:.1f} ms (budget 200 ms)")
    print(f"  gallery: {out_dir / f'gallery_{args.size}px_s{args.seed}.png'}")

    if args.samples:
        sdir = Path(args.samples)
        sdir.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        for kind, ch in SAMPLE_MAPS:
            maps = tm.generate_maps(kind, size=args.size, seed=args.seed)
            p = sdir / tm.map_filename(kind, ch, args.size, args.seed)
            Image.fromarray(maps[ch]).save(p)
            print(f"  sample : {p}")

    if not args.skip_glb:
        print("Textured GLB demo (wood/metal/marble table):")
        textured_glb_demo(out_dir, args.size, args.seed)


if __name__ == "__main__":
    main()
