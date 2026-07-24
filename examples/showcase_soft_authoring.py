"""Showcase example 3 — soft authoring: cloth towel + frangible vase.

Uses ``generation.soft_author`` (the iemodel/3 non-rigid authoring API) to
create a pinned cotton towel with a ``soft_body`` block and two ceramic
vessels with ``fracture`` blocks, then writes GLBs and iemodel/3 manifests
to ``examples/out/soft_authoring/`` — the same metadata SceneEditor/Sim
consume for soft-body and frangible physics.

Run:  python examples/showcase_soft_authoring.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _showcase_common import OUT_ROOT, try_render

from ironengine_3d_creator.generation import soft_author


def main() -> int:
    out_dir = OUT_ROOT / "soft_authoring"
    out_dir.mkdir(parents=True, exist_ok=True)

    towel = soft_author.author_cloth(material="cotton", width=0.46, depth=0.60,
                                     resolution=(28, 20), pins="corners",
                                     n_points=4000, seed=5)
    vase = soft_author.author_frangible_vessel(material="ceramic", radius=0.070,
                                               height=0.150, wall_thickness=0.006,
                                               n_points=5000, seed=3)
    pot = soft_author.author_frangible_vessel(material="brick", radius=0.048,
                                              height=0.095, wall_thickness=0.007,
                                              n_points=3000, seed=8)

    for name, obj in (("towel", towel), ("vase", vase), ("pot", pot)):
        obj.write_glb(out_dir / f"{name}.glb")
        manifest = obj.build_manifest()
        (out_dir / f"{name}.iemodel.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        body = manifest.get("physics", {}).get("body_type")
        extra_keys = [k for k in ("soft_body", "fracture", "articulation")
                      if k in manifest]
        print(f"{name}: body_type={body}, extras={extra_keys}, "
              f"points={len(obj.positions)}")

    try_render([towel.parts[0], vase.parts[0], pot.parts[0]],
               out_dir / "soft_authoring.png", azimuth=35, elevation=18,
               albedo_overrides={"cloth_sheet": (0.80, 0.75, 0.66),
                                 "vessel_lathe": (0.72, 0.42, 0.30)})
    print(f"outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
