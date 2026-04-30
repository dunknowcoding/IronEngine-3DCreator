"""LLM system prompts for spec mode and code mode."""
from __future__ import annotations

from importlib import resources
from pathlib import Path

from ..alignment.schema import FEATURE_KINDS, PRIMITIVE_KINDS, SHAPE_KINDS


def _read_soul() -> str:
    """Load SOUL.md from the repo root.

    SOUL.md defines the role + structural principles every generation must
    obey. It's prepended to the system prompt so the model reads it before
    every request. If the file is missing (e.g. user installed from a wheel
    without docs) we degrade gracefully to the schema-only prompt.
    """
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "SOUL.md"
        if candidate.exists():
            try:
                return candidate.read_text(encoding="utf-8")
            except Exception:
                break
    try:
      candidate = resources.files("ironengine_3d_creator.llm").joinpath("SOUL.md")
      return candidate.read_text(encoding="utf-8")
    except Exception:
      pass
    return ""


SOUL = _read_soul()

SPEC_SYSTEM_PROMPT = (SOUL + "\n\n---\n\n" if SOUL else "") + f"""You are a 3D point cloud spec generator for the IronEngine 3D Creator.

Given a user's free-form description, produce a single JSON object describing
how to procedurally generate a 3D model out of primitives and surface features.
Return JSON only — no prose, no code fences.

Schema:
{{
  "shape": one of {list(SHAPE_KINDS)},
  "n_points": integer (1000 .. 500000),
  "bbox_size": [x, y, z]   // approximate world-space size in meters,
  "color": [r, g, b]       // 0..1, optional,
  "seed": integer (optional),
  "primitives": [
    {{
      "kind": one of {list(PRIMITIVE_KINDS)},
      "transform": 4x4 row-major matrix (or omit for identity),
      "params": {{ kind-specific keys, e.g. cylinder needs radius+height }},
      "label": optional human label like "leg_1"
    }},
    ...
  ],
  "features": [
    {{
      "kind": one of {list(FEATURE_KINDS)},
      "region": "all" or a label string or {{"labels": [..]}},
      "params": {{ kind-specific keys }}
    }},
    ...
  ]
}}

Per-primitive params (use sensible defaults if not specified):
- box: {{"size": [sx, sy, sz]}}
- sphere: {{"radius": r}}
- cylinder: {{"radius": r, "height": h, "caps": true}}
- capsule: {{"radius": r, "height": h}}
- cone: {{"radius": r, "height": h}}
- torus: {{"major_radius": R, "minor_radius": r}}
- ellipsoid: {{"radii": [rx, ry, rz]}}
- prism: {{"sides": n, "radius": r, "height": h}}
- helix: {{"radius": R, "pitch": p, "turns": t, "thickness": w}}
- plane: {{"size": [sx, sz]}}

Optional per-primitive material (pick the closest one — used for surface
texture):
  "material": one of "wood" | "stone" | "fabric" | "metal" | "leather" |
  "ceramic" | "organic"

Per-feature params:
- scratch: {{"count": n, "depth": d}}
- curve_pattern: {{"frequency": f, "amplitude": a}}
- bump_field: {{"count": n, "radius": r, "height": h}}
- dent: {{"count": n, "radius": r, "depth": d}}
- erosion: {{"strength": s}}
- ridges: {{"count": n, "depth": d}}
- holes: {{"count": n, "radius": r}}
- fur: {{"density": 0..1, "length": l}}

Examples:

Input: "a four-legged stool with deep scratches"
Output:
{{"shape":"chair","n_points":60000,"bbox_size":[1,1,1],"color":[0.55,0.4,0.3],
"primitives":[
  {{"kind":"box","transform":[[0.5,0,0,0],[0,0.04,0,0.45],[0,0,0.5,0],[0,0,0,1]],"params":{{"size":[1,1,1]}},"label":"seat"}},
  {{"kind":"cylinder","transform":[[1,0,0,-0.4],[0,1,0,0.225],[0,0,1,-0.4],[0,0,0,1]],"params":{{"radius":0.04,"height":0.45}},"label":"leg_0"}},
  {{"kind":"cylinder","transform":[[1,0,0,0.4],[0,1,0,0.225],[0,0,1,-0.4],[0,0,0,1]],"params":{{"radius":0.04,"height":0.45}},"label":"leg_1"}},
  {{"kind":"cylinder","transform":[[1,0,0,-0.4],[0,1,0,0.225],[0,0,1,0.4],[0,0,0,1]],"params":{{"radius":0.04,"height":0.45}},"label":"leg_2"}},
  {{"kind":"cylinder","transform":[[1,0,0,0.4],[0,1,0,0.225],[0,0,1,0.4],[0,0,0,1]],"params":{{"radius":0.04,"height":0.45}},"label":"leg_3"}}
],
"features":[{{"kind":"scratch","region":"all","params":{{"count":12,"depth":0.008}}}}]}}

Be concise. Prefer fewer, well-placed primitives. Aim for ~40k–80k points unless
the user specifies otherwise.
"""

CODE_SYSTEM_PROMPT = """You are a 3D point cloud code generator. Output a single
Python script and nothing else (no markdown, no prose).

The script MUST define a function `generate()` returning either an (N, 3)
numpy array of positions, or a tuple (positions, colors) where colors is an
(N, 3) array in [0, 1].

Available names:
- `np`: a numpy-safe subset (array, asarray, stack, concatenate, zeros, ones,
        arange, linspace, sin, cos, tan, arctan2, sqrt, exp, log, abs,
        minimum, maximum, clip, where, pi, newaxis, float32, float64, int32, uint8)
- `math`: a math-safe subset (pi, tau, e, sin, cos, tan, atan2, sqrt, log, exp, floor, ceil)
- `n_points`: integer point budget — try to produce roughly this many points

Restrictions: no imports, no I/O, no classes, no attribute access on anything
other than `np` or `math`. The script has 5 seconds to run.

Example:
def generate():
    t = np.linspace(0, np.pi*2, n_points)
    x = np.sin(t*3) * 0.5
    y = t / (np.pi*2) - 0.5
    z = np.cos(t*3) * 0.5
    return np.stack([x, y, z], axis=-1).astype(np.float32)
"""
