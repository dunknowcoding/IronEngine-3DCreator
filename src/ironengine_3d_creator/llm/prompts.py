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
- superellipsoid: {{"radii": [rx, ry, rz], "exponents": [e1, e2]}}
  // rounded box/ellipsoid hybrid: e = 1 gives an ellipsoid, e ~ 0.5–0.8
  // gives plump rounded-box forms. Use for rounded furniture, cushions,
  // pillows, poufs, and weathered stones.
- tube: {{"path": [[x,y,z], [x,y,z], ...], "radius": r, "radius2": r2 (optional taper), "caps": true}}
  // a pipe swept along a 3D polyline (>= 2 points). Use for curved handles,
  // handrails, grab bars, faucet spouts, and bent pipes. Without "path" it is
  // a straight vertical bar of "height" (a curvable cylinder).
- sweep: same params as tube — a (possibly tapered) pipe along a "path".
  // Use for rails, cables, and trim that follows an edge.
- arch: {{"major_radius": R, "minor_radius": r, "arc": angle_rad (default 3.1416 = a ∩ half-arch with both feet at y=0), "start_angle": a0, "caps": true}}
  // a torus segment standing upright in the local XY plane. Use for arcades,
  // archways, bridge arches, and curved chair backrests — never fake an arch
  // with a full torus.
- panel: {{"size": [width, height], "thickness": t, "bend": angle_rad}}
  // a thin plate; bend = 0 is flat, bend != 0 curls it into a cylindrical
  // shell segment (|bend| < ~2.98 rad). Use for curved shells, chair backs,
  // door leaves, tabletops, and windscreens.

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

More examples — complex, curved, and hollow objects:

Input: "a small stone footbridge with a single arch"
Output:
{{"shape":"abstract","n_points":90000,"bbox_size":[3.2,1.1,0.8],"color":[0.6,0.58,0.55],
"primitives":[
  {{"kind":"box","transform":[[1,0,0,-1.2],[0,1,0,0.4],[0,0,1,0],[0,0,0,1]],"params":{{"size":[0.4,0.8,0.8]}},"label":"pier_left","material":"stone"}},
  {{"kind":"box","transform":[[1,0,0,1.2],[0,1,0,0.4],[0,0,1,0],[0,0,0,1]],"params":{{"size":[0.4,0.8,0.8]}},"label":"pier_right","material":"stone"}},
  {{"kind":"torus","transform":[[0,0,1,0],[1,0,0,0],[0,1,0,0.35],[0,0,0,1]],"params":{{"major_radius":1.2,"minor_radius":0.18}},"label":"arch","material":"stone"}},
  {{"kind":"box","transform":[[1,0,0,0],[0,1,0,0.95],[0,0,1,0],[0,0,0,1]],"params":{{"size":[3.2,0.12,0.8]}},"label":"deck","material":"stone"}},
  {{"kind":"box","transform":[[1,0,0,0],[0,1,0,1.12],[0,0,1,0.36],[0,0,0,1]],"params":{{"size":[3.2,0.22,0.08]}},"label":"rail_front","material":"stone"}},
  {{"kind":"box","transform":[[1,0,0,0],[0,1,0,1.12],[0,0,1,-0.36],[0,0,0,1]],"params":{{"size":[3.2,0.22,0.08]}},"label":"rail_back","material":"stone"}}
],
"features":[{{"kind":"bump_field","region":"all","params":{{"count":30,"radius":0.03,"height":0.01}}}}]}}

Input: "a ceramic mug with a handle"
Output:
{{"shape":"vase","n_points":50000,"bbox_size":[0.13,0.11,0.09],"color":[0.85,0.82,0.75],
"primitives":[
  {{"kind":"cylinder","transform":[[1,0,0,0],[0,1,0,0.05],[0,0,1,0],[0,0,0,1]],"params":{{"radius":0.045,"height":0.1}},"label":"body","material":"ceramic"}},
  {{"kind":"cylinder","transform":[[0.9,0,0,0],[0,1,0,0.052],[0,0,0.9,0],[0,0,0,1]],"params":{{"radius":0.04,"height":0.096}},"label":"interior","material":"ceramic"}},
  {{"kind":"torus","transform":[[1,0,0,0.062],[0,0,1,0],[0,1,0,0.055],[0,0,0,1]],"params":{{"major_radius":0.03,"minor_radius":0.008}},"label":"handle","material":"ceramic"}}
]}}

Input: "a wooden stool with a curved seat"
Output:
{{"shape":"chair","n_points":60000,"bbox_size":[0.45,0.5,0.45],"color":[0.5,0.35,0.2],
"primitives":[
  {{"kind":"ellipsoid","transform":[[1,0,0,0],[0,1,0,0.46],[0,0,1,0],[0,0,0,1]],"params":{{"radii":[0.22,0.035,0.22]}},"label":"seat","material":"wood"}},
  {{"kind":"capsule","transform":[[1,0,0,-0.15],[0,1,0,0.22],[0,0,1,-0.15],[0,0,0,1]],"params":{{"radius":0.025,"height":0.42}},"label":"leg_0","material":"wood"}},
  {{"kind":"capsule","transform":[[1,0,0,0.15],[0,1,0,0.22],[0,0,1,-0.15],[0,0,0,1]],"params":{{"radius":0.025,"height":0.42}},"label":"leg_1","material":"wood"}},
  {{"kind":"capsule","transform":[[1,0,0,-0.15],[0,1,0,0.22],[0,0,1,0.15],[0,0,0,1]],"params":{{"radius":0.025,"height":0.42}},"label":"leg_2","material":"wood"}},
  {{"kind":"capsule","transform":[[1,0,0,0.15],[0,1,0,0.22],[0,0,1,0.15],[0,0,0,1]],"params":{{"radius":0.025,"height":0.42}},"label":"leg_3","material":"wood"}},
  {{"kind":"torus","transform":[[0,1,0,0],[1,0,0,0],[0,0,1,0.12],[0,0,0,1]],"params":{{"major_radius":0.19,"minor_radius":0.012}},"label":"stretcher_ring","material":"wood"}}
]}}

Input: "a garden gate with an arched top and a curved tube handle"
Output:
{{"shape":"gate","n_points":70000,"bbox_size":[1.4,1.6,0.15],"color":[0.3,0.32,0.34],
"primitives":[
  {{"kind":"box","transform":[[1,0,0,-0.65],[0,1,0,0.65],[0,0,1,0],[0,0,0,1]],"params":{{"size":[0.12,1.3,0.12]}},"label":"post_left","material":"metal"}},
  {{"kind":"box","transform":[[1,0,0,0.65],[0,1,0,0.65],[0,0,1,0],[0,0,0,1]],"params":{{"size":[0.12,1.3,0.12]}},"label":"post_right","material":"metal"}},
  {{"kind":"arch","transform":[[1,0,0,0],[0,1,0,1.3],[0,0,1,0],[0,0,0,1]],"params":{{"major_radius":0.65,"minor_radius":0.04}},"label":"arch_top","material":"metal"}},
  {{"kind":"panel","transform":[[1,0,0,0],[0,1,0,0.7],[0,0,1,0],[0,0,0,1]],"params":{{"size":[1.18,1.1],"thickness":0.03,"bend":0.0}},"label":"leaf","material":"metal"}},
  {{"kind":"tube","transform":[[1,0,0,0.45],[0,1,0,0.9],[0,0,1,0.08],[0,0,0,1]],"params":{{"path":[[0,0,0],[0.1,0.05,0.04],[0.2,0,0]],"radius":0.015}},"label":"handle","material":"metal"}}
]}}

For complex prompts, compose curved primitives (superellipsoid / tube / sweep /
arch / panel / ellipsoid / torus / capsule / helix) — a curved handle is a
`tube` with a bent path, an archway is an `arch`, a curved shell or backrest
is a bent `panel`, a cushion is a `superellipsoid` — express holes as empty
space between labelled parts, give every part its own label and material, and
keep real-world proportions. Otherwise be concise and prefer fewer,
well-placed primitives. Aim for ~40k–80k points unless the user specifies
otherwise.
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
