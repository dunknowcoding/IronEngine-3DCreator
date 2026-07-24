# PROMPTING.md — How to Prompt for Complex, Exquisite 3D Models

A guide for the LLM (and for users writing prompts) on decomposing real-world
objects into **part graphs**: named parts, attachment transforms, symmetry
instancing, and sliced cross-sections. Follow this and 3DCreator produces
objects that are *structured*, not blobs.

---

## 1. The mental model: objects are assemblies, not silhouettes

Every exquisite object decomposes along five axes. Run through this checklist
before writing a single primitive:

### 1.1 Part analysis — "what is it made of?"
Name the **functional parts**, not the shapes. A chair is not "a box with
sticks"; it is `seat`, `leg_fl/fr/bl/br`, `back_posts`, `back_slats`,
`cross_braces`, `cushion`. A robot is `torso`, `head`, `shoulder_l/r`,
`upper_arm_l/r`, `elbow_l/r`, `forearm_l/r`, `hand_l/r`, `hip_l/r`,
`thigh_l/r`, `knee_l/r`, `shin_l/r`, `foot_l/r`.

Rules of thumb:
- **One part per material change** (wood seat vs. metal frame vs. fabric cushion).
- **One part per articulation point** (a joint that could move gets its own
  node — name it after the joint: `elbow_l`, not `arm_piece_3`).
- **One part per symmetry element** — define it once, instance it (see 1.3).

### 1.2 Silhouette analysis — "what is its cross-section story?"
If the object has a continuous skin (vase, tower, hull, fuselage, lamp
shade), do NOT approximate it with stacked ellipsoids. Describe it as
**slices**: pick 4–8 heights, give the radius/width at each, and loft them
(`generation.slicer`). The eye reads the silhouette curve; 5 well-chosen
slices beat 20 random primitives.

### 1.3 Symmetry analysis — "what repeats?"
Almost everything man-made is symmetric:
- **Bilateral** (chairs, robots, ships, computers): define the `_l` part,
  mirror it to `_r` across the symmetry plane.
- **Radial** (fence pickets, table legs, greeble arrays, wheel spokes):
  define one element, make a rotational array.
Instancing is not just convenience — it guarantees the symmetry is *exact*.

### 1.4 Hierarchy analysis — "what rides on what?"
Attachments form a tree: the hand rides on the forearm, the forearm on the
upper arm. Parent parts to their carrier so sub-assemblies move together and
bounding boxes nest logically.

### 1.5 Scale analysis — "how big is it, really?"
Work in metres, real-world scale. Anchors:

| Object | Key dimension | Value |
|---|---|---|
| Chair seat height | y of seat top | 0.45 m |
| Table top height | y of top surface | 0.72–0.78 m |
| Fence post | total height | ~1.1 m |
| Classical column | total height | 2.6–3.2 m |
| Desk monitor | screen width | ~0.54 m (24") |
| Computer tower | height | ~0.44 m |
| Toy spaceship | nose-to-tail length | 2.0 m |
| Desktop robot | total height | 0.6–0.75 m |
| Vase | total height | 0.25–0.40 m |

Get ONE anchor dimension right and derive the rest from proportion rules
(see §2). A chair whose seat is at 0.45 m looks right even if everything else
is approximate; a chair with a random scale looks like a toy.

---

## 2. Style vocabularies

Speak the style's grammar. Each style below lists its signature parts,
proportions, and materials (materials must be one of: `wood`, `stone`,
`fabric`, `metal`, `leather`, `ceramic`, `organic`).

### Rococo / Baroque (fence, gate, furniture)
- **Signature parts**: lathed posts (plinth → base ring → shaft → collar →
  urn finial), scrollwork curls (spiral tube paths), shallow arch infills,
  spear tips, vertical bars.
- **Proportions**: post height ~1.1 m; panel span 1.3–1.7 m; rails at ~25%
  and ~80% of height; arch rise 10–15% of span.
- **Materials**: `metal` (wrought iron), `stone` (plinths). Dark iron,
  bronze, verdigris, antique gold colors.

### Neoclassical (column, temple, facade)
- **Signature parts**: stepped plinth, base mouldings (2 tori), **sliced
  shaft** (5–7 slices with entasis), necking ring, echinus, abacus,
  volutes (ionic), entablature.
- **Proportions**: shaft radius = 1/16–1/18 of height; top radius ≈ 0.78 ×
  base radius; entasis bulge ≈ +3.5% radius at 1/3 height.
- **Materials**: `stone`. Marble white, limestone, sandstone.

### Modern Luxury (console, monolith, interior object)
- **Signature parts**: beveled monoliths (superellipsoids, exponents
  0.25–0.45), razor-thin metal inlay bars (1–2 cm), floating glass shelf,
  one sculptural accent (a ring or vessel).
- **Proportions**: monolith W:H:D ≈ 5:4:2; trim ≤ 2% of the main dimension.
- **Materials**: `stone` (marble), `metal` (brass/champagne), `ceramic` (glass).

### Futurist (chair, product)
- **Signature parts**: cantilever or pedestal base, shell seat/back
  (superellipsoids with exponents 0.35–0.5, or *bent panels*), wrap-around
  back (panel with `bend` 0.4–0.7 rad), cushion, thin tube armrests.
- **Proportions**: seat at 0.45 m; shell thickness ≤ 10% of width.

### Mechanical / Sci-fi (spaceship, robot)
- **Signature parts**: sliced hull (nose cone → hull slices → nozzle),
  canopy, wing plane, greeble arrays (small boxes/cylinders scattered on the
  hull), engine glow disc; or jointed limbs with articulation-ready naming.
- **Proportions**: spaceship length 2 m (toy model), hull radius ≈ 12% of
  length, wingspan ≈ 70% of length.
- **Materials**: `metal` hull, `ceramic` canopy/glow/greebles.

---

## 3. Spec mechanics (what the pipeline accepts)

### 3.1 GenerationSpec primitives
```json
{
  "kind": "cylinder",
  "transform": [[...4x4 row-major...]],
  "params": {"radius": 0.075, "height": 0.9, "caps": true, "material": "metal"},
  "label": "post_l"
}
```
Kinds: `box sphere cylinder capsule cone torus ellipsoid prism helix plane
superellipsoid tube sweep arch panel`. Every part **must** carry a snake_case
`label`. Transforms are T·Ry·Rx·Rz·S (radians). Bent sheets use `panel`
(`size` + `thickness` + `bend`); curled tubes use `tube` with a `path` of
3D points; rounded boxes use `superellipsoid` with low exponents.

### 3.2 Part graphs (complex assemblies)
For genuinely complex objects, emit a part graph instead of a flat list:
nodes are primitives or lofts, edges are attachment transforms, and symmetry
is instancing:

```json
{
  "name": "robot",
  "nodes": [
    {"name": "thigh_l", "kind": "capsule",
     "params": {"radius": 0.036, "height": 0.12},
     "material": "metal", "translate": [-0.075, 0.245, 0]},
    {"name": "thigh_r", "mirror_of": {"node": "thigh_l", "axis": "x"}}
  ]
}
```
The builder (`generation.complex_builder.PartGraph`) supports
`add_primitive`, `add_loft`, `attach(child, parent)`, `mirror(name, axis)`,
and `array_radial(name, count, axis)`. Instances share one mesh — a 6-petal
flower costs one petal.

### 3.3 Lofts (slicing builder)
For continuous skins, describe slices instead of primitives:

```json
{
  "profile": {"type": "circle", "segments": 48},
  "slices": [
    {"position": 0.00, "scale": [0.55, 0.55]},
    {"position": 0.08, "scale": [1.00, 1.00]},
    {"position": 0.18, "scale": [0.80, 0.80], "rotation": 0.2},
    {"position": 0.32, "scale": [0.45, 0.45], "offset": [0.01, 0.0]}
  ],
  "axis": "y"
}
```
Profiles: `circle`, `rounded_rect`, `superellipse`, or a custom polygon.
Per-slice `scale` (taper), `rotation` (twist), `offset` (drift/lean).
`generation.slicer.loft` turns this into a watertight mesh with smooth
normals and UVs.

---

## 4. Worked examples (full specs)

### Example 1 — Baroque vase (loft, 0.32 m)

Analysis: continuous skin → slices, not stacked ellipsoids. Silhouette:
foot → belly → shoulder → neck → flared rim. One material (`ceramic`).
Bilateral about every vertical plane → a circular profile.

```python
from ironengine_3d_creator.generation import slicer
from ironengine_3d_creator.generation.complex_builder import PartGraph

g = PartGraph("baroque_vase")
profile = slicer.profile_circle(0.05, 48)          # unit: 5 cm radius base
g.add_loft(
    "body", profile,
    slicer.radius_slices(
        positions=[0.00, 0.02, 0.08, 0.18, 0.26, 0.30, 0.32],
        radii=    [0.55, 0.78, 1.00, 0.82, 0.42, 0.40, 0.48],
    ),
    material="ceramic",
)
g.add_primitive("rim", "torus",
                {"major_radius": 0.025, "minor_radius": 0.006},
                material="ceramic", translate=(0, 0.32, 0))
result = g.build()     # 2 parts, watertight loft + rim, ~2.6k tris
```

### Example 2 — Desktop robot (part graph with instancing, 0.7 m)

Analysis: bilateral symmetry → define left limbs, mirror. Articulation → one
node per joint, named after it. Hierarchy → segments parented to torso.

```python
from ironengine_3d_creator.generation.complex_builder import PartGraph

g = PartGraph("robot")
g.add_primitive("torso", "superellipsoid",
                {"radii": [0.14, 0.12, 0.095], "exponents": [0.5, 0.5]},
                material="ceramic", translate=(0, 0.44, 0))
g.add_primitive("head", "superellipsoid",
                {"radii": [0.085, 0.07, 0.08], "exponents": [0.5, 0.5]},
                material="ceramic", parent="torso", translate=(0, 0.205, 0))

# Left leg chain: hip → thigh → knee → shin → foot.
g.add_primitive("hip_l", "sphere", {"radius": 0.048},
                material="metal", translate=(-0.085, 0.32, 0))
g.add_primitive("thigh_l", "capsule", {"radius": 0.036, "height": 0.12},
                material="metal", parent="hip_l", translate=(0.01, -0.075, 0))
g.add_primitive("knee_l", "sphere", {"radius": 0.038},
                material="metal", parent="thigh_l", translate=(0, -0.075, 0))
g.add_primitive("shin_l", "capsule", {"radius": 0.030, "height": 0.11},
                material="metal", parent="knee_l", translate=(0, -0.06, 0))
# ... left arm chain: shoulder_l → upper_arm_l → elbow_l → forearm_l → hand_l

# Bilateral instancing: the entire left side defined once.
for part in ("hip_l", "thigh_l", "knee_l", "shin_l"):
    g.mirror(part, axis="x")
result = g.build()
# Every mirrored part shares its mesh with the left-side definition:
# result.parts[i].vertices is result.parts[0].vertices → near-zero extra memory.
# result.aabbs()["thigh_l"] → per-named-part world bounding box for tracking.
```

### Example 3 — Toy spaceship (sliced hull + greebles, 2 m)

Analysis: hull is a slice story along Z (nose → mid → aft → nozzle); wings
bilateral; greebles are a bounded random array with a strict triangle budget.

```python
from ironengine_3d_creator.generation import slicer
from ironengine_3d_creator.generation.complex_builder import PartGraph

g = PartGraph("spaceship")
hull_profile = slicer.profile_superellipse(0.25, 0.22, 2.4, 40)
g.add_loft(
    "hull", hull_profile,
    slicer.radius_slices(
        positions=[-1.00, -0.82, -0.30, 0.30, 0.72, 0.95, 1.00],
        radii=    [0.30,  0.62,  0.84, 1.00, 0.72, 0.28, 0.05],
    ),
    axis="z", material="metal",
)
g.add_primitive("wing_l", "box", {"size": [0.62, 0.035, 0.40]},
                material="metal", translate=(-0.55, 0.0, -0.10), rz=0.06)
g.mirror("wing_l", axis="x")
g.add_primitive("canopy", "ellipsoid", {"radii": [0.10, 0.09, 0.20]},
                material="ceramic", parent="hull", translate=(0, 0.24, 0.30))
# Greeble array: deterministic positions, 4–8 small boxes on the spine.
import numpy as np
rng = np.random.default_rng(7)
for i in range(6):
    z = float(rng.uniform(-0.55, 0.5))
    g.add_primitive(f"greeble_{i}", "box",
                    {"size": [0.06, 0.03, 0.10]},
                    material="ceramic", translate=(0, 0.25, z))
result = g.build()
print(result.triangle_count())   # keep under ~20k tris for a toy model
```

---

## 5. Prompt-writing checklist (for users)

A good prompt names: **object + style + anchor dimension + material story +
one signature detail**.

- ❌ "a chair" → a generic chair.
- ✅ "a futurist shell chair, seat at 0.45 m, white ceramic shell with a
  chrome pedestal and a wrap-around back" → the builder knows the family,
  the scale anchor, the materials, and the signature bent-panel back.
- ✅ "a rococo iron fence panel, posts 1.1 m tall, scrollwork between two
  rails, bronze" → posts + scroll tubes + arch infill fall out directly.
- ✅ "a 2 m toy spaceship, sliced hull, twin wing pods, greebles on the
  spine" → loft + instancing + array.

Decompose first, name every part, anchor one real-world dimension, instance
everything that repeats. That is the whole skill.
