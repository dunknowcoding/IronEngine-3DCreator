# CR_ComplexBuilder — execution plan

Ownership: generation/slicer.py (new), generation/complex_builder.py (new),
generation/style_families.py (extend only), docs/PROMPTING.md (new),
tests/test_slicer.py, tests/test_complex_builder.py, tests/test_exquisite_families.py.

## Stage 1 — generation/slicer.py (lofting / "3D-printing style")
- Profile library: circle, rounded_rect, superellipse, custom polygon (CCW validated).
- `Slice` dataclass: position along axis + per-slice scale/rotation/offset.
- `loft(profile, slices, axis)` → watertight analytic mesh: smooth geometric
  normals (oriented vs. ring centroid), planar UVs (u around, v along axis),
  centroid-fan caps, winding fixed via signed volume.
- `loft_volume` (trapezoid of scaled profile area) for test cross-checks.

## Stage 2 — generation/complex_builder.py (part graphs)
- PartGraph: nodes (primitive / loft / assembly via parent chains), edges =
  attachment transforms composed down the tree.
- Symmetry instancing: mirror / radial arrays — one mesh, many instance
  transforms; instances SHARE vertex/normal/uv/face arrays (zero-copy).
- Per-named-part metadata: conservative world AABBs (corner-transform of the
  local AABB), per-part triangle counts, materials, volumes.
- `bake()` → AnalyticPart-compatible world-space parts for the exporter
  (mirror instances get winding flipped).

## Stage 3 — style_families.py extension (7 exquisite families)
rococo_fence (post ~1.1 m), neoclassical_column (~3 m, sliced shaft),
modern_luxury (beveled superellipsoid monoliths), futurist_chair (seat 0.45 m),
desktop_computer, spaceship (2 m sliced hull + greebles), robot (~0.6 m,
articulation-ready joint naming). All registered into FAMILY_BUILDERS
(existing entries untouched); palettes extended. Grammars: core <= 5 parts,
budget-gated richness (respect ctx.room()), real-world dimensions pre-fit.

## Stage 4 — docs/PROMPTING.md
Prompting-skills guide: part-graph decomposition method, style vocabularies,
3 worked examples with full specs (loft vase, robot graph, spaceship graph).

## Stage 5 — tests + full suite
- test_slicer.py: profiles valid/CCW, loft watertight (2-manifold edges),
  volume vs trapezoid, normals outward, UV bounds, twist/taper/offset.
- test_complex_builder.py: attachments, mirror/array instancing shares mesh
  memory, AABBs per named part, bake winding for mirrors.
- test_exquisite_families.py: each family validator-clean, >= 3 parts,
  analytic mesh build within triangle budgets, real-world dims pre-fit.
- Full pytest run; report failures marked mine vs theirs. COMMIT NOTHING.
