# CR_ComplexGeometry — execution plan

Repo: IronEngine-3DCreator. Ownership: alignment/**, generation/{primitives,features,compositor,analytic_mesh,textures}.py, tests/ (new tests).

## Stage 1 — New primitive kinds (schema + samplers + meshes + validator)
- schema.py: PRIMITIVE_KINDS += superellipsoid, tube, sweep (alias of tube), arch, panel.
- primitives.py: samplers (area-weighted), primitive_area, inside_primitive (for CSG point filtering).
  - superellipsoid: radii + exponents [e1,e2]; rejection sampler w/ implicit-normal weight.
  - tube/sweep: path polyline + radius (+radius2 taper) + caps; default path from height.
  - arch: major/minor radius + arc + start_angle + caps (replaces full-torus hack in defaults.py:206-210).
  - panel: size + thickness + bend (curved sheet, bend=0 → box).
- analytic_mesh.py: exact meshes (analytic normals, UVs, watertight), volumes
  (superellipsoid V = (2/3)e1e2·rxryrz·B(e2/2,e2/2)·B(e1,e1/2)), local_aabb.
- validator.py: param defaults + range clamps for new kinds.
- defaults.py: archway template uses `arch` kind.

## Stage 2 — CSG-lite subtraction
- Spec: primitive with params {"role": "subtract"} (+ optional "target": label).
- analytic_mesh.py: carve straight-through tunnels for box/panel(bend=0)/prism hosts
  with cylinder(→ellipse)/box(→rect) cutters, axis-aligned in host frame:
  ring caps (subdivided outer boundary ↔ inner hole loop), matching side strips,
  inward-normal tunnel walls. One mesh-level hole per host; containment margin
  validation (no sever → no orphan geometry); volume bookkeeping.
- compositor.py: skip sampling cutters; filter host points inside cutter
  (inside_primitive in cutter local frame); GenerationResult.warnings.

## Stage 3 — Composition reliability
- integrity.py: _repair_rotated_verticals — upright elongated leg/vbar/stem parts
  whose long axis is >40° off vertical (yaw-preserving rotation rebuild, scale kept).
- integrity.py: interpenetration flagging (AABB overlap depth) as warnings;
  intentional gaps preserved (30%-of-longest-bbox SOUL tolerance, vbar pairs).

## Stage 4 — Proportion rules
- validator.py: _PROPORTION_RULES table (chair/table/lamp/creature/tree/fence…);
  soft clamps of part thickness in local frame + warnings.

## Stage 5 — Tests + demo
- tests/test_new_primitives.py: samplers (count/bounds), meshes (watertight, volume
  vs analytic), areas.
- tests/test_subtraction.py: cylinder-through-panel (handle hole), box-through-box
  (arch opening), cylinder-through-prism cap, point-level hollow mug, orphan/sever warnings.
- tests/test_integrity_rotation.py: 90°-rotated leg uprighted + snapped; gap preservation;
  proportion clamp warnings.
- tools/demo_complex.py: headless arched chair + handled mug, OBJ export + stats JSON.
- Run full pytest (63 baseline must stay green).
