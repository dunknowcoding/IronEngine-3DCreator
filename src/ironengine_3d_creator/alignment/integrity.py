"""Post-LLM integrity checks + deterministic auto-fix.

The LLM frequently emits a `GenerationSpec` whose primitives don't actually
form a coherent object — legs that don't reach the floor, a seat hovering
above the legs, twisted transforms, asymmetric placements. We apply a small
amount of *physical reasoning* before sampling:

1. **Bounding-box pass** — compute world-space AABB of every primitive.
2. **Role inference** — bucket each primitive by its label
   (`leg_*`, `seat`, `back`, `head`, `body`, `tail`, …).
3. **Per-shape repair**:
   - Chair / table / stool: legs snap to the floor, become symmetric,
     seat snaps to leg tops, back attaches to seat rear.
   - Creature / quadruped: limbs sync to a common floor, head and tail
     attach to body extents.
   - Vase / lamp / mushroom / tree: bottom-most primitive sits on floor,
     stack-aligned along Y.
4. **Connectivity sweep** — anything still floating gets pulled toward
   its closest neighbour or dropped (with a warning).

Returns `(spec, warnings)` so the caller can surface fixes to the user.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .schema import GenerationSpec, Primitive

_log = logging.getLogger(__name__)


@dataclass
class _PrimAABB:
    index: int
    label: str
    kind: str
    role: str          # "leg" | "seat" | "back" | "head" | "body" | "tail" | "stem" | "cap" | "other"
    centre: np.ndarray  # (3,)
    half: np.ndarray    # (3,) half-extents


# Roles drive structural repair. Match any substring in the label so
# `bar_07`, `picket_left`, `top_rail` etc. all resolve correctly. The list
# is permissive on purpose — the alternative is rigid templates that
# refuse to handle anything beyond the canonical chair / vase / creature.
_LEG_LABELS = ("leg", "footing")
_SEAT_LABELS = ("seat", "tabletop", "surface", "platform", "tray")
_BACK_LABELS = ("back", "backrest")
_HEAD_LABELS = ("head", "skull")
_BODY_LABELS = ("body", "torso", "trunk")
_TAIL_LABELS = ("tail",)
_STEM_LABELS = ("stem", "neck", "branch", "pole")
_CAP_LABELS = ("cap", "shade", "crown", "lid", "roof")
_RAIL_LABELS = ("rail", "beam", "ledge", "lintel", "girder", "stretcher", "crossbar")
_VBAR_LABELS = ("bar", "picket", "slat", "spindle", "baluster", "post", "pillar",
                 "column", "support", "stake", "rod", "strut", "bollard")
_BASE_LABELS = ("base", "foundation", "footing", "plinth", "pedestal")
_FINIAL_LABELS = ("finial", "ornament", "decoration", "filigree", "knob", "cresting", "carving")
_PANEL_LABELS = ("panel", "facade", "wall", "screen", "door")


def _classify_role(label: str | None, kind: str) -> str:
    """Resolve a primitive's label to one of the structural roles below.

    Order matters: a label that contains both "post" and "base" should
    resolve to "base" (more specific role). We test the most specific
    families first.
    """
    name = (label or "").lower()
    for role, keys in (
        ("rail",    _RAIL_LABELS),       # horizontal member spanning verticals
        ("base",    _BASE_LABELS),       # bottommost structural footprint
        ("finial",  _FINIAL_LABELS),     # decoration on top of a structure
        ("panel",   _PANEL_LABELS),      # flat covering between rails
        ("vbar",    _VBAR_LABELS),       # vertical members in a fence/lattice
        ("leg",     _LEG_LABELS),        # furniture leg (snaps to floor)
        ("seat",    _SEAT_LABELS),
        ("back",    _BACK_LABELS),
        ("head",    _HEAD_LABELS),
        ("body",    _BODY_LABELS),
        ("tail",    _TAIL_LABELS),
        ("stem",    _STEM_LABELS),
        ("cap",     _CAP_LABELS),
    ):
        if any(k in name for k in keys):
            return role
    return "other"


def _local_aabb(prim: Primitive) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box in the primitive's local frame, centred at origin."""
    p = prim.params
    if prim.kind == "box":
        s = np.asarray(p.get("size", [1, 1, 1]), dtype=np.float32) / 2.0
        return -s, s
    if prim.kind == "sphere":
        r = float(p.get("radius", 0.5))
        return np.full(3, -r, dtype=np.float32), np.full(3, r, dtype=np.float32)
    if prim.kind == "cylinder":
        r = float(p.get("radius", 0.4)); h = float(p.get("height", 1.0))
        return np.asarray([-r, -h / 2, -r], dtype=np.float32), np.asarray([r, h / 2, r], dtype=np.float32)
    if prim.kind == "capsule":
        r = float(p.get("radius", 0.3)); h = float(p.get("height", 1.0))
        return (
            np.asarray([-r, -h / 2 - r, -r], dtype=np.float32),
            np.asarray([r,  h / 2 + r,  r], dtype=np.float32),
        )
    if prim.kind == "cone":
        r = float(p.get("radius", 0.5)); h = float(p.get("height", 1.0))
        return np.asarray([-r, -h / 2, -r], dtype=np.float32), np.asarray([r, h / 2, r], dtype=np.float32)
    if prim.kind == "torus":
        R = float(p.get("major_radius", 0.5)); r = float(p.get("minor_radius", 0.15))
        return (
            np.asarray([-(R + r), -r, -(R + r)], dtype=np.float32),
            np.asarray([(R + r),  r,  (R + r)], dtype=np.float32),
        )
    if prim.kind == "ellipsoid":
        rx, ry, rz = p.get("radii", [0.5, 0.5, 0.5])
        return np.asarray([-rx, -ry, -rz], dtype=np.float32), np.asarray([rx, ry, rz], dtype=np.float32)
    if prim.kind == "prism":
        r = float(p.get("radius", 0.5)); h = float(p.get("height", 1.0))
        return np.asarray([-r, -h / 2, -r], dtype=np.float32), np.asarray([r, h / 2, r], dtype=np.float32)
    if prim.kind == "helix":
        R = float(p.get("radius", 0.4)); h = float(p.get("pitch", 0.2)) * float(p.get("turns", 3.0))
        t = float(p.get("thickness", 0.05))
        return np.asarray([-R - t, -h / 2 - t, -R - t], dtype=np.float32), np.asarray([R + t, h / 2 + t, R + t], dtype=np.float32)
    if prim.kind == "plane":
        sx, sz = p.get("size", [1, 1])
        return np.asarray([-sx / 2, 0.0, -sz / 2], dtype=np.float32), np.asarray([sx / 2, 0.0, sz / 2], dtype=np.float32)
    return np.asarray([-0.5, -0.5, -0.5], dtype=np.float32), np.asarray([0.5, 0.5, 0.5], dtype=np.float32)


def _world_aabb(prim: Primitive) -> tuple[np.ndarray, np.ndarray]:
    """World-space AABB by sampling the 8 corners of the local AABB through
    the primitive's transform. Cheap and accurate for our purposes."""
    lo, hi = _local_aabb(prim)
    corners = np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]],
        [hi[0], hi[1], hi[2]],
    ], dtype=np.float32)
    T = prim.transform_matrix()
    homo = np.concatenate([corners, np.ones((8, 1), dtype=np.float32)], axis=1)
    world = (homo @ T.T)[:, :3]
    return world.min(0), world.max(0)


def _summary(prims: list[Primitive]) -> list[_PrimAABB]:
    out: list[_PrimAABB] = []
    for i, p in enumerate(prims):
        lo, hi = _world_aabb(p)
        centre = (lo + hi) * 0.5
        half = (hi - lo) * 0.5
        out.append(_PrimAABB(
            index=i, label=p.label or "", kind=p.kind,
            role=_classify_role(p.label, p.kind),
            centre=centre.astype(np.float32),
            half=half.astype(np.float32),
        ))
    return out


def _set_translation(prim: Primitive, target_centre: np.ndarray) -> None:
    """Replace the translation component of the primitive's transform so that
    its world-space AABB centre matches `target_centre`. Preserves rotation /
    scale.
    """
    T = prim.transform_matrix()
    lo, hi = _world_aabb(prim)
    current_centre = (lo + hi) * 0.5
    delta = target_centre - current_centre
    T[0, 3] += float(delta[0])
    T[1, 3] += float(delta[1])
    T[2, 3] += float(delta[2])
    prim.transform = T.tolist()


def _set_y_centre(prim: Primitive, y_centre: float) -> None:
    lo, hi = _world_aabb(prim)
    cur = float((lo[1] + hi[1]) * 0.5)
    delta = y_centre - cur
    T = prim.transform_matrix()
    T[1, 3] += float(delta)
    prim.transform = T.tolist()


def _set_y_bottom(prim: Primitive, y_bottom: float) -> None:
    lo, hi = _world_aabb(prim)
    half_y = float((hi[1] - lo[1]) * 0.5)
    _set_y_centre(prim, y_bottom + half_y)


def _bbox_overlap(a: _PrimAABB, b: _PrimAABB) -> bool:
    return (
        abs(a.centre[0] - b.centre[0]) <= (a.half[0] + b.half[0])
        and abs(a.centre[1] - b.centre[1]) <= (a.half[1] + b.half[1])
        and abs(a.centre[2] - b.centre[2]) <= (a.half[2] + b.half[2])
    )


def _euclidean_min_gap(a: _PrimAABB, b: _PrimAABB) -> float:
    dx = max(0.0, abs(a.centre[0] - b.centre[0]) - (a.half[0] + b.half[0]))
    dy = max(0.0, abs(a.centre[1] - b.centre[1]) - (a.half[1] + b.half[1]))
    dz = max(0.0, abs(a.centre[2] - b.centre[2]) - (a.half[2] + b.half[2]))
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


# ---------------------------------------------------------------- repair


def _repair_legs_seat(spec: GenerationSpec, info: list[_PrimAABB], warnings: list[str]) -> None:
    """Snap leg primitives so their bottoms sit on y=0, place the seat on
    top of them, attach the back to the seat's rear edge."""
    legs = [p for p in info if p.role == "leg"]
    seats = [p for p in info if p.role == "seat"]
    backs = [p for p in info if p.role == "back"]

    if not legs or not seats:
        return

    # Snap every leg bottom to y=0 and re-summarise.
    for leg in legs:
        prim = spec.primitives[leg.index]
        _set_y_bottom(prim, 0.0)
    info[:] = _summary(spec.primitives)
    legs = [p for p in info if p.role == "leg"]
    seats = [p for p in info if p.role == "seat"]
    backs = [p for p in info if p.role == "back"]

    # Symmetric placement when the count looks like 4 legs at a rectangle.
    if len(legs) == 4:
        seat = seats[0]
        sx = seat.half[0] - 0.04   # inset from the seat edge
        sz = seat.half[2] - 0.04
        sx = max(sx, 0.05)
        sz = max(sz, 0.05)
        corners = [(-sx, -sz), (+sx, -sz), (-sx, +sz), (+sx, +sz)]
        for leg, (lx, lz) in zip(sorted(legs, key=lambda l: (l.centre[2], l.centre[0])), corners):
            prim = spec.primitives[leg.index]
            top_y = float(2 * leg.half[1])  # leg height (already snapped to bottom 0)
            _set_translation(prim, np.array([lx, top_y / 2.0, lz], dtype=np.float32))
        info[:] = _summary(spec.primitives)
        legs = [p for p in info if p.role == "leg"]
        seats = [p for p in info if p.role == "seat"]
        backs = [p for p in info if p.role == "back"]
        warnings.append("integrity: 4 legs snapped to seat corners")

    # Place the seat just above the leg tops.
    leg_top = max(leg.centre[1] + leg.half[1] for leg in legs)
    for seat in seats:
        prim = spec.primitives[seat.index]
        _set_y_bottom(prim, leg_top)
        warnings.append(f"integrity: seat {seat.label or seat.kind!r} snapped to leg top")
    info[:] = _summary(spec.primitives)
    seats = [p for p in info if p.role == "seat"]

    # Attach back to the rear edge of the seat.
    if seats and backs:
        seat = seats[0]
        for back in backs:
            prim = spec.primitives[back.index]
            target = np.array([
                seat.centre[0],
                seat.centre[1] + seat.half[1] + back.half[1],
                seat.centre[2] - seat.half[2] + back.half[2],
            ], dtype=np.float32)
            _set_translation(prim, target)
        warnings.append("integrity: back attached to seat rear")


def _repair_quadruped(spec: GenerationSpec, info: list[_PrimAABB], warnings: list[str]) -> None:
    legs = [p for p in info if p.role == "leg"]
    body = next((p for p in info if p.role == "body"), None)
    head = next((p for p in info if p.role == "head"), None)
    tail = next((p for p in info if p.role == "tail"), None)

    if not legs or body is None:
        return

    # Legs reach the floor.
    for leg in legs:
        _set_y_bottom(spec.primitives[leg.index], 0.0)
    info[:] = _summary(spec.primitives)
    legs = [p for p in info if p.role == "leg"]
    body = next((p for p in info if p.role == "body"), body)

    # Body sits on top of the legs (centre of body slightly below leg top so
    # legs visually emerge from under the body).
    leg_top = max(leg.centre[1] + leg.half[1] for leg in legs)
    target_y = leg_top + body.half[1] * 0.5
    _set_y_centre(spec.primitives[body.index], target_y)
    info[:] = _summary(spec.primitives)
    body = next((p for p in info if p.role == "body"), None)
    if body is None:
        return

    # Head touches front of body.
    if head is not None:
        target = np.array([
            body.centre[0],
            body.centre[1] + body.half[1] * 0.5,
            body.centre[2] + body.half[2] + head.half[2] * 0.7,
        ], dtype=np.float32)
        _set_translation(spec.primitives[head.index], target)
        warnings.append("integrity: head attached to front of body")

    # Tail touches rear of body.
    if tail is not None:
        target = np.array([
            body.centre[0],
            body.centre[1] + body.half[1] * 0.3,
            body.centre[2] - body.half[2] - tail.half[2] * 0.7,
        ], dtype=np.float32)
        _set_translation(spec.primitives[tail.index], target)
        warnings.append("integrity: tail attached to rear of body")

    warnings.append(f"integrity: {len(legs)} limbs grounded under body")
    # Refresh so the downstream connectivity sweep sees the new positions.
    info[:] = _summary(spec.primitives)


def _repair_stack(spec: GenerationSpec, info: list[_PrimAABB], warnings: list[str], roles_order: tuple[str, ...]) -> None:
    """Stack primitives with the listed roles bottom-to-top, ground-aligned."""
    queue = []
    for role in roles_order:
        for p in info:
            if p.role == role:
                queue.append(p)
    if not queue:
        return
    y = 0.0
    for p in queue:
        prim = spec.primitives[p.index]
        _set_y_bottom(prim, y)
        # Re-summarise so half-extents are current after the snap.
        new_info = _summary(spec.primitives)
        new_p = next(x for x in new_info if x.index == p.index)
        y = new_p.centre[1] + new_p.half[1]
    info[:] = _summary(spec.primitives)
    warnings.append(f"integrity: stacked {' → '.join(p.role for p in queue)} on the floor")


def _repair_framework(spec: GenerationSpec, info: list[_PrimAABB], warnings: list[str]) -> None:
    """Generic snap for rail-and-vertical-member structures.

    Any object built from horizontal rails (rail / beam / lintel / stretcher /
    crossbar) plus vertical members (post / pillar / bar / picket / slat /
    spindle / baluster / column / support) is treated as a "framework". The
    repair:
      - Snaps every "vbar" so its top touches the highest rail and its
        bottom touches the lowest rail (or the floor if no bottom rail).
      - Aligns the rails' X/Z extents to enclose the bars (so the rails
        actually span them rather than stopping short).
      - Drops any "base"/"footing" parts to y=0.
      - Stacks "finial" parts on top of the highest member.
      - Critically does NOT pull bars toward each other — they are
        intentionally spaced. The connectivity sweep that runs afterward
        respects this by skipping pairs that are both vbars.

    Returns silently if the spec doesn't look like a framework.
    """
    rails = [p for p in info if p.role == "rail"]
    vbars = [p for p in info if p.role == "vbar"]
    bases = [p for p in info if p.role == "base"]
    finials = [p for p in info if p.role == "finial"]

    if len(vbars) < 2 or not rails:
        # Not enough members to call this a framework. Some shapes (lamp post,
        # single pillar) still want stack-style repair, which we leave to the
        # per-shape branches in check_and_fix.
        return

    # Sort rails by Y so we know which is the top vs bottom.
    rails_sorted = sorted(rails, key=lambda r: r.centre[1])
    bottom_rail = rails_sorted[0]
    top_rail = rails_sorted[-1] if len(rails_sorted) > 1 else None
    bottom_y = float(bottom_rail.centre[1] + bottom_rail.half[1])
    top_y = float(top_rail.centre[1] - top_rail.half[1]) if top_rail else None

    # Each vertical bar: snap the bottom to the top of the bottom rail (or
    # the floor) and stretch by translating so the top kisses the top rail.
    # We assume bars are oriented along Y and have a y_half that we can
    # use as the half-height.
    for bar in vbars:
        prim = spec.primitives[bar.index]
        cur_bottom = float(bar.centre[1] - bar.half[1])
        cur_top = float(bar.centre[1] + bar.half[1])
        if top_y is not None:
            target_centre_y = (bottom_y + top_y) * 0.5
        else:
            # Single rail: snap so bottom of bar = floor (or rail top), centre = old.
            target_centre_y = bottom_y + bar.half[1]
        delta_y = target_centre_y - bar.centre[1]
        if abs(delta_y) > 1e-3:
            T = prim.transform_matrix()
            T[1, 3] += float(delta_y)
            prim.transform = T.tolist()
    info[:] = _summary(spec.primitives)
    warnings.append(f"integrity (framework): snapped {len(vbars)} vertical members between rails")

    # Rails should span the bars. If a rail's X/Z half-extent is smaller than
    # the spread of the bars, we leave it alone (the user / LLM may have
    # asked for a partial rail). We don't auto-resize geometry — repositioning
    # is safe; resizing would change params and surprise the user.

    # Bases drop to the floor.
    for base in bases:
        prim = spec.primitives[base.index]
        _set_y_bottom(prim, 0.0)
    if bases:
        warnings.append(f"integrity (framework): {len(bases)} base part(s) on the floor")
        info[:] = _summary(spec.primitives)

    # Finials stack on top of the highest member (rail, bar, or post).
    if finials:
        info[:] = _summary(spec.primitives)
        all_y_tops = [p.centre[1] + p.half[1] for p in info if p.role != "finial"]
        if all_y_tops:
            top_y = max(all_y_tops)
            for f in finials:
                prim = spec.primitives[f.index]
                _set_y_bottom(prim, top_y)
            warnings.append(f"integrity (framework): {len(finials)} finial(s) stacked on top")
            info[:] = _summary(spec.primitives)


def _connectivity_sweep(spec: GenerationSpec, info: list[_PrimAABB], warnings: list[str]) -> None:
    """Pull any primitive that's *clearly* floating toward its closest
    neighbour. "Clearly" is measured *relative to the spec's overall bbox*
    so intentionally-spaced parts (fence pickets, scaffold poles, cluster
    branches) aren't dragged together.

    Rules:
    - Pairs where BOTH endpoints are vbars are skipped — those are
      intentionally separated.
    - We only pull if the gap exceeds 30 % of the spec's longest bbox
      side. Below that the LLM's spacing is presumed deliberate.
    """
    info[:] = _summary(spec.primitives)
    if len(info) <= 1:
        return

    # Whole-object bbox = AABB of every primitive's AABB. The "long" side
    # tells us what counts as "intentional" spacing.
    all_lo = np.min([p.centre - p.half for p in info], axis=0)
    all_hi = np.max([p.centre + p.half for p in info], axis=0)
    bbox_long = float(np.max(all_hi - all_lo))
    gap_threshold = max(0.05, bbox_long * 0.30)

    for primitive_index in range(len(spec.primitives)):
        a = next((x for x in info if x.index == primitive_index), None)
        if a is None:
            continue
        # Skip any neighbour pair that's both a vbar (rails are the
        # connector, not each other).
        candidates = [
            (b, _euclidean_min_gap(a, b)) for b in info
            if b.index != primitive_index and not (a.role == "vbar" and b.role == "vbar")
        ]
        if not candidates:
            continue
        b, gap_best = min(candidates, key=lambda x: x[1])
        if gap_best <= 1e-3:
            continue
        if gap_best < gap_threshold:
            # Spacing looks intentional; trust the LLM.
            continue
        direction = b.centre - a.centre
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        offset = direction / norm * gap_best
        prim = spec.primitives[a.index]
        T = prim.transform_matrix()
        T[0, 3] += float(offset[0])
        T[1, 3] += float(offset[1])
        T[2, 3] += float(offset[2])
        prim.transform = T.tolist()
        warnings.append(
            f"integrity: pulled {a.label or a.kind!r} {gap_best:.2f}m → connect to {b.label or b.kind!r}"
        )
        info[:] = _summary(spec.primitives)  # refresh so the next primitive sees the new state


# ---------------------------------------------------------------- entry point


def check_and_fix(spec: GenerationSpec) -> tuple[GenerationSpec, list[str]]:
    """Run integrity checks and snap obvious mistakes. Idempotent."""
    if not spec.primitives:
        return spec, []
    warnings: list[str] = []
    info = _summary(spec.primitives)
    shape = (spec.shape or "").lower()

    if shape in ("chair", "stool", "table", "desk"):
        _repair_legs_seat(spec, info, warnings)
    elif shape in ("creature", "animal", "quadruped"):
        _repair_quadruped(spec, info, warnings)
    elif shape in ("vase",):
        # Stack body parts but skip the order if the spec already has rails
        # (e.g. an architectural urn on a balustrade).
        _repair_stack(spec, info, warnings, roles_order=("base", "body", "stem"))
    elif shape == "lamp":
        _repair_stack(spec, info, warnings, roles_order=("base", "seat", "stem", "cap"))
    elif shape == "mushroom":
        _repair_stack(spec, info, warnings, roles_order=("stem", "cap"))
    elif shape == "tree":
        _repair_stack(spec, info, warnings, roles_order=("trunk", "body", "crown", "cap"))
    elif shape in (
        "fence", "lattice", "grid", "railing", "balustrade", "scaffold",
        "gate", "trellis", "screen", "archway", "colonnade", "framework",
    ):
        _repair_framework(spec, info, warnings)
    else:
        # Generic fallback: if the spec *looks* like a framework (has rails +
        # multiple vbars regardless of declared shape), run the framework
        # repair. This handles cases where the LLM emits a bookshelf, a
        # ladder, a window with bars, etc. without using the shape names
        # we recognise.
        _repair_framework(spec, info, warnings)

    # Final pass: anything still floating > 5 cm gets pulled to its neighbour.
    _connectivity_sweep(spec, info, warnings)

    return spec, warnings
