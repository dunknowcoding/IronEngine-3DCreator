"""Parametric building architecture — floor plans, interiors, slice views.

Closes the "buildings are too simple / pillar blocked a door / doors have no
style / buildings need interiors / 3D-printing slice view" complaints.

Pipeline::

    plan = generate_plan(seed=7, floors=2, style="neoclassical")
    report = validate_plan(plan)            # sweep/connectivity/area checks
    built = compile_building(plan)          # part-graph spec + meshes
    slices = slice_building(built)          # per-floor '3D-print' slices
    result = build_building({...})          # all of the above, one call

Layout model (axis-aligned, metres, Y-up, footprint in XZ):

- corridor spine along the front wall; rooms in the strip behind it;
- entrance door (front facade) → corridor → every room door → stairwell →
  upper floors, so the whole building is enterable and connected;
- walls are panels with REAL thickness (exterior 0.20–0.30 m, interior
  0.10–0.15 m) decomposed into piers / lintels / sills so every doorway is
  a REAL hole; contained window bays may instead be carved by a
  ``role: "subtract"`` cutter through the analytic-mesh path;
- staircases: straight / L / U with rise ≈ 0.17 m, going ≈ 0.28 m and
  railings (handrail + balusters + newels);
- exterior detail per style (baroque / neoclassical / modern): cornice,
  base plinth, corner quoins, downspouts, lintels, balcony with railing.

Validation (:func:`validate_plan`) — violations are auto-relocated with
warnings, never silently kept:

1. door-swing sweep test: no structural column/pillar inside a door's
   swing sector (radius = leaf width, ROM 0–110°);
2. corridor path: no column inside the corridor strip;
3. connectivity: BFS from ``outside`` must reach every room on every floor;
4. room areas ≥ ``min_area`` and aspect ratio ≤ ``max_aspect``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..alignment.schema import GenerationSpec, Primitive
from .analytic_mesh import AnalyticPart, build_spec_meshes_with_report
from .complex_builder import T
from . import doors as doorlib

TAU = 2.0 * math.pi

# ---------------------------------------------------------------------------
# style presets
# ---------------------------------------------------------------------------

STYLE_PRESETS: dict[str, dict] = {
    "baroque": {
        "wall_ext": 0.30, "wall_int": 0.15,
        "cornice": "dentil", "quoins": True, "plinth_h": 0.55,
        "lintel": "keystone", "downspout": "round",
    },
    "neoclassical": {
        "wall_ext": 0.24, "wall_int": 0.12,
        "cornice": "band", "quoins": True, "plinth_h": 0.45,
        "lintel": "flat", "downspout": "round",
    },
    "modern": {
        "wall_ext": 0.20, "wall_int": 0.10,
        "cornice": "thin", "quoins": False, "plinth_h": 0.30,
        "lintel": "none", "downspout": "square",
    },
}

ROOM_NAME_POOL = ("living", "bedroom", "kitchen", "study", "bath", "dining")
MIN_ROOM_AREA = 3.5
MAX_ROOM_ASPECT = 4.0
RISE_TARGET = 0.17
GOING = 0.28
DOOR_ROM_DEG = 110.0


# ---------------------------------------------------------------------------
# plan dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Opening:
    kind: str                      # "door" | "window"
    offset: float                  # centre position along wall from start (m)
    width: float
    height: float
    sill: float = 0.0
    open_method: str = "hinged_single"
    style: str = "panel_wood"
    hinge_side: str = "left"
    swing_into: str = "b"          # door swings into room_b ("a"/"b" on wall sides)
    room_a: str = ""               # space on the wall-normal-negative side
    room_b: str = ""               # space on the wall-normal-positive side
    decorations: tuple = ()

    def to_dict(self) -> dict:
        return dict(kind=self.kind, offset=round(self.offset, 4), width=self.width,
                    height=self.height, sill=self.sill, open_method=self.open_method,
                    style=self.style, hinge_side=self.hinge_side,
                    room_a=self.room_a, room_b=self.room_b,
                    decorations=list(self.decorations))


@dataclass
class Wall:
    floor: int
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    height: float
    exterior: bool
    openings: list[Opening] = field(default_factory=list)
    label: str = ""

    @property
    def length(self) -> float:
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def direction(self) -> tuple[float, float]:
        L = max(self.length, 1e-9)
        return ((self.end[0] - self.start[0]) / L, (self.end[1] - self.start[1]) / L)

    def to_dict(self) -> dict:
        return dict(floor=self.floor, start=list(self.start), end=list(self.end),
                    thickness=self.thickness, height=self.height,
                    exterior=self.exterior, label=self.label,
                    openings=[o.to_dict() for o in self.openings])


@dataclass
class Room:
    name: str
    floor: int
    rect: tuple[float, float, float, float]        # x, z, w, d
    furniture: list[dict] = field(default_factory=list)

    @property
    def area(self) -> float:
        return self.rect[2] * self.rect[3]

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.rect[0] + self.rect[2] / 2, self.rect[1] + self.rect[3] / 2)

    def to_dict(self) -> dict:
        return dict(name=self.name, floor=self.floor, rect=[round(v, 4) for v in self.rect],
                    area=round(self.area, 3), furniture=self.furniture)


@dataclass
class Staircase:
    kind: str                      # "straight" | "L" | "U"
    floor_from: int
    well: tuple[float, float, float, float]        # x, z, w, d
    risers: int
    rise: float
    going: float
    steps: list[dict] = field(default_factory=list)     # {rect, top_y}
    landings: list[dict] = field(default_factory=list)  # {rect, top_y}

    def to_dict(self) -> dict:
        return dict(kind=self.kind, floor_from=self.floor_from,
                    well=list(self.well), risers=self.risers,
                    rise=round(self.rise, 4), going=round(self.going, 4),
                    steps=len(self.steps), landings=len(self.landings))


@dataclass
class Column:
    floor: int
    x: float
    z: float
    size: float = 0.24
    structural: bool = True

    @property
    def half(self) -> float:
        return self.size / 2


@dataclass
class BuildingPlan:
    width: float
    depth: float
    floors: int
    floor_height: float
    style: str
    seed: int
    min_area: float = MIN_ROOM_AREA
    walls: list[Wall] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    stairs: list[Staircase] = field(default_factory=list)
    columns: list[Column] = field(default_factory=list)
    entrance: dict = field(default_factory=dict)
    corridor_width: float = 1.4

    def floor_base(self, floor: int) -> float:
        return floor * self.floor_height

    def room(self, name: str, floor: int) -> Room | None:
        for r in self.rooms:
            if r.name == name and r.floor == floor:
                return r
        return None

    def to_dict(self) -> dict:
        return dict(width=self.width, depth=self.depth, floors=self.floors,
                    floor_height=self.floor_height, style=self.style, seed=self.seed,
                    walls=[w.to_dict() for w in self.walls],
                    rooms=[r.to_dict() for r in self.rooms],
                    stairs=[s.to_dict() for s in self.stairs],
                    columns=[dict(floor=c.floor, x=round(c.x, 4), z=round(c.z, 4),
                                  size=c.size) for c in self.columns],
                    entrance=self.entrance)


# ---------------------------------------------------------------------------
# plan generator
# ---------------------------------------------------------------------------


def _split_strip(rng: np.random.Generator, width: float, n: int, min_w: float = 2.6) -> list[float]:
    """Split `width` into n segments ≥ min_w (jittered, deterministic)."""
    cuts = np.sort(rng.uniform(0.35, 0.65, size=n - 1)) * width
    segs = np.diff([0.0, *cuts, width])
    for _ in range(24):
        if segs.min() >= min_w:
            break
        i = int(segs.argmin())
        j = int(segs.argmax())
        move = min((segs[j] - min_w) / 2, (min_w - segs[i]))
        segs[j] -= move
        segs[i] += move
    return [float(s) for s in segs]


def _assign_names(rng: np.random.Generator, rects: list[tuple], has_stairwell: bool) -> list[str]:
    order = sorted(range(len(rects)), key=lambda i: -(rects[i][2] * rects[i][3]))
    names = [""] * len(rects)
    pool = list(ROOM_NAME_POOL)
    if has_stairwell:
        names[order[-1]] = "stairwell"        # smallest room becomes the stairwell
        order = order[:-1]
    for rank, i in enumerate(order):
        names[i] = pool[rank] if rank < len(pool) else f"room{i}"
    return names


def generate_plan(*, seed: int = 0, floors: int = 2, width: float = 10.0,
                  depth: float = 8.0, floor_height: float = 3.0,
                  style: str = "neoclassical", rooms_per_floor: int | None = None,
                  corridor_width: float = 1.4, stair_kind: str = "U",
                  min_area: float = MIN_ROOM_AREA,
                  balcony: bool = True) -> BuildingPlan:
    """Deterministic seeded floor-plan generator.

    Layout: corridor spine along the front wall (z ∈ [0, corridor_width]),
    room strip behind it; stairwell is the smallest room; entrance centred on
    the front facade. Every room gets a corridor door and a facade window.
    """
    rng = np.random.default_rng(int(seed))
    st = STYLE_PRESETS.get(style, STYLE_PRESETS["neoclassical"])
    floors = max(1, int(floors))
    t_ext, t_int = st["wall_ext"], st["wall_int"]
    W, D, fh = float(width), float(depth), float(floor_height)
    cw = float(corridor_width)
    plan = BuildingPlan(width=W, depth=D, floors=floors, floor_height=fh,
                        style=style, seed=int(seed), min_area=float(min_area),
                        corridor_width=cw)

    n_rooms = rooms_per_floor or int(rng.integers(3, 5))       # 3–4 rooms
    n_rooms = max(2, min(5, n_rooms))
    segs = _split_strip(rng, W, n_rooms)
    xs = [0.0]
    for s in segs:
        xs.append(xs[-1] + s)
    rects = [(xs[i], cw, segs[i], D - cw) for i in range(n_rooms)]
    names = _assign_names(rng, rects, has_stairwell=floors > 1)

    # ---- rooms + interior partition walls + corridor wall -----------------
    for f in range(floors):
        # corridor wall (with room doors)
        corr = Wall(floor=f, start=(0.0, cw), end=(W, cw), thickness=t_int,
                    height=fh, exterior=False, label=f"f{f}_corridor_wall")
        for i, rect in enumerate(rects):
            plan.rooms.append(Room(name=names[i], floor=f, rect=rect))
            room_name = f"f{f}_{names[i]}"
            cx = rect[0] + rect[2] / 2
            dw = 0.86 if names[i] != "stairwell" else 0.80
            corr.openings.append(Opening(
                kind="door", offset=cx, width=dw, height=2.04,
                open_method="hinged_single", style="panel_wood",
                hinge_side="left" if (i + f) % 2 == 0 else "right",
                room_a=f"f{f}_corridor", room_b=room_name))
        plan.walls.append(corr)
        # partitions between rooms
        for i in range(1, n_rooms):
            plan.walls.append(Wall(floor=f, start=(xs[i], cw), end=(xs[i], D),
                                   thickness=t_int, height=fh, exterior=False,
                                   label=f"f{f}_partition{i}"))

    # ---- exterior walls (per floor) ---------------------------------------
    stair_i = names.index("stairwell") if "stairwell" in names else -1
    for f in range(floors):
        front = Wall(floor=f, start=(0.0, 0.0), end=(W, 0.0), thickness=t_ext,
                     height=fh, exterior=True, label=f"f{f}_front")
        if f == 0:
            front.openings.append(Opening(
                kind="door", offset=W / 2, width=1.20, height=2.16,
                open_method="hinged_double", style="panel_wood",
                hinge_side="left", room_a="outside", room_b="f0_corridor",
                decorations=("moldings", "transom")))
            plan.entrance = {"wall": front.label, "offset": W / 2,
                             "room": "f0_corridor"}
        else:
            front.openings.append(Opening(
                kind="window", offset=W / 2, width=1.30, height=1.40, sill=0.90,
                open_method="casement", style="wood",
                room_a="outside", room_b=f"f{f}_corridor"))
        plan.walls.append(front)

        back = Wall(floor=f, start=(0.0, D), end=(W, D), thickness=t_ext,
                    height=fh, exterior=True, label=f"f{f}_back")
        for i, rect in enumerate(rects):
            cx = rect[0] + rect[2] / 2
            is_balcony_door = (balcony and f == 1 and i == 0)
            if is_balcony_door:
                back.openings.append(Opening(
                    kind="door", offset=cx, width=1.50, height=2.10, sill=0.0,
                    open_method="french", style="glass",
                    room_a="outside", room_b=f"f{f}_{names[i]}",
                    decorations=("muntins",)))
            else:
                back.openings.append(Opening(
                    kind="window", offset=cx, width=1.20, height=1.40, sill=0.90,
                    open_method="casement", style="wood",
                    room_a="outside", room_b=f"f{f}_{names[i]}"))
        plan.walls.append(back)

        for side, x0 in (("left", 0.0), ("right", W)):
            w = Wall(floor=f, start=(x0, 0.0), end=(x0, D), thickness=t_ext,
                     height=fh, exterior=True, label=f"f{f}_{side}")
            if stair_i >= 0 and ((side == "right" and stair_i == n_rooms - 1)
                                 or (side == "left" and stair_i == 0)):
                rect = rects[stair_i]
                w.openings.append(Opening(
                    kind="window", offset=rect[1] + rect[3] / 2, width=1.00,
                    height=1.30, sill=1.00, open_method="casement", style="wood",
                    room_a="outside", room_b=f"f{f}_stairwell"))
            plan.walls.append(w)

    # ---- staircases ---------------------------------------------------------
    if floors > 1 and stair_i >= 0:
        rect = rects[stair_i]
        well = (rect[0] + 0.15, rect[1] + 0.35, rect[2] - 0.30, rect[3] - 0.60)
        for f in range(floors - 1):
            plan.stairs.append(layout_staircase(stair_kind, fh, well, f))

    # ---- structural columns -------------------------------------------------
    # junction columns (sit inside wall crossings — safe by construction)
    for f in range(floors):
        for px in xs[1:-1]:
            plan.columns.append(Column(f, px, cw, size=t_int + 0.12))
            plan.columns.append(Column(f, px, D - t_ext / 2, size=t_ext + 0.10))
        for cx_, cz_ in ((t_ext / 2, t_ext / 2), (W - t_ext / 2, t_ext / 2),
                         (t_ext / 2, D - t_ext / 2), (W - t_ext / 2, D - t_ext / 2)):
            plan.columns.append(Column(f, cx_, cz_, size=t_ext + 0.10))
        # interior columns in wide rooms (span > 4.2 m) — placed near the
        # door zone on purpose for ~half the seeds, so the validator's
        # sweep-and-relocate is genuinely exercised (the original complaint
        # was "a pillar BLOCKED a door")
        widest = max(range(len(rects)), key=lambda i: rects[i][2] * rects[i][3])
        for i, rect in enumerate(rects):
            if rect[2] > 4.2 or i == widest:
                near_door = bool(rng.integers(0, 2))
                if near_door:
                    # squarely in the door swing zone: just inside the room,
                    # next to the corridor door (validator must relocate it)
                    cx_ = rect[0] + rect[2] / 2 + rng.uniform(-0.35, 0.35)
                    cz_ = cw + rng.uniform(0.45, 0.95)
                else:
                    cx_ = rect[0] + rng.uniform(0.30, 0.70) * rect[2]
                    cz_ = rect[1] + rng.uniform(0.35, 0.75) * rect[3]
                plan.columns.append(Column(f, cx_, cz_, size=0.26))

    return plan


# ---------------------------------------------------------------------------
# staircase layout
# ---------------------------------------------------------------------------


def layout_staircase(kind: str, floor_height: float, well: tuple, floor_from: int,
                     *, rise_target: float = RISE_TARGET, going: float = GOING) -> Staircase:
    """Compute steps/landings for a straight / L / U stair inside `well`.

    Guarantees: rise in [0.15, 0.185], going in [0.25, 0.32], total rise ==
    floor_height. Steps are recorded as plan rects with their top height.
    """
    x0, z0, ww, wd = (float(v) for v in well)
    fh = float(floor_height)
    risers = max(6, int(round(fh / rise_target)))
    rise = fh / risers
    kind = str(kind).upper()
    steps: list[dict] = []
    landings: list[dict] = []

    def _flight(n, cx0, cz0, dx, dz, width, g, i0):
        """n steps starting at flight-origin (cx0, cz0) — the CENTRE of the
        first step's run direction axis — moving along (dx, dz) with going g.
        Step i top is at rise*(i0+i+1)."""
        run = []
        for i in range(n):
            cx = cx0 + dx * (i + 0.5) * g
            cz = cz0 + dz * (i + 0.5) * g
            if abs(dx) > 0:
                rect = (cx - g / 2, cz - width / 2, g, width)
            else:
                rect = (cx - width / 2, cz - g / 2, width, g)
            run.append({"rect": rect, "top_y": rise * (i0 + i + 1)})
        return run

    if kind == "STRAIGHT":
        n = risers
        need = n * going + 0.5
        going_eff = min(0.32, (wd - 0.5) / n) if need > wd else going
        going_eff = max(0.25, going_eff)
        steps = _flight(n, x0 + ww / 2, z0, 0, 1, min(0.95, ww - 0.1), going_eff, 0)
        return Staircase("straight", floor_from, (x0, z0, ww, wd), n, rise,
                         going_eff, steps, [])

    if kind == "L":
        n1 = max(3, int(round(risers * 0.65)))
        n2 = risers - n1
        fw = min(0.95, (ww - 0.1), (wd - 0.1) / 2)
        # flight 1 along +z in the left strip, landing in the corner, flight
        # 2 along +x beside it
        steps += _flight(n1, x0 + fw / 2, z0, 0, 1, fw, going, 0)
        landings.append({"rect": (x0, z0 + n1 * going, fw, fw), "top_y": rise * n1})
        steps += _flight(n2, x0 + fw, z0 + n1 * going + fw / 2, 1, 0, fw, going, n1)
        return Staircase("L", floor_from, (x0, z0, ww, wd), risers, rise, going, steps, landings)

    # default "U": two parallel flights along +z / −z with a far landing
    fw = min(0.95, (ww - 0.15) / 2)
    n1 = (risers + 1) // 2
    n2 = risers - n1
    max_run = wd - fw                       # flight + landing must fit well depth
    going_eff = going
    if n1 * going > max_run:
        going_eff = max(0.25, min(0.32, max_run / n1))
    steps += _flight(n1, x0 + 0.05 + fw / 2, z0, 0, 1, fw, going_eff, 0)
    land_z = z0 + n1 * going_eff
    landings.append({"rect": (x0 + 0.05, land_z, 2 * fw + 0.10, fw), "top_y": rise * n1})
    steps += _flight(n2, x0 + 0.05 + fw + 0.10 + fw / 2, land_z + fw, 0, -1, fw, going_eff, n1)
    return Staircase("U", floor_from, (x0, z0, ww, wd), risers, rise, going_eff, steps, landings)


# ---------------------------------------------------------------------------
# validation — sweep tests, connectivity, areas; violations auto-relocated
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    ok: bool
    warnings: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(ok=self.ok, warnings=self.warnings, fixes=self.fixes,
                    checks=self.checks)


def _door_swing_sector(plan: BuildingPlan, wall: Wall, op: Opening) -> dict | None:
    """Swing-sector descriptor for a hinged door opening on `wall`.

    Returns hinge point, closed direction (unit), inward normal (unit), and
    radius = leaf width. Doors swing AWAY from the corridor / INTO room_b
    for interior doors, and into the corridor for the entrance.
    """
    if op.kind != "door" or op.open_method in ("sliding", "garage", "revolving"):
        return None
    d = wall.direction
    leaves = 2 if op.open_method in ("hinged_double", "french") else 1
    leaf_w = op.width / leaves
    ux, uz = d
    # inward normal: rotate direction by -90° → (uz, -ux)? Both normals are
    # possible; pick the one pointing into the swing room. Interior doors on
    # the corridor wall swing into the room (+z); the entrance swings into
    # the building (+z). Convention: swing normal = (+uz, -ux) flipped to
    # point at the room_b side — room_b is always on the normal-positive
    # side of the wall in this generator.
    nx, nz = uz, -ux
    # ensure the normal points into the building interior for exterior walls
    if wall.exterior:
        cx = (wall.start[0] + wall.end[0]) / 2 + nx
        cz = (wall.start[1] + wall.end[1]) / 2 + nz
        if not (0.0 <= cx <= plan.width and 0.0 <= cz <= plan.depth):
            nx, nz = -nx, -nz
    else:
        # interior corridor wall: rooms are behind it; normal must point from
        # corridor toward rooms. Use the room_b rect centroid as reference.
        room = None
        for r in plan.rooms:
            if f"f{r.floor}_{r.name}" == op.room_b:
                room = r
                break
        if room is not None:
            mx = (wall.start[0] + wall.end[0]) / 2
            mz = (wall.start[1] + wall.end[1]) / 2
            rcx, rcz = room.centroid
            if (rcx - mx) * nx + (rcz - mz) * nz < 0:
                nx, nz = -nx, -nz
    # hinge point at opening edge
    if op.hinge_side == "left":
        hx = wall.start[0] + ux * (op.offset - op.width / 2)
        hz = wall.start[1] + uz * (op.offset - op.width / 2)
        closed = (ux, uz)
    else:
        hx = wall.start[0] + ux * (op.offset + op.width / 2)
        hz = wall.start[1] + uz * (op.offset + op.width / 2)
        closed = (-ux, -uz)
    return {"hinge": (hx, hz), "closed": closed, "normal": (nx, nz),
            "radius": leaf_w, "rom_deg": DOOR_ROM_DEG}


def _point_in_sector(px: float, pz: float, sector: dict, inflate: float = 0.0) -> bool:
    """Is (px, pz) inside the door sweep sector (radius inflated by
    `inflate`)? Angle measured from the closed-leaf direction toward the
    swing normal."""
    hx, hz = sector["hinge"]
    vx, vz = px - hx, pz - hz
    dist = math.hypot(vx, vz)
    radius = sector["radius"] + inflate
    if dist > radius or dist < 1e-9:
        return False
    cx, cz = sector["closed"]
    nx, nz = sector["normal"]
    along = vx * cx + vz * cz
    side = vx * nx + vz * nz
    ang = math.degrees(math.atan2(side, along))
    # angular inflation for the column's finite size
    infl_ang = math.degrees(math.asin(min(1.0, inflate / dist))) if inflate > 0 else 0.0
    return -infl_ang <= ang <= sector["rom_deg"] + infl_ang


def door_swing_conflicts(plan: BuildingPlan) -> list[dict]:
    """All (column, door) pairs where the column stands inside the door's
    sweep sector on the same floor."""
    conflicts: list[dict] = []
    for wall in plan.walls:
        for op in wall.openings:
            sector = _door_swing_sector(plan, wall, op)
            if sector is None:
                continue
            for col in plan.columns:
                if col.floor != wall.floor:
                    continue
                if _point_in_sector(col.x, col.z, sector, inflate=col.half):
                    conflicts.append({"column": col, "wall": wall.label,
                                      "opening": op, "sector": sector})
    return conflicts


def corridor_conflicts(plan: BuildingPlan) -> list[Column]:
    """Columns standing inside the corridor walking strip.

    Columns EMBEDDED in a wall (their centre lies within the wall's
    centreline band) are pilasters, not obstacles — exempt.
    """
    out = []
    t_int = STYLE_PRESETS.get(plan.style, STYLE_PRESETS["neoclassical"])["wall_int"]
    for col in plan.columns:
        if not (0.0 < col.x < plan.width):
            continue
        # AABB overlap with the clear walking strip z ∈ (0, corridor_width)
        if col.z - col.half >= plan.corridor_width or col.z + col.half <= 0.0:
            continue
        # embedded in the front exterior wall or the corridor partition?
        embedded = False
        for wall in plan.walls:
            if wall.floor != col.floor:
                continue
            (x0, z0), (x1, z1) = wall.start, wall.end
            L = wall.length or 1.0
            ux, uz = (x1 - x0) / L, (z1 - z0) / L
            along = (col.x - x0) * ux + (col.z - z0) * uz
            if not (-col.half <= along <= L + col.half):
                continue
            across = abs((col.x - x0) * uz - (col.z - z0) * ux)
            if across <= (wall.thickness + col.size) / 2 + 0.02:
                embedded = True
                break
        if not embedded:
            out.append(col)
    return out


def connectivity(plan: BuildingPlan) -> tuple[set[str], set[str]]:
    """BFS from 'outside' across door openings + stair links.

    Returns (reachable, unreachable) space names."""
    adj: dict[str, set[str]] = {}
    for wall in plan.walls:
        for op in wall.openings:
            if op.kind != "door":
                continue
            a, b = op.room_a or "outside", op.room_b
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    for st in plan.stairs:
        a = f"f{st.floor_from}_stairwell"
        b = f"f{st.floor_from + 1}_stairwell"
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen = {"outside"}
    frontier = ["outside"]
    while frontier:
        cur = frontier.pop()
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    all_spaces = {f"f{r.floor}_{r.name}" for r in plan.rooms}
    all_spaces |= {f"f{f}_corridor" for f in range(plan.floors)}
    return seen, all_spaces - seen


def area_violations(plan: BuildingPlan) -> list[str]:
    out = []
    for r in plan.rooms:
        if r.name == "stairwell":
            continue
        w, d = r.rect[2], r.rect[3]
        if r.area < plan.min_area:
            out.append(f"f{r.floor}_{r.name}: area {r.area:.2f} m² < {plan.min_area} m²")
        if max(w, d) / max(min(w, d), 1e-9) > MAX_ROOM_ASPECT:
            out.append(f"f{r.floor}_{r.name}: aspect {max(w, d) / min(w, d):.1f} > {MAX_ROOM_ASPECT}")
    return out


def _relocate_column(plan: BuildingPlan, col: Column, sector: dict | None,
                     fixes: list[str]) -> None:
    """Push a column out of a swing sector / corridor, keeping it structural
    (prefer sliding along the room away from the hinge)."""
    old = (round(col.x, 3), round(col.z, 3))
    if sector is not None:
        hx, hz = sector["hinge"]
        nx, nz = sector["normal"]
        need = sector["radius"] + col.half + 0.06
        # move away from the hinge along the wall direction first
        vx, vz = col.x - hx, col.z - hz
        dist = math.hypot(vx, vz)
        if dist < 1e-6:
            vx, vz = nx, nz
            dist = 1.0
        push = need - dist
        col.x += vx / dist * push
        col.z += vz / dist * push
        # keep inside the footprint
        col.x = min(max(col.x, 0.3), plan.width - 0.3)
        col.z = min(max(col.z, plan.corridor_width + 0.3), plan.depth - 0.3)
        fixes.append(f"column f{col.floor}@{old} moved out of door swing → "
                     f"({col.x:.3f}, {col.z:.3f})")
    else:
        # corridor violation: push into the room strip, off the walking path
        col.z = plan.corridor_width + col.half + 0.06
        fixes.append(f"column f{col.floor}@{old} moved out of corridor path → "
                     f"({col.x:.3f}, {col.z:.3f})")


def validate_plan(plan: BuildingPlan, *, max_iters: int = 12) -> ValidationReport:
    """Run all layout checks; auto-relocate offending columns with warnings.

    Checks: (1) door-swing sweep, (2) corridor path, (3) entrance
    connectivity, (4) room areas. Returns a report; ``ok`` means no
    UNRESOLVED violations remain."""
    warnings: list[str] = []
    fixes: list[str] = []

    for _ in range(max_iters):
        moved = False
        for c in door_swing_conflicts(plan):
            col = c["column"]
            warnings.append(
                f"door swing conflict: column f{col.floor}@({col.x:.2f},{col.z:.2f}) "
                f"inside sweep of door on {c['wall']} (offset {c['opening'].offset:.2f})")
            _relocate_column(plan, col, c["sector"], fixes)
            moved = True
        for col in corridor_conflicts(plan):
            warnings.append(
                f"corridor conflict: column f{col.floor}@({col.x:.2f},{col.z:.2f}) "
                "inside corridor walking path")
            _relocate_column(plan, col, None, fixes)
            moved = True
        if not moved:
            break

    residual_swing = door_swing_conflicts(plan)
    residual_corr = corridor_conflicts(plan)

    reachable, unreachable = connectivity(plan)
    if unreachable:
        warnings.append(f"unreachable spaces from entrance: {sorted(unreachable)}")

    areas = area_violations(plan)
    warnings.extend(areas)

    n_doors = sum(1 for w in plan.walls for o in w.openings if o.kind == "door")
    checks = {
        "door_swings_checked": n_doors,
        "swing_violations_resolved": len([f for f in fixes if "swing" in f]),
        "corridor_violations_resolved": len([f for f in fixes if "corridor" in f]),
        "rooms_total": len(plan.rooms),
        "spaces_reachable": len(reachable & {f"f{r.floor}_{r.name}" for r in plan.rooms}
                                | {f"f{f}_corridor" for f in range(plan.floors)}),
        "spaces_unreachable": sorted(unreachable),
        "area_violations": areas,
    }
    ok = not residual_swing and not residual_corr and not unreachable and not areas
    return ValidationReport(ok=ok, warnings=warnings, fixes=fixes, checks=checks)


# ---------------------------------------------------------------------------
# compilation — plan → part-graph spec (panels with real thickness)
# ---------------------------------------------------------------------------


@dataclass
class CompiledBuilding:
    plan: BuildingPlan
    spec: GenerationSpec
    extras: dict
    metadata: dict
    parts: list[AnalyticPart] = field(default_factory=list)
    mesh_warnings: list[str] = field(default_factory=list)

    def build_parts(self) -> list[AnalyticPart]:
        self.parts, self.mesh_warnings = build_spec_meshes_with_report(self.spec)
        return self.parts

    def write_glb(self, path) -> object:
        from ..core.exporter import write_glb_parts

        if not self.parts:
            self.build_parts()
        return write_glb_parts(path, self.parts)


def _wall_ry(wall: Wall) -> float:
    dx, dz = wall.direction
    return math.atan2(-dz, dx)


def _wall_point(wall: Wall, u: float) -> tuple[float, float]:
    dx, dz = wall.direction
    return (wall.start[0] + dx * u, wall.start[1] + dz * u)


def _prim(kind: str, params: dict, transform: np.ndarray, label: str) -> Primitive:
    return Primitive(kind=kind, transform=np.asarray(transform, dtype=np.float32).tolist(),
                     params=params, label=label)


def _panel_prim(wall: Wall, u0: float, u1: float, y0: float, y1: float,
                label: str, material: str) -> Primitive:
    w = u1 - u0
    cx, cz = _wall_point(wall, (u0 + u1) / 2)
    m = T(translate=(cx, (y0 + y1) / 2, cz), ry=_wall_ry(wall))
    return _prim("panel", {"size": [w, y1 - y0], "thickness": wall.thickness,
                           "material": material}, m, label)


def _rect_hole_max_radius(w: float, h: float) -> float:
    """Half-diagonal of a rect hole — mirrors analytic_mesh._Hole.max_radius."""
    return math.hypot(w / 2, h / 2)


def _compile_wall(plan: BuildingPlan, wall: Wall, prims: list[Primitive],
                  cutter_prims: list[Primitive], *, use_subtraction: bool,
                  wall_material: str, warnings: list[str]) -> None:
    """Decompose one wall into pier/lintel/sill panels (real holes for every
    opening). Window bays whose hole is fully contained may instead be a
    single bay panel carved by a `role: "subtract"` box cutter."""
    y_base = plan.floor_base(wall.floor)
    H = wall.height
    L = wall.length
    openings = sorted(wall.openings, key=lambda o: o.offset)
    cursor = 0.0
    k = 0

    def pier(a: float, b: float) -> None:
        nonlocal k
        if b - a < 0.02:
            return
        prims.append(_panel_prim(wall, a, b, y_base, y_base + H,
                                 f"{wall.label}_pier{k}", wall_material))
        k += 1

    for oi, op in enumerate(openings):
        a, b = op.offset - op.width / 2, op.offset + op.width / 2
        y_top = y_base + op.sill + op.height
        carved = False
        if (use_subtraction and op.kind == "window" and op.sill > 0.05):
            # feasibility: hole half-diagonal must clear every bay edge with
            # the carver's containment margin (0.1% of max extent)
            bay_a, bay_b = cursor, (openings[oi + 1].offset - openings[oi + 1].width / 2
                                    if oi + 1 < len(openings) else L)
            r_max = _rect_hole_max_radius(op.width, op.height)
            margin = 1e-3 * max(bay_b - bay_a, H)
            cy = op.sill + op.height / 2
            ok = (op.offset - r_max > bay_a + margin
                  and op.offset + r_max < bay_b - margin
                  and cy - r_max > 0.0 + margin
                  and cy + r_max < H - margin)
            if ok:
                pier(cursor, bay_a) if bay_a > cursor else None
                cx, cz = _wall_point(wall, (bay_a + bay_b) / 2)
                bay_label = f"{wall.label}_bay{oi}"
                prims.append(_prim(
                    "panel", {"size": [bay_b - bay_a, H], "thickness": wall.thickness,
                              "material": wall_material},
                    T(translate=(cx, y_base + H / 2, cz), ry=_wall_ry(wall)), bay_label))
                ox, oz = _wall_point(wall, op.offset)
                cutter_prims.append(_prim(
                    "box", {"size": [op.width, op.height, wall.thickness * 2.5],
                            "role": "subtract", "target": bay_label},
                    T(translate=(ox, y_base + cy, oz), ry=_wall_ry(wall)),
                    f"{wall.label}_cutter{oi}"))
                carved = True
                cursor = bay_b
            else:
                warnings.append(
                    f"subtraction infeasible for {wall.label} opening {oi} "
                    "(hole too close to bay edges) — decomposed instead")
        if not carved:
            pier(cursor, a)
            # lintel above the opening
            if y_top < y_base + H - 0.02:
                prims.append(_panel_prim(wall, a, b, y_top, y_base + H,
                                         f"{wall.label}_lintel{oi}", wall_material))
            # sill below windows
            if op.sill > 0.02:
                prims.append(_panel_prim(wall, a, b, y_base, y_base + op.sill,
                                         f"{wall.label}_sill{oi}", wall_material))
            cursor = b
    pier(cursor, L)


# ---------------------------------------------------------------------------
# stairs geometry
# ---------------------------------------------------------------------------


def _compile_stairs(plan: BuildingPlan, st: Staircase, prims: list[Primitive]) -> None:
    y0 = plan.floor_base(st.floor_from)
    wx, wz, ww, wd = st.well
    cx_mid = wx + ww / 2
    for i, s in enumerate(st.steps):
        rx, rz, rw, rd = s["rect"]
        top = y0 + s["top_y"]
        prims.append(_prim("box", {"size": [rw, 0.06, rd], "material": "wood"},
                           T(translate=(rx + rw / 2, top - 0.03, rz + rd / 2)),
                           f"f{st.floor_from}_stair{st.kind}_step{i}"))
        # baluster on every second step (outer edge)
        if i % 2 == 0:
            bx = rx + rw - 0.03 if rw > rd else rx + rw / 2
            bz = rz + rd - 0.03 if rd >= rw else rz + rd / 2
            prims.append(_prim("box", {"size": [0.025, 0.85, 0.025], "material": "metal"},
                               T(translate=(bx, top + 0.425, bz)),
                               f"f{st.floor_from}_stair{st.kind}_bal{i}"))
    for li, l in enumerate(st.landings):
        rx, rz, rw, rd = l["rect"]
        top = y0 + l["top_y"]
        prims.append(_prim("box", {"size": [rw, 0.12, rd], "material": "wood"},
                           T(translate=(rx + rw / 2, top - 0.06, rz + rd / 2)),
                           f"f{st.floor_from}_stair{st.kind}_landing{li}"))
    # sloped handrails per contiguous run of steps
    runs: list[list[dict]] = []
    for s in st.steps:
        if runs and abs(s["top_y"] - runs[-1][-1]["top_y"] - st.rise) < 1e-6:
            runs[-1].append(s)
        else:
            runs.append([s])
    for ri, run in enumerate(runs):
        if len(run) < 2:
            continue
        a, b = run[0], run[-1]
        ax, az = a["rect"][0] + a["rect"][2] / 2, a["rect"][1] + a["rect"][3] / 2
        bx2, bz2 = b["rect"][0] + b["rect"][2] / 2, b["rect"][1] + b["rect"][3] / 2
        ya, yb = y0 + a["top_y"] + 0.90, y0 + b["top_y"] + 0.90
        run_len = math.hypot(bx2 - ax, bz2 - az)
        if run_len < 1e-6:
            continue
        pitch = math.atan2(yb - ya, run_len)
        along_x = abs(bx2 - ax) > abs(bz2 - az)
        ry = 0.0 if along_x else math.pi / 2
        rz_rot = pitch if along_x else 0.0
        rx_rot = -pitch if not along_x else 0.0
        length = math.hypot(run_len, yb - ya)
        # offset the rail to the outer edge of the flight
        off = (a["rect"][2] / 2 - 0.04) if along_x else (a["rect"][3] / 2 - 0.04)
        ox = (ax + bx2) / 2 + (0.0 if along_x else off)
        oz = (az + bz2) / 2 + (off if along_x else 0.0)
        prims.append(_prim("box", {"size": [length, 0.05, 0.07], "material": "wood"},
                           T(translate=(ox, (ya + yb) / 2, oz), ry=ry, rx=rx_rot, rz=rz_rot),
                           f"f{st.floor_from}_stair{st.kind}_rail{ri}"))
        # newel posts at both ends
        for ni, (px, pz, py) in enumerate(((ax, az, ya), (bx2, bz2, yb))):
            prims.append(_prim("box", {"size": [0.07, 1.0, 0.07], "material": "wood"},
                               T(translate=(px + (0 if along_x else off),
                                            py - 0.55, pz + (off if along_x else 0))),
                               f"f{st.floor_from}_stair{st.kind}_newel{ri}_{ni}"))


# ---------------------------------------------------------------------------
# exterior detail — cornice / plinth / quoins / downspouts / lintels / balcony
# ---------------------------------------------------------------------------


def _perimeter_segments(plan: BuildingPlan) -> list[tuple]:
    """(x, z, length, ry) for the four facades, centred."""
    W, D = plan.width, plan.depth
    return [
        (W / 2, 0.0, W, 0.0),
        (W / 2, D, W, 0.0),
        (0.0, D / 2, D, math.pi / 2),
        (W, D / 2, D, math.pi / 2),
    ]


def _compile_exterior(plan: BuildingPlan, prims: list[Primitive], *,
                      balcony: bool) -> None:
    st = STYLE_PRESETS.get(plan.style, STYLE_PRESETS["neoclassical"])
    t = st["wall_ext"]
    y_top = plan.floors * plan.floor_height

    # base plinth
    ph = st["plinth_h"]
    for i, (cx, cz, L, ry) in enumerate(_perimeter_segments(plan)):
        prims.append(_prim("box", {"size": [L + 2 * t + 0.12, ph, t + 0.12],
                                   "material": "stone"},
                           T(translate=(cx, ph / 2, cz), ry=ry), f"plinth_{i}"))

    # cornice
    kind = st["cornice"]
    bands = [("cornice_main", y_top - 0.22, 0.22, 0.16)]
    if kind == "dentil":
        bands.append(("cornice_crown", y_top - 0.02, 0.10, 0.24))
    elif kind == "thin":
        bands = [("cornice_thin", y_top - 0.06, 0.08, 0.07)]
    for i, (cx, cz, L, ry) in enumerate(_perimeter_segments(plan)):
        for name, yc, hh, protr in bands:
            prims.append(_prim("box", {"size": [L + 2 * t + 2 * protr, hh, t + 2 * protr],
                                       "material": "stone"},
                               T(translate=(cx, yc + hh / 2, cz), ry=ry),
                               f"{name}_{i}"))
        if kind == "dentil":
            n = max(4, int(L / 0.42))
            for j in range(n):
                u = -L / 2 + L * (j + 0.5) / n
                px = cx + (u if ry == 0 else 0.0)
                pz = cz + (0.0 if ry == 0 else u)
                prims.append(_prim("box", {"size": [0.12, 0.12, t + 0.20],
                                           "material": "stone"},
                                   T(translate=(px, y_top - 0.30, pz), ry=ry),
                                   f"dentil_{i}_{j}"))
    if plan.style == "modern":
        # parapet
        for i, (cx, cz, L, ry) in enumerate(_perimeter_segments(plan)):
            prims.append(_prim("box", {"size": [L + 2 * t, 0.45, t],
                                       "material": "brick"},
                               T(translate=(cx, y_top + 0.225, cz), ry=ry), f"parapet_{i}"))

    # corner quoins (alternating blocks up the corners)
    if st["quoins"]:
        y0 = ph
        h = y_top - 0.34 - y0
        n = max(3, int(h / 0.40))
        for ci, (qx, qz) in enumerate(((0, 0), (plan.width, 0),
                                       (0, plan.depth), (plan.width, plan.depth))):
            for j in range(n):
                long = (j % 2 == 0)
                sx = 0.42 if long else 0.30
                y = y0 + h * (j + 0.5) / n
                prims.append(_prim("box", {"size": [sx, h / n * 0.92, t + 0.10],
                                           "material": "stone"},
                                   T(translate=(qx, y, qz)), f"quoin_{ci}_{j}"))

    # downspouts at the two front corners
    for ci, sx in enumerate((0.35, plan.width - 0.35)):
        if st["downspout"] == "square":
            prims.append(_prim("box", {"size": [0.09, y_top - ph, 0.09],
                                       "material": "metal"},
                               T(translate=(sx, (y_top - ph) / 2 + ph, -t / 2 - 0.06)),
                               f"downspout_{ci}"))
        else:
            prims.append(_prim("cylinder", {"radius": 0.045, "height": y_top - ph,
                                            "material": "metal"},
                               T(translate=(sx, (y_top - ph) / 2 + ph, -t / 2 - 0.06)),
                               f"downspout_{ci}"))
            prims.append(_prim("cylinder", {"radius": 0.045, "height": 0.30,
                                            "material": "metal"},
                               T(translate=(sx, y_top - 0.15, -t / 2 + 0.05), rx=math.pi / 2),
                               f"downspout_elbow_{ci}"))
        for bi, by in enumerate((ph + 0.4, y_top * 0.5, y_top - 0.6)):
            prims.append(_prim("box", {"size": [0.14, 0.03, 0.10], "material": "metal"},
                               T(translate=(sx, by, -t / 2 - 0.03)),
                               f"downspout_strap_{ci}_{bi}"))

    # lintel decor above exterior openings
    if st["lintel"] != "none":
        for wall in plan.walls:
            if not wall.exterior:
                continue
            ry = _wall_ry(wall)
            y_base = plan.floor_base(wall.floor)
            for oi, op in enumerate(wall.openings):
                y = y_base + op.sill + op.height
                ox, oz = _wall_point(wall, op.offset)
                prims.append(_prim("box", {"size": [op.width + 0.26, 0.13,
                                                    wall.thickness + 0.10],
                                           "material": "stone"},
                                   T(translate=(ox, y + 0.065, oz), ry=ry),
                                   f"{wall.label}_linteldecor{oi}"))
                if st["lintel"] == "keystone":
                    prims.append(_prim("box", {"size": [0.16, 0.24, wall.thickness + 0.14],
                                               "material": "stone"},
                                       T(translate=(ox, y + 0.10, oz), ry=ry),
                                       f"{wall.label}_keystone{oi}"))

    # balcony with railing (in front of the floor-1 french door)
    if balcony and plan.floors > 1:
        door_op = None
        back = next((w for w in plan.walls if w.label == "f1_back"), None)
        if back:
            for op in back.openings:
                if op.open_method == "french":
                    door_op = op
                    break
        if door_op is not None:
            ox, oz = _wall_point(back, door_op.offset)
            y = plan.floor_base(1)
            slab_d = 0.95
            zc = oz + slab_d / 2          # slab extends outward from the back wall (+z)
            prims.append(_prim("box", {"size": [door_op.width + 0.6, 0.12, slab_d],
                                       "material": "stone"},
                               T(translate=(ox, y - 0.06, zc)), "balcony_slab"))
            # railing: top rail + posts + balusters on 3 open sides
            rail_y = y + 1.0
            w2 = (door_op.width + 0.6) / 2
            prims.append(_prim("box", {"size": [door_op.width + 0.6, 0.05, 0.06],
                                       "material": "metal"},
                               T(translate=(ox, rail_y, oz + slab_d - 0.04)),
                               "balcony_rail_front"))
            for sgn, tag in ((-1, "l"), (1, "r")):
                prims.append(_prim("box", {"size": [0.06, 0.05, slab_d],
                                           "material": "metal"},
                                   T(translate=(ox + sgn * (w2 - 0.04), rail_y,
                                                oz + slab_d / 2)),
                                   f"balcony_rail_{tag}"))
            n_bal = int((door_op.width + 0.6) / 0.16)
            for j in range(n_bal + 1):
                bx = ox - w2 + (door_op.width + 0.6) * j / max(1, n_bal)
                prims.append(_prim("box", {"size": [0.02, 1.0, 0.02],
                                           "material": "metal"},
                                   T(translate=(bx, y + 0.5, oz + slab_d - 0.04)),
                                   f"balcony_bal_{j}"))
            for sgn, tag in ((-1, "l"), (1, "r")):
                prims.append(_prim("box", {"size": [0.06, 1.05, 0.06],
                                           "material": "metal"},
                                   T(translate=(ox + sgn * (w2 - 0.04), y + 0.52,
                                                oz + slab_d - 0.04)),
                                   f"balcony_post_{tag}"))


# ---------------------------------------------------------------------------
# furniture (interior markers — real low-poly placeholders + slice footprints)
# ---------------------------------------------------------------------------

FURNITURE_PRESETS: dict[str, list[dict]] = {
    "living":  [{"kind": "sofa", "size": (1.90, 0.85, 0.45)}, {"kind": "table", "size": (1.10, 0.60, 0.42)}],
    "bedroom": [{"kind": "bed", "size": (1.60, 2.00, 0.50)}, {"kind": "wardrobe", "size": (1.20, 0.60, 2.00)}],
    "kitchen": [{"kind": "counter", "size": (2.20, 0.65, 0.90)}],
    "study":   [{"kind": "desk", "size": (1.40, 0.70, 0.75)}, {"kind": "shelf", "size": (0.90, 0.35, 1.80)}],
    "bath":    [{"kind": "tub", "size": (1.60, 0.75, 0.55)}],
    "dining":  [{"kind": "table", "size": (1.60, 0.90, 0.74)}],
    "stairwell": [],
}


def _furnish(plan: BuildingPlan, prims: list[Primitive], *, enabled: bool) -> None:
    for room in plan.rooms:
        items = FURNITURE_PRESETS.get(room.name, [{"kind": "table", "size": (1.2, 0.7, 0.72)}])
        x, z, w, d = room.rect
        for j, item in enumerate(items):
            fw, fd, fhh = item["size"]
            fx = min(max(x + 0.45 + fw / 2 + j * (fw + 0.35), x + fw / 2 + 0.2), x + w - fw / 2 - 0.2)
            fz = z + d - fd / 2 - 0.35
            rect = (fx - fw / 2, fz - fd / 2, fw, fd)
            room.furniture.append({"kind": item["kind"], "rect": [round(v, 3) for v in rect]})
            if enabled:
                y0 = plan.floor_base(room.floor)
                prims.append(_prim("box", {"size": [fw, fhh, fd], "material": "wood"},
                                   T(translate=(fx, y0 + fhh / 2, fz)),
                                   f"f{room.floor}_{room.name}_furn_{item['kind']}{j}"))


# ---------------------------------------------------------------------------
# compile — the whole building
# ---------------------------------------------------------------------------


def compile_building(plan: BuildingPlan, *, use_subtraction: bool = True,
                     interiors: bool = True, furniture: bool = True,
                     balcony: bool = True) -> CompiledBuilding:
    """Compile a validated plan into a part-graph spec + iemodel extras."""
    st = STYLE_PRESETS.get(plan.style, STYLE_PRESETS["neoclassical"])
    prims: list[Primitive] = []
    cutters: list[Primitive] = []
    warnings: list[str] = []
    wall_mat = "brick" if plan.style != "modern" else "stone"

    # ---- walls (piers/lintels/sills, or carved window bays) ----------------
    for wall in plan.walls:
        _compile_wall(plan, wall, prims, cutters, use_subtraction=use_subtraction,
                      wall_material=wall_mat if wall.exterior else "stone",
                      warnings=warnings)

    # ---- floor / roof slabs -------------------------------------------------
    # Slabs above a stair get a REAL well hole via 4-way decomposition (the
    # mesh carver cannot tunnel near slab edges — a large well hole close to
    # the facade would sever the host; decomposition is exact and always
    # valid). Window bays still demonstrate `role: "subtract"` carving.
    W, D = plan.width, plan.depth
    well_holes: dict[int, tuple] = {}
    for st_case in plan.stairs:
        well_holes[st_case.floor_from + 1] = st_case.well
    for f in range(plan.floors + 1):
        y = plan.floor_base(f)
        label = f"slab_f{f}" if f < plan.floors else "roof_slab"
        hole = well_holes.get(f)
        if hole is None:
            prims.append(_prim("box", {"size": [W + 2 * st["wall_ext"], 0.16, D + 2 * st["wall_ext"]],
                                       "material": "stone"},
                               T(translate=(W / 2, y - 0.08, D / 2)), label))
        else:
            wx, wz, ww, wd = hole
            pieces = [
                (f"{label}_front", 0.0, 0.0, W, wz),
                (f"{label}_back", 0.0, wz + wd, W, D - wz - wd),
                (f"{label}_left", 0.0, wz, wx, wd),
                (f"{label}_right", wx + ww, wz, W - wx - ww, wd),
            ]
            for pname, px, pz, pw, pd in pieces:
                if pw < 0.02 or pd < 0.02:
                    continue
                prims.append(_prim("box", {"size": [pw, 0.16, pd], "material": "stone"},
                                   T(translate=(px + pw / 2, y - 0.08, pz + pd / 2)), pname))

    # ---- staircases -----------------------------------------------------------
    for st_case in plan.stairs:
        _compile_stairs(plan, st_case, prims)

    # ---- columns ----------------------------------------------------------------
    for ci, col in enumerate(plan.columns):
        y0 = plan.floor_base(col.floor)
        prims.append(_prim("box", {"size": [col.size, plan.floor_height - 0.16, col.size],
                                   "material": "stone"},
                           T(translate=(col.x, y0 + (plan.floor_height - 0.16) / 2, col.z)),
                           f"f{col.floor}_column{ci}"))

    # ---- door & window assemblies ------------------------------------------------
    door_parts: list[AnalyticPart] = []
    joints: list[dict] = []
    door_meta: list[dict] = []
    for wall in plan.walls:
        ry = _wall_ry(wall)
        y_base = plan.floor_base(wall.floor)
        for oi, op in enumerate(wall.openings):
            ox, oz = _wall_point(wall, op.offset)
            M = np.asarray(T(translate=(ox, y_base + op.sill, oz), ry=ry), dtype=np.float64)
            prefix = f"{wall.label}_op{oi}_"
            try:
                if op.kind == "door":
                    deco = tuple(op.decorations)
                    if wall.label == plan.entrance.get("wall"):
                        deco = tuple(deco) + ("house_number",)
                    kwargs = dict(width=op.width, height=op.height, style=op.style,
                                  hinge_side=op.hinge_side, decorations=deco,
                                  prefix=prefix)
                    if op.open_method in ("hinged_double", "french") and "hinge_side" in kwargs:
                        kwargs.pop("hinge_side") if op.open_method == "french" else None
                    res = doorlib.build_door(op.open_method, **kwargs)
                else:
                    res = doorlib.build_window(op.open_method if op.open_method in doorlib.WINDOW_BUILDERS else "casement",
                                               width=op.width, height=op.height,
                                               style=op.style if op.style in doorlib.WINDOW_STYLES else "wood",
                                               prefix=prefix)
            except TypeError:
                # builders with different signatures (revolving/garage lack hinge_side)
                safe = {k: v for k, v in dict(width=op.width, height=op.height,
                                              style=op.style, decorations=tuple(op.decorations),
                                              prefix=prefix).items()}
                try:
                    res = doorlib.build_door(op.open_method, **safe)
                except TypeError:
                    safe.pop("decorations", None)
                    safe.pop("style", None)
                    res = doorlib.build_door(op.open_method, **safe)
            door_parts.extend(doorlib.place(res.parts, M))
            joints.extend(res.extras.get("articulation", {}).get("joints", []))
            door_meta.append({"wall": wall.label, "offset": op.offset,
                              **{k: v for k, v in res.metadata.items() if k != "swing"},
                              "swing": res.metadata.get("swing")})

    # ---- exterior detail -----------------------------------------------------------
    _compile_exterior(plan, prims, balcony=balcony)

    # ---- furniture --------------------------------------------------------------------
    if interiors:
        _furnish(plan, prims, enabled=furniture)

    # ---- spec --------------------------------------------------------------------------
    spec = GenerationSpec(shape="building", n_points=2000,
                          bbox_size=(W, plan.floors * plan.floor_height, D),
                          primitives=prims + cutters, seed=plan.seed)

    extras = {
        "physics": {"body_type": "articulated"},
        "articulation": {"joints": joints, "open_method": "building"},
    }
    metadata = {
        "style": plan.style,
        "floors": plan.floors,
        "floor_height": plan.floor_height,
        "footprint": [W, D],
        "rooms": [r.to_dict() for r in plan.rooms],
        "doors": door_meta,
        "door_parts": door_parts,
        "staircases": [s.to_dict() for s in plan.stairs],
        "entrance": plan.entrance,
        "enterable": bool(plan.entrance) and all(
            any(o.kind == "door" for o in w.openings)
            for w in plan.walls if w.label.endswith("corridor_wall")),
        "compile_warnings": warnings,
    }
    return CompiledBuilding(plan=plan, spec=spec, extras=extras, metadata=metadata)


def build_parts(built: CompiledBuilding) -> list[AnalyticPart]:
    """Meshes for the whole building: spec parts + detailed door/window parts."""
    spec_parts, warn = build_spec_meshes_with_report(built.spec)
    built.mesh_warnings = warn
    built.parts = spec_parts + list(built.metadata.get("door_parts", []))
    return built.parts


# ---------------------------------------------------------------------------
# slice view — the '3D-printing slice' per floor
# ---------------------------------------------------------------------------


@dataclass
class SliceResult:
    floors: list[dict]
    scale: float = 55.0               # px per metre in SVG output

    def to_dict(self) -> dict:
        return {"floors": self.floors, "scale": self.scale}


def slice_building(built: CompiledBuilding, heights: list[float] | None = None) -> SliceResult:
    """Horizontal cross-section per floor at `heights` (default 1.2 m above
    each floor slab): wall distribution, openings (door swings!), room
    labels, furniture footprints, stair runs."""
    plan = built.plan
    floors_out: list[dict] = []
    for f in range(plan.floors):
        y_base = plan.floor_base(f)
        cut = (heights[f] if heights and f < len(heights) else 1.2)
        cut_local = cut  # height above floor slab, in wall-local terms

        wall_segs = []
        door_marks = []
        win_marks = []
        for wall in plan.walls:
            if wall.floor != f:
                continue
            L = wall.length
            spans = sorted((o for o in wall.openings
                            if o.sill - 1e-6 < cut_local < o.sill + o.height + 1e-6),
                           key=lambda o: o.offset)
            cursor = 0.0
            for o in spans:
                a, b = o.offset - o.width / 2, o.offset + o.width / 2
                if a > cursor:
                    wall_segs.append({"wall": wall.label, "u": [cursor, a],
                                      "exterior": wall.exterior,
                                      "thickness": wall.thickness,
                                      "start": wall.start, "end": wall.end})
                if o.kind == "door":
                    sector = _door_swing_sector(plan, wall, o)
                    mark = {"offset": o.offset, "width": o.width, "wall": wall.label,
                            "open_method": o.open_method, "start": wall.start,
                            "end": wall.end}
                    if sector:
                        mark["hinge"] = list(sector["hinge"])
                        mark["closed"] = list(sector["closed"])
                        mark["normal"] = list(sector["normal"])
                        mark["radius"] = sector["radius"]
                        mark["rom_deg"] = sector["rom_deg"]
                    door_marks.append(mark)
                else:
                    win_marks.append({"offset": o.offset, "width": o.width,
                                      "wall": wall.label, "start": wall.start,
                                      "end": wall.end})
                cursor = b
            if cursor < L:
                wall_segs.append({"wall": wall.label, "u": [cursor, L],
                                  "exterior": wall.exterior,
                                  "thickness": wall.thickness,
                                  "start": wall.start, "end": wall.end})

        rooms = []
        furn = []
        for r in plan.rooms:
            if r.floor != f:
                continue
            rooms.append({"name": r.name, "rect": list(r.rect),
                          "area": round(r.area, 2), "centroid": list(r.centroid)})
            for item in r.furniture:
                furn.append({"kind": item["kind"], "rect": item["rect"]})

        stairs = []
        for st_case in plan.stairs:
            if st_case.floor_from != f:
                continue
            stairs.append({"kind": st_case.kind,
                           "steps": [s["rect"] for s in st_case.steps],
                           "landings": [l["rect"] for l in st_case.landings],
                           "rise": round(st_case.rise, 4), "going": round(st_case.going, 4)})

        columns = [{"x": c.x, "z": c.z, "size": c.size}
                   for c in plan.columns if c.floor == f]
        floors_out.append({"floor": f, "cut_height": round(y_base + cut, 3),
                           "wall_segments": wall_segs, "doors": door_marks,
                           "windows": win_marks, "rooms": rooms,
                           "furniture": furn, "stairs": stairs, "columns": columns})
    return SliceResult(floors=floors_out)


# ---------------------------------------------------------------------------
# SVG schematic
# ---------------------------------------------------------------------------

_SVG_WALL = "#26221e"
_SVG_WALL_INT = "#4a443d"
_SVG_DOOR = "#b03a2e"
_SVG_WIN = "#2471a3"
_SVG_FURN = "#7d6608"
_SVG_STAIR = "#1e8449"


def _svg_arc(hinge, closed, normal, radius, rom_deg, scale, ox, oy):
    """SVG path for a door swing arc + open leaf line."""
    hx, hz = hinge
    pts = []
    for i in range(0, int(rom_deg) + 1, 10):
        a = math.radians(i)
        ca, sa = math.cos(a), math.sin(a)
        dx = closed[0] * ca + normal[0] * sa
        dz = closed[1] * ca + normal[1] * sa
        pts.append((ox + (hx + dx * radius) * scale, oy + (hz + dz * radius) * scale))
    path = "M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts)
    a100 = math.radians(rom_deg * 0.9)
    lx = hx + (closed[0] * math.cos(a100) + normal[0] * math.sin(a100)) * radius
    lz = hz + (closed[1] * math.cos(a100) + normal[1] * math.sin(a100)) * radius
    line = (f"M {ox + hx * scale:.1f} {oy + hz * scale:.1f} "
            f"L {ox + lx * scale:.1f} {oy + lz * scale:.1f}")
    return path, line


def slice_to_svg(slices: SliceResult, path) -> str:
    """Write all floor slices as one labelled SVG sheet (floors in a row)."""
    from pathlib import Path

    s = slices.scale
    pad = 30.0
    title_h = 46.0
    sheets = []
    x_cursor = pad
    max_h = 0.0
    for fl in slices.floors:
        max_x = max((seg["end"][0] for seg in fl["wall_segments"]), default=10.0)
        max_z = max((seg["end"][1] for seg in fl["wall_segments"]), default=8.0)
        w_px = max_x * s + 2 * pad
        h_px = max_z * s + 2 * pad + title_h
        ox, oy = x_cursor, title_h + pad
        parts = [f'<g font-family="Georgia, serif">']
        parts.append(f'<text x="{x_cursor + 4:.0f}" y="{title_h - 14:.0f}" font-size="17" '
                     f'fill="#1c1a17">floor {fl["floor"]} — slice @ {fl["cut_height"]:.2f} m</text>')
        # wall segments as thick lines
        for seg in fl["wall_segments"]:
            (x0, z0), (x1, z1) = seg["start"], seg["end"]
            L = math.hypot(x1 - x0, z1 - z0) or 1.0
            ux, uz = (x1 - x0) / L, (z1 - z0) / L
            ax, az = x0 + ux * seg["u"][0], z0 + uz * seg["u"][0]
            bx, bz = x0 + ux * seg["u"][1], z0 + uz * seg["u"][1]
            col = _SVG_WALL if seg["exterior"] else _SVG_WALL_INT
            parts.append(f'<line x1="{ox + ax * s:.1f}" y1="{oy + az * s:.1f}" '
                         f'x2="{ox + bx * s:.1f}" y2="{oy + bz * s:.1f}" stroke="{col}" '
                         f'stroke-width="{max(2.0, seg["thickness"] * s):.1f}" stroke-linecap="square"/>')
        # columns
        for c in fl["columns"]:
            parts.append(f'<rect x="{ox + (c["x"] - c["size"] / 2) * s:.1f}" y="{oy + (c["z"] - c["size"] / 2) * s:.1f}" '
                         f'width="{c["size"] * s:.1f}" height="{c["size"] * s:.1f}" fill="#191512"/>')
        # windows
        for wm in fl["windows"]:
            (x0, z0), (x1, z1) = wm["start"], wm["end"]
            L = math.hypot(x1 - x0, z1 - z0) or 1.0
            ux, uz = (x1 - x0) / L, (z1 - z0) / L
            a, b = wm["offset"] - wm["width"] / 2, wm["offset"] + wm["width"] / 2
            for off in (-0.05, 0.05):
                nx, nz = uz * off, -ux * off
                parts.append(f'<line x1="{ox + (x0 + ux * a + nx) * s:.1f}" y1="{oy + (z0 + uz * a + nz) * s:.1f}" '
                             f'x2="{ox + (x0 + ux * b + nx) * s:.1f}" y2="{oy + (z0 + uz * b + nz) * s:.1f}" '
                             f'stroke="{_SVG_WIN}" stroke-width="1.4"/>')
        # door swings
        for dm in fl["doors"]:
            if "hinge" in dm:
                arc, line = _svg_arc(dm["hinge"], dm["closed"], dm["normal"],
                                     dm["radius"], dm["rom_deg"], s, ox, oy)
                parts.append(f'<path d="{arc}" fill="none" stroke="{_SVG_DOOR}" '
                             f'stroke-width="1.3" stroke-dasharray="4 3"/>')
                parts.append(f'<path d="{line}" fill="none" stroke="{_SVG_DOOR}" stroke-width="2.0"/>')
        # stairs
        for stc in fl["stairs"]:
            for rect in stc["steps"] + stc["landings"]:
                parts.append(f'<rect x="{ox + rect[0] * s:.1f}" y="{oy + rect[1] * s:.1f}" '
                             f'width="{rect[2] * s:.1f}" height="{rect[3] * s:.1f}" fill="none" '
                             f'stroke="{_SVG_STAIR}" stroke-width="1.1"/>')
            if stc["steps"]:
                r0 = stc["steps"][0]
                parts.append(f'<text x="{ox + (r0[0] + r0[2] / 2) * s:.1f}" y="{oy + (r0[1] + r0[3] / 2) * s:.1f}" '
                             f'font-size="10" fill="{_SVG_STAIR}" text-anchor="middle">UP</text>')
        # furniture
        for fu in fl["furniture"]:
            rect = fu["rect"]
            parts.append(f'<rect x="{ox + rect[0] * s:.1f}" y="{oy + rect[1] * s:.1f}" '
                         f'width="{rect[2] * s:.1f}" height="{rect[3] * s:.1f}" fill="none" '
                         f'stroke="{_SVG_FURN}" stroke-width="1.0" stroke-dasharray="3 3"/>')
            parts.append(f'<text x="{ox + (rect[0] + rect[2] / 2) * s:.1f}" y="{oy + (rect[1] + rect[3] / 2) * s:.1f}" '
                         f'font-size="8" fill="{_SVG_FURN}" text-anchor="middle">{fu["kind"]}</text>')
        # room labels
        for rm in fl["rooms"]:
            cxp = ox + rm["centroid"][0] * s
            czp = oy + rm["centroid"][1] * s
            parts.append(f'<text x="{cxp:.1f}" y="{czp:.1f}" font-size="13" fill="#211d19" '
                         f'text-anchor="middle" font-weight="bold">{rm["name"]}</text>')
            parts.append(f'<text x="{cxp:.1f}" y="{czp + 13:.1f}" font-size="9.5" fill="#5b544c" '
                         f'text-anchor="middle">{rm["area"]:.1f} m²</text>')
        parts.append("</g>")
        sheets.append("".join(parts))
        x_cursor += w_px
        max_h = max(max_h, h_px)

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{x_cursor + pad:.0f}" '
           f'height="{max_h + pad:.0f}" viewBox="0 0 {x_cursor + pad:.0f} {max_h + pad:.0f}">'
           f'<rect width="100%" height="100%" fill="#faf8f4"/>'
           + "".join(sheets) + "</svg>")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(svg, encoding="utf-8")
    return str(p)


def slice_to_png(slices: SliceResult, path) -> str:
    """Render the slice sheet to PNG via matplotlib (Agg)."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(slices.floors)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.8), squeeze=False)
    fig.patch.set_facecolor("#faf8f4")
    for ax, fl in zip(axes[0], slices.floors):
        ax.set_facecolor("#faf8f4")
        for seg in fl["wall_segments"]:
            (x0, z0), (x1, z1) = seg["start"], seg["end"]
            L = math.hypot(x1 - x0, z1 - z0) or 1.0
            ux, uz = (x1 - x0) / L, (z1 - z0) / L
            ax.plot([x0 + ux * seg["u"][0], x0 + ux * seg["u"][1]],
                    [z0 + uz * seg["u"][0], z0 + uz * seg["u"][1]],
                    color=_SVG_WALL if seg["exterior"] else _SVG_WALL_INT,
                    lw=7 if seg["exterior"] else 4.5, solid_capstyle="butt")
        for c in fl["columns"]:
            ax.add_patch(Rectangle((c["x"] - c["size"] / 2, c["z"] - c["size"] / 2),
                                   c["size"], c["size"], color="#191512"))
        for dm in fl["doors"]:
            if "hinge" not in dm:
                continue
            th = np.linspace(0, math.radians(dm["rom_deg"]), 24)
            hx, hz = dm["hinge"]
            cx, cz = dm["closed"]
            nx, nz = dm["normal"]
            xs = hx + (cx * np.cos(th) + nx * np.sin(th)) * dm["radius"]
            zs = hz + (cz * np.cos(th) + nz * np.sin(th)) * dm["radius"]
            ax.plot(xs, zs, color=_SVG_DOOR, lw=1.1, ls="--")
            a9 = math.radians(dm["rom_deg"] * 0.9)
            ax.plot([hx, hx + (cx * math.cos(a9) + nx * math.sin(a9)) * dm["radius"]],
                    [hz, hz + (cz * math.cos(a9) + nz * math.sin(a9)) * dm["radius"]],
                    color=_SVG_DOOR, lw=1.8)
        for wm in fl["windows"]:
            (x0, z0), (x1, z1) = wm["start"], wm["end"]
            L = math.hypot(x1 - x0, z1 - z0) or 1.0
            ux, uz = (x1 - x0) / L, (z1 - z0) / L
            a, b = wm["offset"] - wm["width"] / 2, wm["offset"] + wm["width"] / 2
            ax.plot([x0 + ux * a, x0 + ux * b], [z0 + uz * a, z0 + uz * b],
                    color=_SVG_WIN, lw=2.2)
        for stc in fl["stairs"]:
            for rect in stc["steps"] + stc["landings"]:
                ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3],
                                       fill=False, ec=_SVG_STAIR, lw=0.9))
            if stc["steps"]:
                r0 = stc["steps"][0]
                ax.text(r0[0] + r0[2] / 2, r0[1] + r0[3] / 2, "UP",
                        color=_SVG_STAIR, fontsize=8, ha="center", va="center")
        for fu in fl["furniture"]:
            rect = fu["rect"]
            ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3],
                                   fill=False, ec=_SVG_FURN, lw=0.9, ls="--"))
            ax.text(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, fu["kind"],
                    color=_SVG_FURN, fontsize=6.5, ha="center", va="center")
        for rm in fl["rooms"]:
            ax.text(rm["centroid"][0], rm["centroid"][1],
                    f"{rm['name']}\n{rm['area']:.1f} m²", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#211d19")
        ax.set_title(f"floor {fl['floor']} — slice @ {fl['cut_height']:.2f} m", fontsize=12)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def build_building(params: dict | None = None, **kw) -> dict:
    """One-call pipeline: plan → validate → compile → meshes → slices.

    ``params`` keys (all optional): seed, floors, width, depth, floor_height,
    style ("baroque" | "neoclassical" | "modern"), rooms_per_floor,
    stair_kind ("straight" | "L" | "U"), min_area, use_subtraction,
    interiors, furniture, balcony.

    Returns a dict: ``plan`` (dict), ``validation`` (report dict), ``built``
    (:class:`CompiledBuilding`), ``spec`` (:class:`GenerationSpec`),
    ``parts`` (meshes), ``extras`` (iemodel articulation), ``slices``
    (:class:`SliceResult`), ``metadata``.
    """
    params = dict(params or {}, **kw)
    plan = generate_plan(
        seed=int(params.get("seed", 0)),
        floors=int(params.get("floors", 2)),
        width=float(params.get("width", 10.0)),
        depth=float(params.get("depth", 8.0)),
        floor_height=float(params.get("floor_height", 3.0)),
        style=str(params.get("style", "neoclassical")),
        rooms_per_floor=params.get("rooms_per_floor"),
        stair_kind=str(params.get("stair_kind", "U")),
        min_area=float(params.get("min_area", MIN_ROOM_AREA)),
        balcony=bool(params.get("balcony", True)),
    )
    report = validate_plan(plan)
    built = compile_building(
        plan,
        use_subtraction=bool(params.get("use_subtraction", True)),
        interiors=bool(params.get("interiors", True)),
        furniture=bool(params.get("furniture", True)),
        balcony=bool(params.get("balcony", True)),
    )
    parts = build_parts(built)
    slices = slice_building(built, heights=params.get("slice_heights"))
    built.metadata["validation"] = report.to_dict()
    built.metadata["n_parts"] = len(parts)
    return {
        "plan": plan.to_dict(),
        "validation": report.to_dict(),
        "built": built,
        "spec": built.spec,
        "parts": parts,
        "extras": built.extras,
        "slices": slices,
        "metadata": built.metadata,
    }
