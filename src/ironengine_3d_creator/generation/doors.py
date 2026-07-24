"""Door & window style library — detailed, articulated, decorated assemblies.

Every builder returns a :class:`DoorResult` bundling

- ``parts`` — world-ready :class:`AnalyticPart` meshes (frame, leaf(s),
  *visible* barrel hinges, handle, track/rollers, moldings, kick plate,
  transom, house-number plaque, muntin grids, sills …),
- ``extras`` — an iemodel/3-style ``articulation`` block (same shape as
  ``generation.soft_author``: ``physics.body_type`` + ``articulation.joints``
  with ``name`` / ``kind`` / ``parent`` / ``child`` / ``axis`` /
  ``limits_deg``), plus ``open_method`` metadata,
- ``metadata`` — swing-arc / travel metadata used by the floor-plan
  validator (hinge point, sweep radius, ROM 0–110°, prismatic travel).

Local frame for every assembly: the opening is centred at ``(0, 0, 0)``,
floor at ``y = 0``, the wall plane is local XY and the leaf closes along
local +X. Use :func:`place` to move an assembly into a building wall.

Door open methods: ``hinged_single`` / ``hinged_double`` (ROM 0–110°),
``sliding`` (track + rollers, prismatic), ``french`` (double hinged,
glazed), ``revolving`` (continuous centre pivot), ``garage`` (sectioned,
inter-section hinges).

Windows: ``casement`` (visible hinge + crank), ``sash`` (double-hung,
prismatic), ``fixed``; all with muntin grids and a sill.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .analytic_mesh import AnalyticPart, build_part_mesh
from .complex_builder import T

# ---------------------------------------------------------------------------
# small mesh helpers
# ---------------------------------------------------------------------------


def _box(label: str, size, center, material: str, *, ry=0.0, rx=0.0, rz=0.0) -> AnalyticPart:
    return build_part_mesh(
        "box", {"size": [float(v) for v in size]},
        T(translate=tuple(float(c) for c in center), ry=ry, rx=rx, rz=rz),
        label, material,
    )


def _cyl(label: str, radius: float, height: float, center, material: str, *, axis="y") -> AnalyticPart:
    rx = rz = 0.0
    if axis == "x":
        rz = math.pi / 2
    elif axis == "z":
        rx = math.pi / 2
    return build_part_mesh(
        "cylinder", {"radius": float(radius), "height": float(height)},
        T(translate=tuple(float(c) for c in center), rx=rx, rz=rz),
        label, material,
    )


def place(parts: list[AnalyticPart], transform: np.ndarray, *, prefix: str = "") -> list[AnalyticPart]:
    """Apply a world transform to an assembly (door/window) built in its
    local opening frame. Optionally prefixes part labels."""
    from .analytic_mesh import apply_transform

    M = np.asarray(transform, dtype=np.float64)
    out: list[AnalyticPart] = []
    for p in parts:
        v, n = apply_transform(p.vertices, p.normals, M)
        out.append(AnalyticPart(
            label=f"{prefix}{p.label}" if prefix else p.label,
            kind=p.kind, material=p.material,
            vertices=v, normals=n, uvs=p.uvs, faces=p.faces,
            aabb_min=v.min(axis=0), aabb_max=v.max(axis=0),
            solid_volume_m3=p.solid_volume_m3,
        ))
    return out


# ---------------------------------------------------------------------------
# result bundle
# ---------------------------------------------------------------------------


@dataclass
class DoorResult:
    """Parts + iemodel articulation + validation metadata for one assembly."""

    parts: list[AnalyticPart]
    extras: dict
    metadata: dict = field(default_factory=dict)

    @property
    def open_method(self) -> str:
        return str(self.metadata.get("open_method", ""))


def _articulation(joints: list[dict], open_method: str) -> dict:
    """iemodel/3 articulation block (mirrors generation.soft_author shape)."""
    return {
        "physics": {"body_type": "articulated"},
        "articulation": {"joints": joints, "open_method": open_method},
    }


# ---------------------------------------------------------------------------
# style tables
# ---------------------------------------------------------------------------

DOOR_STYLES: dict[str, dict] = {
    "panel_wood":     {"leaf": "wood",  "frame": "wood",  "moldings": True,  "kick": False, "glazed": False},
    "glass":          {"leaf": "glass", "frame": "metal", "moldings": False, "kick": False, "glazed": True},
    "metal_security": {"leaf": "metal", "frame": "metal", "moldings": False, "kick": True,  "glazed": False},
}

WINDOW_STYLES: dict[str, dict] = {
    "wood":  {"frame": "wood",  "sash": "wood"},
    "metal": {"frame": "metal", "sash": "metal"},
    "upvc":  {"frame": "plastic", "sash": "plastic"},
}

HINGE_YS = (0.30, 1.05, 1.80)       # barrel-hinge heights on a 2.1 m leaf
ROM_DEG = (0.0, 110.0)              # hinge range of motion


# ---------------------------------------------------------------------------
# shared hardware
# ---------------------------------------------------------------------------


def _barrel_hinge(prefix: str, x: float, y: float, z: float, material: str = "metal") -> list[AnalyticPart]:
    """Visible barrel hinge: pin barrel + two leaf plates."""
    return [
        _cyl(f"{prefix}_barrel", 0.011, 0.085, (x, y, z), material),
        _cyl(f"{prefix}_pintop", 0.016, 0.012, (x, y + 0.049, z), material),
        _cyl(f"{prefix}_pinbot", 0.016, 0.012, (x, y - 0.049, z), material),
        _box(f"{prefix}_plate_leaf", (0.055, 0.085, 0.004), (x + 0.030, y, z), material),
        _box(f"{prefix}_plate_frame", (0.004, 0.085, 0.045), (x, y, z + 0.024), material),
    ]


def _lever_handle(prefix: str, x: float, y: float, z: float, material: str = "metal") -> list[AnalyticPart]:
    """Lever handle: rosette + spindle + lever."""
    return [
        _cyl(f"{prefix}_rosette", 0.024, 0.006, (x, y, z), material, axis="z"),
        _box(f"{prefix}_lever", (0.115, 0.016, 0.014), (x - 0.045, y, z + 0.018), material),
        _box(f"{prefix}_lever_back", (0.115, 0.016, 0.014), (x - 0.045, y, z - 0.018), material),
        _cyl(f"{prefix}_rosette_back", 0.024, 0.006, (x, y, z - 0.036), material, axis="z"),
    ]


def _molded_panels(prefix: str, w: float, h: float, z: float, material: str,
                   *, both_sides: bool = True) -> list[AnalyticPart]:
    """Raised molding frames on the leaf face(s) (panel-wood style).

    `both_sides` (default) molds the back face too, so entrance doors read
    as panelled from outside — hinges stay on the swing side only."""
    parts = []
    rows = [(0.12, h * 0.52), (h * 0.56, h - 0.12)]  # (y0, y1) of the two panels
    faces = [("", z)] + ([("b", -z)] if both_sides else [])
    for ftag, fz in faces:
        for i, (y0, y1) in enumerate(rows):
            ph = y1 - y0
            for side, (mx, mw, mh) in {
                "t": (0.0, w * 0.62, 0.030), "b": (0.0, w * 0.62, 0.030),
            }.items():
                my = y1 - 0.015 if side == "t" else y0 + 0.015
                parts.append(_box(f"{prefix}_mold{ftag}_{i}{side}", (mw, mh, 0.012), (mx, my, fz), material))
            for side, mx in (("l", -w * 0.31 + 0.015), ("r", w * 0.31 - 0.015)):
                parts.append(_box(f"{prefix}_mold{ftag}_{i}{side}", (0.030, ph - 0.06, 0.012), (mx, (y0 + y1) / 2, fz), material))
    return parts


def _muntin_grid(prefix: str, w: float, h: float, cols: int, rows: int, z: float,
                 y0: float, material: str, bar: float = 0.018) -> list[AnalyticPart]:
    """Muntin (glazing-bar) grid over a glazed area."""
    parts = []
    for c in range(1, max(1, cols)):
        x = -w / 2 + w * c / cols
        parts.append(_box(f"{prefix}_muntin_v{c}", (bar, h, bar), (x, y0 + h / 2, z), material))
    for r in range(1, max(1, rows)):
        y = y0 + h * r / rows
        parts.append(_box(f"{prefix}_muntin_h{r}", (w, bar, bar), (0.0, y, z), material))
    return parts


def _transom(prefix: str, w: float, y: float, material_frame: str) -> list[AnalyticPart]:
    """Glazed transom light above the door head."""
    th = 0.35
    parts = [
        _box(f"{prefix}_transom_glass", (w - 0.08, th - 0.08, 0.008), (0.0, y + th / 2, 0.0), "glass"),
        _box(f"{prefix}_transom_rail_t", (w, 0.04, 0.045), (0.0, y + th - 0.02, 0.0), material_frame),
        _box(f"{prefix}_transom_rail_b", (w, 0.04, 0.045), (0.0, y + 0.02, 0.0), material_frame),
        _box(f"{prefix}_transom_stile_l", (0.04, th, 0.045), (-w / 2 + 0.02, y + th / 2, 0.0), material_frame),
        _box(f"{prefix}_transom_stile_r", (0.04, th, 0.045), (w / 2 - 0.02, y + th / 2, 0.0), material_frame),
    ]
    parts += _muntin_grid(f"{prefix}_transom", w - 0.08, th - 0.08, 3, 1, 0.0, y + 0.04, material_frame)
    return parts


def _house_number(prefix: str, text: str, x: float, y: float, z: float) -> list[AnalyticPart]:
    """House-number plaque (geometry) — the digits live in metadata."""
    return [_box(f"{prefix}_number_plaque", (0.16, 0.10, 0.012), (x, y, z), "metal")]


# ---------------------------------------------------------------------------
# doors
# ---------------------------------------------------------------------------


def hinged_door(width: float = 0.92, height: float = 2.10, *, style: str = "panel_wood",
                double: bool = False, hinge_side: str = "left",
                decorations=("moldings",), house_number: str | None = None,
                prefix: str = "door") -> DoorResult:
    """Hinged single/double door: frame, leaf(s), visible barrel hinges,
    lever handle, swing-arc metadata, iemodel hinge joint (ROM 0–110°)."""
    st = DOOR_STYLES.get(style, DOOR_STYLES["panel_wood"])
    fm, lm = st["frame"], st["leaf"]
    parts: list[AnalyticPart] = []
    jw = 0.055                                        # jamb width
    # frame: 2 jambs + head (+ threshold for security doors)
    parts.append(_box(f"{prefix}_jamb_l", (jw, height + 0.02, 0.10), (-width / 2 - jw / 2, (height + 0.02) / 2, 0.0), fm))
    parts.append(_box(f"{prefix}_jamb_r", (jw, height + 0.02, 0.10), (width / 2 + jw / 2, (height + 0.02) / 2, 0.0), fm))
    parts.append(_box(f"{prefix}_head", (width + 2 * jw, jw, 0.10), (0.0, height + 0.02 + jw / 2, 0.0), fm))
    if st["kick"]:
        parts.append(_box(f"{prefix}_threshold", (width + 2 * jw, 0.025, 0.12), (0.0, 0.0125, 0.0), fm))

    joints: list[dict] = []
    leaves = 2 if double else 1
    leaf_w = (width - 0.02) / leaves
    for li in range(leaves):
        side = hinge_side if li == 0 else ("right" if hinge_side == "left" else "left")
        sgn = -1.0 if side == "left" else 1.0
        # leaf closed along +X; left leaf occupies [-w/2, -w/2+leaf_w]
        cx = -width / 2 + leaf_w / 2 + li * leaf_w
        lx = cx - 0.0
        lp = f"{prefix}_leaf{li}"
        if st["glazed"]:
            # glazed leaf: stiles/rails + glass pane + muntins
            parts.append(_box(f"{lp}_stile_l", (0.07, height - 0.04, 0.042), (lx - leaf_w / 2 + 0.035, height / 2, 0.0), lm if not st["glazed"] else fm))
            parts.append(_box(f"{lp}_stile_r", (0.07, height - 0.04, 0.042), (lx + leaf_w / 2 - 0.035, height / 2, 0.0), fm))
            parts.append(_box(f"{lp}_rail_t", (leaf_w - 0.14, 0.07, 0.042), (lx, height - 0.075, 0.0), fm))
            parts.append(_box(f"{lp}_rail_b", (leaf_w - 0.14, 0.12, 0.042), (lx, 0.08, 0.0), fm))
            parts.append(_box(f"{lp}_glass", (leaf_w - 0.16, height - 0.24, 0.008), (lx, height / 2, 0.0), "glass"))
            parts += _muntin_grid(lp, leaf_w - 0.16, height - 0.24, 2, 4, 0.0, 0.12, fm)
        else:
            parts.append(_box(lp, (leaf_w, height - 0.03, 0.044), (lx, (height - 0.03) / 2 + 0.01, 0.0), lm))
            if st["moldings"] and "moldings" in decorations:
                parts += _molded_panels(lp, leaf_w, height - 0.10, 0.028, lm)
        if st["kick"]:
            parts.append(_box(f"{lp}_kickplate", (leaf_w - 0.04, 0.18, 0.004), (lx, 0.11, 0.026), "metal"))
            parts.append(_box(f"{lp}_kickplate_b", (leaf_w - 0.04, 0.18, 0.004), (lx, 0.11, -0.026), "metal"))
        # visible barrel hinges on the hinge jamb
        hx = -width / 2 + (0.0 if side == "left" else width)
        for hi, hy in enumerate(HINGE_YS):
            parts += _barrel_hinge(f"{lp}_hinge{hi}", hx + (-0.012 if side == "left" else 0.012), hy, 0.0)
        # lever handle on the opposite stile
        hx2 = -width / 2 + (leaf_w - 0.09 if side == "left" else width - leaf_w + 0.09)
        parts += _lever_handle(f"{lp}_handle", hx2, 1.02, 0.028)
        hinge_x = hx
        joints.append({
            "name": f"{prefix}_hinge_joint{li}",
            "kind": "revolute",
            "parent": f"{prefix}_jamb_{'l' if side == 'left' else 'r'}",
            "child": lp if not st["glazed"] else f"{lp}_stile_l",
            "axis": [0, 1, 0],
            "origin": [round(hinge_x, 4), 0.0, 0.0],
            "limits_deg": list(ROM_DEG) if side == "left" else [-ROM_DEG[1], -ROM_DEG[0]],
        })

    if "transom" in decorations:
        parts += _transom(prefix, width, height + 0.02 + jw, fm)
    if house_number:
        parts += _house_number(prefix, house_number, 0.0, height * 0.62, 0.055)

    metadata = {
        "open_method": "hinged_double" if double else "hinged_single",
        "style": style,
        "clear_opening": [width, height],
        "swing": {
            "type": "arc",
            "hinge_x": -width / 2 if hinge_side == "left" else width / 2,
            "radius": leaf_w,
            "angle_deg": list(ROM_DEG),
            "side": hinge_side,
        },
        "decorations": list(decorations) + (["house_number"] if house_number else []),
    }
    return DoorResult(parts, _articulation(joints, metadata["open_method"]), metadata)


def sliding_door(width: float = 1.60, height: float = 2.05, *, style: str = "glass",
                 decorations=("track_cover",), prefix: str = "door") -> DoorResult:
    """Sliding door: leaf + overhead track + rollers (prismatic metadata)."""
    st = DOOR_STYLES.get(style, DOOR_STYLES["glass"])
    fm = st["frame"]
    parts: list[AnalyticPart] = []
    # track (double length) + cover + floor guide
    parts.append(_box(f"{prefix}_track", (width * 2, 0.05, 0.06), (width / 2, height + 0.06, 0.0), "metal"))
    if "track_cover" in decorations:
        parts.append(_box(f"{prefix}_track_cover", (width * 2, 0.10, 0.08), (width / 2, height + 0.13, 0.0), fm))
    parts.append(_box(f"{prefix}_floor_guide", (width, 0.012, 0.03), (0.0, 0.006, 0.0), "metal"))
    # leaf (glazed for "glass" style)
    lp = f"{prefix}_leaf0"
    parts.append(_box(f"{lp}_stile_l", (0.06, height, 0.040), (-width / 2 + 0.03, height / 2, 0.0), fm))
    parts.append(_box(f"{lp}_stile_r", (0.06, height, 0.040), (width / 2 - 0.03, height / 2, 0.0), fm))
    parts.append(_box(f"{lp}_rail_t", (width - 0.12, 0.06, 0.040), (0.0, height - 0.03, 0.0), fm))
    parts.append(_box(f"{lp}_rail_b", (width - 0.12, 0.10, 0.040), (0.0, 0.05, 0.0), fm))
    parts.append(_box(f"{lp}_glass", (width - 0.14, height - 0.20, 0.008), (0.0, height / 2, 0.0), "glass"))
    # rollers on the leaf top edge
    for i, rx in enumerate((-width / 4, width / 4)):
        parts.append(_cyl(f"{lp}_roller{i}", 0.028, 0.024, (rx, height + 0.028, 0.0), "metal", axis="z"))
    # recessed pull
    parts.append(_box(f"{lp}_pull", (0.025, 0.16, 0.02), (-width / 2 + 0.035, 1.02, 0.026), "metal"))
    joints = [{
        "name": f"{prefix}_slide_joint",
        "kind": "prismatic",
        "parent": f"{prefix}_track",
        "child": f"{lp}_stile_l",
        "axis": [1, 0, 0],
        "origin": [0.0, 0.0, 0.0],
        "limits_m": [0.0, round(width, 4)],
        "limits_deg": [0, 0],
    }]
    metadata = {
        "open_method": "sliding",
        "style": style,
        "clear_opening": [width, height],
        "swing": {"type": "slide", "travel": width, "axis": [1, 0, 0]},
        "hardware": {"track_length": width * 2, "rollers": 2},
        "decorations": list(decorations),
    }
    return DoorResult(parts, _articulation(joints, "sliding"), metadata)


def french_door(width: float = 1.50, height: float = 2.10, *, style: str = "glass",
                decorations=("muntins",), prefix: str = "door") -> DoorResult:
    """French windows-doors: glazed double hinged leaves with muntin grids."""
    res = hinged_door(width, height, style="glass", double=True,
                      decorations=(), prefix=prefix)
    if "muntins" in decorations:
        res.metadata["decorations"] = ["muntins"]
    res.metadata["open_method"] = "french"
    res.extras["articulation"]["open_method"] = "french"
    res.metadata["style"] = style
    return res


def revolving_door(width: float = 1.80, height: float = 2.20, *, wings: int = 4,
                   style: str = "glass", prefix: str = "door") -> DoorResult:
    """Revolving door: drum enclosure + centre pivot + radiating wings
    (continuous revolute metadata)."""
    parts: list[AnalyticPart] = []
    r = width / 2
    # drum: two half-cylinder enclosure walls (left/right quadrant walls)
    for sgn, tag in ((-1, "l"), (1, "r")):
        parts.append(_cyl(f"{prefix}_drum_{tag}", r + 0.06, 0.10, (0.0, height + 0.05, 0.0), "metal"))
    parts.append(_box(f"{prefix}_canopy", (width + 0.24, 0.10, width + 0.24), (0.0, height + 0.10, 0.0), "metal"))
    # centre pivot
    parts.append(_cyl(f"{prefix}_pivot", 0.045, height, (0.0, height / 2, 0.0), "metal"))
    # wings
    for i in range(max(3, int(wings))):
        a = i * math.pi / max(3, int(wings))
        parts.append(_box(f"{prefix}_wing{i}", (width - 0.08, height - 0.06, 0.012),
                          (0.0, height / 2, 0.0), "glass", ry=a))
        parts.append(_box(f"{prefix}_wing{i}_rail", (width - 0.08, 0.08, 0.03),
                          (0.0, 0.86, 0.0), "metal", ry=a))
    joints = [{
        "name": f"{prefix}_pivot_joint",
        "kind": "revolute",
        "continuous": True,
        "parent": f"{prefix}_pivot",
        "child": f"{prefix}_wing0",
        "axis": [0, 1, 0],
        "origin": [0.0, 0.0, 0.0],
        "limits_deg": [0, 360],
    }]
    metadata = {
        "open_method": "revolving",
        "style": style,
        "clear_opening": [width, height],
        "swing": {"type": "revolve", "radius": r},
        "wings": int(wings),
    }
    return DoorResult(parts, _articulation(joints, "revolving"), metadata)


def garage_door(width: float = 2.40, height: float = 2.10, *, sections: int = 4,
                style: str = "metal_security", prefix: str = "door") -> DoorResult:
    """Sectioned garage door: stacked panels with inter-section hinges and
    side tracks (overhead sectional metadata)."""
    st = DOOR_STYLES.get(style, DOOR_STYLES["metal_security"])
    parts: list[AnalyticPart] = []
    n = max(3, int(sections))
    sh = height / n
    # side tracks + rollers
    for sgn, tag in ((-1, "l"), (1, "r")):
        parts.append(_box(f"{prefix}_track_{tag}", (0.06, height + 0.10, 0.08),
                          (sgn * (width / 2 + 0.06), (height + 0.10) / 2, 0.0), "metal"))
    joints: list[dict] = []
    prev = None
    for i in range(n):
        y = sh * i + sh / 2
        sp = f"{prefix}_section{i}"
        parts.append(_box(sp, (width - 0.04, sh - 0.015, 0.045), (0.0, y, 0.0), st["leaf"]))
        # section ribs
        parts.append(_box(f"{sp}_rib", (width - 0.10, 0.04, 0.006), (0.0, y, 0.026), st["leaf"]))
        for sgn, tag in ((-1, "l"), (1, "r")):
            parts.append(_cyl(f"{sp}_roller_{tag}", 0.022, 0.03,
                              (sgn * (width / 2 + 0.02), y, 0.0), "metal", axis="x"))
        if prev is not None:
            joints.append({
                "name": f"{prefix}_section_hinge{i}",
                "kind": "revolute",
                "parent": prev,
                "child": sp,
                "axis": [1, 0, 0],
                "origin": [0.0, round(sh * i, 4), 0.0],
                "limits_deg": [0, 90],
            })
        prev = sp
    # handle on bottom section
    parts.append(_box(f"{prefix}_lift_handle", (0.20, 0.03, 0.03), (0.0, sh * 0.5, 0.035), "metal"))
    joints.append({
        "name": f"{prefix}_travel",
        "kind": "prismatic",
        "parent": f"{prefix}_track_l",
        "child": f"{prefix}_section{n - 1}",
        "axis": [0, 1, 0],
        "origin": [0.0, 0.0, 0.0],
        "limits_m": [0.0, round(height, 4)],
        "limits_deg": [0, 0],
    })
    metadata = {
        "open_method": "garage",
        "style": style,
        "sections": n,
        "clear_opening": [width, height],
        "swing": {"type": "overhead", "travel": height},
    }
    return DoorResult(parts, _articulation(joints, "garage"), metadata)


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


def _window_frame(prefix: str, w: float, h: float, material: str) -> list[AnalyticPart]:
    fw = 0.05
    return [
        _box(f"{prefix}_frame_l", (fw, h, 0.09), (-w / 2 + fw / 2, h / 2, 0.0), material),
        _box(f"{prefix}_frame_r", (fw, h, 0.09), (w / 2 - fw / 2, h / 2, 0.0), material),
        _box(f"{prefix}_frame_t", (w, fw, 0.09), (0.0, h - fw / 2, 0.0), material),
        _box(f"{prefix}_frame_b", (w, fw, 0.09), (0.0, fw / 2, 0.0), material),
        # protruding sill with slight outward slope
        _box(f"{prefix}_sill", (w + 0.10, 0.045, 0.16), (0.0, -0.0225, 0.02), "stone", rx=math.radians(-6)),
    ]


def casement_window(width: float = 1.10, height: float = 1.40, *, style: str = "wood",
                    muntins=(2, 3), sashes: int = 2, prefix: str = "win") -> DoorResult:
    """Casement window: side-hung sashes with *visible* barrel hinges and a
    crank handle; muntin grid per sash."""
    st = WINDOW_STYLES.get(style, WINDOW_STYLES["wood"])
    parts = _window_frame(prefix, width, height, st["frame"])
    joints: list[dict] = []
    n = max(1, int(sashes))
    sw = (width - 0.10) / n
    for i in range(n):
        sp = f"{prefix}_sash{i}"
        cx = -width / 2 + 0.05 + sw / 2 + i * sw
        parts.append(_box(f"{sp}_glass", (sw - 0.05, height - 0.15, 0.006), (cx, height / 2, 0.0), "glass"))
        for sgn, tag in ((-1, "l"), (1, "r")):
            parts.append(_box(f"{sp}_stile_{tag}", (0.035, height - 0.12, 0.030),
                              (cx + sgn * (sw / 2 - 0.0175), height / 2, 0.0), st["sash"]))
        parts.append(_box(f"{sp}_rail_t", (sw, 0.035, 0.030), (cx, height - 0.0775, 0.0), st["sash"]))
        parts.append(_box(f"{sp}_rail_b", (sw, 0.035, 0.030), (cx, 0.0775, 0.0), st["sash"]))
        parts += _muntin_grid(sp, sw - 0.05, height - 0.15, muntins[0], muntins[1], 0.0, 0.075, st["sash"], bar=0.014)
        # visible hinges on the outer stile
        hinge_x = -width / 2 + 0.05 + (0.0 if i == 0 else sw) + (0.0 if n > 1 else 0.0)
        side = "left" if i == 0 else "right"
        hx = -width / 2 + 0.05 if side == "left" else width / 2 - 0.05
        for hi, hy in enumerate((0.30, height - 0.30)):
            parts += _barrel_hinge(f"{sp}_hinge{hi}", hx, hy, 0.0)
        joints.append({
            "name": f"{sp}_hinge_joint",
            "kind": "revolute",
            "parent": f"{prefix}_frame_{'l' if side == 'left' else 'r'}",
            "child": f"{sp}_stile_{'l' if side == 'left' else 'r'}",
            "axis": [0, 1, 0],
            "origin": [round(hx, 4), 0.0, 0.0],
            "limits_deg": [0, 90] if side == "left" else [-90, 0],
        })
    # crank handle at sill centre
    parts.append(_cyl(f"{prefix}_crank_hub", 0.018, 0.03, (0.0, 0.10, 0.03), "metal", axis="z"))
    parts.append(_box(f"{prefix}_crank_arm", (0.012, 0.09, 0.012), (0.0, 0.06, 0.045), "metal", rx=math.radians(20)))
    metadata = {
        "open_method": "casement",
        "style": style,
        "clear_opening": [width, height],
        "swing": {"type": "arc", "radius": sw, "angle_deg": [0, 90]},
        "muntins": list(muntins),
    }
    return DoorResult(parts, _articulation(joints, "casement"), metadata)


def sash_window(width: float = 1.00, height: float = 1.50, *, style: str = "wood",
                muntins=(2, 2), prefix: str = "win") -> DoorResult:
    """Double-hung sash window: two vertically sliding sashes (prismatic
    metadata) with muntin grids and sill."""
    st = WINDOW_STYLES.get(style, WINDOW_STYLES["wood"])
    parts = _window_frame(prefix, width, height, st["frame"])
    joints: list[dict] = []
    sh = (height - 0.10) / 2
    for i, (y0, z) in enumerate(((0.05, 0.012), (sh + 0.05, -0.012))):
        sp = f"{prefix}_sash{i}"
        parts.append(_box(f"{sp}_glass", (width - 0.15, sh - 0.07, 0.006), (0.0, y0 + sh / 2, z), "glass"))
        for sgn, tag in ((-1, "l"), (1, "r")):
            parts.append(_box(f"{sp}_stile_{tag}", (0.035, sh - 0.02, 0.028),
                              (sgn * (width / 2 - 0.0925), y0 + sh / 2, z), st["sash"]))
        parts.append(_box(f"{sp}_rail_t", (width - 0.15, 0.035, 0.028), (0.0, y0 + sh - 0.0275, z), st["sash"]))
        parts.append(_box(f"{sp}_rail_b", (width - 0.15, 0.035, 0.028), (0.0, y0 + 0.0275, z), st["sash"]))
        parts += _muntin_grid(sp, width - 0.15, sh - 0.07, muntins[0], muntins[1], z, y0 + 0.035, st["sash"], bar=0.014)
        joints.append({
            "name": f"{sp}_slide_joint",
            "kind": "prismatic",
            "parent": f"{prefix}_frame_l",
            "child": f"{sp}_stile_l",
            "axis": [0, 1, 0],
            "origin": [0.0, 0.0, 0.0],
            "limits_m": [0.0, round(sh, 4)],
            "limits_deg": [0, 0],
        })
    # sash lock at meeting rail
    parts.append(_box(f"{prefix}_lock", (0.07, 0.025, 0.03), (0.0, sh + 0.05, 0.02), "metal"))
    metadata = {
        "open_method": "sash",
        "style": style,
        "clear_opening": [width, height],
        "swing": {"type": "slide_vertical", "travel": sh},
        "muntins": list(muntins),
    }
    return DoorResult(parts, _articulation(joints, "sash"), metadata)


def fixed_window(width: float = 1.20, height: float = 1.20, *, style: str = "wood",
                 muntins=(3, 2), prefix: str = "win") -> DoorResult:
    """Fixed picture window: frame + glass + muntin grid + sill (no joints)."""
    st = WINDOW_STYLES.get(style, WINDOW_STYLES["wood"])
    parts = _window_frame(prefix, width, height, st["frame"])
    parts.append(_box(f"{prefix}_glass", (width - 0.10, height - 0.10, 0.006), (0.0, height / 2, 0.0), "glass"))
    parts += _muntin_grid(prefix, width - 0.10, height - 0.10, muntins[0], muntins[1], 0.0, 0.05, st["frame"])
    metadata = {
        "open_method": "fixed",
        "style": style,
        "clear_opening": [width, height],
        "muntins": list(muntins),
    }
    return DoorResult(parts, _articulation([], "fixed"), metadata)


# ---------------------------------------------------------------------------
# registries
# ---------------------------------------------------------------------------

DOOR_BUILDERS = {
    "hinged_single": lambda **kw: hinged_door(double=False, **kw),
    "hinged_double": lambda **kw: hinged_door(double=True, **kw),
    "sliding": sliding_door,
    "french": french_door,
    "revolving": revolving_door,
    "garage": garage_door,
}

WINDOW_BUILDERS = {
    "casement": casement_window,
    "sash": sash_window,
    "fixed": fixed_window,
}


def build_door(open_method: str, **kw) -> DoorResult:
    """Build a door assembly by open method name (see :data:`DOOR_BUILDERS`)."""
    if open_method not in DOOR_BUILDERS:
        raise KeyError(f"unknown door open method {open_method!r}; expected one of {sorted(DOOR_BUILDERS)}")
    return DOOR_BUILDERS[open_method](**kw)


def build_window(kind: str, **kw) -> DoorResult:
    """Build a window assembly by kind (see :data:`WINDOW_BUILDERS`)."""
    if kind not in WINDOW_BUILDERS:
        raise KeyError(f"unknown window kind {kind!r}; expected one of {sorted(WINDOW_BUILDERS)}")
    return WINDOW_BUILDERS[kind](**kw)
