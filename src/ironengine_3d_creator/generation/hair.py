"""Hair for the parametric human: scalp shell, visible hairline, strand clusters.

Hairstyle library (`HAIRSTYLES`): bald, buzz, curly, twin_ponytails,
slicked (gelled), horseshoe (male-pattern baldness), long_straight, bob.

Construction model
------------------
* **Scalp shell** — a lofted cap conforming to the cranium silhouette
  recorded by `human_anatomy.HumanBuilder.head_sections`, tilted so the rim
  sits higher at the front hairline than at the nape. Partial-coverage
  styles (horseshoe) use small *embedded* closed shells: their inner half
  hides inside the skull, only the hair side emerges (lofts can't do open
  partial profiles).
* **Visible hairline** — tiny tapered follicle-direction strokes along the
  front rim (they read as baby hairs / the hairline edge, not a hard seam).
* **Strand clusters** — every lock is an independent tapered-tube loft part
  with a cubic-bezier centreline. Top strands respect a **part line**
  (offset from the midline) and a **crown whorl** (swirl field around a
  whorl centre just right of the crown midline).

Wind physics (per-strand metadata `wind_response`)
--------------------------------------------------
Each strand part carries spring-chain parameters for the Sim to drive in
wind: point masses, rest lengths, angular stiffness, damping ratio and
aerodynamic drag area. Intended mapping from wind speed to tip deflection
(see `WIND_SPEED_MAPPING`):

* indoor still air, v < 0.2 m/s → negligible (< 2°, rest pose dominates)
* desk fan, v = 1–3 m/s → visible sway of long locks (≈ 5–25°); short
  styles (buzz / slicked) stay put
* outdoor gusts, v = 5–15 m/s → long hair streams (≈ 30–85°, near the
  chain's travel limit)

Quasi-static model per chain joint:
``deflection_deg(v) = clamp(0.5 * rho_air * v^2 * C_d * A * L / k_theta, 0, 85)``
with ``rho_air = 1.225 kg/m^3`` and ``C_d ≈ 1.1`` for a hair lock; the Sim
interpolates deflection along the chain from root (stiff) to tip (free).
"""
from __future__ import annotations

import math

import numpy as np

from . import slicer
from .human_anatomy import (CROWN_Y, HAIRLINE_Y, HEAD_CENTER_Y, HumanBuilder,
                            bezier, body_section, head_section, path_slices,
                            spheroid_loft)

TAU = 2.0 * math.pi

HAIRSTYLES = ("bald", "buzz", "curly", "twin_ponytails", "slicked",
              "horseshoe", "long_straight", "bob")

RHO_AIR = 1.225           # kg/m^3
DRAG_CD = 1.1             # drag coefficient of a hair lock
LOCK_LINEAR_DENSITY_G_M = 4.0   # g per metre for a reference 1.6 mm lock

WIND_SPEED_MAPPING: dict = {
    "indoor_still": {
        "v_m_s": (0.0, 0.2), "tip_deflection_deg": (0.0, 2.0),
        "note": "negligible — gravity/rest pose dominates; springs may be skipped",
    },
    "desk_fan": {
        "v_m_s": (1.0, 3.0), "tip_deflection_deg": (5.0, 25.0),
        "note": "visible sway of long locks; buzz/slicked stay put (stiff, low area)",
    },
    "outdoor_gust": {
        "v_m_s": (5.0, 15.0), "tip_deflection_deg": (30.0, 85.0),
        "note": "long hair streams; chain approaches its angular travel limit",
    },
    "model": ("quasi-static drag balance per joint: deflection_deg(v) = "
              "clamp(0.5*rho_air*v^2*C_d*A*L/k_theta, 0, 85), rho_air=1.225 "
              "kg/m^3, C_d≈1.1; interpolate along the chain root→tip"),
}


# ---------------------------------------------------------------------------
# scalp sampling helpers
# ---------------------------------------------------------------------------


def _skin_at(builder: HumanBuilder, y: float) -> tuple[float, float]:
    """Interpolated (hw, hd) of the head skin surface at world height y."""
    secs = builder.head_sections
    if not secs:
        return 0.043 * builder.H, 0.056 * builder.H
    ys = np.array([s[0] for s in secs])
    hw = np.interp(y, ys, [s[1] for s in secs])
    hd = np.interp(y, ys, [s[2] for s in secs])
    return float(hw), float(hd)


def _scalp_point(builder: HumanBuilder, y: float, theta: float,
                 grow: float = 1.035) -> tuple[np.ndarray, np.ndarray]:
    """(point, outward normal) on the scalp at (height y, angle theta)."""
    hw, hd = _skin_at(builder, y)
    x = hw * math.cos(theta) * grow
    z = hd * math.sin(theta) * grow
    n = np.array([math.cos(theta) / max(hw, 1e-6), 0.0,
                  math.sin(theta) / max(hw, 1e-6)])
    n /= np.linalg.norm(n) + 1e-12
    return np.array([x, y, z]), n


# ---------------------------------------------------------------------------
# physics metadata
# ---------------------------------------------------------------------------


def _wind_response(pts: np.ndarray, r0: float, root_local: np.ndarray) -> dict:
    """Spring-chain + drag parameters for one strand lock (see module docs)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = float(seg_len.sum())
    mu = LOCK_LINEAR_DENSITY_G_M * (r0 / 0.0016) ** 2 * 1e-3   # kg/m
    masses = (mu * L * 1e3 / len(pts)) * np.ones(len(pts))     # g per joint
    k_theta = 3e-6 * (r0 / 0.0016) ** 3 / max(L / 0.25, 0.2)   # N·m/rad per joint
    drag_area = L * 2.0 * r0 * 1.5                              # m^2 (clump factor)
    return {
        "model": "spring_chain",
        "segments": int(len(pts) - 1),
        "rest_lengths_m": [float(s) for s in seg_len],
        "point_masses_g": [float(m) for m in masses],
        "angular_stiffness_n_m_rad": float(k_theta),
        "damping_ratio": 0.35,
        "drag_area_m2": float(drag_area),
        "drag_cd": DRAG_CD,
        "anchor_part": "head",
        "root_offset_head_local": [float(c) for c in root_local],
        "length_m": float(L),
        "wind_mapping": "generation.hair.WIND_SPEED_MAPPING",
    }


# ---------------------------------------------------------------------------
# strand + shell builders
# ---------------------------------------------------------------------------


def _add_strand(builder: HumanBuilder, name: str, pts: np.ndarray,
                radii, color, physics: bool = True) -> None:
    seg = builder.seg(7)
    prof = slicer.profile_circle(1.0, seg)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    # strand centrelines are generated root→tip with decreasing y; sort +
    # re-spread as a safety net so the y-axis loft never sees duplicates
    order = np.argsort(-pts[:, 1], kind="stable")
    pts = pts[order]
    pts[:, 1] = np.linspace(pts[:, 1].max(), pts[:, 1].min(), pts.shape[0])
    slices = path_slices(pts, radii, axis="y")
    head_c = HEAD_CENTER_Y * builder.H
    md = {"anatomy": "hair_strand"}
    if physics:
        md["wind_response"] = _wind_response(
            pts, float(np.max(radii)), pts[0] - np.array([0.0, head_c, 0.0]))
    builder.add_loft(name, prof, slices, parent="head", albedo=color,
                     metadata=md)


def _scalp_shell(builder: HumanBuilder, rim_y: float, tilt_deg: float,
                 color, name: str = "hair_scalp", grow: float = 1.045) -> None:
    """Lofted cap from the (tilted) hairline rim up over the crown."""
    H = builder.H
    seg = builder.seg(40)
    crown = CROWN_Y * H + 0.0015 * H
    prof = head_section(1.0, 1.0, seg, n=2.8)
    slices = []
    ys = np.linspace(rim_y, crown, max(4, builder.seg(8)))
    for y in ys:
        hw, hd = _skin_at(builder, y)
        slices.append(slicer.Slice(y - rim_y, (hw * grow, hd * grow)))
    builder.add_loft(name, prof, slices, parent="head", caps=True,
                     translate=(0.0, rim_y, 0.0),
                     rx=-math.radians(tilt_deg),   # front rim rises
                     albedo=color, metadata={"anatomy": "scalp_shell"})


def _hairline_strokes(builder: HumanBuilder, color, count: int) -> None:
    """Follicle-direction baby-hair strokes along the front hairline edge."""
    H = builder.H
    rng = builder.rng
    y_hl = HAIRLINE_Y * H
    hw, hd = _skin_at(builder, y_hl)
    n = max(6, int(round(count * builder.d)))
    seg = builder.seg(5)
    prof = slicer.profile_circle(1.0, seg)
    for i in range(n):
        t = (i + 0.5) / n
        th = math.pi * (0.18 + 0.64 * t)          # front arc only (z > 0)
        x = hw * math.cos(th) * 1.03
        z = hd * math.sin(th) * 1.03
        r = (0.00045 + 0.00015 * rng.random()) * H
        ln = (0.0045 + 0.0015 * rng.random()) * H
        # follicles at the edge point down + slightly forward/outward
        pts = np.array([
            [x, y_hl + 0.5 * ln, z - 0.0006 * H],
            [x * 1.01, y_hl - 0.1 * ln, z + 0.0003 * H],
            [x * 1.015, y_hl - 0.6 * ln, z + 0.0006 * H],
        ])
        slices = path_slices(pts, [r, r * 0.7, r * 0.25], axis="y")
        builder.add_loft(f"hairline_{i:02d}", prof, slices, parent="head",
                         albedo=color, metadata={"anatomy": "hairline_stroke"})


def _flow_strand(builder: HumanBuilder, root: np.ndarray, normal: np.ndarray,
                 length: float, flow: np.ndarray, curl: float = 0.0,
                 n_pts: int = 7) -> np.ndarray:
    """Bezier centreline: root → normal lift → style flow → tip."""
    flow = np.asarray(flow, dtype=np.float64)
    flow /= np.linalg.norm(flow) + 1e-12
    p1 = root + normal * (0.25 * length)
    p2 = root + normal * (0.15 * length) + flow * (0.45 * length)
    p3 = root + normal * (0.05 * length) + flow * length
    pts = bezier(root, p1, p2, p3, n_pts)
    if curl > 0.0:  # lateral helix wiggle ⊥ flow (keeps y monotonic-ish)
        u = np.cross(flow, [0.0, 1.0, 0.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(flow, u)
        v /= np.linalg.norm(v)
        t = np.linspace(0.0, 1.0, n_pts)
        wob = curl * length * 0.12
        off = (np.cos(TAU * 2.2 * t)[:, None] * u + np.sin(TAU * 2.2 * t)[:, None] * v)
        pts = pts + wob * (t * (1 - t) * 4.0)[:, None] * off
    # enforce monotone-decreasing y for the loft
    if pts[-1, 1] < pts[0, 1]:
        pts[:, 1] = np.linspace(pts[0, 1], pts[-1, 1], n_pts)
    return pts


# ---------------------------------------------------------------------------
# style builders — each returns (strand part names, extra metadata)
# ---------------------------------------------------------------------------


def _style_bald(builder: HumanBuilder, color) -> list[str]:
    return []


def _style_buzz(builder: HumanBuilder, color) -> list[str]:
    _scalp_shell(builder, rim_y=0.946 * builder.H, tilt_deg=9.0,
                 color=tuple(c * 0.85 for c in color), grow=1.018)
    _hairline_strokes(builder, color, 14)
    return []


def _roots_grid(builder: HumanBuilder, n_theta: int, y_lo: float,
                y_hi: float, rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Jittered (theta, y) root samples over the cap region."""
    rng = builder.rng
    out = []
    for r in range(rows):
        y = y_lo + (y_hi - y_lo) * (r + 0.5) / rows
        for i in range(n_theta):
            th = TAU * (i + 0.5) / n_theta + rng.normal(0, 0.06)
            pt, n = _scalp_point(builder, y + rng.normal(0, 0.002 * builder.H), th)
            out.append((pt, n))
    return out


def _apply_part_and_whorl(builder: HumanBuilder, root: np.ndarray,
                          normal: np.ndarray, part_x: float | None,
                          whorl: np.ndarray | None) -> np.ndarray:
    """Initial strand direction: surface normal bent by part line + whorl."""
    d = normal.copy()
    if part_x is not None and root[1] > 0.955 * builder.H:
        side = 1.0 if root[0] >= part_x else -1.0
        d = d + np.array([side * 0.9, -0.25, -0.15])
    if whorl is not None:
        flat = root - whorl
        flat[1] = 0.0
        dist = np.linalg.norm(flat)
        if dist < 0.022 * builder.H and dist > 1e-9:
            tang = np.array([-flat[2], 0.0, flat[0]]) / dist     # swirl
            d = d * 0.4 + (flat / dist * 0.5 + tang * 0.9)
    d /= np.linalg.norm(d) + 1e-12
    return d


def _long_straight(builder: HumanBuilder, color, length_m: float,
                   n_theta: int, rows: int,
                   curl: float = 0.0, bob_under: bool = False) -> list[str]:
    H = builder.H
    rng = builder.rng
    names = []
    part_x = -0.006 * H
    whorl = np.array([0.006 * H, 0.988 * H, -0.015 * H])
    roots = _roots_grid(builder, n_theta, 0.948 * H, 0.990 * H, rows)
    for i, (root, normal) in enumerate(roots):
        d0 = _apply_part_and_whorl(builder, root, normal, part_x, whorl)
        # main flow: down, slightly outward at the sides, back at the nape.
        # Front (face-sector) roots are parted to the SIDES so the locks
        # frame the face instead of hanging over it.
        if root[2] > 0.020 * H and root[1] < 0.968 * H:
            side = 1.0 if root[0] >= part_x else -1.0
            out = np.array([side * 0.95, -0.80, -0.10])
        else:
            out = np.array([root[0] * 0.6, -1.0, root[2] * 0.35 - 0.25])
        out /= np.linalg.norm(out)
        ln = length_m * (0.9 + 0.2 * rng.random())
        pts = _flow_strand(builder, root, d0, ln, out, curl=curl,
                           n_pts=max(6, builder.seg(8)))
        if bob_under:  # tips tuck inward toward the neck
            inward = np.array([-root[0], 0.0, -abs(root[2]) * 0.4])
            inward /= np.linalg.norm(inward) + 1e-12
            pts[-1] += inward * 0.018 * H
            pts[-2] += inward * 0.010 * H
            pts[:, 1] = np.linspace(pts[0, 1], pts[-1, 1], pts.shape[0])
        r0 = (0.0019 + 0.0005 * rng.random()) * H
        radii = np.linspace(r0, r0 * 0.22, pts.shape[0])
        name = f"hair_strand_{i:03d}"
        _add_strand(builder, name, pts, radii,
                    _tint(color, rng, 0.10))
        names.append(name)
    return names


def _tint(color, rng, amt: float = 0.08):
    f = 1.0 + rng.uniform(-amt, amt)
    return tuple(min(1.0, max(0.0, c * f)) for c in color)


def _style_long(builder: HumanBuilder, color) -> list[str]:
    n_theta = max(8, int(round(14 * builder.d)))
    rows = max(3, int(round(5 * builder.d)))
    return _long_straight(builder, color, length_m=0.30 * builder.H,
                          n_theta=n_theta, rows=rows)


def _style_bob(builder: HumanBuilder, color) -> list[str]:
    n_theta = max(8, int(round(13 * builder.d)))
    rows = max(2, int(round(4 * builder.d)))
    return _long_straight(builder, color, length_m=0.115 * builder.H,
                          n_theta=n_theta, rows=rows,
                          bob_under=True)


def _style_curly(builder: HumanBuilder, color) -> list[str]:
    n_theta = max(8, int(round(13 * builder.d)))
    rows = max(3, int(round(5 * builder.d)))
    return _long_straight(builder, color, length_m=0.14 * builder.H,
                          n_theta=n_theta, rows=rows,
                          curl=1.0)


def _style_slicked(builder: HumanBuilder, color) -> list[str]:
    """Gelled: short strands combed straight back over the crown."""
    H = builder.H
    rng = builder.rng
    names = []
    n = max(10, int(round(30 * builder.d)))
    for i in range(n):
        t = (i + 0.5) / n
        th = math.pi * (0.15 + 0.70 * t)          # front half of the scalp
        y = (0.958 + 0.030 * rng.random()) * H
        root, normal = _scalp_point(builder, y, th)
        flow = np.array([root[0] * 0.15, -0.35, -1.0])
        pts = _flow_strand(builder, root, normal * 0.4 + np.array([0, 0.5, -0.6]),
                           0.070 * H * (0.85 + 0.3 * rng.random()), flow,
                           n_pts=max(5, builder.seg(6)))
        r0 = (0.0012 + 0.0003 * rng.random()) * H
        radii = np.linspace(r0, r0 * 0.3, pts.shape[0])
        name = f"hair_strand_{i:03d}"
        _add_strand(builder, name, pts, radii, _tint(color, rng, 0.06))
        names.append(name)
    return names


def _style_twin_ponytails(builder: HumanBuilder, color) -> list[str]:
    H = builder.H
    rng = builder.rng
    names: list[str] = []
    n_per = max(7, int(round(14 * builder.d)))
    seg = builder.seg(8)
    for side, sgn in (("l", -1.0), ("r", 1.0)):
        tie = np.array([sgn * 0.052 * H, 0.940 * H, -0.006 * H])
        # gathered strands fan from the tie point down past the shoulder
        for i in range(n_per):
            j = rng.normal(0, 0.004 * H, 3)
            root = tie + j
            tip = tie + np.array([sgn * (0.020 + 0.030 * rng.random()) * H,
                                  -(0.22 + 0.06 * rng.random()) * H,
                                  (-0.010 - 0.03 * rng.random()) * H])
            mid1 = root + np.array([sgn * 0.018 * H, -0.05 * H, -0.006 * H])
            mid2 = root + (tip - root) * 0.55 + np.array([sgn * 0.012 * H, 0, 0])
            pts = bezier(root, mid1, mid2, tip, max(6, builder.seg(8)))
            pts[:, 1] = np.linspace(pts[0, 1], pts[-1, 1], pts.shape[0])
            r0 = (0.0016 + 0.0004 * rng.random()) * H
            radii = np.linspace(r0, r0 * 0.22, pts.shape[0])
            name = f"hair_tail_{side}_{i:02d}"
            _add_strand(builder, name, pts, radii, _tint(color, rng, 0.10))
            names.append(name)
        # hair tie band around the gathered root
        prof = slicer.profile_circle(1.0, seg)
        slices = [slicer.Slice(-0.004 * H, (0.0135 * H, 0.0135 * H)),
                  slicer.Slice(0.004 * H, (0.0135 * H, 0.0135 * H))]
        builder.add_loft(f"hair_tie_{side}", prof, slices, parent="head",
                         translate=tuple(tie), rz=sgn * math.pi / 2,
                         albedo=(0.62, 0.16, 0.18), material="fabric",
                         metadata={"anatomy": "hair_tie"})
    return names


def _style_horseshoe(builder: HumanBuilder, color) -> list[str]:
    """Male-pattern: bare crown, hair ring around the sides + nape.

    Coverage comes from three small closed shells *embedded* in the skull
    (only the hair side emerges) plus short cropped strands hanging off the
    band region.
    """
    H = builder.H
    rng = builder.rng
    seg = builder.seg(10)
    col = _tint(color, rng, 0.0)
    # side patches (above the ears) + nape patch
    for side, sgn in (("l", -1.0), ("r", 1.0)):
        prof, slices = spheroid_loft(0.011 * H, 0.024 * H, 0.028 * H, seg, 5,
                                     margin=0.3)
        builder.add_loft(f"hair_patch_{side}", prof, slices, parent="head",
                         translate=(sgn * 0.0365 * H, 0.922 * H, 0.0),
                         albedo=col, metadata={"anatomy": "scalp_patch"})
    prof, slices = spheroid_loft(0.027 * H, 0.026 * H, 0.011 * H, seg, 5,
                                 margin=0.3)
    builder.add_loft("hair_patch_nape", prof, slices, parent="head",
                     translate=(0.0, 0.918 * H, -0.0435 * H),
                     albedo=col, metadata={"anatomy": "scalp_patch"})
    # cropped strands ringing the band (back 240°, none on top)
    names = []
    n = max(10, int(round(22 * builder.d)))
    for i in range(n):
        th = math.pi * (1.08 + 0.84 * (i + 0.5) / n)   # back/side arc only
        y = (0.915 + 0.025 * rng.random()) * H
        root, normal = _scalp_point(builder, y, th)
        flow = np.array([root[0] * 0.3, -1.0, root[2] * 0.3 - 0.2])
        pts = _flow_strand(builder, root, normal, 0.022 * H, flow,
                           n_pts=max(4, builder.seg(5)))
        r0 = (0.0011 + 0.0003 * rng.random()) * H
        radii = np.linspace(r0, r0 * 0.3, pts.shape[0])
        name = f"hair_strand_{i:03d}"
        _add_strand(builder, name, pts, radii, _tint(color, rng, 0.08))
        names.append(name)
    return names


_STYLE_FUNCS = {
    "bald": _style_bald,
    "buzz": _style_buzz,
    "curly": _style_curly,
    "twin_ponytails": _style_twin_ponytails,
    "slicked": _style_slicked,
    "horseshoe": _style_horseshoe,
    "long_straight": _style_long,
    "bob": _style_bob,
}

# Styles that get a full scalp shell + hairline strokes.
_SHELL_STYLES = ("curly", "twin_ponytails", "slicked", "long_straight", "bob")


def add_hair(builder: HumanBuilder) -> dict:
    """Attach the requested hairstyle to `builder.graph`; return extras."""
    style = builder.p.hair_style
    if style not in HAIRSTYLES:
        raise ValueError(f"hair_style must be one of {HAIRSTYLES}")
    color = builder.p.hair_color

    strand_parts: list[str] = []
    if style in _SHELL_STYLES:
        _scalp_shell(builder, rim_y=0.946 * builder.H, tilt_deg=9.0, color=color)
        _hairline_strokes(builder, color, 16)
    strand_parts = _STYLE_FUNCS[style](builder, color)

    return {
        "hair": {
            "style": style,
            "color": color,
            "strand_parts": list(strand_parts),
            "strand_count": len(strand_parts),
            "wind_speed_mapping": WIND_SPEED_MAPPING,
        }
    }
