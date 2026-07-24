"""Wearable garments for the parametric human — separate swappable parts.

Each garment is a distinct set of named parts (root part named exactly after
the garment: `tshirt`, `shirt`, `pants`, `dress`, `jacket`) with its own
world AABB, parented ("bound") to the Sim bone region it wears:

    tshirt → chest (+ sleeves on upper arms)      shirt → chest (+ collar/buttons)
    pants  → pelvis (+ legs)                      dress → chest (bodice) + skirt
    jacket → chest (open front panels + sleeves)

Cloth shells are open lofts (caps=False at neck / hem / cuffs) offset a
garment-specific *ease* beyond the skin sections. Swapping garments is a
pure rebuild: `swap_garments(spec, ["dress"])` returns a fresh HumanSpec
with identical skeleton/face/hair parts (same params, same seed).
"""
from __future__ import annotations

import math

import numpy as np

from . import slicer
from .human_anatomy import (ANKLE_Y, ELBOW_Y, HIP_HALF_X, HIP_Y, KNEE_Y,
                            SHOULDER_X, SHOULDER_Y, THORAX_Y, WAIST_Y,
                            WRIST_Y, HumanBuilder, body_section, path_slices)

TAU = 2.0 * math.pi

GARMENTS = ("tshirt", "shirt", "pants", "dress", "jacket")

DEFAULT_CLOTH_COLORS: dict[str, tuple[float, float, float]] = {
    "tshirt": (0.86, 0.86, 0.87),
    "shirt":  (0.72, 0.80, 0.89),
    "pants":  (0.24, 0.27, 0.34),
    "dress":  (0.52, 0.20, 0.26),
    "jacket": (0.28, 0.30, 0.28),
}


def _cloth_color(builder: HumanBuilder, garment: str):
    return builder.p.cloth_colors.get(garment, DEFAULT_CLOTH_COLORS[garment])


def _chest_dims(builder: HumanBuilder) -> tuple[float, float]:
    """(half width, half depth) of the clothed chest, mirroring _torso."""
    p, H = builder.p, builder.H
    chest_hw = (0.088 + 0.012 * (1.0 - p.gender)) * H * p.shoulder_scale
    chest_hw *= 0.94 + 0.06 * p.bulk
    chest_hd = 0.060 * H * p.bulk
    return chest_hw, chest_hd


def _torso_shell(builder: HumanBuilder, name: str, color, *, y_top: float,
                 y_hem: float, ease: float, parent: str = "chest",
                 neck_r: float | None = None, metadata: dict | None = None):
    """Open cloth tube around the torso (neck hole at the open top)."""
    H, p = builder.H, builder.p
    seg = builder.seg(36)
    chw, chd = _chest_dims(builder)
    waist_hw = (0.078 - 0.010 * p.gender) * H * (0.5 * (p.bulk + 1.0))
    hip_hw = HIP_HALF_X * p.hip_scale * H + 0.013 * H * p.bulk
    bust = 0.34 * p.bust + 0.10 * (1.0 - p.gender) * p.muscle
    prof = body_section(1.0, 1.0, seg, n=3.0)
    # (y, hw, hd, z_push) stations from neck to hem
    stations = [
        (y_top, neck_r or 0.052 * H, (neck_r or 0.052 * H) * 0.85, 0.0),
        (0.832 * H, SHOULDER_X * p.shoulder_scale * H * 0.88 + ease,
         chd * 0.92 + ease, 0.0),
        (0.790 * H, chw + ease, chd * (1.0 + bust) + ease, 0.010 * H * p.bust),
        (0.740 * H, chw * 0.99 + ease, chd + ease, 0.0),
        (0.690 * H, waist_hw * 1.06 + ease, chd * 0.90 + ease, 0.0),
        (0.640 * H, hip_hw * 1.04 + ease, chd * 1.02 + ease, 0.0),
        (0.580 * H, hip_hw * 1.05 + ease, chd * 1.04 + ease, 0.0),
        (y_hem, hip_hw * 1.08 + ease, chd * 1.06 + ease, 0.0),
    ]
    stations = [s for s in stations if y_hem - 1e-6 < s[0] <= y_top + 1e-6]
    stations.sort(key=lambda s: -s[0])
    slices = [slicer.Slice(y, (hw, hd), offset=(0.0, zp))
              for (y, hw, hd, zp) in stations]
    builder.add_loft(name, prof, slices, parent=parent, caps=False,
                     albedo=color, material="fabric",
                     metadata={"garment": name.split("_")[0],
                               "bind_bone": parent, **(metadata or {})})


def _sleeve(builder: HumanBuilder, name: str, side: str, color, *,
            y_top: float, y_cuff: float, ease: float, bind: str):
    """Open cloth tube over an arm (shoulder → cuff)."""
    H, p = builder.H, builder.p
    seg = builder.seg(20)
    sx = SHOULDER_X * p.shoulder_scale * H
    sgn = -1.0 if side == "l" else 1.0
    r0 = 0.020 * H * p.bulk + ease
    prof = body_section(1.0, 1.0, seg, n=2.8)
    slices = [
        slicer.Slice(y_top, (r0 * 1.32, r0 * 1.26)),
        slicer.Slice((y_top + y_cuff) / 2, (r0 * 1.16, r0 * 1.20)),
        slicer.Slice(y_cuff, (r0 * 1.05, r0 * 1.08)),
    ]
    builder.add_loft(name, prof, slices, parent=bind, caps=False,
                     translate=(sgn * sx, 0.0, 0.0), albedo=color,
                     material="fabric",
                     metadata={"garment": name.split("_")[0], "bind_bone": bind})


# ---------------------------------------------------------------------------
# garments
# ---------------------------------------------------------------------------


def _tshirt(builder: HumanBuilder, color) -> list[str]:
    H = builder.H
    ease = 0.008 * H
    _torso_shell(builder, "tshirt", color, y_top=0.848 * H, y_hem=0.470 * H,
                 ease=ease, neck_r=0.058 * H)
    parts = ["tshirt"]
    for side in ("l", "r"):
        _sleeve(builder, f"tshirt_sleeve_{side}", side, color,
                y_top=0.845 * H, y_cuff=ELBOW_Y * H + 0.055 * H, ease=ease,
                bind=f"upper_arm_{side}")
        parts.append(f"tshirt_sleeve_{side}")
    return parts


def _shirt(builder: HumanBuilder, color) -> list[str]:
    H = builder.H
    ease = 0.009 * H
    _torso_shell(builder, "shirt", color, y_top=0.852 * H, y_hem=0.455 * H,
                 ease=ease, neck_r=0.054 * H)
    parts = ["shirt"]
    for side in ("l", "r"):
        _sleeve(builder, f"shirt_sleeve_{side}", side, color,
                y_top=0.845 * H, y_cuff=WRIST_Y * H + 0.015 * H, ease=ease,
                bind=f"upper_arm_{side}")
        parts.append(f"shirt_sleeve_{side}")
        # cuff band
        seg = builder.seg(12)
        prof = body_section(1.0, 1.0, seg, n=2.8)
        r = 0.020 * H * builder.p.bulk + ease + 0.002 * H
        sx = SHOULDER_X * builder.p.shoulder_scale * H
        sgn = -1.0 if side == "l" else 1.0
        slices = [slicer.Slice(WRIST_Y * H + 0.015 * H, (r, r)),
                  slicer.Slice(WRIST_Y * H - 0.005 * H, (r * 0.97, r * 0.97))]
        builder.add_loft(f"shirt_cuff_{side}", prof, slices,
                         parent=f"lower_arm_{side}", caps=False,
                         translate=(sgn * sx, 0.0, 0.0), albedo=color,
                         material="fabric",
                         metadata={"garment": "shirt", "bind_bone": f"lower_arm_{side}"})
        parts.append(f"shirt_cuff_{side}")
    # collar: stand + fall ring around the neck base
    seg = builder.seg(22)
    prof = body_section(1.0, 1.0, seg, n=3.0)
    neck_r = 0.058 * H
    slices = [
        slicer.Slice(0.846 * H, (neck_r * 1.02, neck_r * 0.90)),
        slicer.Slice(0.856 * H, (neck_r * 1.04, neck_r * 0.92)),
        slicer.Slice(0.862 * H, (neck_r * 1.18, neck_r * 1.05)),   # collar fall
    ]
    builder.add_loft("shirt_collar", prof, slices, parent="chest", caps=False,
                     albedo=color, material="fabric",
                     metadata={"garment": "shirt", "bind_bone": "chest"})
    parts.append("shirt_collar")
    # button placket: thin strip down the front + real button discs
    seg8 = builder.seg(10)
    prof = body_section(0.0075 * H, 0.0016 * H, seg8, n=3.0)
    slices = [slicer.Slice(0.845 * H, (1.0, 1.0)),
              slicer.Slice(0.660 * H, (1.0, 1.0))]
    chw, chd = _chest_dims(builder)
    z_front = chd * 1.05 + 0.012 * H
    builder.add_loft("shirt_placket", prof, slices, parent="chest", caps=True,
                     translate=(0.0, 0.0, z_front), albedo=color,
                     material="fabric",
                     metadata={"garment": "shirt", "bind_bone": "chest"})
    parts.append("shirt_placket")
    seg6 = builder.seg(6)
    prof = slicer.profile_circle(1.0, seg6)
    for i, by in enumerate(np.linspace(0.830 * H, 0.680 * H, 6)):
        slices = [slicer.Slice(-0.0009 * H, (0.0026 * H, 0.0026 * H)),
                  slicer.Slice(0.0009 * H, (0.0026 * H, 0.0026 * H))]
        builder.add_loft(f"shirt_button_{i}", prof, slices, parent="chest",
                         caps=True, translate=(0.0, float(by), z_front + 0.0022 * H),
                         rx=math.pi / 2, albedo=(0.92, 0.92, 0.90),
                         material="ceramic",
                         metadata={"garment": "shirt", "bind_bone": "chest"})
        parts.append(f"shirt_button_{i}")
    return parts


def _pants(builder: HumanBuilder, color) -> list[str]:
    H, p = builder.H, builder.p
    ease = 0.006 * H
    seg = builder.seg(32)
    hip_hw = HIP_HALF_X * p.hip_scale * H + 0.013 * H * p.bulk
    waist_hw = (0.078 - 0.010 * p.gender) * H * (0.5 * (p.bulk + 1.0))
    prof = body_section(1.0, 1.0, seg, n=3.0)
    slices = [
        slicer.Slice(0.645 * H, (waist_hw * 1.06 + ease, 0.056 * H * p.bulk + ease)),
        slicer.Slice(0.600 * H, (hip_hw * 1.00 + ease, 0.062 * H * p.bulk + ease)),
        slicer.Slice(0.545 * H, (hip_hw * 1.01 + ease, 0.063 * H * p.bulk + ease)),
        slicer.Slice(0.515 * H, (hip_hw * 0.90 + ease, 0.058 * H * p.bulk + ease)),
    ]
    builder.add_loft("pants", prof, slices, parent="pelvis", caps=False,
                     albedo=color, material="fabric",
                     metadata={"garment": "pants", "bind_bone": "pelvis"})
    parts = ["pants"]
    # legs down to the ankles (enough ease to swallow the calf bulge)
    for side in ("l", "r"):
        sgn = -1.0 if side == "l" else 1.0
        hx = HIP_HALF_X * p.hip_scale * H
        r_leg = 0.042 * H * p.bulk + 0.011 * H
        seg14 = builder.seg(22)
        prof = body_section(1.0, 1.0, seg14, n=3.0)
        slices = [
            slicer.Slice(0.510 * H, (r_leg * 1.02, r_leg * 1.05)),
            slicer.Slice(KNEE_Y * H, (r_leg * 0.92, r_leg * 1.00)),
            slicer.Slice(0.200 * H, (r_leg * 0.86, r_leg * 1.02)),
            slicer.Slice(0.120 * H, (r_leg * 0.72, r_leg * 0.78)),
            slicer.Slice(ANKLE_Y * H + 0.012 * H, (r_leg * 0.62, r_leg * 0.64)),
        ]
        builder.add_loft(f"pants_leg_{side}", prof, slices,
                         parent=f"upper_leg_{side}", caps=False,
                         translate=(sgn * hx, 0.0, 0.0), albedo=color,
                         material="fabric",
                         metadata={"garment": "pants",
                                   "bind_bone": f"upper_leg_{side}"})
        parts.append(f"pants_leg_{side}")
    return parts


def _dress(builder: HumanBuilder, color) -> list[str]:
    H, p = builder.H, builder.p
    ease = 0.006 * H
    # fitted bodice (main named part) — shoulders to waist
    _torso_shell(builder, "dress", color, y_top=0.848 * H, y_hem=0.620 * H,
                 ease=ease, neck_r=0.062 * H)
    parts = ["dress"]
    # flared skirt with 8-gore pleat modulation, waist → knee
    seg = builder.seg(40)
    th, x, z = None, None, None
    base = body_section(1.0, 1.0, seg, n=3.0)
    ang = np.arctan2(base[:, 1], base[:, 0])
    gore = 1.0 + 0.045 * np.cos(8.0 * ang)
    prof = base * gore[:, None]
    waist_hw = (0.078 - 0.010 * p.gender) * H * (0.5 * (p.bulk + 1.0))
    slices = [
        slicer.Slice(0.625 * H, (waist_hw * 1.10 + ease, 0.058 * H * p.bulk + ease)),
        slicer.Slice(0.540 * H, (waist_hw * 1.28 + 0.020 * H, 0.075 * H + 0.010 * H)),
        slicer.Slice(0.450 * H, (waist_hw * 1.45 + 0.035 * H, 0.088 * H + 0.016 * H)),
        slicer.Slice(0.330 * H, (waist_hw * 1.60 + 0.050 * H, 0.100 * H + 0.022 * H)),
    ]
    builder.add_loft("dress_skirt", prof, slices, parent="pelvis", caps=False,
                     albedo=color, material="fabric",
                     metadata={"garment": "dress", "bind_bone": "pelvis"})
    parts.append("dress_skirt")
    return parts


def _jacket(builder: HumanBuilder, color) -> list[str]:
    H, p = builder.H, builder.p
    ease = 0.012 * H
    chw, chd = _chest_dims(builder)
    waist_hw = (0.078 - 0.010 * p.gender) * H * (0.5 * (p.bulk + 1.0))
    hip_hw = HIP_HALF_X * p.hip_scale * H + 0.013 * H * p.bulk
    # Back + sides: a torso tube whose front is pinched INTO the chest
    # (negative front bulge hides the ring's front inside the body), so the
    # two front panels below read as an open jacket.
    seg = builder.seg(36)
    prof = body_section(1.0, 1.0, seg, n=3.0, front=-0.58)
    stations = [
        (0.850 * H, 0.056 * H, 0.050 * H),
        (0.832 * H, SHOULDER_X * p.shoulder_scale * H * 0.90 + ease, chd * 0.92 + ease),
        (0.790 * H, chw + ease, chd + ease),
        (0.710 * H, chw * 0.99 + ease, chd * 0.97 + ease),
        (0.640 * H, hip_hw * 1.05 + ease, chd * 1.02 + ease),
        (0.545 * H, hip_hw * 1.08 + ease, chd * 1.06 + ease),
    ]
    slices = [slicer.Slice(y, (hw, hd)) for (y, hw, hd) in stations]
    builder.add_loft("jacket", prof, slices, parent="chest", caps=False,
                     albedo=color, material="fabric",
                     metadata={"garment": "jacket", "bind_bone": "chest"})
    parts = ["jacket"]
    # open front panels: flattened lens lofts over the left/right chest,
    # split at the centre front (the jacket opening)
    seg = builder.seg(16)
    for side, sgn in (("l", -1.0), ("r", 1.0)):
        prof = body_section(0.052 * H, 0.0022 * H, seg, n=2.6)
        stations = [  # (y, z at the cloth surface)
            (0.845 * H, chd * 0.55 + ease),
            (0.790 * H, chd * (1.0 + 0.30 * p.bust) + ease),
            (0.710 * H, chd * 0.98 + ease),
            (0.620 * H, chd * 0.95 + ease),
            (0.550 * H, chd * 1.00 + ease),
        ]
        slices = [slicer.Slice(y, (1.0, 1.0),
                               offset=(sgn * (0.058 * H + 0.004 * H), z))
                  for (y, z) in stations]
        builder.add_loft(f"jacket_front_{side}", prof, slices, parent="chest",
                         caps=True, albedo=color, material="fabric",
                         metadata={"garment": "jacket", "bind_bone": "chest"})
        parts.append(f"jacket_front_{side}")
    # sleeves + collar
    for side in ("l", "r"):
        _sleeve(builder, f"jacket_sleeve_{side}", side, color,
                y_top=0.848 * H, y_cuff=WRIST_Y * H + 0.020 * H,
                ease=ease, bind=f"upper_arm_{side}")
        parts.append(f"jacket_sleeve_{side}")
    seg = builder.seg(14)
    prof = body_section(1.0, 1.0, seg, n=3.0)
    neck_r = 0.060 * H
    slices = [
        slicer.Slice(0.848 * H, (neck_r * 1.05, neck_r * 0.92)),
        slicer.Slice(0.860 * H, (neck_r * 1.22, neck_r * 1.06)),
    ]
    builder.add_loft("jacket_collar", prof, slices, parent="chest", caps=False,
                     albedo=color, material="fabric",
                     metadata={"garment": "jacket", "bind_bone": "chest"})
    parts.append("jacket_collar")
    return parts


_GARMENT_FUNCS = {
    "tshirt": _tshirt,
    "shirt": _shirt,
    "pants": _pants,
    "dress": _dress,
    "jacket": _jacket,
}


def add_garments(builder: HumanBuilder) -> dict:
    """Attach every garment listed in `builder.p.clothes`; return extras."""
    worn: dict[str, list[str]] = {}
    for garment in builder.p.clothes:
        g = str(garment).lower()
        if g not in GARMENTS:
            raise ValueError(f"unknown garment {garment!r} (have {GARMENTS})")
        worn[g] = _GARMENT_FUNCS[g](builder, _cloth_color(builder, g))
    return {"clothes": {"garments": list(worn), "parts": worn}}


def swap_garments(spec, clothes) -> "object":
    """Rebuild `spec`'s human wearing `clothes` (same params + seed).

    Skeleton, face and hair parts come out byte-identical; only the garment
    parts change. Returns a fresh `human_anatomy.HumanSpec`.
    """
    from .human_anatomy import build_human

    params = dict(spec.params.__dict__)
    params["clothes"] = tuple(clothes)
    return build_human(params)
