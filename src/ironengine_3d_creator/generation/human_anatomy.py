"""Anatomically detailed parametric human generator (replaces the 'wooden puppet').

`build_human(params)` returns a :class:`HumanSpec` wrapping a
:class:`~ironengine_3d_creator.generation.complex_builder.PartGraph` whose
named parts match the Sim doll bone names defined in
``IronEngine-Sim/src/ironengine_sim/world/skeleton.py`` (read-only reference):
``pelvis, spine, chest, neck, head, clavicle_l/r, upper_arm_l/r,
lower_arm_l/r, hand_l/r, upper_leg_l/r, lower_leg_l/r, foot_l/r``.

Landmark heights replicate skeleton.py's Winter/Dempster fractions exactly
(feet at y=0, facing +Z, arms hanging) so the visual mesh aligns with the
physics skeleton:

    ankle 0.045 · knee 0.291 · hip 0.536 · waist 0.626 · thorax 0.716
    neck 0.846 · head-base 0.891 · shoulder 0.825 · elbow 0.655 · wrist 0.495

Anatomy beyond the puppet:

* FACE — eyeballs with iris/pupil, eyelid panels per eye shape
  (almond / round / hooded / monolid), eyebrows as clusters of tiny tapered
  hair strokes (never painted slabs), lofted nose (bridge → tip → nostril
  wings + nostril parts), ears (helix C-tube + inner ridge + lobe), mouth
  (upper/lower lip lofts, teeth + oral cavity visible when `mouth_open`>0),
  jaw / cheekbone shaping baked into the head cross-section profiles.
* BODY — lofted muscle shells with natural curves: deltoid caps, biceps
  bulge, calf taper, gender-parametric chest (bust) / hip / buttock
  silhouettes, hands with 5 fingers × 3 phalanx segments + fingernails,
  feet with 5 toes + toenails.

Every organic part is a slicing-builder loft (watertight where capped), so
triangle density is fully controllable through the `detail` LOD knob.

Vertex-color realism: each part carries `metadata["albedo"]` (linear 0..1
RGB — skin tones, lip pink, teeth white, nail beds, iris colors).
``HumanSpec.vertex_colors(result)`` expands those into per-vertex COLOR_0
arrays with subtle seeded albedo noise (raw albedo, no baked lighting, per
the W8 export contract in `generation.colorize`).

Hair and clothes live in sibling modules (`generation.hair`,
`generation.clothing`) and are attached by `build_human`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import slicer
from .analytic_mesh import apply_transform
from .complex_builder import BuildResult, PartGraph, T

TAU = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Sim skeleton mirror (IronEngine-Sim .../world/skeleton.py — keep in sync)
# ---------------------------------------------------------------------------

SIM_BONE_NAMES: tuple[str, ...] = (
    "pelvis", "spine", "chest", "neck", "head",
    "clavicle_l", "clavicle_r",
    "upper_arm_l", "upper_arm_r", "lower_arm_l", "lower_arm_r",
    "hand_l", "hand_r",
    "upper_leg_l", "upper_leg_r", "lower_leg_l", "lower_leg_r",
    "foot_l", "foot_r",
)

# Landmark fractions of total height (skeleton.py values).
ANKLE_Y, KNEE_Y, HIP_Y = 0.045, 0.291, 0.536
WAIST_Y, THORAX_Y, NECK_Y = 0.626, 0.716, 0.846
SHOULDER_Y, ELBOW_Y, WRIST_Y = 0.825, 0.655, 0.495
SHOULDER_X, HIP_HALF_X = 0.130, 0.085

# Face landmarks (fractions of H) — anatomical: eyes at half head height.
CHIN_Y, CROWN_Y = 0.870, 1.000
EYE_Y, BROW_Y = 0.935, 0.946
NOSE_TIP_Y, MOUTH_Y, EAR_Y = 0.917, 0.901, 0.930
HAIRLINE_Y = 0.956

HEAD_HW, HEAD_HD = 0.043, 0.056       # half width / half depth of cranium
HEAD_CENTER_Y = 0.5 * (CHIN_Y + CROWN_Y)

EYE_SHAPES = ("almond", "round", "hooded", "monolid")
BODY_TYPES = ("slim", "average", "athletic", "heavy")
DETAIL_SCALES = {"low": 0.62, "medium": 1.0, "high": 1.5}

# ---------------------------------------------------------------------------
# color palettes (linear albedo 0..1)
# ---------------------------------------------------------------------------

SKIN_TONES: dict[str, tuple[float, float, float]] = {
    "porcelain": (0.93, 0.82, 0.74),
    "light":     (0.88, 0.71, 0.60),
    "tan":       (0.76, 0.57, 0.44),
    "olive":     (0.68, 0.54, 0.42),
    "brown":     (0.48, 0.33, 0.24),
    "dark":      (0.30, 0.20, 0.15),
}
EYE_COLORS: dict[str, tuple[float, float, float]] = {
    "brown": (0.30, 0.17, 0.08),
    "hazel": (0.45, 0.32, 0.14),
    "green": (0.28, 0.46, 0.28),
    "blue":  (0.24, 0.44, 0.64),
    "gray":  (0.44, 0.47, 0.50),
}
HAIR_COLORS: dict[str, tuple[float, float, float]] = {
    "black":      (0.055, 0.045, 0.045),
    "dark_brown": (0.16, 0.10, 0.06),
    "brown":      (0.30, 0.19, 0.10),
    "auburn":     (0.42, 0.20, 0.09),
    "blonde":     (0.70, 0.54, 0.30),
    "red":        (0.52, 0.22, 0.10),
    "gray":       (0.52, 0.52, 0.54),
    "white":      (0.84, 0.84, 0.86),
}

LIP_PINK = (0.72, 0.38, 0.38)
TEETH_WHITE = (0.94, 0.93, 0.88)
NAIL_PINK = (0.90, 0.76, 0.72)
MOUTH_CAVITY = (0.32, 0.10, 0.10)
NOSTRIL_DARK = (0.22, 0.12, 0.10)
PUPIL_BLACK = (0.02, 0.02, 0.02)
SCLERA_WHITE = (0.96, 0.95, 0.93)


def resolve_color(value, palette: dict[str, tuple[float, float, float]],
                  default: str) -> tuple[float, float, float]:
    """Accept a palette name or an explicit (r, g, b) triple."""
    if value is None:
        return palette[default]
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in palette:
            raise ValueError(f"unknown color name {value!r} (have {sorted(palette)})")
        return palette[key]
    v = tuple(float(c) for c in value)
    if len(v) != 3:
        raise ValueError(f"color triple expected, got {value!r}")
    return v


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


@dataclass
class HumanParams:
    """Resolved parametric-human description (all real-world scale, metres)."""

    height_m: float = 1.75
    body_type: str = "average"               # slim|average|athletic|heavy
    gender: float = 1.0                      # 0.0 masculine … 1.0 feminine
    skin_tone: tuple[float, float, float] = SKIN_TONES["light"]
    eye_color: tuple[float, float, float] = EYE_COLORS["brown"]
    eye_shape: str = "almond"                # almond|round|hooded|monolid
    hair_style: str = "long_straight"        # see generation.hair.HAIRSTYLES
    hair_color: tuple[float, float, float] = HAIR_COLORS["dark_brown"]
    mouth_open: float = 0.0                  # 0 closed … 1 fully open
    clothes: tuple[str, ...] = ("tshirt", "pants")
    cloth_colors: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    detail: str = "medium"                   # low|medium|high (LOD knob)
    seed: int = 7
    # effective shaping (derived in resolve_params)
    bulk: float = 1.0
    shoulder_scale: float = 1.0
    hip_scale: float = 1.0
    bust: float = 1.0                        # bust prominence 0..1
    muscle: float = 1.0                      # muscle definition 0.7..1.3


def resolve_params(params: dict | HumanParams | None = None, **kw) -> HumanParams:
    """Merge a dict / HumanParams / kwargs into a validated HumanParams."""
    raw: dict = {}
    if isinstance(params, HumanParams):
        raw.update(params.__dict__)
    elif isinstance(params, dict):
        raw.update(params)
    elif params is not None:
        raise TypeError(f"params must be dict|HumanParams|None, got {type(params)!r}")
    raw.update(kw)

    body_type = str(raw.get("body_type", "average")).lower()
    if body_type not in BODY_TYPES:
        raise ValueError(f"body_type must be one of {BODY_TYPES}")
    bulk, shoulder, hip, muscle = {
        "slim":     (0.84, 0.94, 0.95, 0.85),
        "average":  (1.00, 1.00, 1.00, 1.00),
        "athletic": (1.07, 1.09, 0.98, 1.25),
        "heavy":    (1.26, 1.04, 1.12, 0.75),
    }[body_type]

    gender = raw.get("gender", 1.0)
    if isinstance(gender, str):
        g = gender.strip().lower()
        gender = {"male": 0.0, "m": 0.0, "female": 1.0, "f": 1.0,
                  "neutral": 0.5, "androgynous": 0.5}[g]
    gender = float(min(1.0, max(0.0, gender)))
    # Gender shapes the silhouette envelope (kept inside the Sim skeleton's
    # shoulder/hip scale ranges so the physics proxy still fits).
    shoulder *= 1.0 + (0.09 * (1.0 - gender) - 0.06 * gender)
    hip *= 1.0 + (0.09 * gender - 0.04 * (1.0 - gender))
    bust = float(raw.get("bust", gender))
    bust = float(min(1.0, max(0.0, bust)))

    eye_shape = str(raw.get("eye_shape", "almond")).lower()
    if eye_shape not in EYE_SHAPES:
        raise ValueError(f"eye_shape must be one of {EYE_SHAPES}")
    detail = str(raw.get("detail", "medium")).lower()
    if detail not in DETAIL_SCALES:
        raise ValueError(f"detail must be one of {tuple(DETAIL_SCALES)}")

    return HumanParams(
        height_m=float(raw.get("height_m", 1.75)),
        body_type=body_type,
        gender=gender,
        skin_tone=resolve_color(raw.get("skin_tone"), SKIN_TONES, "light"),
        eye_color=resolve_color(raw.get("eye_color"), EYE_COLORS, "brown"),
        eye_shape=eye_shape,
        hair_style=str(raw.get("hair_style", "long_straight")).lower(),
        hair_color=resolve_color(raw.get("hair_color"), HAIR_COLORS, "dark_brown"),
        mouth_open=float(min(1.0, max(0.0, raw.get("mouth_open", 0.0)))),
        clothes=tuple(raw.get("clothes", ("tshirt", "pants"))),
        cloth_colors=dict(raw.get("cloth_colors", {})),
        detail=detail,
        seed=int(raw.get("seed", 7)),
        bulk=float(raw.get("bulk", bulk)),
        shoulder_scale=float(raw.get("shoulder_scale", shoulder)),
        hip_scale=float(raw.get("hip_scale", hip)),
        bust=bust,
        muscle=float(raw.get("muscle", muscle)),
    )


# ---------------------------------------------------------------------------
# profile / slice helpers (all lofts, all controllable tessellation)
# ---------------------------------------------------------------------------


def _superellipse(a: float, b: float, seg: int, n: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(theta, x, y) for |x/a|^n + |y/b|^n = 1, CCW."""
    th = np.linspace(0.0, TAU, max(5, int(seg)), endpoint=False)
    c, s = np.cos(th), np.sin(th)
    x = a * np.sign(c) * np.abs(c) ** (2.0 / n)
    y = b * np.sign(s) * np.abs(s) ** (2.0 / n)
    return th, x, y


def body_section(hw: float, hd: float, seg: int, n: float = 3.2,
                 front: float = 0.0, back: float = 0.0, side: float = 0.0,
                 front_push: float = 0.0) -> np.ndarray:
    """Anatomical horizontal cross-section (x = width, z = depth; +z = front).

    front/back: fractional depth bulges on the front/back halves (bust,
    buttocks, calves); side: fractional width bulge (deltoids); front_push:
    forward z-shift weighted toward the midline (bust point, nose of a
    profile). All bulge values are fractions of the half-depth/width.
    """
    th, x, z = _superellipse(hw, hd, seg, n)
    s = np.sin(th)
    wf = np.clip(s, 0.0, 1.0) ** 2
    wb = np.clip(-s, 0.0, 1.0) ** 2
    z = z * (1.0 + front * wf + back * wb) + front_push * np.clip(s, 0.0, 1.0) ** 3
    x = x * (1.0 + side * np.clip(np.abs(np.cos(th)), 0.0, 1.0) ** 2)
    return np.stack([x, z], axis=-1)


def head_section(hw: float, hd: float, seg: int, n: float = 2.8,
                 cheek: float = 0.0, jaw_wide: float = 0.0) -> np.ndarray:
    """Skull/face cross-section; cheek = zygomatic side push (front-side),
    jaw_wide = extra width on the lower face sides (masseter region)."""
    th, x, z = _superellipse(hw, hd, seg, n)
    s, c = np.sin(th), np.cos(th)
    wf = np.clip(s, 0.0, 1.0)
    x = x * (1.0 + cheek * wf ** 2 * np.clip(np.abs(c), 0.0, 1.0)
             + jaw_wide * np.clip(-s, 0.0, 1.0) * np.clip(np.abs(c), 0.0, 1.0))
    return np.stack([x, z], axis=-1)


def spheroid_loft(rx: float, ry: float, rz: float, seg: int, rings: int,
                  margin: float = 0.10) -> tuple[np.ndarray, list[slicer.Slice]]:
    """Lathe a spheroid along Y: circle profile + radius slices."""
    profile = slicer.profile_circle(1.0, seg)
    phis = np.linspace(-math.pi / 2 + margin, math.pi / 2 - margin, max(3, rings))
    slices = [slicer.Slice(position=float(ry * math.sin(p)),
                           scale=(max(rx * math.cos(p), 1e-4),
                                  max(rz * math.cos(p), 1e-4)))
              for p in phis]
    return profile, slices


def path_slices(pts: np.ndarray, radii, axis: str = "y") -> list[slicer.Slice]:
    """Slices that follow a 3D polyline (must be monotonic along `axis`)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (pts.shape[0],))
    ai = "xyz".index(axis)
    oi = [i for i in range(3) if i != ai]
    return [slicer.Slice(position=float(p[ai]), scale=(float(max(r, 1e-4)),) * 2,
                         offset=(float(p[oi[0]]), float(p[oi[1]])))
            for p, r in zip(pts, radii)]


def bezier(p0, p1, p2, p3, n: int) -> np.ndarray:
    """Cubic bezier points (n, 3)."""
    t = np.linspace(0.0, 1.0, max(2, n))[:, None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------


class HumanBuilder:
    """Accumulates loft nodes into a PartGraph with bone-name parts."""

    def __init__(self, params: HumanParams):
        self.p = params
        self.H = params.height_m
        self.d = DETAIL_SCALES[params.detail]
        self.rng = np.random.default_rng(params.seed)
        self.graph = PartGraph("human")
        self.worlds: dict[str, np.ndarray] = {}
        self.head_sections: list[tuple[float, float, float]] = []  # (y, hw, hd)
        self.scalp_top_y = CROWN_Y * self.H

    # -- placement ----------------------------------------------------------
    def seg(self, base: int) -> int:
        return max(5, int(round(base * self.d)))

    def add_loft(self, name: str, profile: np.ndarray, slices: list[slicer.Slice],
                 *, axis: str = "y", caps: bool = True, parent: str | None = None,
                 translate=(0.0, 0.0, 0.0), ry: float = 0.0, rx: float = 0.0,
                 rz: float = 0.0, material: str = "organic",
                 albedo=LIP_PINK, metadata: dict | None = None) -> None:
        md = dict(metadata or {})
        md["albedo"] = tuple(float(c) for c in albedo)
        node = self.graph.add_loft(name, profile, slices, axis=axis, caps=caps,
                                   material=material, parent=parent,
                                   metadata=md)
        world = T(translate=translate, ry=ry, rx=rx, rz=rz)
        if parent is not None:
            node.local = np.linalg.inv(self.worlds[parent]) @ world
            self.worlds[name] = self.worlds[parent] @ node.local
        else:
            node.local = world
            self.worlds[name] = world

    # -- whole body ---------------------------------------------------------
    def build(self) -> None:
        self._torso()
        self._neck()
        self._head()
        self._arms()
        self._hands()
        self._legs()
        self._feet()
        self._face()

    # ------------------------------------------------------------------ torso
    def _torso(self) -> None:
        H, p = self.H, self.p
        seg = self.seg(44)
        hip_half = HIP_HALF_X * p.hip_scale * H
        pelvis_c = (HIP_Y + 0.045) * H                    # skeleton _PELVIS_C

        # pelvis: hip flare + buttock back bulge, narrowing to the waist
        prof = body_section(1.0, 1.0, seg, n=3.0)
        pw = hip_half + 0.013 * H * p.bulk
        pd = 0.060 * H * p.bulk
        slices = [
            slicer.Slice(-0.055 * H, (pw * 0.86, pd * 0.92), offset=(0, -0.004 * H)),   # glute fold
            slicer.Slice(-0.030 * H, (pw * 1.00, pd * 1.00), offset=(0, -0.006 * H)),   # hip joints
            slicer.Slice(-0.008 * H, (pw * 1.02, pd * 1.02), offset=(0, -0.002 * H)),   # trochanter max
            slicer.Slice(0.018 * H, (pw * 0.97, pd * 0.98)),
            slicer.Slice(0.045 * H, (pw * 0.88, pd * 0.90)),                            # iliac crest
        ]
        # buttock prominence (posterior = -z): female-leaning curve
        butt = 0.22 + 0.18 * p.gender
        slices = [slicer.Slice(s.position, s.scale, s.rotation,
                               (s.offset[0], s.offset[1] - butt * 0.012 * H
                                * math.exp(-((s.position + 0.030 * H) / (0.025 * H)) ** 2)))
                  for s in slices]
        self.add_loft("pelvis", prof, slices, parent=None,
                      translate=(0.0, pelvis_c, 0.0), albedo=self.p.skin_tone)

        # spine / waist: narrow, gender-tapered
        waist_hw = (0.078 - 0.010 * p.gender) * H * (0.5 * (p.bulk + 1.0))
        waist_hd = 0.052 * H * p.bulk
        spine_c = (WAIST_Y + 0.045) * H
        prof = body_section(1.0, 1.0, seg, n=3.0)
        slices = [
            slicer.Slice(-0.045 * H, (waist_hw * 1.06, waist_hd * 1.02)),
            slicer.Slice(-0.020 * H, (waist_hw * 0.96, waist_hd * 0.96)),   # waist min
            slicer.Slice(0.005 * H, (waist_hw * 1.00, waist_hd)),
            slicer.Slice(0.045 * H, (waist_hw * 1.10, waist_hd * 1.06)),
        ]
        self.add_loft("spine", prof, slices, parent="pelvis",
                      translate=(0.0, spine_c, 0.0), albedo=self.p.skin_tone)

        # chest: thorax up to the neck; female bust via front bulge slices,
        # male pecs via a shallow uniform front bulge.
        chest_c = (THORAX_Y + 0.065) * H                  # skeleton _CHEST_C
        chest_hw = (0.088 + 0.012 * (1.0 - p.gender)) * H * p.shoulder_scale
        chest_hw *= 0.94 + 0.06 * p.bulk
        chest_hd = 0.060 * H * p.bulk
        bust_amt = 0.34 * p.bust + 0.10 * (1.0 - p.gender) * p.muscle
        stations = [(-0.065, 0.86, 0.94, 0.00), (-0.040, 0.94, 0.98, 0.30),
                    (-0.015, 1.00, 1.00, 0.85), (0.008, 1.01, 1.02, 1.00),
                    (0.030, 1.00, 1.00, 0.55), (0.050, 0.99, 0.97, 0.10),
                    (0.065, 0.97, 0.94, 0.00)]
        # Slicer lofts share ONE profile, so the per-station shaping is baked
        # into (scale, offset): depth scale carries the bust/pec bulge and the
        # z-offset pushes it forward.
        prof = body_section(1.0, 1.0, seg, n=3.0)
        slices = []
        for dy, fw, fd, bw in stations:
            slices.append(slicer.Slice(
                dy * H, (chest_hw * fw, chest_hd * fd * (1.0 + bust_amt * bw)),
                offset=(0.0, 0.010 * H * p.bust * bw - 0.002 * H)))
        # shoulder slope: widen top toward the deltoids
        slices.append(slicer.Slice(0.066 * H, (SHOULDER_X * p.shoulder_scale * H * 0.86,
                                               chest_hd * 0.90)))
        self.add_loft("chest", prof, slices, parent="spine",
                      translate=(0.0, chest_c, 0.0), albedo=self.p.skin_tone)

        # clavicles: slim tubes from sternum to the shoulder joints
        seg8 = self.seg(12)
        prof = slicer.profile_circle(1.0, seg8)
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            x0, x1 = 0.030 * H, SHOULDER_X * p.shoulder_scale * H
            y = SHOULDER_Y * H
            pts = np.array([[sgn * x0, y + 0.006 * H, 0.028 * H],
                            [sgn * (x0 + x1) / 2, y + 0.010 * H, 0.024 * H],
                            [sgn * x1, y, 0.008 * H]])
            r = 0.011 * H * p.bulk
            slices = path_slices(pts, [r * 0.8, r, r * 0.9], axis="x")
            self.add_loft(f"clavicle_{side}", prof, slices, axis="x", parent="chest",
                          albedo=self.p.skin_tone)

    # ------------------------------------------------------------------- neck
    def _neck(self) -> None:
        H, p = self.H, self.p
        seg = self.seg(24)
        neck_c = (NECK_Y + 0.0225) * H
        r = 0.022 * H * (0.9 + 0.1 * p.bulk)
        prof = body_section(1.0, 1.0, seg, n=2.6)
        slices = [
            slicer.Slice(-0.024 * H, (r * 1.55, r * 1.45), offset=(0, -0.010 * H)),  # trapezius
            slicer.Slice(-0.010 * H, (r * 1.15, r * 1.10), offset=(0, -0.003 * H)),
            slicer.Slice(0.008 * H, (r, r * 1.05)),
            slicer.Slice(0.024 * H, (r * 0.92, r * 0.95)),
        ]
        self.add_loft("neck", prof, slices, parent="chest",
                      translate=(0.0, neck_c, 0.0), albedo=self.p.skin_tone)

    # ------------------------------------------------------------------- head
    def _head(self) -> None:
        H, p = self.H, self.p
        seg = self.seg(56)
        hw, hd = HEAD_HW * H, HEAD_HD * H
        jaw = 0.10 * (1.0 - p.gender)                     # squarer male jaw
        cheek = 0.05 + 0.06 * p.gender                    # zygomatic push
        # (y, half-width, half-depth, cheek, jaw) — chin → crown, local frame
        # centred at HEAD_CENTER_Y so the loft's local coords are symmetric.
        stations = [
            (CHIN_Y, 0.52, 0.72, 0.00, jaw * 0.6),        # chin
            (0.886, 0.72, 0.82, 0.02, jaw),               # jaw line
            (0.905, 0.86, 0.90, 0.05, jaw),               # mouth level
            (0.925, 0.97, 0.97, cheek, 0.03),             # cheekbones
            (0.935, 1.00, 1.00, cheek, 0.00),             # eye level (max)
            (0.948, 0.97, 1.00, 0.03, 0.00),              # brow ridge
            (0.962, 0.94, 0.98, 0.00, 0.00),              # forehead
            (0.978, 0.86, 0.92, 0.00, 0.00),              # upper cranium
            (0.990, 0.66, 0.74, 0.00, 0.00),              # crown curve
            (0.997, 0.34, 0.40, 0.00, 0.00),              # crown
        ]
        self.head_sections = []
        for y, fw, fd, ck, jw in stations:
            # store for the hair module (world y, hw, hd of the skin surface)
            self.head_sections.append((y * H, hw * fw, hd * fd))
        # Slicer lofts share ONE profile: the unit head section carries the
        # averaged cheek/jaw shaping and each slice scales it to the station's
        # (hw, hd) outline.
        prof = head_section(1.0, 1.0, seg, n=2.8,
                            cheek=cheek * 0.6, jaw_wide=jaw * 0.6)
        slices = [
            slicer.Slice((y - HEAD_CENTER_Y) * H, (hw * fw, hd * fd))
            for (y, fw, fd, ck, jw) in stations
        ]
        self.add_loft("head", prof, slices, parent="neck",
                      translate=(0.0, HEAD_CENTER_Y * H, 0.0),
                      albedo=self.p.skin_tone,
                      metadata={"anatomy": "skull_jaw_cheekbones"})

    # ------------------------------------------------------------------- arms
    def _arms(self) -> None:
        H, p = self.H, self.p
        seg = self.seg(28)
        sx = SHOULDER_X * p.shoulder_scale * H
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            # upper arm: deltoid cap → biceps bulge → elbow
            r0 = 0.020 * H * p.bulk
            prof = body_section(1.0, 1.0, seg, n=2.8)
            c = 0.5 * (SHOULDER_Y + ELBOW_Y) * H
            delt = 1.55 + 0.25 * (p.muscle - 1.0)
            slices = [
                slicer.Slice(0.100 * H, (r0 * delt, r0 * delt * 0.95), offset=(sgn * 0.004 * H, 0)),  # deltoid cap top
                slicer.Slice(0.085 * H, (r0 * delt * 1.02, r0 * delt), offset=(sgn * 0.006 * H, 0)),   # deltoid max
                slicer.Slice(0.045 * H, (r0 * 1.18, r0 * 1.28), offset=(0, 0.002 * H)),                # biceps front
                slicer.Slice(0.000 * H, (r0 * 1.12, r0 * 1.22), offset=(0, 0.002 * H)),
                slicer.Slice(-0.050 * H, (r0 * 1.02, r0 * 1.06)),
                slicer.Slice(-0.085 * H, (r0 * 0.95, r0 * 0.95)),                                      # elbow
            ]
            self.add_loft(f"upper_arm_{side}", prof, slices, parent=f"clavicle_{side}",
                          translate=(sgn * sx, c, 0.0), albedo=self.p.skin_tone,
                          metadata={"anatomy": "deltoid_biceps"})
            # forearm: extensor mass near elbow → slim wrist
            prof = body_section(1.0, 1.0, seg, n=2.8)
            c = 0.5 * (ELBOW_Y + WRIST_Y) * H
            slices = [
                slicer.Slice(0.080 * H, (r0 * 0.98, r0)),
                slicer.Slice(0.050 * H, (r0 * 1.10, r0 * 1.14), offset=(0, -0.001 * H)),   # extensor
                slicer.Slice(0.000 * H, (r0 * 0.92, r0 * 0.95)),
                slicer.Slice(-0.050 * H, (r0 * 0.76, r0 * 0.78)),
                slicer.Slice(-0.080 * H, (r0 * 0.66, r0 * 0.68)),                          # wrist
            ]
            self.add_loft(f"lower_arm_{side}", prof, slices, parent=f"upper_arm_{side}",
                          translate=(sgn * sx, c, 0.0), albedo=self.p.skin_tone)

    # ------------------------------------------------------------------ hands
    def _hands(self) -> None:
        H, p = self.H, self.p
        seg = self.seg(18)
        sx = SHOULDER_X * p.shoulder_scale * H
        palm_len, palm_hw, palm_ht = 0.056 * H, 0.021 * H, 0.0085 * H
        wrist_y = WRIST_Y * H
        finger_len = {"middle": 0.050, "ring": 0.046, "index": 0.044,
                      "thumb": 0.036, "pinky": 0.034}
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            palm_c = np.array([sgn * sx, wrist_y - palm_len * 0.55, 0.0])
            prof = body_section(palm_hw, palm_ht, seg, n=3.6)
            slices = [
                slicer.Slice(0.5 * palm_len, (0.86, 0.92)),      # wrist edge
                slicer.Slice(0.15 * palm_len, (1.0, 1.02)),
                slicer.Slice(-0.25 * palm_len, (1.04, 1.00)),    # knuckles
                slicer.Slice(-0.5 * palm_len, (0.94, 0.88)),     # finger bases
            ]
            # Palm slab: width along z (medial palm faces the thigh), thin x.
            self.add_loft(f"hand_{side}", prof, slices, parent=f"lower_arm_{side}",
                          translate=tuple(palm_c), ry=(math.pi / 2 * sgn),
                          albedo=self.p.skin_tone,
                          metadata={"anatomy": "palm"})
            base_y = palm_c[1] - 0.5 * palm_len
            # finger layout: (name, z offset from palm centre, base radius)
            fingers = [("index", 0.0165, 0.0052), ("middle", 0.0055, 0.0055),
                       ("ring", -0.0055, 0.0052), ("pinky", -0.0155, 0.0044)]
            for fname, zoff, fr in fingers:
                self._finger(side, sgn, fname, fr * H,
                             np.array([palm_c[0], base_y, zoff * H]),
                             finger_len[fname] * H, curl_sgn=sgn)
            # thumb: on the front (+z) edge, angled forward/outward
            self._finger(side, sgn, "thumb", 0.0058 * H,
                         np.array([palm_c[0], base_y + 0.010 * H, 0.021 * H]),
                         finger_len["thumb"] * H, curl_sgn=sgn, thumb=True)

    def _finger(self, side: str, sgn: float, fname: str, fr: float,
                base: np.ndarray, flen: float, curl_sgn: float,
                thumb: bool = False) -> None:
        seg = self.seg(9)
        prof = slicer.profile_circle(1.0, seg)
        # 3 phalanx segments (mission spec: 3 for every finger incl. thumb)
        splits = (0.42, 0.33, 0.25)
        curls = (8.0, 20.0, 34.0) if not thumb else (4.0, 12.0, 20.0)
        pos = base.copy()
        for k in range(3):
            seg_len = flen * splits[k]
            ang = math.radians(sum(curls[: k + 1])) * curl_sgn
            # segment direction: hanging down, curling toward ±x about Z
            direction = np.array([math.sin(ang), -math.cos(ang), 0.0])
            r0, r1 = fr * (1.0 - 0.12 * k), fr * (1.0 - 0.12 * (k + 1))
            # loft built downward (local −y) so rz=ang maps +y′→direction
            slices = [
                slicer.Slice(-seg_len, (max(r1, 1e-4), max(r1 * 0.90, 1e-4))),
                slicer.Slice(-seg_len * 0.45, (r0 * 0.98, r0 * 0.92)),
                slicer.Slice(0.0, (r0, r0 * 0.94)),
            ]
            self.add_loft(f"finger_{side}_{fname}_{k + 1}", prof, slices,
                          parent=f"hand_{side}",
                          translate=tuple(pos),
                          rz=ang, albedo=self.p.skin_tone,
                          metadata={"anatomy": f"phalanx_{k + 1}"})
            pos = pos + direction * seg_len
        # fingernail on the dorsal side (+x for right hand) of the distal tip
        nail_w, nail_t = fr * 1.15, fr * 0.30
        seg6 = self.seg(8)
        prof = body_section(nail_w, nail_w * 0.8, seg6, n=3.4)
        tip = pos - direction * (flen * splits[2] * 0.25)
        dorsal = np.array([curl_sgn * (fr * 0.72), 0.0, 0.0])
        slices = [slicer.Slice(-nail_t, (1.0, 1.0)), slicer.Slice(nail_t, (0.9, 0.9))]
        self.add_loft(f"nail_{side}_{fname}", prof, slices, parent=f"hand_{side}",
                      translate=tuple(tip + dorsal), rz=-curl_sgn * math.pi / 2,
                      albedo=NAIL_PINK, material="ceramic",
                      metadata={"anatomy": "fingernail"})

    # ------------------------------------------------------------------- legs
    def _legs(self) -> None:
        H, p = self.H, self.p
        seg = self.seg(30)
        hx = HIP_HALF_X * p.hip_scale * H
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            # thigh: glute fold → quad mass → knee
            r0 = 0.042 * H * p.bulk
            c = 0.5 * (HIP_Y + KNEE_Y) * H
            prof = body_section(1.0, 1.0, seg, n=3.0)
            quad = 0.10 + 0.10 * p.muscle
            slices = [
                slicer.Slice(0.122 * H, (r0 * 1.06, r0 * 1.10), offset=(0, -0.004 * H)),   # glute fold
                slicer.Slice(0.070 * H, (r0 * 1.04, r0 * (1.04 + quad * 0.4))),
                slicer.Slice(0.000 * H, (r0 * 0.98, r0 * (1.00 + quad))),                  # quad max
                slicer.Slice(-0.070 * H, (r0 * 0.86, r0 * 0.92)),
                slicer.Slice(-0.122 * H, (r0 * 0.74, r0 * 0.76)),                          # knee
            ]
            self.add_loft(f"upper_leg_{side}", prof, slices, parent="pelvis",
                          translate=(sgn * hx, c, 0.0), albedo=self.p.skin_tone,
                          metadata={"anatomy": "quadriceps"})
            # calf: high back bulge tapering to a slim ankle (calf taper)
            r1 = 0.030 * H * p.bulk
            c = 0.5 * (KNEE_Y + ANKLE_Y) * H
            prof = body_section(1.0, 1.0, seg, n=3.0)
            slices = [
                slicer.Slice(0.123 * H, (r1 * 0.92, r1 * 0.95)),
                slicer.Slice(0.075 * H, (r1 * 1.00, r1 * (1.0 + 0.42))),                   # gastrocnemius
                slicer.Slice(0.010 * H, (r1 * 0.88, r1 * (1.0 + 0.30)), offset=(0, -0.003 * H)),
                slicer.Slice(-0.060 * H, (r1 * 0.62, r1 * 0.68)),
                slicer.Slice(-0.123 * H, (r1 * 0.46, r1 * 0.50)),                          # ankle
            ]
            self.add_loft(f"lower_leg_{side}", prof, slices, parent=f"upper_leg_{side}",
                          translate=(sgn * hx, c, 0.0), albedo=self.p.skin_tone,
                          metadata={"anatomy": "calf_taper"})

    # ------------------------------------------------------------------- feet
    def _feet(self) -> None:
        H, p = self.H, self.p
        seg = self.seg(22)
        hx = HIP_HALF_X * p.hip_scale * H
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            # foot shell lofted along z (heel → toes), profile (x, y)
            prof = body_section(1.0, 1.0, seg, n=3.4)
            hw = 0.024 * H
            slices = [
                slicer.Slice(-0.030 * H, (hw * 0.80, 0.020 * H), offset=(0, 0.021 * H)),   # heel
                slicer.Slice(-0.010 * H, (hw * 0.92, 0.030 * H), offset=(0, 0.031 * H)),   # ankle line
                slicer.Slice(0.030 * H, (hw * 0.95, 0.024 * H), offset=(0, 0.025 * H)),    # arch
                slicer.Slice(0.075 * H, (hw * 1.05, 0.020 * H), offset=(0, 0.021 * H)),    # ball
                slicer.Slice(0.105 * H, (hw * 0.92, 0.014 * H), offset=(0, 0.015 * H)),    # toe end
            ]
            self.add_loft(f"foot_{side}", prof, slices, axis="z",
                          parent=f"lower_leg_{side}",
                          translate=(sgn * hx, 0.0, 0.0), albedo=self.p.skin_tone,
                          metadata={"anatomy": "heel_arch_forefoot"})
            # toes: 5 small tapered lofts at the front; big toe on the medial
            # side (toward the midline = -sgn for the right foot is -x)
            toe_data = [("big", 0.0072, 0.020, -0.0115), ("2nd", 0.0056, 0.017, -0.0055),
                        ("3rd", 0.0052, 0.016, 0.0005), ("4th", 0.0048, 0.014, 0.0062),
                        ("5th", 0.0044, 0.012, 0.0118)]
            seg6 = self.seg(8)
            prof_t = slicer.profile_circle(1.0, seg6)
            for tname, tr, tlen, tx in toe_data:
                r = tr * H
                slices = [
                    slicer.Slice(0.0, (r, r * 0.80)),
                    slicer.Slice(tlen * H * 0.6, (r * 0.92, r * 0.74)),
                    slicer.Slice(tlen * H, (r * 0.72, r * 0.58)),
                ]
                self.add_loft(f"toe_{side}_{tname}", prof_t, slices, axis="z",
                              parent=f"foot_{side}",
                              translate=(sgn * hx + tx * H * sgn * -1.0, 0.008 * H,
                                         0.105 * H),
                              albedo=self.p.skin_tone,
                              metadata={"anatomy": "toe"})
                # toenail on top of the tip
                seg5 = self.seg(5)
                prof_n = body_section(r * 0.72, r * 0.55, seg5, n=3.2)
                slices_n = [slicer.Slice(0.0, (1.0, 1.0)),
                            slicer.Slice(0.0016 * H, (0.85, 0.85))]
                self.add_loft(f"toenail_{side}_{tname}", prof_n, slices_n,
                              parent=f"foot_{side}",
                              translate=(sgn * hx + tx * H * sgn * -1.0,
                                         0.008 * H + r * 0.62,
                                         0.105 * H + tlen * H * 0.75),
                              albedo=NAIL_PINK, material="ceramic",
                              metadata={"anatomy": "toenail"})

    # ------------------------------------------------------------------- face
    def _face(self) -> None:
        H, p = self.H, self.p
        self._eyes()
        self._eyebrows()
        self._nose()
        self._ears()
        self._mouth()

    def _eyes(self) -> None:
        H, p = self.H, self.p
        seg = self.seg(22)
        r_ball = 0.0072 * H
        z_face = 0.046 * H
        # eye-shape parameters: (upper-lid coverage, lid overhang, canthus tilt)
        shape_cfg = {
            "almond":  (0.28, 0.0010, 6.0),
            "round":   (0.14, 0.0006, 0.0),
            "hooded":  (0.46, 0.0026, -4.0),
            "monolid": (0.42, 0.0012, 2.0),
        }[p.eye_shape]
        coverage, overhang, tilt_deg = shape_cfg
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            cx, cy = sgn * 0.0185 * H, EYE_Y * H
            # eyeball (sclera)
            prof, slices = spheroid_loft(r_ball, r_ball * 0.92, r_ball * 0.92,
                                         seg, self.seg(10))
            self.add_loft(f"eye_{side}", prof, slices, parent="head",
                          translate=(cx, cy, z_face), albedo=SCLERA_WHITE,
                          material="ceramic", metadata={"anatomy": "eyeball"})
            # iris: coloured disc on the ball front (slightly domed spheroid)
            r_iris = 0.0034 * H
            prof, slices = spheroid_loft(r_iris, r_iris, r_iris * 0.28,
                                         self.seg(10), 3, margin=0.55)
            self.add_loft(f"iris_{side}", prof, slices, parent="head",
                          translate=(cx, cy, z_face + r_ball * 0.90),
                          albedo=p.eye_color, material="ceramic",
                          metadata={"anatomy": "iris"})
            # pupil
            r_pup = 0.0015 * H
            prof, slices = spheroid_loft(r_pup, r_pup, r_pup * 0.25,
                                         self.seg(8), 3, margin=0.55)
            self.add_loft(f"pupil_{side}", prof, slices, parent="head",
                          translate=(cx, cy, z_face + r_ball * 0.99),
                          albedo=PUPIL_BLACK, material="ceramic",
                          metadata={"anatomy": "pupil"})
            # eyelid panels: thin curved slabs hugging the ball, coverage set
            # by eye shape (this is what differentiates almond/round/hooded/
            # monolid — the same eyeball shows through differently).
            lid_w = 0.0115 * H
            seg8 = self.seg(12)
            # Panel height grows with coverage: hooded/monolid = tall skin
            # flap over the ball, round = thin rim, almond = in between.
            lid_half_h = r_ball * (coverage + 0.22)
            prof = body_section(lid_half_h, 0.0008 * H, seg8, n=3.0)
            n_st = 9
            th = np.linspace(-1.0, 1.0, n_st)
            y_edge = cy + r_ball * (1.0 - 2.0 * coverage)     # lower rim
            y_cent = y_edge + lid_half_h
            slices = []
            for t in th:
                x = cx + t * lid_w * 0.5
                y = y_cent + (1 - t ** 2) * r_ball * 0.10 \
                    + math.radians(tilt_deg) * t * lid_w * 0.5 * sgn
                z = z_face + r_ball * (0.86 - 0.10 * t ** 2) + 0.0006 * H \
                    + overhang * (abs(t) ** 1.5)
                slices.append(slicer.Slice(x, (1.0, 1.0), offset=(y, z)))
            self.add_loft(f"eyelid_{side}", prof, slices, axis="x", caps=True,
                          parent="head", albedo=p.skin_tone,
                          metadata={"anatomy": "upper_eyelid",
                                    "eye_shape": p.eye_shape})
            # lower lid: subtle thin rim under the ball
            prof = body_section(0.0008 * H, 0.0006 * H, seg8, n=3.0)
            slices = []
            for t in th:
                x = cx + t * lid_w * 0.46
                y = cy - r_ball * 0.86 + (1 - t ** 2) * r_ball * 0.05
                z = z_face + r_ball * (0.78 - 0.08 * t ** 2) + 0.0005 * H
                slices.append(slicer.Slice(x, (1.0, 1.0), offset=(y, z)))
            self.add_loft(f"eyelid_lower_{side}", prof, slices, axis="x",
                          caps=True, parent="head", albedo=p.skin_tone,
                          metadata={"anatomy": "lower_eyelid"})

    def _eyebrows(self) -> None:
        """Eyebrows as clusters of tiny tapered hair strokes — never slabs."""
        H, p = self.H, self.p
        seg = self.seg(6)
        prof = slicer.profile_circle(1.0, seg)
        n_strokes = max(6, int(round(18 * self.d)))
        brow_col = tuple(c * 0.9 for c in p.hair_color)
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            # brow arc: inner (near nose) slightly lower, tail tapered down
            x0, x1 = sgn * 0.0075 * H, sgn * 0.0275 * H
            y0, y1 = BROW_Y * H, (BROW_Y - 0.004) * H
            z0, z1 = 0.0525 * H, 0.0485 * H
            for i in range(n_strokes):
                t = (i + 0.5) / n_strokes
                # strokes fan: inner strokes point up, tail strokes lie down
                lean = 0.35 + 0.65 * t
                cx = x0 + (x1 - x0) * t
                cy = y0 + (y1 - y0) * t + 0.0012 * H * math.sin(math.pi * t)
                cz = z0 + (z1 - z0) * t
                ln = (0.0085 - 0.0025 * t) * H
                jx, jy = self.rng.normal(0, 0.0006 * H, 2)
                dx = sgn * ln * 0.55 * (0.5 + 0.5 * lean)
                dy = ln * (1.0 - 0.55 * lean)
                pts = np.array([
                    [cx + jx, cy + jy, cz],
                    [cx + jx + dx * 0.5, cy + jy + dy * 0.55, cz + 0.0006 * H],
                    [cx + jx + dx, cy + jy + dy * 0.9, cz + 0.0004 * H],
                ])
                r = (0.00055 + 0.00015 * self.rng.random()) * H
                slices = path_slices(pts, [r, r * 0.7, r * 0.25], axis="x")
                self.add_loft(f"brow_{side}_{i:02d}", prof, slices, axis="x",
                              parent="head",
                              albedo=brow_col, metadata={"anatomy": "eyebrow_stroke"})

    def _nose(self) -> None:
        H, p = self.H, self.p
        seg = self.seg(18)
        # nose lofted along z: bridge → tip bulb → nostril wings, then the
        # column curves back under (slice offsets dip in y).
        stations = [  # (z, y, half-width, half-height)
            (0.0490, 0.9380, 0.0042, 0.0075),   # bridge top (between brows)
            (0.0535, 0.9300, 0.0044, 0.0078),   # bridge
            (0.0580, 0.9220, 0.0050, 0.0072),   # dorsum
            (0.0620, 0.9170, 0.0062, 0.0062),   # tip bulb
            (0.0605, 0.9110, 0.0070, 0.0052),   # tip underside
            (0.0555, 0.9070, 0.0105, 0.0040),   # nostril wings (ala)
            (0.0515, 0.9055, 0.0115, 0.0030),   # base on the face
        ]
        prof = body_section(1.0, 1.0, seg, n=2.6)
        slices = [slicer.Slice(z * H, (hw * H, hh * H), offset=(0.0, y * H))
                  for (z, y, hw, hh) in stations]
        self.add_loft("nose", prof, slices, axis="z", parent="head",
                      albedo=p.skin_tone,
                      metadata={"anatomy": "bridge_tip_nostrils"})
        # nostrils: two small dark pits under the tip
        seg6 = self.seg(8)
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            prof, slices = spheroid_loft(0.0019 * H, 0.0012 * H, 0.0010 * H,
                                         seg6, 3, margin=0.5)
            self.add_loft(f"nostril_{side}", prof, slices, parent="head",
                          translate=(sgn * 0.0042 * H, 0.9095 * H, 0.0588 * H),
                          albedo=NOSTRIL_DARK, metadata={"anatomy": "nostril"})

    def _ears(self) -> None:
        H, p = self.H, self.p
        seg = self.seg(7)
        prof = slicer.profile_circle(1.0, seg)
        hw = HEAD_HW * H
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            cx = sgn * (hw + 0.001 * H)
            cy, cz = EAR_Y * H, 0.002 * H
            # helix rim: C opening toward the face — top (slightly front),
            # sweeping back at mid-height, ending at the lobe. Monotonic in
            # y so the y-axis loft follows the rim in order.
            rr = 0.0105 * H
            n_pt = 11
            ang = np.linspace(math.radians(80), math.radians(-85), n_pt)
            pts = np.stack([
                cx + sgn * (0.0016 * H + 0.0014 * H * np.abs(np.cos(ang))),
                cy + rr * np.sin(ang),
                cz - rr * 0.80 * np.cos(ang),
            ], axis=-1)
            pts = pts[np.argsort(-pts[:, 1])]
            r_t = 0.0015 * H
            slices = path_slices(pts, np.linspace(r_t, r_t * 0.8, n_pt), axis="y")
            self.add_loft(f"ear_helix_{side}", prof, slices, parent="head",
                          albedo=p.skin_tone, metadata={"anatomy": "helix"})
            # inner ridge (antihelix): smaller, shifted forward + inward
            pts2 = pts.copy()
            pts2[:, 0] -= sgn * 0.0011 * H
            pts2[:, 2] += 0.0016 * H
            pts2[:, 1] = cy + (pts2[:, 1] - cy) * 0.60
            slices = path_slices(pts2, np.linspace(r_t * 0.7, r_t * 0.5, n_pt), axis="y")
            self.add_loft(f"ear_ridge_{side}", prof, slices, parent="head",
                          albedo=p.skin_tone, metadata={"anatomy": "antihelix"})
            # lobe: small soft spheroid at the bottom
            prof2, slices = spheroid_loft(0.0035 * H, 0.0048 * H, 0.0032 * H,
                                          self.seg(8), 4, margin=0.35)
            self.add_loft(f"ear_lobe_{side}", prof2, slices, parent="head",
                          translate=(cx + sgn * 0.0018 * H, cy - rr * 0.95, cz),
                          albedo=p.skin_tone, metadata={"anatomy": "ear_lobe"})
            # concha shadow: tiny dark dimple
            prof3, slices = spheroid_loft(0.0016 * H, 0.0022 * H, 0.0012 * H,
                                          self.seg(6), 3, margin=0.5)
            self.add_loft(f"ear_concha_{side}", prof3, slices, parent="head",
                          translate=(cx + sgn * 0.0008 * H, cy - 0.001 * H,
                                     cz - 0.0005 * H),
                          albedo=NOSTRIL_DARK, metadata={"anatomy": "concha"})

    def _mouth(self) -> None:
        H, p = self.H, self.p
        seg = self.seg(16)
        open_gap = 0.010 * H * p.mouth_open
        half_w = 0.0145 * H
        z_face = 0.0510 * H
        n_st = 13
        th = np.linspace(-1.0, 1.0, n_st)
        for lname, lsgn, thick in (("lip_upper", 1.0, 0.0016),
                                   ("lip_lower", -1.0, 0.0021)):
            prof = body_section(thick * H, thick * 0.62 * H, seg, n=2.6)
            slices = []
            for t in th:
                x = t * half_w
                # gentle smile arc + cupid's bow dip for the upper lip
                y = MOUTH_Y * H + lsgn * (0.0018 * H + open_gap * 0.5) \
                    + (1 - t ** 2) * 0.0006 * H * lsgn \
                    - (lsgn > 0) * 0.0007 * H * math.exp(-(t / 0.22) ** 2)
                z = z_face + (1 - t ** 2) * 0.0035 * H + 0.0009 * H
                slices.append(slicer.Slice(x, (1.0, 1.0), offset=(y, z)))
            self.add_loft(lname, prof, slices, axis="x", parent="head",
                          albedo=LIP_PINK, metadata={"anatomy": lname})
        # teeth: two white arcs behind the lips (visible when the mouth opens)
        for tname, tsgn, ty in (("teeth_upper", 1.0, MOUTH_Y * H + 0.0004 * H),
                                ("teeth_lower", -1.0,
                                 MOUTH_Y * H - open_gap - 0.0004 * H)):
            prof = body_section(0.0016 * H, 0.0012 * H, self.seg(10), n=3.0)
            slices = []
            for t in th:
                x = t * half_w * 0.82
                y = ty - tsgn * (1 - t ** 2) * 0.0008 * H
                z = z_face + (1 - t ** 2) * 0.0026 * H - 0.0004 * H
                slices.append(slicer.Slice(x, (1.0, 1.0), offset=(y, z)))
            self.add_loft(tname, prof, slices, axis="x", parent="head",
                          albedo=TEETH_WHITE, material="ceramic",
                          metadata={"anatomy": tname})
        # oral cavity: dark interior disc behind the teeth
        prof, slices = spheroid_loft(half_w * 0.8, 0.0035 * H + open_gap * 0.5,
                                     0.0012 * H, self.seg(10), 3, margin=0.6)
        self.add_loft("mouth_cavity", prof, slices, parent="head",
                      translate=(0.0, MOUTH_Y * H - open_gap * 0.5,
                                 z_face - 0.001 * H),
                      albedo=MOUTH_CAVITY, metadata={"anatomy": "oral_cavity"})


# ---------------------------------------------------------------------------
# the spec (public API)
# ---------------------------------------------------------------------------


@dataclass
class HumanSpec:
    """Result of `build_human`: part graph + appearance extras."""

    graph: PartGraph
    params: HumanParams
    bone_names: tuple[str, ...] = SIM_BONE_NAMES
    extras: dict = field(default_factory=dict)

    # -- conveniences -------------------------------------------------------
    @property
    def appearance(self) -> dict:
        p = self.params
        return {
            "skin_tone": p.skin_tone,
            "eye_color": p.eye_color,
            "eye_shape": p.eye_shape,
            "hair_style": p.hair_style,
            "hair_color": p.hair_color,
            "body_type": p.body_type,
            "height_m": p.height_m,
            "gender": p.gender,
            "detail": p.detail,
        }

    def build(self) -> BuildResult:
        return self.graph.build()

    def part_albedos(self) -> dict[str, tuple[float, float, float]]:
        """Part name → base albedo (from node metadata)."""
        return {n: tuple(node.metadata.get("albedo", (0.7, 0.7, 0.7)))
                for n, node in self.graph.nodes.items()}

    def vertex_colors(self, result: BuildResult | None = None,
                      noise: float = 0.035) -> dict[str, np.ndarray]:
        """Per-instance (V, 3) float32 COLOR_0 arrays keyed by part label.

        Raw albedo with subtle seeded per-vertex variation (no baked
        lighting), matching the W8 export contract in `generation.colorize`.
        """
        result = result or self.build()
        rng = np.random.default_rng(self.params.seed ^ 0xC010)
        albedo = self.part_albedos()
        out: dict[str, np.ndarray] = {}
        for part in result.parts:
            base = np.asarray(albedo.get(part.name, (0.7, 0.7, 0.7)), dtype=np.float64)
            n = part.vertices.shape[0]
            var = rng.uniform(1.0 - noise, 1.0 + noise, n)[:, None]
            out[part.label] = np.clip(base[None, :] * var, 0.0, 1.0).astype(np.float32)
        return out

    def colored_world_meshes(self, result: BuildResult | None = None):
        """World-space (positions, indices, normals, colors, name) tuples —
        the exact shape BonaFide's `Mesh.from_arrays` consumes."""
        result = result or self.build()
        colors = self.vertex_colors(result)
        meshes = []
        for part in result.parts:
            vw, nw = apply_transform(part.vertices, part.normals, part.transform)
            f = part.faces
            if float(np.linalg.det(part.transform[:3, :3])) < 0.0:
                f = f[:, [0, 2, 1]]
            meshes.append((np.asarray(vw, dtype=np.float32),
                           np.asarray(f, dtype=np.int64),
                           np.asarray(nw, dtype=np.float32),
                           colors[part.label], part.label))
        return meshes


def build_human(params: dict | HumanParams | None = None, **kw) -> HumanSpec:
    """Build an anatomically detailed parametric human.

    Parameters (dict or kwargs — see :class:`HumanParams`): `height_m`,
    `body_type` (slim/average/athletic/heavy), `gender` (0 male … 1 female),
    `skin_tone`, `eye_color`, `eye_shape` (almond/round/hooded/monolid),
    `hair_style` + `hair_color` (see `generation.hair.HAIRSTYLES`),
    `mouth_open` (0..1), `clothes` (tuple of garment names, see
    `generation.clothing.GARMENTS`), `cloth_colors`, `detail`
    (low/medium/high LOD knob), `seed`.

    Returns a :class:`HumanSpec` whose `graph` contains the 19 Sim-bone-named
    body parts plus face / hair / garment sub-parts, with `appearance`
    extras and (when hair is present) per-strand wind-response physics
    metadata (see `generation.hair.WIND_SPEED_MAPPING`).
    """
    p = resolve_params(params, **kw)
    builder = HumanBuilder(p)
    builder.build()

    extras: dict = {"appearance_hint": "see HumanSpec.appearance"}

    from . import hair  # deferred: hair imports helpers from this module
    hair_extras = hair.add_hair(builder)

    from . import clothing
    clothing_extras = clothing.add_garments(builder)

    extras.update(hair_extras)
    extras.update(clothing_extras)
    return HumanSpec(graph=builder.graph, params=p, extras=extras)
