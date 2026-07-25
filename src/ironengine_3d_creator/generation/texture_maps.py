"""Tileable procedural texture-map generators (albedo / bump / roughness / alpha).

Mission: real-world surface richness *without* geometric cost — image maps
replace computationally heavy micro-geometry (weaves, pores, scrollwork,
panel grooves). Every generator here is:

- **Deterministic** — seeded via ``numpy.random.default_rng(seed)``; the same
  (kind, size, seed) triple always yields byte-identical maps.
- **Tileable** — all noise is periodic value noise on integer-frequency
  lattices and all patterns use integer repeats per tile, so the right/bottom
  edges wrap seamlessly onto the left/top edges (wrap = "repeat").
- **Fast** — vectorized NumPy only; each channel map renders in well under
  200 ms at 512x512 on a desktop CPU (typically < 20 ms).

Channel contract (returned as a ``dict[str, np.ndarray]``):

- ``"albedo"``     — (H, W, 3) uint8 sRGB base colour, always present.
- ``"bump"``       — (H, W) uint8 height map (128 = neutral), when meaningful.
- ``"roughness"``  — (H, W) uint8 roughness (0 = mirror, 255 = matte), when meaningful.
- ``"alpha"``      — (H, W) uint8 coverage mask (ornament decals), when meaningful.

See docs/TEXTURES.md for the full contract and the bake doctrine.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

TAU = 2.0 * math.pi

MIN_SIZE = 64
MAX_SIZE = 1024

# Channels each kind produces besides albedo (metadata for tooling/tests).
KIND_CHANNELS: dict[str, tuple[str, ...]] = {}


# ---------------------------------------------------------------------------
# tileable noise primitives
# ---------------------------------------------------------------------------


def _periodic_value_noise(
    size: int, fx: int, fy: int, rng: np.random.Generator
) -> np.ndarray:
    """Smooth value noise on a wrapping (fx, fy) lattice — exactly tileable.

    Separable interpolation (x within lattice rows, then y) keeps the large
    temporaries at (fy, size) instead of 4x (size, size) — roughly half the
    memory traffic of direct 2-D fancy indexing.
    """
    fx = max(1, int(fx))
    fy = max(1, int(fy))
    g = rng.random((fy, fx)).astype(np.float32)
    t = (np.arange(size, dtype=np.float32) + 0.5) / size
    tx = t * fx
    ty = t * fy
    ix0 = np.floor(tx).astype(np.int64) % fx
    iy0 = np.floor(ty).astype(np.int64) % fy
    ix1 = (ix0 + 1) % fx
    iy1 = (iy0 + 1) % fy
    sx = tx - np.floor(tx)
    sy = ty - np.floor(ty)
    sx = sx * sx * (3.0 - 2.0 * sx)
    sy = sy * sy * (3.0 - 2.0 * sy)
    # Interpolate along x for every lattice row: (fy, size).
    gx = g[:, ix0] + (g[:, ix1] - g[:, ix0]) * sx[None, :]
    # Interpolate along y between wrapped lattice rows: (size, size).
    out = gx[iy0] + (gx[iy1] - gx[iy0]) * sy[:, None]
    return out.astype(np.float32)


def _fbm(
    size: int,
    base_freq: int,
    octaves: int,
    rng: np.random.Generator,
    *,
    gain: float = 0.5,
) -> np.ndarray:
    """Fractional Brownian motion over tileable octaves (lacunarity 2)."""
    total = np.zeros((size, size), dtype=np.float32)
    amp = 1.0
    norm = 0.0
    f = max(1, int(base_freq))
    for _ in range(max(1, int(octaves))):
        total += amp * _periodic_value_noise(size, f, f, rng)
        norm += amp
        amp *= gain
        f *= 2
    return total / max(norm, 1e-9)


def _aniso_noise(size: int, fx: int, fy: int, rng: np.random.Generator, octaves: int = 2) -> np.ndarray:
    """Stretched fbm (e.g. brushed streaks, wood grain)."""
    total = np.zeros((size, size), dtype=np.float32)
    amp = 1.0
    norm = 0.0
    for _ in range(max(1, octaves)):
        total += amp * _periodic_value_noise(size, fx, fy, rng)
        norm += amp
        amp *= 0.5
        fx *= 2
        fy *= 2
    return total / max(norm, 1e-9)


def _grids(size: int) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) in [0, 1); u varies along columns, v along rows."""
    t = (np.arange(size, dtype=np.float32) + 0.5) / size
    u, v = np.meshgrid(t, t)  # (H, W) each
    return u, v


def _hash01(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic per-cell hash in [0, 1) from integer cell indices."""
    h = np.sin(ix * 127.1 + iy * 311.7 + float(seed) * 74.7) * 43758.5453
    return (h - np.floor(h)).astype(np.float32)


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def _rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack([_u8(r), _u8(g), _u8(b)], axis=-1)


def _lerp3(dark: tuple, light: tuple, t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)[..., None]
    d = np.asarray(dark, dtype=np.float32)
    li = np.asarray(light, dtype=np.float32)
    return _u8(d[None, None, :] + (li - d)[None, None, :] * t)


def _check_size(size: int) -> int:
    size = int(size)
    if not (MIN_SIZE <= size <= MAX_SIZE):
        raise ValueError(f"size must be within [{MIN_SIZE}, {MAX_SIZE}], got {size}")
    return size


# ---------------------------------------------------------------------------
# material map generators: (size, seed) -> channel dict
# ---------------------------------------------------------------------------


def _wood(size: int, seed: int, *, rings: int, light: tuple, dark: tuple,
          pore_thresh: float | None, waviness: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, _v = _grids(size)
    distort = _fbm(size, 4, 4, rng)
    fine = _aniso_noise(size, rings * 8, 2, rng, octaves=2)
    # Integer `rings` keeps the stripe phase tileable across u.
    phase = rings * u + waviness * (distort - 0.5) + 0.15 * (fine - 0.5)
    grain = 0.5 + 0.5 * np.sin(TAU * phase)
    grain = grain ** 1.4
    tone = 0.75 * grain + 0.25 * fine
    albedo = _lerp3(dark, light, tone)
    bump = 0.35 + 0.5 * tone
    if pore_thresh is not None:
        pores = _aniso_noise(size, rings * 16, 3, rng, octaves=2)
        pore_mask = (pores > pore_thresh).astype(np.float32)
        albedo = _u8(albedo.astype(np.float32) / 255.0 * (1.0 - 0.35 * pore_mask[..., None]))
        bump = bump - 0.35 * pore_mask
    return {"albedo": albedo, "bump": _u8(bump)}


def wood_oak(size: int, seed: int) -> dict[str, np.ndarray]:
    """Light open-pore oak, straight grain with visible pores."""
    return _wood(size, seed, rings=9, light=(0.78, 0.62, 0.44), dark=(0.52, 0.36, 0.22),
                 pore_thresh=0.78, waviness=1.2)


def wood_walnut(size: int, seed: int) -> dict[str, np.ndarray]:
    """Dark walnut, broad undulating cathedral bands, closed pore."""
    return _wood(size, seed, rings=5, light=(0.42, 0.30, 0.20), dark=(0.18, 0.11, 0.06),
                 pore_thresh=None, waviness=2.2)


def marble(size: int, seed: int) -> dict[str, np.ndarray]:
    """White Carrara-style marble with turbulence-warped veins."""
    rng = np.random.default_rng(seed)
    u, v = _grids(size)
    turb = _fbm(size, 3, 3, rng)
    vein1 = 1.0 - np.abs(np.sin(TAU * (3.0 * u + 2.0 * v + 3.5 * turb)))
    v2 = vein1 * vein1
    v4 = v2 * v2
    vein1 = v4 * v4 * v2  # ^10, cheap
    vein2 = 1.0 - np.abs(np.sin(TAU * (2.0 * v - u + 2.5 * _fbm(size, 4, 3, rng))))
    w2 = vein2 * vein2
    w4 = w2 * w2
    w8 = w4 * w4
    vein2 = w8 * w8 * w2  # ^18, cheap
    base = 0.90 + 0.07 * _fbm(size, 6, 2, rng)
    veins = np.clip(vein1 * 0.75 + vein2 * 0.5, 0.0, 1.0)
    tone = base - 0.52 * veins
    albedo = _rgb(tone, tone, tone * 1.01)
    bump = _u8(0.55 - 0.25 * veins + 0.06 * turb)
    rough = _u8(np.clip(0.22 + 0.30 * veins, 0, 1))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


def granite(size: int, seed: int) -> dict[str, np.ndarray]:
    """Speckled granite: quartz/feldspar/mica grains."""
    rng = np.random.default_rng(seed)
    base = 0.52 + 0.14 * _fbm(size, 4, 3, rng)
    sp1 = _periodic_value_noise(size, 24, 24, rng)
    sp2 = _periodic_value_noise(size, 28, 28, rng)
    sp3 = _periodic_value_noise(size, 32, 32, rng)
    dark = (sp1 > 0.60).astype(np.float32)
    light = (sp2 > 0.64).astype(np.float32)
    feldspar = (sp3 > 0.70).astype(np.float32) * (1.0 - dark)
    r = base - 0.22 * dark + 0.28 * light + 0.20 * feldspar
    g = base - 0.22 * dark + 0.28 * light + 0.10 * feldspar
    b = base - 0.20 * dark + 0.26 * light + 0.06 * feldspar
    bump = _u8(0.5 + 0.3 * (sp1 - 0.5) + 0.2 * (sp2 - 0.5))
    rough = _u8(np.full_like(base, 0.82) + 0.1 * (sp3 - 0.5))
    return {"albedo": _rgb(r, g, b), "bump": bump, "roughness": rough}


def stone(size: int, seed: int) -> dict[str, np.ndarray]:
    """Cut limestone / sandstone: soft tonal bands + fine grain."""
    rng = np.random.default_rng(seed)
    grain = _fbm(size, 24, 2, rng)
    tone = 0.58 + 0.16 * _fbm(size, 3, 3, rng) + 0.06 * grain
    bands = 0.05 * np.sin(TAU * (3.0 * _grids(size)[1] + 1.2 * _fbm(size, 2, 2, rng)))
    tone = tone + bands
    albedo = _rgb(tone, tone * 0.985, tone * 0.94)
    bump = _u8(0.5 + 0.35 * (grain - 0.5) + 0.15 * bands)
    return {"albedo": albedo, "bump": bump}


def brick(size: int, seed: int) -> dict[str, np.ndarray]:
    """Running-bond brickwork; integer courses/columns keep it tileable."""
    rng = np.random.default_rng(seed)
    rows, cols = 8, 4
    u, v = _grids(size)
    row = np.floor(v * rows).astype(np.int64)
    shifted = u * cols + 0.5 * (row % 2)
    bx = np.floor(shifted).astype(np.int64) % cols  # wrap -> same brick across seam
    fx = shifted - np.floor(shifted)
    fy = v * rows - row
    mw, mh = 0.07, 0.10
    mortar = (fx < mw) | (fx > 1.0 - mw) | (fy < mh) | (fy > 1.0 - mh)
    tint = 0.78 + 0.42 * _hash01(bx, row, seed)
    wear = _fbm(size, 12, 3, rng)
    brick_r = 0.60 * tint + 0.10 * (wear - 0.5)
    brick_g = 0.27 * tint + 0.06 * (wear - 0.5)
    brick_b = 0.19 * tint + 0.05 * (wear - 0.5)
    mort_tone = 0.72 + 0.08 * (wear - 0.5)
    m = mortar[..., None]
    albedo = np.where(m, np.stack([_u8(mort_tone)] * 3, axis=-1),
                      _rgb(brick_r, brick_g, brick_b))
    bump = _u8(np.where(mortar, 0.28 + 0.1 * wear, 0.62 + 0.18 * wear))
    rough = _u8(np.where(mortar, 0.92, 0.85) + 0.05 * (wear - 0.5))
    return {"albedo": albedo.astype(np.uint8), "bump": bump, "roughness": rough}


def linen(size: int, seed: int) -> dict[str, np.ndarray]:
    """Plain-weave linen: balanced warp/weft with over-under checker."""
    rng = np.random.default_rng(seed)
    f = 48  # threads per tile (integer -> tileable)
    u, v = _grids(size)
    warp = 0.5 + 0.5 * np.sin(TAU * f * u)
    weft = 0.5 + 0.5 * np.sin(TAU * f * v)
    over = ((np.floor(u * f) + np.floor(v * f)) % 2).astype(np.float32)
    height = over * warp + (1.0 - over) * weft
    thread_noise = _aniso_noise(size, f, f, rng, octaves=1)
    tone = 0.86 + 0.10 * (height - 0.5) + 0.06 * (thread_noise - 0.5)
    albedo = _rgb(tone, tone * 0.985, tone * 0.93)
    bump = _u8(0.30 + 0.45 * height)
    rough = _u8(np.full((size, size), 0.9, dtype=np.float32))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


def denim(size: int, seed: int) -> dict[str, np.ndarray]:
    """Indigo denim: diagonal 2/1 twill ridges + white weft flecks."""
    rng = np.random.default_rng(seed)
    f = 40
    u, v = _grids(size)
    twill = 0.5 + 0.5 * np.sin(TAU * f * (u + v))
    weave = 0.5 + 0.5 * np.sin(TAU * f * 2 * u) * np.sin(TAU * f * 2 * v)
    fleck = _hash01(np.floor(u * f * 2), np.floor(v * f * 2), seed)
    fleck = (fleck > 0.88).astype(np.float32)
    height = 0.65 * twill + 0.35 * weave
    base_r = 0.14 + 0.10 * height
    base_g = 0.22 + 0.12 * height
    base_b = 0.36 + 0.16 * height
    albedo = _rgb(base_r + 0.5 * fleck, base_g + 0.5 * fleck, base_b + 0.45 * fleck)
    bump = _u8(0.30 + 0.45 * height)
    return {"albedo": albedo, "bump": bump}


def leather(size: int, seed: int) -> dict[str, np.ndarray]:
    """Pebbled leather with crease cracks."""
    rng = np.random.default_rng(seed)
    pebble = _fbm(size, 14, 3, rng)
    pebble = np.clip((pebble - 0.5) * 2.2 + 0.5, 0, 1)
    ridged = 1.0 - np.abs(2.0 * _fbm(size, 8, 2, rng) - 1.0)
    cracks = (ridged > 0.93).astype(np.float32)
    tone = 0.80 + 0.28 * pebble - 0.18 * cracks
    albedo = _rgb(0.42 * tone, 0.28 * tone, 0.18 * tone)
    bump = _u8(np.clip(0.30 + 0.5 * pebble - 0.35 * cracks, 0, 1))
    rough = _u8(np.clip(0.72 - 0.15 * pebble, 0, 1))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


def brushed_metal(size: int, seed: int) -> dict[str, np.ndarray]:
    """Brushed aluminium: fine anisotropic streaks along v."""
    rng = np.random.default_rng(seed)
    streak = _aniso_noise(size, 128, 2, rng, octaves=2)
    macro = _fbm(size, 3, 2, rng)
    bright = 0.55 + 0.22 * (streak - 0.5) + 0.10 * (macro - 0.5)
    albedo = _rgb(bright * 0.96, bright * 0.99, bright * 1.04)
    rough = _u8(np.clip(0.30 + 0.30 * streak, 0, 1))
    bump = _u8(0.5 + 0.25 * (streak - 0.5))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


def rust(size: int, seed: int) -> dict[str, np.ndarray]:
    """Rusted steel: blotchy oxidation over grey metal."""
    rng = np.random.default_rng(seed)
    blotch = _fbm(size, 5, 5, rng)
    mask = np.clip((blotch - 0.42) / 0.18, 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)
    fine = _fbm(size, 22, 3, rng)
    metal_tone = 0.46 + 0.08 * (fine - 0.5)
    rust_a = np.array([0.58, 0.26, 0.10], dtype=np.float32)
    rust_b = np.array([0.32, 0.14, 0.05], dtype=np.float32)
    rust_col = rust_b[None, None, :] + (rust_a - rust_b)[None, None, :] * fine[..., None]
    metal_col = np.stack([metal_tone, metal_tone * 1.01, metal_tone * 1.03], axis=-1)
    albedo = _u8(metal_col * (1.0 - mask[..., None]) + rust_col * mask[..., None])
    bump = _u8(np.clip(0.45 + mask * (0.35 * (fine - 0.3)), 0, 1))
    rough = _u8(0.35 + 0.6 * mask)
    return {"albedo": albedo, "bump": bump, "roughness": rough}


def grass(size: int, seed: int) -> dict[str, np.ndarray]:
    """Lawn grass: patchy turf + vertical blade streaks."""
    rng = np.random.default_rng(seed)
    patch = _fbm(size, 4, 4, rng)
    blades = _aniso_noise(size, 6, 96, rng, octaves=2)
    dry = (patch > 0.66).astype(np.float32)
    g_r = 0.12 + 0.16 * patch + 0.10 * dry + 0.06 * (blades - 0.5)
    g_g = 0.34 + 0.22 * patch + 0.02 * dry + 0.14 * (blades - 0.5)
    g_b = 0.09 + 0.08 * patch + 0.02 * dry + 0.04 * (blades - 0.5)
    albedo = _rgb(g_r, g_g, g_b)
    bump = _u8(0.30 + 0.45 * blades)
    return {"albedo": albedo, "bump": bump}


def concrete(size: int, seed: int) -> dict[str, np.ndarray]:
    """Poured concrete: fine aggregate speckle, form stains, hairline cracks."""
    rng = np.random.default_rng(seed)
    speckle = _fbm(size, 32, 2, rng)
    base = 0.62 + 0.09 * _fbm(size, 3, 3, rng) + 0.05 * speckle
    stains = (_fbm(size, 2, 2, rng) > 0.62).astype(np.float32)
    ridged = 1.0 - np.abs(2.0 * _fbm(size, 6, 3, rng) - 1.0)
    cracks = (ridged > 0.965).astype(np.float32)
    tone = base - 0.07 * stains - 0.16 * cracks
    albedo = _rgb(tone, tone, tone * 0.985)
    bump = _u8(np.clip(0.5 + 0.2 * (speckle - 0.5) - 0.3 * cracks, 0, 1))
    rough = _u8(np.full((size, size), 0.9, dtype=np.float32))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


def rococo_ornament(size: int, seed: int) -> dict[str, np.ndarray]:
    """Rococo scrollwork decal: symmetric curled acanthus relief.

    Carries ``alpha`` (ornament coverage) and a strong ``bump`` so carved
    scrollwork can be decaled onto plain geometry instead of being modelled.
    """
    rng = np.random.default_rng(seed)
    u, v = _grids(size)
    k = 3  # integer repeats -> tileable
    swirl_u = np.sin(TAU * (k * u + 0.55 * np.sin(TAU * k * v)))
    swirl_v = np.sin(TAU * (k * v + 0.55 * np.sin(TAU * k * u)))
    scroll = (0.5 + 0.5 * swirl_u) * (0.5 + 0.5 * swirl_v)
    curl = 0.5 + 0.5 * np.sin(TAU * (k * (u + v) + 0.8 * np.sin(TAU * k * (u - v))))
    height = scroll ** 1.6 * 0.75 + curl ** 3.0 * 0.25
    height = np.clip(height, 0, 1)
    grain = _fbm(size, 16, 2, rng)
    height = np.clip(height + 0.06 * (grain - 0.5), 0, 1)
    alpha = np.clip((height - 0.28) / 0.12, 0.0, 1.0)
    gold = np.array([0.74, 0.56, 0.26], dtype=np.float32)
    gold_hi = np.array([0.88, 0.74, 0.42], dtype=np.float32)
    col = gold[None, None, :] + (gold_hi - gold)[None, None, :] * height[..., None]
    albedo = _u8(col)
    return {
        "albedo": albedo,
        "bump": _u8(height),
        "alpha": _u8(alpha),
        "roughness": _u8(np.clip(0.45 - 0.2 * height, 0, 1)),
    }


def scifi_panel(size: int, seed: int) -> dict[str, np.ndarray]:
    """Futuristic hull panelling: recessed grooves, rivets, per-panel tint."""
    rng = np.random.default_rng(seed)
    cols, rows = 4, 4
    u, v = _grids(size)
    cx = np.floor(u * cols).astype(np.int64)
    cy = np.floor(v * rows).astype(np.int64)
    fx = u * cols - cx
    fy = v * rows - cy
    gw = 0.035
    groove = (fx < gw) | (fx > 1.0 - gw) | (fy < gw) | (fy > 1.0 - gw)
    # Rivets near panel corners.
    dcorner = np.minimum.reduce([
        np.hypot(fx, fy), np.hypot(1 - fx, fy),
        np.hypot(fx, 1 - fy), np.hypot(1 - fx, 1 - fy),
    ])
    rivet = (dcorner < 0.10).astype(np.float32)
    tint = 0.88 + 0.24 * _hash01(cx, cy, seed)
    wear = _fbm(size, 10, 3, rng)
    base = 0.46 * tint + 0.05 * (wear - 0.5)
    groove_tone = 0.16 + 0.03 * (wear - 0.5)
    tone = np.where(groove, groove_tone, base)
    albedo = _rgb(tone * 0.92, tone * 0.97, tone * 1.08)
    bump_h = np.where(groove, 0.22, 0.55) + 0.30 * rivet * (1.0 - (dcorner / 0.10) ** 2)
    bump = _u8(np.clip(bump_h + 0.06 * (wear - 0.5), 0, 1))
    rough = _u8(np.where(groove, 0.65, 0.38 + 0.1 * (wear - 0.5)))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


# ---------------------------------------------------------------------------
# CR_TexReal: camo / organic / fabric / masonry generators
# ---------------------------------------------------------------------------


def _camo(size: int, seed: int, palette: list[tuple]) -> dict[str, np.ndarray]:
    """Blotch camouflage: K smooth noise fields, one per palette colour.

    Each pixel takes the colour of its strongest field; where the top two
    fields are close the colours blend, giving organic soft-edged blotches.
    All fields are wrapping fbm, so the pattern tiles exactly.
    """
    rng = np.random.default_rng(seed)
    fields = np.stack([_fbm(size, 3, 4, rng) for _ in palette], axis=0)
    order = np.argsort(-fields, axis=0)
    top = np.take_along_axis(fields, order[:1], axis=0)[0]
    second = np.take_along_axis(fields, order[1:2], axis=0)[0]
    winner = order[0]
    pal = np.asarray(palette, dtype=np.float32)
    blend = np.clip((top - second) / 0.12, 0.0, 1.0)[..., None]
    col = pal[order[1]] * (1.0 - blend) + pal[winner] * blend
    grain = _fbm(size, 28, 2, rng)[..., None]
    col = col * (0.90 + 0.20 * grain)
    return {
        "albedo": _u8(col),
        "bump": _u8(np.clip(0.48 + 0.20 * (grain[..., 0] - 0.5), 0, 1)),
        "roughness": _u8(np.full((size, size), 0.88, dtype=np.float32)),
    }


def woodland_camo(size: int, seed: int) -> dict[str, np.ndarray]:
    """Temperate woodland camo: green/olive/brown/black soft blotches."""
    return _camo(size, seed, [
        (0.30, 0.34, 0.16),   # olive
        (0.15, 0.19, 0.09),   # dark green
        (0.29, 0.21, 0.11),   # earth brown
        (0.09, 0.09, 0.07),   # near-black
    ])


def desert_camo(size: int, seed: int) -> dict[str, np.ndarray]:
    """Arid desert camo: sand/tan/pale blotches."""
    return _camo(size, seed, [
        (0.74, 0.64, 0.44),   # sand
        (0.60, 0.48, 0.30),   # tan
        (0.83, 0.77, 0.61),   # pale khaki
        (0.44, 0.33, 0.19),   # umber
    ])


def skin(size: int, seed: int) -> dict[str, np.ndarray]:
    """Skin with pores, freckles and red/yellow mottling.

    The albedo is centred on a mid tone and the variation is multiplicative-
    friendly: exporters shift the tone lighter/darker via the base-colour
    factor (the per-part tint hook) without regenerating the map.
    """
    rng = np.random.default_rng(seed)
    mottle = _fbm(size, 6, 4, rng)                    # large tonal patches
    fine = _fbm(size, 24, 3, rng)                     # pore field
    pores = np.clip((fine - 0.68) / 0.10, 0.0, 1.0)   # soft pore dots
    freck_field = _periodic_value_noise(size, 20, 20, rng)
    freckles = np.clip((freck_field - 0.80) / 0.07, 0.0, 1.0)
    r = 0.80 + 0.10 * (mottle - 0.5) - 0.05 * pores - 0.16 * freckles
    g = 0.60 + 0.08 * (mottle - 0.5) - 0.06 * pores - 0.12 * freckles
    b = 0.48 + 0.06 * (mottle - 0.5) - 0.06 * pores - 0.08 * freckles
    albedo = _rgb(r, g, b)
    bump = _u8(np.clip(0.50 + 0.25 * (fine - 0.5) - 0.30 * pores, 0, 1))
    rough = _u8(np.clip(0.55 + 0.15 * (mottle - 0.5), 0, 1))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


def knit_wool(size: int, seed: int) -> dict[str, np.ndarray]:
    """Knitted wool: V-stitch chevron rows + fuzzy fibre variation."""
    rng = np.random.default_rng(seed)
    f = 24  # stitch columns per tile (integer -> tileable)
    u, v = _grids(size)
    d1 = 0.5 + 0.5 * np.sin(TAU * f * (0.5 * u + v))
    d2 = 0.5 + 0.5 * np.sin(TAU * f * (-0.5 * u + v))
    chevron = np.maximum(d1, d2)                       # crossing V strands
    fuzz = _aniso_noise(size, f, 6, rng, octaves=2)
    tone = 0.74 + 0.16 * chevron + 0.10 * (fuzz - 0.5)
    albedo = _rgb(tone, tone * 0.96, tone * 0.88)
    bump = _u8(np.clip(0.25 + 0.55 * chevron + 0.12 * (fuzz - 0.5), 0, 1))
    rough = _u8(np.full((size, size), 0.95, dtype=np.float32))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


def plaster_wall(size: int, seed: int) -> dict[str, np.ndarray]:
    """Interior plaster: trowel-sweep ridges, faint stains, fine grain."""
    rng = np.random.default_rng(seed)
    ridged = 1.0 - np.abs(2.0 * _fbm(size, 4, 3, rng) - 1.0)
    sweeps = ridged ** 2                                # soft trowel arcs
    grain = _fbm(size, 30, 2, rng)
    stains = np.clip((_fbm(size, 2, 2, rng) - 0.60) / 0.15, 0.0, 1.0)
    tone = 0.87 + 0.05 * (sweeps - 0.5) + 0.03 * (grain - 0.5) - 0.06 * stains
    albedo = _rgb(tone, tone * 0.985, tone * 0.945)
    bump = _u8(np.clip(0.50 + 0.30 * (sweeps - 0.5) + 0.12 * (grain - 0.5), 0, 1))
    rough = _u8(np.full((size, size), 0.92, dtype=np.float32))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


def snow(size: int, seed: int) -> dict[str, np.ndarray]:
    """Fresh snow: soft drift shading, blue-tinged dips, glinting sparkle."""
    rng = np.random.default_rng(seed)
    drifts = _fbm(size, 4, 3, rng)
    fine = _fbm(size, 18, 2, rng)
    sparkle_field = _periodic_value_noise(size, 64, 64, rng)
    sparkle = np.clip((sparkle_field - 0.975) / 0.015, 0.0, 1.0)  # sparse glints
    shade = 0.90 + 0.10 * drifts + 0.03 * (fine - 0.5)
    r = np.clip(shade - 0.03 * (1.0 - drifts) + 0.35 * sparkle, 0, 1)
    g = np.clip(shade - 0.01 * (1.0 - drifts) + 0.35 * sparkle, 0, 1)
    b = np.clip(shade + 0.03 * (1.0 - drifts) + 0.35 * sparkle, 0, 1)
    albedo = _rgb(r, g, b)
    bump = _u8(np.clip(0.45 + 0.45 * drifts + 0.10 * (fine - 0.5), 0, 1))
    rough = _u8(np.clip(0.60 - 0.45 * sparkle + 0.10 * (fine - 0.5), 0, 1))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


def mud(size: int, seed: int) -> dict[str, np.ndarray]:
    """Wet mud: dark saturated blotches, drying crust cracks, small grit."""
    rng = np.random.default_rng(seed)
    blotch = _fbm(size, 4, 4, rng)
    wet = np.clip((blotch - 0.45) / 0.20, 0.0, 1.0)
    wet = wet * wet * (3.0 - 2.0 * wet)
    ridged = 1.0 - np.abs(2.0 * _fbm(size, 7, 3, rng) - 1.0)
    cracks = np.clip((ridged - 0.90) / 0.05, 0.0, 1.0) * (1.0 - wet)
    grit_field = _periodic_value_noise(size, 48, 48, rng)
    grit = np.clip((grit_field - 0.90) / 0.05, 0.0, 1.0)
    dark = np.array([0.16, 0.11, 0.07], dtype=np.float32)     # wet mud
    light = np.array([0.42, 0.32, 0.21], dtype=np.float32)    # drying crust
    col = light[None, None, :] * (1.0 - wet[..., None]) + dark[None, None, :] * wet[..., None]
    col = col * (1.0 - 0.45 * cracks[..., None]) + 0.20 * grit[..., None]
    albedo = _u8(col)
    bump = _u8(np.clip(0.45 + 0.25 * (blotch - 0.5) - 0.40 * cracks + 0.25 * grit, 0, 1))
    rough = _u8(np.clip(0.95 - 0.45 * wet, 0, 1))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


def chainmail(size: int, seed: int) -> dict[str, np.ndarray]:
    """Interlocking ringmail weave for fantasy armour.

    Two half-offset ring sub-lattices (even/odd rows); every pixel measures
    its wrapped distance to the nearest ring centres, so the annulus pattern
    tiles exactly. Gaps between rings fall to deep shadow.
    """
    rng = np.random.default_rng(seed)
    cols = rows = 16
    u, v = _grids(size)
    cu = u * cols
    cv = v * rows
    iu = np.floor(cu)
    iv = np.floor(cv)

    def _nearest_d2(x_off: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        best = np.full((size, size), np.inf, dtype=np.float32)
        bi = np.zeros((size, size), dtype=np.int64)
        bj = np.zeros((size, size), dtype=np.int64)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                dx = cu - (iu + du + 0.5 + x_off)
                dy = cv - (iv + dv + 0.5)
                d2 = dx * dx + dy * dy
                m = d2 < best
                best = np.where(m, d2, best)
                bi = np.where(m, (iu + du).astype(np.int64), bi)
                bj = np.where(m, (iv + dv).astype(np.int64), bj)
        return best, bi, bj

    d2a, ia, ja = _nearest_d2(0.0)          # even rows
    d2b, ib, jb = _nearest_d2(0.5)          # odd rows, half-offset
    use_a = d2a <= d2b
    dist = np.sqrt(np.where(use_a, d2a, d2b))
    ring_i = np.where(use_a, ia, ib)
    ring_j = np.where(use_a, ja, jb)
    annulus = np.exp(-(((dist - 0.34) / 0.10) ** 2))   # bright ring band
    tint = 0.80 + 0.35 * _hash01(ring_i % cols, ring_j % rows, seed)
    sheen = _fbm(size, 6, 2, rng)
    bright = annulus * tint * (0.50 + 0.25 * (sheen - 0.5))
    tone = 0.10 + 0.55 * bright
    albedo = _rgb(tone * 0.98, tone, tone * 1.06)
    bump = _u8(np.clip(0.12 + 0.75 * annulus, 0, 1))
    rough = _u8(np.clip(0.68 - 0.38 * annulus, 0, 1))
    return {"albedo": albedo, "bump": bump, "roughness": rough}


# ---------------------------------------------------------------------------
# registry + public API
# ---------------------------------------------------------------------------

TEXTURE_GENERATORS = {
    "wood_oak": wood_oak,
    "wood_walnut": wood_walnut,
    "marble": marble,
    "granite": granite,
    "stone": stone,
    "brick": brick,
    "linen": linen,
    "denim": denim,
    "leather": leather,
    "brushed_metal": brushed_metal,
    "rust": rust,
    "grass": grass,
    "concrete": concrete,
    "rococo_ornament": rococo_ornament,
    "scifi_panel": scifi_panel,
    # CR_TexReal
    "woodland_camo": woodland_camo,
    "desert_camo": desert_camo,
    "skin": skin,
    "knit_wool": knit_wool,
    "knit": knit_wool,             # alias
    "plaster_wall": plaster_wall,
    "snow": snow,
    "mud": mud,
    "chainmail": chainmail,
    "rusted_metal": rust,          # alias: rusted steel IS the rust kind
}


def list_texture_kinds() -> list[str]:
    """All registered texture kinds (sorted)."""
    return sorted(TEXTURE_GENERATORS)


def generate_maps(kind: str, size: int = 512, seed: int = 0) -> dict[str, np.ndarray]:
    """Generate all channel maps for `kind`.

    Returns a dict with at least ``"albedo"`` ((size, size, 3) uint8) plus the
    kind's meaningful secondary channels ((size, size) uint8). Deterministic
    for a given (kind, size, seed).
    """
    fn = TEXTURE_GENERATORS.get(str(kind).lower())
    if fn is None:
        raise KeyError(
            f"unknown texture kind {kind!r}; available: {', '.join(list_texture_kinds())}"
        )
    size = _check_size(size)
    maps = fn(size, int(seed))
    assert "albedo" in maps, f"generator {kind!r} did not emit albedo"
    return maps


def map_filename(kind: str, channel: str, size: int, seed: int) -> str:
    """Canonical on-disk name for one channel map."""
    return f"{kind}_{channel}_{size}px_s{int(seed)}.png"


def save_maps(
    maps: dict[str, np.ndarray],
    out_dir: str | Path,
    *,
    kind: str,
    size: int,
    seed: int,
) -> dict[str, str]:
    """Write channel maps as PNGs; returns {channel: path}.

    A kind with an ``alpha`` channel is saved as a single RGBA
    ``<kind>_rgba_*.png`` (albedo + alpha combined) in addition to the
    individual channel PNGs, since decals usually want them packed.
    """
    from PIL import Image  # local import: generators stay NumPy-only

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for channel, arr in maps.items():
        if channel == "alpha":
            continue  # packed into the RGBA file below
        arr = np.asarray(arr)
        img = Image.fromarray(arr, "RGB" if arr.ndim == 3 else "L")
        p = out / map_filename(kind, channel, size, seed)
        img.save(p)
        paths[channel] = str(p)
    if "alpha" in maps:
        rgba = np.concatenate(
            [np.asarray(maps["albedo"]), np.asarray(maps["alpha"])[..., None]], axis=-1
        )
        p = out / map_filename(kind, "rgba", size, seed)
        Image.fromarray(rgba, "RGBA").save(p)
        paths["rgba"] = str(p)
    return paths


def load_map(path: str | Path) -> np.ndarray:
    """Read a saved PNG back as uint8 (round-trip helper for tests/tools)."""
    from PIL import Image

    return np.asarray(Image.open(path))
