"""Apply tileable texture maps to analytic meshes + the textures manifest block.

This module is the bridge between ``generation.texture_maps`` (image-space
procedural maps) and ``generation.analytic_mesh`` (per-primitive UVs). It
deliberately does **not** modify ``core/exporter.py``: the exporter already
supports image textures — ``core.exporter._write_glb_scene`` builds a glTF
``baseColorTexture`` per part by baking per-vertex colours onto the part's
UV grid. Wiring therefore means:

1. Build analytic parts (``analytic_mesh.build_spec_meshes``).
2. For each part, sample the chosen tileable maps at the part's existing UVs
   (``apply_maps_to_part``) to get per-vertex colours, with the bump map
   folded in as cheap self-shadowing so micro-relief survives at vertex
   density.
3. Export through the stock path (``core.exporter.write_glb`` /
   ``write_glb_parts``). Passing the concatenated part vertices as
   ``positions`` and the sampled colours as ``colors`` takes the exporter's
   1:1 colour path, so the baked baseColorTexture *is* the sampled map.
4. Emit the ``textures`` manifest block (``textures_manifest_block``) next to
   the export so downstream tools can re-bind the original full-resolution
   maps (albedo/bump/roughness/alpha) instead of the vertex-baked albedo.

See docs/TEXTURES.md for the manifest schema (``ietexture/1``) and the bake
doctrine (when a texture beats geometry).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .texture_maps import generate_maps, map_filename

TEXTURES_BLOCK_SCHEMA = "ietexture/1"

_CHANNELS = ("albedo", "bump", "roughness", "alpha", "rgba", "normal")


# ---------------------------------------------------------------------------
# UV sampling
# ---------------------------------------------------------------------------


def sample_map(map_arr: np.ndarray, uvs: np.ndarray, *, wrap: bool = True) -> np.ndarray:
    """Bilinear-sample a (H, W) or (H, W, C) map at (N, 2) UVs.

    ``wrap=True`` (the default — all generated maps are tileable) treats UVs
    as repeating; otherwise they are clamped to [0, 1].
    """
    arr = np.asarray(map_arr)
    uvs = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    u = uvs[:, 0]
    v = uvs[:, 1]
    if wrap:
        u = u % 1.0
        v = v % 1.0
    else:
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
    h, w = arr.shape[:2]
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    if wrap:
        x1 = (x0 + 1) % w
        y1 = (y0 + 1) % h
        x0 %= w
        y0 %= h
    else:
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
    c00 = arr[y0, x0].astype(np.float32)
    c01 = arr[y0, x1].astype(np.float32)
    c10 = arr[y1, x0].astype(np.float32)
    c11 = arr[y1, x1].astype(np.float32)
    top = c00 + (c01 - c00) * fx[..., None] if arr.ndim == 3 else c00 + (c01 - c00) * fx
    bot = c10 + (c11 - c10) * fx[..., None] if arr.ndim == 3 else c10 + (c11 - c10) * fx
    out = top + (bot - top) * (fy[..., None] if arr.ndim == 3 else fy)
    return out


def apply_maps_to_part(
    part,
    maps: Mapping[str, np.ndarray],
    *,
    uv_scale: tuple[float, float] = (1.0, 1.0),
    bump_strength: float = 0.35,
) -> np.ndarray:
    """Sample `maps` at an AnalyticPart's UVs -> (V, 3) float32 colours.

    ``uv_scale`` repeats the tileable map across the part (e.g. (4, 1) wraps
    wood grain four times around a cylinder). ``bump_strength`` folds the
    bump/height channel into the albedo as cheap ambient occlusion so
    micro-relief (weave, pores, grooves) reads at export vertex density
    without any added geometry.
    """
    uvs = np.asarray(part.uvs, dtype=np.float64).copy()
    uvs[:, 0] *= float(uv_scale[0])
    uvs[:, 1] *= float(uv_scale[1])
    albedo = maps.get("albedo")
    if albedo is None:
        raise KeyError("maps must contain an 'albedo' channel")
    cols = sample_map(albedo, uvs) / 255.0
    bump = maps.get("bump")
    if bump is not None and bump_strength > 0.0:
        h = sample_map(bump, uvs) / 255.0  # (V,)
        shade = 1.0 - float(bump_strength) * (0.5 - h) * 2.0
        cols = cols * np.clip(shade, 0.0, 2.0)[:, None]
    return np.clip(cols, 0.0, 1.0).astype(np.float32)


def apply_maps_to_parts(
    parts: Sequence,
    assignments: Mapping[str, str | Mapping[str, np.ndarray]],
    *,
    size: int = 512,
    seed: int = 0,
    uv_scale: tuple[float, float] | Mapping[str, tuple[float, float]] = (1.0, 1.0),
    bump_strength: float = 0.35,
) -> tuple[list[np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Texture every part by label.

    ``assignments`` maps a part label to either a texture-kind name (maps are
    generated at ``size``/``seed`` and cached so parts sharing a kind reuse
    one map set) or a pre-generated channel dict.

    Returns ``(per_part_colors, generated_maps)`` where ``generated_maps`` is
    ``{kind: channel_dict}`` for the kinds that were generated — feed it to
    ``textures_manifest_block`` / ``texture_maps.save_maps``.
    """
    colors: list[np.ndarray] = []
    generated: dict[str, dict[str, np.ndarray]] = {}
    for i, part in enumerate(parts):
        label = part.label or f"part_{i}"
        spec = assignments.get(label) or assignments.get("*")
        if spec is None:
            # Untextured part: neutral mid-grey keeps the exporter happy.
            colors.append(np.full((part.vertices.shape[0], 3), 0.7, dtype=np.float32))
            continue
        if isinstance(spec, str):
            if spec not in generated:
                generated[spec] = generate_maps(spec, size=size, seed=seed)
            maps = generated[spec]
        else:
            maps = spec
        scale = uv_scale.get(label, (1.0, 1.0)) if isinstance(uv_scale, Mapping) else uv_scale
        colors.append(
            apply_maps_to_part(part, maps, uv_scale=scale, bump_strength=bump_strength)
        )
    return colors, generated


# ---------------------------------------------------------------------------
# map attachment (CR_TexReal image-map export path)
# ---------------------------------------------------------------------------
#
# ``apply_maps_to_part`` *samples* maps into per-vertex colours (the baked
# path). ``attach_maps_to_part`` instead hands the full-resolution maps to
# the GLB exporter by setting three duck-typed attributes on the part:
#
# - ``part.maps``      — channel dict from ``texture_maps.generate_maps``
#                        (``albedo`` required; ``bump``/``normal`` optional);
# - ``part.uv_scale``  — tile repeats across the part's UVs (exporter scales
#                        TEXCOORD_0, samplers wrap = repeat);
# - ``part.tint``      — optional (r, g, b) multiplicative tint exported as
#                        baseColorFactor + COLOR_0 so the texture still shows
#                        unmodulated (renderers multiply vertex colour ×
#                        texture; white keeps the map as authored).
#
# Parts without ``.maps`` keep the stock vertex-colour bake path untouched.


def attach_maps_to_part(
    part,
    maps: Mapping[str, np.ndarray],
    *,
    uv_scale: tuple[float, float] = (1.0, 1.0),
    tint: tuple[float, float, float] | None = None,
):
    """Attach full-resolution channel maps to a part for image-map GLB export.

    Returns the part (attributes are set in place; `AnalyticPart`/`BuiltPart`
    are plain dataclasses, so dynamic attributes are legal). Raises KeyError
    if the channel dict has no ``albedo``.
    """
    if maps.get("albedo") is None:
        raise KeyError("maps must contain an 'albedo' channel")
    part.maps = maps
    part.uv_scale = (float(uv_scale[0]), float(uv_scale[1]))
    if tint is not None:
        t = np.asarray(tint, dtype=np.float32).reshape(3)
        part.tint = tuple(float(c) for c in np.clip(t, 0.0, 1.0))
    return part


def attach_maps_to_parts(
    parts: Sequence,
    assignments: Mapping[str, str | Mapping[str, np.ndarray]],
    *,
    size: int = 512,
    seed: int = 0,
    uv_scale: tuple[float, float] | Mapping[str, tuple[float, float]] = (1.0, 1.0),
    tints: Mapping[str, tuple[float, float, float]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Attach maps to every part by label (``apply_maps_to_parts`` conventions).

    ``assignments`` maps a part label to a texture-kind name or a ready
    channel dict; ``"*"`` is the wildcard. Parts with no assignment are left
    untouched (vertex-colour bake path). Returns ``{kind: channel_dict}`` for
    the kinds that were generated (same kind shared across parts reuses one
    cached map set).
    """
    generated: dict[str, dict[str, np.ndarray]] = {}
    for i, part in enumerate(parts):
        label = part.label or f"part_{i}"
        spec = assignments.get(label) or assignments.get("*")
        if spec is None:
            continue
        if isinstance(spec, str):
            if spec not in generated:
                generated[spec] = generate_maps(spec, size=size, seed=seed)
            maps = generated[spec]
        else:
            maps = spec
        scale = uv_scale.get(label, (1.0, 1.0)) if isinstance(uv_scale, Mapping) else uv_scale
        tint = tints.get(label) if tints else None
        attach_maps_to_part(part, maps, uv_scale=scale, tint=tint)
    return generated


# ---------------------------------------------------------------------------
# manifest block (ietexture/1)
# ---------------------------------------------------------------------------


def textures_manifest_block(
    assignments: Sequence[Mapping[str, Any]],
    *,
    generated: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    map_files: Mapping[str, str] | None = None,
    size: int = 512,
    seed: int = 0,
    maps_dir: str = "textures",
) -> dict:
    """Build the ``textures`` manifest block mapping texture -> part -> channel.

    ``assignments`` is a list of records::

        {"part": "<label>", "material": "<hint>", "kind": "<texture kind>",
         "channels": ["albedo", "bump", ...],
         "uv": {"wrap": "repeat", "scale": [su, sv]}}     # optional

    ``map_files`` optionally maps ``"<kind>_<channel>"`` (or just a channel
    name for single-map records) to on-disk PNG paths (as produced by
    ``texture_maps.save_maps``); missing entries fall back to the canonical
    ``maps_dir/<kind>_<channel>_<size>px_s<seed>.png`` name so the block stays
    resolvable even before files are written.
    """
    maps: dict[str, dict] = {}
    out_assignments: list[dict] = []
    for rec in assignments:
        part = str(rec["part"])
        kind = str(rec.get("kind") or rec.get("material") or "custom")
        channels = list(rec.get("channels") or ["albedo"])
        uv = dict(rec.get("uv") or {"wrap": "repeat", "scale": [1, 1]})
        chan_ids: dict[str, str] = {}
        for ch in channels:
            map_id = f"{kind}_{ch}"
            chan_ids[ch] = map_id
            if map_id in maps:
                continue
            file_rel = None
            if map_files:
                file_rel = map_files.get(map_id) or map_files.get(ch)
            if file_rel is None:
                file_rel = f"{maps_dir}/{map_filename(kind, ch, size, seed)}"
            maps[map_id] = {
                "file": str(file_rel),
                "kind": kind,
                "channel": ch,
                "size": int(size),
                "seed": int(seed),
                "tileable": True,
                "format": "png",
            }
        out_assignments.append(
            {
                "part": part,
                "material": rec.get("material"),
                "maps": chan_ids,
                "uv": uv,
            }
        )
    return {
        "schema": TEXTURES_BLOCK_SCHEMA,
        "maps": maps,
        "assignments": out_assignments,
    }


def validate_textures_block(block: Mapping[str, Any]) -> list[str]:
    """Structural validation; returns a list of problems ([] = valid)."""
    errors: list[str] = []
    if not isinstance(block, Mapping):
        return ["block is not a mapping"]
    if block.get("schema") != TEXTURES_BLOCK_SCHEMA:
        errors.append(f"schema must be {TEXTURES_BLOCK_SCHEMA!r}")
    maps = block.get("maps")
    assigns = block.get("assignments")
    if not isinstance(maps, Mapping) or not maps:
        errors.append("'maps' must be a non-empty mapping")
        maps = {}
    if not isinstance(assigns, Sequence) or isinstance(assigns, (str, bytes)):
        errors.append("'assignments' must be a list")
        assigns = []
    for map_id, meta in maps.items():
        if not isinstance(meta, Mapping):
            errors.append(f"map {map_id!r}: metadata is not a mapping")
            continue
        ch = meta.get("channel")
        if ch not in _CHANNELS:
            errors.append(f"map {map_id!r}: unknown channel {ch!r}")
        f = meta.get("file")
        if not (isinstance(f, str) and f.lower().endswith(".png")):
            errors.append(f"map {map_id!r}: file must be a .png path")
    for rec in assigns:
        if not isinstance(rec, Mapping) or "part" not in rec:
            errors.append("assignment missing 'part'")
            continue
        chan = rec.get("maps")
        if not isinstance(chan, Mapping) or not chan:
            errors.append(f"assignment {rec.get('part')!r}: 'maps' must be non-empty")
            continue
        for ch, map_id in chan.items():
            if map_id not in maps:
                errors.append(f"assignment {rec['part']!r}: map id {map_id!r} not in 'maps'")
    return errors


# ---------------------------------------------------------------------------
# bake doctrine — geometry detail -> texture maps
# ---------------------------------------------------------------------------

_DetailFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray]


def bake_detail_to_texture(
    detail: _DetailFn | np.ndarray,
    *,
    size: int = 512,
    seed: int = 0,
    base_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
    ao_strength: float = 0.4,
    normal_strength: float = 2.0,
    kind: str = "custom",
) -> dict[str, np.ndarray]:
    """Bake high-frequency procedural detail into maps instead of geometry.

    ``detail`` is either a callable ``detail(u, v, rng) -> height`` evaluated
    on tileable (H, W) grids (u, v in [0, 1), height in [0, 1], integer
    repeats keep it tileable) or a ready (H, W) height array. Returns:

    - ``"albedo"``  — base colour with the height folded in as ambient
      occlusion (crevices darken), (size, size, 3) uint8;
    - ``"bump"``    — the height field itself, (size, size) uint8;
    - ``"normal"``  — tangent-space normal map derived from the height
      gradient, (size, size, 3) uint8.

    This is the canonical replacement for modelling micro-geometry (weave
    threads, carved flutes, pores): zero triangles, one texture lookup.
    See docs/TEXTURES.md ("Bake doctrine") for the performance budget.
    """
    size = int(size)
    rng = np.random.default_rng(seed)
    if callable(detail):
        t = (np.arange(size, dtype=np.float32) + 0.5) / size
        u, v = np.meshgrid(t, t)
        height = np.asarray(detail(u, v, rng), dtype=np.float32)
    else:
        height = np.asarray(detail, dtype=np.float32)
        if height.shape != (size, size):
            raise ValueError(
                f"detail array must be ({size}, {size}); got {height.shape} "
                "(resize it yourself or pass a callable)"
            )
    height = np.clip(height, 0.0, 1.0)

    base = np.asarray(base_color, dtype=np.float32)
    shade = np.clip(1.0 - float(ao_strength) * (0.5 - height) * 2.0, 0.0, 2.0)
    albedo = np.clip(base[None, None, :] * shade[..., None], 0, 1)

    # Tangent-space normal map from the height gradient; wrapping central
    # differences keep the normal map tileable.
    gx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    gy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    nz = np.ones_like(height) / max(float(normal_strength), 1e-6)
    n = np.stack([-gx, -gy, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    normal_u8 = np.clip(n * 0.5 + 0.5, 0, 1)

    return {
        "albedo": (albedo * 255.0).round().astype(np.uint8),
        "bump": (height * 255.0).round().astype(np.uint8),
        "normal": (normal_u8 * 255.0).round().astype(np.uint8),
    }


# --- ready-made high-frequency detail fields (geometry replacements) --------


def weave_detail(threads: int = 48) -> _DetailFn:
    """Over/under thread weave height — replaces modelled cloth threads."""
    f = int(threads)

    def fn(u: np.ndarray, v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        warp = 0.5 + 0.5 * np.sin(2.0 * np.pi * f * u)
        weft = 0.5 + 0.5 * np.sin(2.0 * np.pi * f * v)
        over = ((np.floor(u * f) + np.floor(v * f)) % 2).astype(np.float32)
        return over * warp + (1.0 - over) * weft

    return fn


def flute_detail(flutes: int = 24, depth_profile: float = 1.0) -> _DetailFn:
    """Carved vertical flutes (column shafts, rococo frames) as height."""
    k = int(flutes)

    def fn(u: np.ndarray, v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return (0.5 + 0.5 * np.sin(2.0 * np.pi * k * u)) ** float(depth_profile)

    return fn


def pore_detail(cells: int = 16, threshold: float = 0.75) -> _DetailFn:
    """Pitted pore field (stone, ceramic glaze, leather) as height."""
    c = int(cells)

    def fn(u: np.ndarray, v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        base = 0.6 + 0.4 * np.sin(2.0 * np.pi * (u + v))
        pits = rng.random((c, c)).astype(np.float32)
        ix = (np.floor(u * c).astype(np.int64)) % c
        iy = (np.floor(v * c).astype(np.int64)) % c
        pit = (pits[iy, ix] > threshold).astype(np.float32)
        return np.clip(base * 0.3 + 0.7 * (1.0 - pit), 0, 1)

    return fn
