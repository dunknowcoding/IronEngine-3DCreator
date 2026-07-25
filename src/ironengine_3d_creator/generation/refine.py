"""Post-generation mesh refinement (CR_TexReal).

Hero assets stop looking like stacked low-poly shapes here:

- **Subdivision** — midpoint 1-to-4 triangle splits with Loop-lite smoothing.
  Sharp edges (dihedral angle above ``crease_deg``) and boundary edges are
  detected and preserved: their new points sit at exact edge midpoints and
  crease vertices follow the Loop crease rule (exactly 2 crease neighbours)
  or stay pinned, so a crate stays crisp while skin/cloth/organic parts
  round out.
- **Tangent-aware smoothing** — optional Taubin λ|μ passes (volume-neutral:
  the negative-μ pass re-inflates what the λ pass shrinks) restricted to
  non-crease adjacency; crease and boundary vertices are pinned.
- **Procedural displacement** — offsets along (recomputed) vertex normals
  from a callable, an explicit per-vertex height array, or a
  ``texture_maps`` bump channel sampled at the part UVs
  (``bump_displacement``) — micro-relief for skin/cloth/organic shapes
  without modelling it.
- **Guardrails** — ``tri_budget`` clamps the subdivision level (the largest
  level whose output still fits is used; requested vs. applied levels are
  recorded) and every refinement ends with the
  ``analytic_mesh.weld_mesh`` degenerate-face cleanup.

API::

    refined = refine_mesh(part_or_arrays, levels=1, crease_deg=30.0,
                          displacement=bump_displacement("knit_wool", 0.0006))
    part2 = refine_part(analytic_part, levels=1)          # keeps metadata
    parts2 = refine_garment(parts, weave="knit_wool")     # cloth thickness + weave
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np

from .analytic_mesh import AnalyticPart, signed_volume, weld_mesh

__all__ = [
    "RefinedMesh",
    "refine_mesh",
    "refine_part",
    "bump_displacement",
    "solidify_shell",
    "is_cloth_part",
    "refine_garment",
]

DEFAULT_TRI_BUDGET = 200_000
GARMENT_TRI_BUDGET = 150_000

# Materials treated as cloth by refine_garment (plus any part whose metadata
# carries a "garment" key, e.g. generation.clothing parts).
CLOTH_MATERIALS = frozenset({"fabric", "cloth", "knit", "wool", "denim", "linen"})

_Displacement = (
    Callable[[np.ndarray, np.ndarray, np.ndarray | None, np.random.Generator], np.ndarray]
    | np.ndarray
    | Mapping
)


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class RefinedMesh:
    """Refined arrays plus provenance (level clamping, crease statistics)."""

    vertices: np.ndarray      # (V, 3) float32
    normals: np.ndarray       # (V, 3) float32
    uvs: np.ndarray | None    # (V, 2) float32 or None
    faces: np.ndarray         # (F, 3) int64
    levels_applied: int
    levels_requested: int
    triangles_in: int
    triangles_out: int
    crease_edges: int


# ---------------------------------------------------------------------------
# topology helpers (vectorized)
# ---------------------------------------------------------------------------


def _unique_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique undirected edges + per-edge incident faces.

    Returns ``(edges, face_edges, inverse)``: ``edges`` is (E, 2) int64 with
    i < j; ``face_edges`` is (E, 3) — the faces containing each edge (-1
    padded, so ``face_edges[e, 1] < 0`` marks a boundary edge); ``inverse``
    maps each (F, 3) face-edge to its edge id.
    """
    f = np.asarray(faces, dtype=np.int64)
    fe = np.stack(
        [f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1
    ).reshape(-1, 2)
    fe_sorted = np.sort(fe, axis=1)
    edges, inverse = np.unique(fe_sorted, axis=0, return_inverse=True)
    n_e = edges.shape[0]
    face_ids = np.repeat(np.arange(f.shape[0]), 3)
    order = np.argsort(inverse, kind="stable")
    inv_s = inverse[order]
    face_s = face_ids[order]
    counts = np.bincount(inv_s, minlength=n_e).astype(np.int64)
    starts = np.zeros(n_e, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    rank = np.arange(order.shape[0], dtype=np.int64) - starts[inv_s]
    face_edges = np.full((n_e, 3), -1, dtype=np.int64)
    valid = rank < 3
    face_edges[inv_s[valid], rank[valid]] = face_s[valid]
    return edges, face_edges, inverse.reshape(f.shape[0], 3)


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]].astype(np.float64)
    e1 = vertices[faces[:, 1]].astype(np.float64) - v0
    e2 = vertices[faces[:, 2]].astype(np.float64) - v0
    n = np.cross(e1, e2)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1e-30)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted smooth vertex normals.

    Vertices whose adjacent faces are all degenerate (e.g. duplicated pole
    vertices) fall back to the centroid direction — never a zero vector.
    """
    v = vertices.astype(np.float64)
    v0 = v[faces[:, 0]]
    fn = np.cross(v[faces[:, 1]] - v0, v[faces[:, 2]] - v0)  # |.| = 2 * area
    n = np.zeros_like(v)
    flat = faces.reshape(-1)
    np.add.at(n, flat, np.repeat(fn, 3, axis=0))
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    zero = (norm[:, 0] < 1e-30)
    if zero.any():
        d = v - v.mean(axis=0, keepdims=True)
        dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
        n[zero] = dn[zero]
        norm = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.maximum(norm, 1e-30)).astype(np.float32)


def _crease_mask(
    edges: np.ndarray,
    face_edges: np.ndarray,
    face_nrm: np.ndarray,
    crease_deg: float,
) -> np.ndarray:
    """Boolean (E,): True where the edge is a hard crease or a boundary."""
    cos_thresh = math.cos(math.radians(float(crease_deg)))
    f0 = face_edges[:, 0]
    f1 = face_edges[:, 1]
    boundary = f1 < 0
    dot = np.ones(edges.shape[0], dtype=np.float64)
    shared = ~boundary
    dot[shared] = np.einsum(
        "ij,ij->i", face_nrm[f0[shared]], face_nrm[f1[shared]]
    )
    return boundary | (dot < cos_thresh)


def _adjacency_sums(
    edges: np.ndarray, v: np.ndarray, n_v: int
) -> tuple[np.ndarray, np.ndarray]:
    """(count, position-sum) of edge-neighbours per vertex (vectorized)."""
    if edges.shape[0] == 0:
        return np.zeros(n_v, dtype=np.int64), np.zeros((n_v, 3), dtype=np.float64)
    ei, ej = edges[:, 0], edges[:, 1]
    cnt = np.bincount(np.concatenate([ei, ej]), minlength=n_v).astype(np.int64)
    ssum = np.zeros((n_v, 3), dtype=np.float64)
    np.add.at(ssum, ei, v[ej])
    np.add.at(ssum, ej, v[ei])
    return cnt, ssum


# ---------------------------------------------------------------------------
# subdivision (midpoint split + Loop-lite repositioning)
# ---------------------------------------------------------------------------


def _subdivide_once(
    vertices: np.ndarray,
    uvs: np.ndarray | None,
    faces: np.ndarray,
    crease_deg: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int]:
    v = vertices.astype(np.float64)
    f = np.asarray(faces, dtype=np.int64)
    edges, face_edges, inverse = _unique_edges(f)
    face_nrm = _face_normals(v, f)
    crease = _crease_mask(edges, face_edges, face_nrm, crease_deg)
    n_crease = int(crease.sum())

    # --- new edge points ---------------------------------------------------
    ea = v[edges[:, 0]]
    eb = v[edges[:, 1]]
    new_pts = (ea + eb) * 0.5
    smooth_idx = np.nonzero(~crease)[0]
    if smooth_idx.size:
        # Loop rule: 3/8 (A + B) + 1/8 (C + D) across the two adjacent faces.
        fid0 = face_edges[smooth_idx, 0]
        fid1 = face_edges[smooth_idx, 1]
        tri0 = f[fid0]
        tri1 = f[fid1]
        a = edges[smooth_idx, 0]
        b = edges[smooth_idx, 1]
        # Opposite corner = the one face corner that is neither endpoint
        # (index-sum identity is exact for integer vertex ids).
        c0 = tri0.sum(axis=1) - a - b
        c1 = tri1.sum(axis=1) - a - b
        new_pts[smooth_idx] = 0.375 * (ea[smooth_idx] + eb[smooth_idx]) + 0.125 * (
            v[c0] + v[c1]
        )

    n_v = v.shape[0]
    edge_vid = np.arange(n_v, n_v + edges.shape[0], dtype=np.int64)
    mid = edge_vid[inverse]  # (F, 3) new vertex id per face-edge

    # --- rebuild faces (1 -> 4) ---------------------------------------------
    a_f, b_f, c_f = f[:, 0], f[:, 1], f[:, 2]
    mab, mbc, mca = mid[:, 0], mid[:, 1], mid[:, 2]
    new_faces = np.stack(
        [
            np.stack([a_f, mab, mca], axis=1),
            np.stack([mab, b_f, mbc], axis=1),
            np.stack([mca, mbc, c_f], axis=1),
            np.stack([mab, mbc, mca], axis=1),
        ],
        axis=1,
    ).reshape(-1, 3)

    # --- reposition existing vertices (Loop-lite, crease-aware) -------------
    s_cnt, s_sum = _adjacency_sums(edges[~crease], v, n_v)
    c_cnt, c_sum = _adjacency_sums(edges[crease], v, n_v)
    beta = np.where(
        s_cnt == 3, 3.0 / 16.0, 3.0 / (8.0 * np.maximum(s_cnt, 1))
    )[:, None]
    smooth_target = (1.0 - s_cnt[:, None] * beta) * v + beta * s_sum
    crease_target = 0.75 * v + 0.125 * c_sum
    moved = v.copy()
    # The Loop crease rule assumes the two crease neighbours continue one
    # smooth crease curve. Where the crease takes a sharp turn at the vertex
    # (box corners, panel junctions) the rule would drag the corner inward —
    # pin those instead. Straight crease: neighbours point ~opposite ways.
    is_c2 = c_cnt == 2
    if is_c2.any():
        nbr = np.full((n_v, 2), -1, dtype=np.int64)
        c_edges = edges[crease]
        both = np.concatenate([c_edges, c_edges[:, ::-1]], axis=0)
        vids, nids = both[:, 0], both[:, 1]
        slot0 = nbr[vids, 0] < 0
        nbr[vids[slot0], 0] = nids[slot0]
        nbr[vids[~slot0], 1] = nids[~slot0]
        e1 = v[nbr[:, 0]] - v
        e2 = v[nbr[:, 1]] - v
        cos_turn = np.einsum("ij,ij->i", e1, e2) / (
            np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-30
        )
        smooth_crease = is_c2 & (cos_turn < -math.cos(math.radians(float(crease_deg))))
    else:
        smooth_crease = is_c2
    pinned = (c_cnt > 2) | (is_c2 & ~smooth_crease) | ((c_cnt == 1) & (s_cnt == 0))
    free_smooth = ~smooth_crease & ~pinned & (s_cnt > 0)
    moved[smooth_crease] = crease_target[smooth_crease]
    moved[free_smooth] = smooth_target[free_smooth]

    out_v = np.concatenate([moved, new_pts], axis=0).astype(np.float32)
    if uvs is not None:
        uv = np.asarray(uvs, dtype=np.float64)
        mid_uv = (uv[edges[:, 0]] + uv[edges[:, 1]]) * 0.5
        out_uv = np.concatenate([uv, mid_uv], axis=0).astype(np.float32)
    else:
        out_uv = None
    return out_v, out_uv, new_faces.astype(np.int64), n_crease


# ---------------------------------------------------------------------------
# Taubin smoothing (volume-neutral, crease/boundary pinned)
# ---------------------------------------------------------------------------


def _taubin(
    vertices: np.ndarray,
    faces: np.ndarray,
    crease_deg: float,
    iters: int,
    lam: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    if iters <= 0:
        return vertices
    v = vertices.astype(np.float64)
    f = np.asarray(faces, dtype=np.int64)
    edges, face_edges, _inv = _unique_edges(f)
    crease = _crease_mask(edges, face_edges, _face_normals(v, f), crease_deg)
    keep = edges[~crease]
    n_v = v.shape[0]
    ei, ej = keep[:, 0], keep[:, 1]
    cnt = np.bincount(np.concatenate([ei, ej]), minlength=n_v).astype(np.float64)
    free = cnt > 0
    for step in range(int(iters) * 2):
        factor = lam if step % 2 == 0 else mu
        ssum = np.zeros((n_v, 3), dtype=np.float64)
        np.add.at(ssum, ei, v[ej])
        np.add.at(ssum, ej, v[ei])
        mean = ssum[free] / cnt[free, None]
        v[free] += factor * (mean - v[free])
    return v.astype(np.float32)


# ---------------------------------------------------------------------------
# displacement
# ---------------------------------------------------------------------------


def bump_displacement(
    kind: str,
    scale: float,
    *,
    size: int = 512,
    seed: int = 0,
    uv_scale: tuple[float, float] = (1.0, 1.0),
) -> dict:
    """Displacement spec sampling a ``texture_maps`` bump channel at part UVs.

    ``scale`` is the peak displacement from the mid height in mesh units
    (metres); the bump value 0.5 is neutral so relief goes both ways.
    """
    return {
        "type": "bump",
        "kind": str(kind),
        "scale": float(scale),
        "size": int(size),
        "seed": int(seed),
        "uv_scale": (float(uv_scale[0]), float(uv_scale[1])),
    }


def _resolve_displacement(
    disp: _Displacement,
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """-> (V,) float offsets along the vertex normal."""
    n_v = positions.shape[0]
    if isinstance(disp, Mapping):
        if disp.get("type") != "bump":
            raise ValueError(f"unknown displacement spec {dict(disp)!r}")
        if uvs is None:
            raise ValueError("bump displacement requires mesh UVs")
        from .texture_maps import generate_maps
        from .texture_apply import sample_map

        maps = generate_maps(
            disp["kind"], size=int(disp.get("size", 512)), seed=int(disp.get("seed", 0))
        )
        bump = maps.get("bump")
        if bump is None:
            raise ValueError(f"texture kind {disp['kind']!r} has no bump channel")
        su, sv = disp.get("uv_scale", (1.0, 1.0))
        uv = np.asarray(uvs, dtype=np.float64).copy()
        uv[:, 0] *= su
        uv[:, 1] *= sv
        h = sample_map(bump, uv) / 255.0
        return ((h - 0.5) * float(disp["scale"])).astype(np.float64)
    if callable(disp):
        out = np.asarray(
            disp(positions, normals, uvs, rng), dtype=np.float64
        ).reshape(-1)
    else:
        out = np.asarray(disp, dtype=np.float64).reshape(-1)
    if out.shape[0] != n_v:
        raise ValueError(f"displacement has {out.shape[0]} entries for {n_v} vertices")
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def _extract(mesh):
    """(vertices, normals, uvs, faces) from an AnalyticPart-like object or a
    (vertices, faces) / (vertices, faces, uvs) tuple."""
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return (
            np.asarray(mesh.vertices, dtype=np.float32),
            getattr(mesh, "normals", None),
            getattr(mesh, "uvs", None),
            np.asarray(mesh.faces, dtype=np.int64),
        )
    if isinstance(mesh, (tuple, list)) and len(mesh) in (2, 3):
        v, f = mesh[0], mesh[1]
        uv = mesh[2] if len(mesh) == 3 else None
        return (
            np.asarray(v, dtype=np.float32),
            None,
            np.asarray(uv, dtype=np.float32) if uv is not None else None,
            np.asarray(f, dtype=np.int64),
        )
    raise TypeError(
        "mesh must be an AnalyticPart-like object or a (vertices, faces[, uvs]) tuple"
    )


def refine_mesh(
    mesh,
    levels: int = 1,
    crease_deg: float = 30.0,
    displacement: _Displacement | None = None,
    *,
    tri_budget: int = DEFAULT_TRI_BUDGET,
    smooth_iters: int = 0,
    seed: int = 0,
) -> RefinedMesh:
    """Refine a triangle mesh: crease-aware subdivision + optional displacement.

    ``levels`` is clamped so ``faces * 4**levels_applied <= tri_budget``
    (largest fitting level wins; level 0 = no subdivision — displacement and
    cleanup still run). ``smooth_iters`` adds volume-neutral Taubin passes
    (default 0 — Loop subdivision already smooths). ``displacement`` is a
    ``bump_displacement(...)`` spec, a callable
    ``f(positions, normals, uvs, rng) -> (V,)`` or a (V,) height array;
    deterministic for a given ``seed``.
    """
    v, _n, uv, f = _extract(mesh)
    tris_in = int(f.shape[0])
    if tris_in == 0:
        raise ValueError("cannot refine an empty mesh")
    levels = max(0, int(levels))
    budget = max(4, int(tri_budget))
    applied = 0
    n_crease = 0
    for _lvl in range(levels):
        if f.shape[0] * 4 > budget:
            break  # guardrail: next level would exceed the triangle budget
        v, uv, f, n_crease = _subdivide_once(v, uv, f, crease_deg)
        applied += 1
    if smooth_iters > 0:
        v = _taubin(v, f, crease_deg, int(smooth_iters))

    nrm = _vertex_normals(v, f)
    if displacement is not None:
        rng = np.random.default_rng(int(seed))
        offsets = _resolve_displacement(displacement, v, nrm, uv, rng)
        v = (v.astype(np.float64) + nrm.astype(np.float64) * offsets[:, None]).astype(
            np.float32
        )
        nrm = _vertex_normals(v, f)

    # Degenerate-face cleanup (reuse the repo's weld pass). The computed
    # normals MUST stay in the weld key: per-face [0, 1]^2 UV projections
    # (box, plane) make position+uv keys collide across hard edges, and
    # dropping the normal key would merge those splits and bend the shading.
    v, nrm_w, uv_w, f = weld_mesh(v, nrm, uv, f)
    if nrm_w is not None:
        nrm = nrm_w
    return RefinedMesh(
        vertices=np.asarray(v, dtype=np.float32),
        normals=np.asarray(nrm, dtype=np.float32),
        uvs=np.asarray(uv_w, dtype=np.float32) if uv_w is not None else None,        faces=np.asarray(f, dtype=np.int64),
        levels_applied=applied,
        levels_requested=levels,
        triangles_in=tris_in,
        triangles_out=int(f.shape[0]),
        crease_edges=n_crease,
    )


def refine_part(part: AnalyticPart, *args, **kwargs) -> AnalyticPart:
    """``refine_mesh`` for an ``AnalyticPart`` — metadata is preserved, the
    AABB and solid volume are recomputed."""
    r = refine_mesh(part, *args, **kwargs)
    return replace(
        part,
        vertices=r.vertices,
        normals=r.normals,
        uvs=r.uvs if r.uvs is not None else part.uvs,
        faces=r.faces,
        aabb_min=r.vertices.min(axis=0),
        aabb_max=r.vertices.max(axis=0),
        solid_volume_m3=abs(signed_volume(r.vertices, r.faces)),
    )


# ---------------------------------------------------------------------------
# garment upgrade (cloth thickness + weave)
# ---------------------------------------------------------------------------


def solidify_shell(
    vertices: np.ndarray,
    normals: np.ndarray | None,
    uvs: np.ndarray | None,
    faces: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Give an open shell real thickness: an inner offset copy plus stitched
    boundary walls, yielding a closed two-sided mesh.

    ``thickness`` is the shell gap in mesh units (metres); the inner copy is
    offset along the negated vertex normals. Winding of the stitched walls is
    normalised via the signed-volume sign so the result is outward-facing.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = (
        np.asarray(normals, dtype=np.float64)
        if normals is not None
        else _vertex_normals(v, f).astype(np.float64)
    )
    n_v = v.shape[0]
    inner = v - n * float(thickness)
    out_v = np.concatenate([v, inner], axis=0)
    out_f = [f, f[:, ::-1] + n_v]
    out_uv: np.ndarray | None = (
        np.concatenate([np.asarray(uvs, dtype=np.float64)] * 2, axis=0)
        if uvs is not None
        else None
    )

    # Boundary edges (used by exactly one face) get stitched to their copies.
    edges, face_edges, _inv = _unique_edges(f)
    boundary = edges[face_edges[:, 1] < 0]
    if boundary.shape[0]:
        a, b = boundary[:, 0], boundary[:, 1]
        walls = np.stack(
            [
                np.stack([a, b, b + n_v], axis=1),
                np.stack([a, b + n_v, a + n_v], axis=1),
            ],
            axis=1,
        ).reshape(-1, 3)
        out_f.append(walls)
    out_f = np.concatenate(out_f, axis=0).astype(np.int64)
    if signed_volume(out_v, out_f) < 0.0:
        out_f = out_f[:, ::-1].copy()
    out_n = _vertex_normals(out_v, out_f)
    out_v, out_n_w, out_uv_w, out_f = weld_mesh(
        out_v.astype(np.float32), out_n, out_uv, out_f
    )
    return (
        np.asarray(out_v, dtype=np.float32),
        np.asarray(out_n_w if out_n_w is not None else _vertex_normals(out_v, out_f)),
        np.asarray(out_uv_w, dtype=np.float32) if out_uv_w is not None else None,
        np.asarray(out_f, dtype=np.int64),
    )


def is_cloth_part(part) -> bool:
    """True for garment/cloth parts (metadata 'garment' or a cloth material)."""
    md = getattr(part, "metadata", None) or {}
    if md.get("garment"):
        return True
    return str(getattr(part, "material", "")).lower() in CLOTH_MATERIALS


def refine_garment(
    parts: Sequence,
    *,
    thickness: float = 0.002,
    levels: int = 0,
    crease_deg: float = 28.0,
    weave: str | None = "knit_wool",
    weave_uv_scale: tuple[float, float] = (8.0, 8.0),
    weave_seed: int = 0,
    displacement_scale: float = 0.0,
    tri_budget: int = GARMENT_TRI_BUDGET,
):
    """Upgrade cloth garment parts: real thickness + optional weave texture.

    For every cloth part (``is_cloth_part``): the shell is solidified by
    ``thickness`` (open cloth tubes become closed two-sided garments),
    optionally subdivided (``levels`` > 0) and displaced by the weave bump
    (``displacement_scale`` > 0), and — when ``weave`` names a texture kind —
    the full-resolution weave maps are attached for the image-map GLB export
    path (``texture_apply.attach_maps_to_part``). Non-cloth parts pass
    through unchanged (same object). Returns a new list.

    Works on any part duck-type with ``vertices``/``normals``/``uvs``/
    ``faces`` (``BuiltPart`` from ``PartGraph.build()`` or ``AnalyticPart``);
    the input part type is preserved via ``dataclasses.replace``.
    """
    from .texture_apply import attach_maps_to_part
    from .texture_maps import generate_maps

    out = []
    for part in parts:
        if not is_cloth_part(part):
            out.append(part)
            continue
        v, n, uv, f = solidify_shell(
            part.vertices, getattr(part, "normals", None),
            getattr(part, "uvs", None), part.faces, thickness,
        )
        if levels > 0 or displacement_scale > 0.0:
            disp = None
            if displacement_scale > 0.0 and weave:
                disp = bump_displacement(
                    weave, displacement_scale, seed=weave_seed, uv_scale=weave_uv_scale
                )
            r = refine_mesh(
                (v, f, uv) if uv is not None else (v, f),
                levels=levels,
                crease_deg=crease_deg,
                displacement=disp,
                tri_budget=tri_budget,
                seed=weave_seed,
            )
            v, n, uv, f = r.vertices, r.normals, r.uvs, r.faces
        kwargs: dict = dict(vertices=v, normals=n, faces=f)
        if uv is not None:
            kwargs["uvs"] = uv
        # Refresh AABBs / volume when the part carries them.
        if hasattr(part, "local_aabb_min"):
            kwargs["local_aabb_min"] = v.min(axis=0)
            kwargs["local_aabb_max"] = v.max(axis=0)
        elif hasattr(part, "aabb_min"):
            kwargs["aabb_min"] = v.min(axis=0)
            kwargs["aabb_max"] = v.max(axis=0)
        if hasattr(part, "solid_volume_m3"):
            kwargs["solid_volume_m3"] = abs(signed_volume(v, f))
        new_part = replace(part, **kwargs)
        if weave:
            maps = generate_maps(weave, size=512, seed=weave_seed)
            attach_maps_to_part(new_part, maps, uv_scale=weave_uv_scale)
        out.append(new_part)
    return out
