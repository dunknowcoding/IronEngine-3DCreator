"""Multi-view projection QA for generated models (specs and meshes).

Extends `generation.reference` (silhouette IoU / parts coverage, REUSED here,
not modified) from single-view similarity to a full multi-view audit:

* projection to 6 canonical orthographic views (front/back/left/right/
  top/bottom) plus 2 deterministic perspective 3/4 views;
* per-view silhouette boundary completeness — a contour-closure score that
  flags open/broken boundaries caused by missing faces or NaN geometry.
  The renderer keeps a painter-overwritten *visible winding sign* per pixel
  (nearest surface wins): a closed solid shows front-facing surface
  everywhere, while an opening exposes back-facing interior. A global
  winding vote makes the metric immune to mesh winding convention;
* mesh-level open-edge detection on position-welded topology (analytic
  meshes duplicate vertices along seams, so edges are counted after welding);
* internal detail density — wireframe edge pixels inside the silhouette,
  catching "too simple" outputs (a blob scores far below a detailed model);
* bounding-box sanity against the declared real-world scale and a fuzzy
  category plausibility table (flags e.g. a 10 m cat);
* part-label projection check — every named part must be visible from at
  least one canonical view;
* reference comparison per canonical view against the real-object corpus
  (`reference.load_reference_masks`, IoU + contour chamfer distance), with an
  LLM-generated-reference fallback cached under
  ``<corpus>/_generated/<object>/<view>.png`` (provenance recorded).

Public API:
    as_parts(obj) -> list[MeshPart]
    project_views(obj, views=ALL_VIEWS, size=CANVAS) -> dict[str, ViewRender]
    boundary_completeness(render) -> dict
    detail_density(render) -> dict
    geometry_integrity(parts) -> dict
    scale_sanity(parts, declared_bbox, object_name) -> dict
    part_visibility(renders, parts) -> dict
    compare_to_reference(obj, object_name, size) -> dict
    ensure_llm_reference(object_name, view, prompt, generator) -> Path | None
    qa_report(obj, object_name=None, declared_bbox=None, size) -> dict
    worst_issues(report, n) -> list[dict]
    render_contact_sheet(renders, out_path, title) -> Path

Everything is headless (numpy + cv2 painter rasterization), deterministic
(fixed view transforms, no RNG), and degrades gracefully when the corpus or
optional inputs are missing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import reference  # reused, unmodified
from .analytic_mesh import build_spec_meshes

_log = logging.getLogger(__name__)

CANVAS = reference.CANVAS  # 256, keep parity with reference.py

ORTHO_VIEWS = ("front", "back", "left", "right", "top", "bottom")
PERSP_VIEWS = ("three_quarter", "three_quarter_back")
ALL_VIEWS = ORTHO_VIEWS + PERSP_VIEWS

# corpus view tag -> multiview canonical view
_CORPUS_VIEW_MAP = {"front": "front", "back": "back", "side": "right", "top": "top"}

MIN_PART_PX = 24          # a part counts as visible in a view from this many px
HOLE_MIN_PX = 8           # ignore enclosed background specks smaller than this
WELD_EPS = 1e-6           # vertex weld tolerance (m) for topology analysis
SCALE_WARN = 1.5          # category-range soft tolerance factor
SCALE_ERR = 3.0           # category-range hard tolerance factor
BBOX_WARN = 2.0           # declared-bbox ratio soft tolerance
BBOX_ERR = 4.0            # declared-bbox ratio hard tolerance
COMPLETE_WARN = 0.75      # per-view completeness soft floor
COMPLETE_ERR = 0.50       # per-view completeness hard floor

# fuzzy category -> plausible (min, max) longest-dimension in meters
CATEGORY_SCALE_RANGES: dict[str, tuple[float, float]] = {
    "ladybug": (0.004, 0.03), "butterfly": (0.02, 0.3), "spider": (0.005, 0.35),
    "ring": (0.01, 0.06), "goldfish": (0.02, 0.45),
    "apple": (0.04, 0.2), "orange": (0.05, 0.16), "pear": (0.06, 0.2),
    "banana": (0.1, 0.35), "carrot": (0.1, 0.4), "cheese": (0.05, 0.45),
    "bread": (0.15, 0.6), "grape": (0.1, 0.5),
    "mug": (0.05, 0.25), "sunglasses": (0.08, 0.3),
    "sneaker": (0.15, 0.5), "shoe": (0.15, 0.5), "boot": (0.2, 0.6),
    "hat": (0.15, 0.55), "cap": (0.1, 0.4), "handbag": (0.15, 0.65),
    "necklace": (0.1, 0.7), "clock": (0.1, 1.2),
    "cat": (0.35, 1.4), "bird": (0.05, 1.7), "dog": (0.35, 1.9),
    "horse": (1.5, 3.2), "flower": (0.05, 1.1), "rose": (0.05, 1.1),
    "leaf": (0.02, 0.6), "fern": (0.1, 1.4), "grass": (0.05, 1.2),
    "monstera": (0.2, 2.2), "bush": (0.3, 3.2), "hedge": (0.3, 3.2),
    "sunflower": (0.5, 3.8), "tree": (2.0, 40.0), "oak": (2.0, 40.0),
    "pine": (2.0, 45.0), "vase": (0.08, 1.2), "lamp": (0.12, 2.4),
    "pendant": (0.15, 2.2), "human": (1.2, 2.4), "person": (1.2, 2.4),
    "chair": (0.45, 1.7), "bench": (0.9, 2.8), "table": (0.45, 3.2),
    "sofa": (1.1, 3.4), "shelf": (0.7, 3.2), "bookshelf": (0.7, 3.2),
    "door": (1.5, 2.8), "window": (0.35, 4.2), "cart": (0.7, 1.8),
    "stall": (1.4, 5.5), "vending": (0.9, 2.4), "sign": (0.15, 4.5),
    "bicycle": (1.2, 2.4), "scooter": (1.1, 2.4), "sedan": (3.4, 5.8),
    "sports_car": (3.6, 5.4), "suv": (4.0, 6.0), "truck": (4.0, 10.0),
    "bus": (7.0, 16.0), "car": (3.0, 6.2),
    "fence": (0.7, 4.5), "gate": (0.9, 5.5), "column": (1.2, 18.0),
    "cottage": (4.0, 30.0), "house": (4.0, 35.0), "villa": (6.0, 45.0),
    "facade": (3.5, 45.0), "tower": (2.5, 80.0), "warehouse": (7.0, 70.0),
    "shopfront": (2.5, 25.0), "robot": (0.08, 3.5), "arch": (1.5, 8.0),
}


# ---------------------------------------------------------------------------
# input normalization
# ---------------------------------------------------------------------------


@dataclass
class MeshPart:
    """Minimal per-part mesh used by the multi-view renderer."""
    label: str
    vertices: np.ndarray  # (V, 3) float, world meters
    faces: np.ndarray     # (F, 3) int

    @property
    def finite(self) -> np.ndarray:
        return np.isfinite(self.vertices).all(axis=1)

    @property
    def aabb_min(self) -> np.ndarray:
        v = self.vertices[self.finite]
        return v.min(axis=0) if len(v) else np.zeros(3)

    @property
    def aabb_max(self) -> np.ndarray:
        v = self.vertices[self.finite]
        return v.max(axis=0) if len(v) else np.zeros(3)


def as_parts(obj) -> list[MeshPart]:
    """Normalize a spec / mesh / scene / (v, f) tuple / GLB path to MeshParts."""
    # GenerationSpec (duck-typed: has primitives) -----------------------------
    if hasattr(obj, "primitives") and hasattr(obj, "bbox_size"):
        return [
            MeshPart(p.label, np.asarray(p.vertices), np.asarray(p.faces))
            for p in build_spec_meshes(obj)
        ]
    # list of AnalyticPart / MeshPart -----------------------------------------
    if isinstance(obj, (list, tuple)) and obj and hasattr(obj[0], "vertices"):
        return [MeshPart(p.label, np.asarray(p.vertices), np.asarray(p.faces)) for p in obj]
    # (vertices, faces) tuple ---------------------------------------------------
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        v, f = obj
        return [MeshPart("mesh", np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64))]
    # path to a mesh file -------------------------------------------------------
    if isinstance(obj, (str, Path)) and Path(obj).exists():
        import trimesh

        obj = trimesh.load(str(obj))
    # trimesh scene / mesh ------------------------------------------------------
    try:
        import trimesh
    except ImportError:  # pragma: no cover
        trimesh = None
    if trimesh is not None:
        if isinstance(obj, trimesh.Scene):
            parts: list[MeshPart] = []
            for node_name in obj.graph.nodes_geometry:
                T, geom_name = obj.graph.get(node_name)
                geom = obj.geometry.get(geom_name)
                if geom is None or not hasattr(geom, "vertices"):
                    continue
                v = trimesh.transformations.transform_points(geom.vertices, T)
                parts.append(MeshPart(str(geom_name), np.asarray(v), np.asarray(geom.faces)))
            return parts
        if isinstance(obj, trimesh.Trimesh):
            return [MeshPart("mesh", np.asarray(obj.vertices), np.asarray(obj.faces))]
    raise TypeError(f"cannot interpret {type(obj).__name__} as spec/mesh parts")


# ---------------------------------------------------------------------------
# view transforms + renderer
# ---------------------------------------------------------------------------

_R3 = {
    # world -> view rotation: view x = right, view y = up, view z = toward viewer
    "front": np.eye(3),
    "back": np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]]),
    "right": np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]),
    "left": np.array([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]]),
    "top": np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
    "bottom": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),
}

# deterministic 3/4 perspective rigs: (azimuth_deg, elevation_deg, dist_factor)
_PERSP_RIG = {
    "three_quarter": (38.0, 24.0, 3.4),
    "three_quarter_back": (218.0, 24.0, 3.4),
}


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0.0, 1.0, 0.0], [-s, 0, c]])


def _rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0, c, -s], [0, s, c]])


@dataclass
class ViewRender:
    """Raster products of one projected view (all (size, size)).

    `visible_sign` records the projected winding sign of the *nearest*
    triangle at each pixel (painter overwrite, far->near), normalized so the
    mesh's majority surface orientation is +1. A closed solid therefore reads
    +1 across its whole silhouette from every view; an opening reads -1.
    """
    view: str
    silhouette: np.ndarray    # uint8 0/1, filled (painter)
    visible_sign: np.ndarray  # int8 -1/0/+1, nearest-triangle facing
    wireframe: np.ndarray     # uint8 0/1, all projected triangle edges
    part_ids: np.ndarray      # int32, -1 background else part index
    perspective: bool

    @property
    def facing(self) -> np.ndarray:
        """uint8 0/1 mask of pixels whose nearest surface is front-facing."""
        return (self.visible_sign > 0).astype(np.uint8) & self.silhouette


def _blank_render(view: str, size: int) -> ViewRender:
    return ViewRender(
        view,
        np.zeros((size, size), np.uint8),
        np.zeros((size, size), np.int8),
        np.zeros((size, size), np.uint8),
        np.full((size, size), -1, np.int32),
        view in _PERSP_RIG,
    )


def project_views(obj, views=ALL_VIEWS, size: int = CANVAS) -> dict[str, ViewRender]:
    """Project spec/mesh to the canonical views; deterministic raster products.

    Non-finite vertices are excluded from rendering (counted by qa_report).
    The visible-sign maps are winding-normalized via a majority vote across
    the orthographic views before returning.
    """
    import cv2

    parts = obj if isinstance(obj, list) and (not obj or isinstance(obj[0], MeshPart)) else as_parts(obj)
    renders: dict[str, ViewRender] = {}
    if not parts:
        return {view: _blank_render(view, size) for view in views}

    for view in views:
        sil = np.zeros((size, size), np.uint8)
        sign = np.zeros((size, size), np.int8)
        wire = np.zeros((size, size), np.uint8)
        ids = np.full((size, size), -1, np.int32)

        persp = view in _PERSP_RIG
        if persp:
            az, el, dist_f = _PERSP_RIG[view]
            R = _rot_x(el) @ _rot_y(az)
        else:
            R = _R3[view]
        all_centers = []
        part_depths: list[float] = []
        per_part: list[tuple[np.ndarray, np.ndarray]] = []  # view-space (v, f)
        for part in parts:
            v = part.vertices
            f = part.faces
            if len(v) == 0 or len(f) == 0:
                per_part.append((np.zeros((0, 3)), np.zeros((0, 3), np.int64)))
                part_depths.append(0.0)
                continue
            vw = v @ R.T
            per_part.append((vw, f))
            ctr = vw[np.isfinite(vw).all(axis=1)]
            all_centers.append(ctr)
            part_depths.append(float(ctr[:, 2].mean()) if len(ctr) else 0.0)
        if not all_centers or sum(len(c) for c in all_centers) == 0:
            renders[view] = _blank_render(view, size)
            continue
        allc = np.concatenate(all_centers)
        centre = allc.mean(axis=0)
        radius = max(float(np.linalg.norm(allc - centre, axis=1).max()), 1e-9)
        cam_dist = dist_f * radius if persp else 0.0
        focal = 1.35 * radius  # narrow-ish lens => mild, readable perspective

        def project(vw: np.ndarray) -> np.ndarray:
            if persp:
                z = np.maximum(cam_dist - vw[:, 2], 1e-6)
                s = focal / z
                return np.stack([vw[:, 0] * s, vw[:, 1] * s], axis=1)
            return vw[:, :2].copy()

        # per-triangle records carry their part index so the id map stays
        # correct after the painter sort (depth, pts2d, facing sign, part idx)
        tri_list: list[tuple[float, np.ndarray, float, int]] = []
        lo = np.full(2, np.inf)
        hi = np.full(2, -np.inf)
        for pi, (vw, f) in enumerate(per_part):
            if len(vw):
                p2 = project(vw)
                finite_tri = np.isfinite(p2[f]).all(axis=(1, 2))
                tris2 = p2[f][finite_tri]
                tris3 = vw[f][finite_tri]
                # facing: view-space winding (z of (b-a)x(c-a)); +z faces viewer
                e1 = tris3[:, 1] - tris3[:, 0]
                e2 = tris3[:, 2] - tris3[:, 0]
                nz = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
                depths = tris3[:, :, 2].mean(axis=1)
                for t, d, s_ in zip(tris2, depths, nz):
                    tri_list.append((float(d), t, float(np.sign(s_)), pi))
                if len(tris2):
                    lo = np.minimum(lo, tris2.reshape(-1, 2).min(axis=0))
                    hi = np.maximum(hi, tris2.reshape(-1, 2).max(axis=0))
        if not tri_list:
            renders[view] = _blank_render(view, size)
            continue
        span = np.maximum(hi - lo, 1e-9)
        scale = (size * 0.92) / span.max()
        mid = (lo + hi) / 2.0

        def to_px(p: np.ndarray) -> np.ndarray:
            q = (p - mid) * scale + size / 2.0
            q = q.copy()
            q[:, 1] = size - q[:, 1]
            return q.astype(np.int32)

        tri_list.sort(key=lambda td: td[0])  # far first (painter, stable sort)
        for d, t, sgn, pi in tri_list:
            px = to_px(t)
            cv2.fillConvexPoly(sil, px, 1)
            cv2.fillConvexPoly(sign, px, 1 if sgn > 0 else -1)
            cv2.fillConvexPoly(ids, px, pi)
            for k in range(3):
                cv2.line(wire, tuple(px[k]), tuple(px[(k + 1) % 3]), 1, 1)
        renders[view] = ViewRender(view, sil, sign, wire, ids, persp)

    # winding vote: the mesh's majority surface orientation becomes +1, so
    # boundary_completeness is immune to the mesh's winding convention
    pos = neg = 0
    for view in ORTHO_VIEWS:
        r = renders.get(view)
        if r is None:
            continue
        inside = r.silhouette.astype(bool)
        pos += int((r.visible_sign[inside] > 0).sum())
        neg += int((r.visible_sign[inside] < 0).sum())
    if neg > pos:
        for r in renders.values():
            r.visible_sign = (-r.visible_sign).astype(np.int8)
    return renders


# ---------------------------------------------------------------------------
# per-view metrics
# ---------------------------------------------------------------------------


def hole_mask(silhouette: np.ndarray, min_px: int = HOLE_MIN_PX) -> np.ndarray:
    """Enclosed background components (>= min_px) strictly inside the
    silhouette — missed faces show through as holes."""
    import cv2

    if not silhouette.any():
        return np.zeros_like(silhouette)
    h, w = silhouette.shape
    flood = silhouette.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    holes = ((silhouette == 0) & (flood != 2)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    keep = np.zeros_like(holes)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            keep[labels == i] = 1
    return keep


def boundary_completeness(render: ViewRender) -> dict:
    """Contour-closure score for one view.

    `facing_ratio` = silhouette fraction whose *nearest* surface is
    front-facing (after the winding vote) — drops when the viewer can see
    *through* an open boundary (missing faces expose back-facing interior).
    `hole_ratio` catches enclosed background gaps.
    completeness = facing_ratio * (1 - hole_ratio); 1.0 = fully closed.
    """
    area = float(render.silhouette.sum())
    if area == 0:
        return {"silhouette_area_px": 0, "facing_ratio": 0.0, "hole_ratio": 0.0,
                "completeness": 0.0}
    inside = render.silhouette.astype(bool)
    facing_ratio = float((render.visible_sign[inside] > 0).sum()) / area
    hole_ratio = float(hole_mask(render.silhouette).sum()) / area
    completeness = max(0.0, min(1.0, facing_ratio)) * (1.0 - min(hole_ratio, 1.0))
    return {
        "silhouette_area_px": int(area),
        "facing_ratio": round(facing_ratio, 4),
        "hole_ratio": round(hole_ratio, 4),
        "completeness": round(completeness, 4),
    }


def detail_density(render: ViewRender, erode_px: int = 2) -> dict:
    """Internal detail: projected wireframe edges strictly inside the
    silhouette. Catches "too simple" models (a blob scores far below a
    detailed build)."""
    import cv2

    sil = render.silhouette
    area = float(sil.sum())
    if area == 0:
        return {"internal_edge_px": 0, "silhouette_area_px": 0, "density": 0.0}
    inner = cv2.erode(sil, np.ones((2 * erode_px + 1,) * 2, np.uint8))
    internal = float(np.logical_and(render.wireframe, inner).sum())
    return {
        "internal_edge_px": int(internal),
        "silhouette_area_px": int(area),
        "density": round(internal / area, 4),
    }


# ---------------------------------------------------------------------------
# geometry integrity + scale
# ---------------------------------------------------------------------------


def _welded_faces(part: MeshPart) -> tuple[np.ndarray, int, int]:
    """Position-welded faces (WELD_EPS), plus (nan_vertex, degenerate_face)
    counts. Analytic meshes duplicate vertices along seams; topology must be
    analyzed after welding or closed solids look open."""
    v = part.vertices
    f = part.faces
    finite = part.finite
    nan = int((~finite).sum())
    if len(f) == 0 or len(v) == 0:
        return np.zeros((0, 3), np.int64), nan, 0
    f = f[finite[f].all(axis=1)]
    if len(f) == 0:
        return np.zeros((0, 3), np.int64), nan, 0
    vf = np.where(finite[:, None], v, 0.0)
    _, inv = np.unique(np.round(vf / WELD_EPS).astype(np.int64), axis=0,
                       return_inverse=True)
    fw = inv[f]
    deg_w = int(((fw[:, 0] == fw[:, 1]) | (fw[:, 1] == fw[:, 2])
                 | (fw[:, 2] == fw[:, 0])).sum())
    fw = fw[(fw[:, 0] != fw[:, 1]) & (fw[:, 1] != fw[:, 2]) & (fw[:, 2] != fw[:, 0])]
    # geometric zero-area faces (exact, pre-weld)
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    deg = int((np.linalg.norm(np.cross(e1, e2), axis=1) < 1e-12).sum())
    return fw, nan, max(deg, deg_w)


def geometry_integrity(parts: list[MeshPart]) -> dict:
    """NaN vertices, degenerate faces, and open (boundary) edges per part.

    A closed solid has every welded edge shared by exactly 2 faces; edges
    used once are open boundaries (a cube missing one face has 4).
    """
    out = {"nan_vertices": 0, "degenerate_faces": 0, "open_edges_total": 0,
           "open_parts": [], "watertight_parts": 0, "n_parts": len(parts)}
    for part in parts:
        fw, nan, deg = _welded_faces(part)
        out["nan_vertices"] += nan
        out["degenerate_faces"] += deg
        if len(fw) == 0:
            out["open_parts"].append({"part": part.label, "open_edges": -1,
                                      "note": "no valid geometry"})
            continue
        edges: dict[tuple[int, int], int] = {}
        for tri in fw:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                key = (int(min(tri[a], tri[b])), int(max(tri[a], tri[b])))
                edges[key] = edges.get(key, 0) + 1
        open_edges = sum(1 for c in edges.values() if c == 1)
        out["open_edges_total"] += open_edges
        if open_edges:
            out["open_parts"].append({"part": part.label,
                                      "open_edges": int(open_edges)})
        else:
            out["watertight_parts"] += 1
    out["watertight_fraction"] = (
        round(out["watertight_parts"] / len(parts), 4) if parts else 0.0
    )
    return out


def _category_range(object_name: str | None) -> tuple[str | None, tuple[float, float] | None]:
    if not object_name:
        return None, None
    name = object_name.lower()
    if name in CATEGORY_SCALE_RANGES:
        return name, CATEGORY_SCALE_RANGES[name]
    best = None
    for key in CATEGORY_SCALE_RANGES:
        if key in name and (best is None or len(key) > len(best)):
            best = key
    if best is None:
        for key in CATEGORY_SCALE_RANGES:
            if name in key and (best is None or len(key) > len(best)):
                best = key
    return best, CATEGORY_SCALE_RANGES.get(best) if best else None


def scale_sanity(parts: list[MeshPart], declared_bbox=None, object_name: str | None = None) -> dict:
    """Bounding-box sanity vs declared real-world scale + category plausibility."""
    finite_pts = [p.vertices[p.finite] for p in parts if p.finite.any()]
    if not finite_pts:
        return {"measured_aabb_m": [0, 0, 0], "longest_m": 0.0, "empty": True,
                "issues": [{"severity": "error", "code": "empty_geometry",
                            "message": "no finite geometry to measure"}]}
    allv = np.concatenate(finite_pts)
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    dims = hi - lo
    longest = float(dims.max())
    issues: list[dict] = []
    out = {
        "measured_aabb_m": [round(float(d), 4) for d in dims],
        "longest_m": round(longest, 4),
        "empty": False,
        "declared_bbox_m": list(declared_bbox) if declared_bbox else None,
        "bbox_match": None,
        "category": None,
        "category_range_m": None,
        "plausible": None,
        "issues": issues,
    }
    if declared_bbox is not None:
        decl = np.asarray(declared_bbox, dtype=float)
        ok = True
        worst = 1.0
        for i in range(3):
            if decl[i] <= 1e-9:
                continue
            ratio = float(dims[i] / decl[i])
            worst = max(worst, ratio, 1.0 / ratio)
            if ratio > BBOX_ERR or ratio < 1.0 / BBOX_ERR:
                issues.append({"severity": "error", "code": "bbox_mismatch",
                               "message": f"axis {'xyz'[i]}: mesh {dims[i]:.3f} m vs declared "
                                          f"{decl[i]:.3f} m ({ratio:.2f}x)"})
                ok = False
            elif ratio > BBOX_WARN or ratio < 1.0 / BBOX_WARN:
                issues.append({"severity": "warning", "code": "bbox_mismatch",
                               "message": f"axis {'xyz'[i]}: mesh {dims[i]:.3f} m vs declared "
                                          f"{decl[i]:.3f} m ({ratio:.2f}x)"})
                ok = False
        out["bbox_match"] = ok
        out["bbox_worst_axis_ratio"] = round(worst, 3)
    cat, rng = _category_range(object_name)
    if rng is not None:
        out["category"] = cat
        out["category_range_m"] = list(rng)
        lo_err, hi_err = rng[0] / SCALE_ERR, rng[1] * SCALE_ERR
        lo_warn, hi_warn = rng[0] / SCALE_WARN, rng[1] * SCALE_WARN
        if longest < lo_err or longest > hi_err:
            issues.append({"severity": "error", "code": "scale_implausible",
                           "message": f"longest dim {longest:.2f} m is implausible for "
                                      f"{cat!r} (expected ~{rng[0]:.3g}-{rng[1]:.3g} m)"})
            out["plausible"] = False
        elif longest < lo_warn or longest > hi_warn:
            issues.append({"severity": "warning", "code": "scale_implausible",
                           "message": f"longest dim {longest:.2f} m is borderline for "
                                      f"{cat!r} (expected ~{rng[0]:.3g}-{rng[1]:.3g} m)"})
            out["plausible"] = False
        else:
            out["plausible"] = True
    return out


def part_visibility(renders: dict[str, ViewRender], parts: list[MeshPart]) -> dict:
    """Every named part must be visible (>= MIN_PART_PX, occlusion-aware via
    the per-view id map) from at least one canonical view."""
    vis: dict[str, list[str]] = {}
    totals: dict[str, int] = {}
    for idx, part in enumerate(parts):
        seen = []
        total = 0
        for view, r in renders.items():
            n = int((r.part_ids == idx).sum())
            total += n
            if n >= MIN_PART_PX:
                seen.append(view)
        vis[part.label or f"part_{idx}"] = seen
        totals[part.label or f"part_{idx}"] = total
    invisible = [lbl for lbl, seen in vis.items() if not seen]
    return {
        "visible_views": vis,
        "px_totals": totals,
        "invisible_parts": invisible,
        "all_parts_visible": not invisible,
    }


# ---------------------------------------------------------------------------
# reference comparison (corpus masks per canonical view + LLM fallback)
# ---------------------------------------------------------------------------


def derive_silhouette(img: np.ndarray) -> np.ndarray | None:
    """Silhouette mask (0/1) from a photo/render: alpha -> border-bg distance
    -> GrabCut (same heuristic ladder as the corpus' make_masks.py)."""
    import cv2

    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        if alpha.min() < 250:
            m = (alpha > 30).astype(np.uint8)
            return _fill_and_clean(m)
    bgr = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    border = np.concatenate([
        bgr[0].reshape(-1, 3), bgr[-1].reshape(-1, 3),
        bgr[:, 0].reshape(-1, 3), bgr[:, -1].reshape(-1, 3),
    ]).astype(np.float32)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)
    if border.std() < 25:
        fg = (dist > max(30, dist.mean() + dist.std())).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = _fill_and_clean(fg)
        if 0.02 < m.mean() < 0.95:
            return m
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mx, my = int(w * 0.05), int(h * 0.05)
    gc[my:h - my, mx:w - mx] = cv2.GC_PR_FGD
    bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, None, bgm, fgm, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    m = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    m = _fill_and_clean(m)
    return m if 0.005 < m.mean() < 0.98 else None


def _fill_and_clean(mask: np.ndarray) -> np.ndarray:
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == i).astype(np.uint8)
    h, w = mask.shape
    flood = mask.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    return ((mask == 1) | (flood != 2)).astype(np.uint8)


def generated_dir(object_name: str) -> Path:
    return reference.corpus_root() / "_generated" / object_name


def ensure_llm_reference(
    object_name: str,
    view: str,
    prompt: str | None = None,
    generator=None,
) -> Path | None:
    """Return the cached LLM-generated reference image for (object, view).

    Cache: ``<corpus>/_generated/<object>/<view>.png`` + ``provenance.json``.
    On a cache miss, `generator(prompt, out_path)` is invoked when provided
    (the agent wires this to the image_generation plugin); provenance is
    recorded so AI-generated references are never mistaken for licensed
    corpus photos. Returns None when unavailable.
    """
    gdir = generated_dir(object_name)
    img_path = gdir / f"{view}.png"
    if img_path.exists():
        return img_path
    if generator is None:
        _log.info("no cached LLM reference for %s/%s and no generator given",
                  object_name, view)
        return None
    gdir.mkdir(parents=True, exist_ok=True)
    prompt = prompt or (
        f"Clean silhouette-style reference of a {object_name.replace('_', ' ')}, "
        f"canonical {view} view, whole object centered on a plain white background, "
        "no shadow, no text, orthographic-like framing, product-catalog style"
    )
    out = generator(prompt, img_path)
    if out is None or not img_path.exists():
        return None
    prov = {
        "object": object_name,
        "view": view,
        "file": img_path.name,
        "prompt": prompt,
        "generator": "image_generation plugin (Kimi agent-gw)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": ("AI-generated reference — plausibility proxy only; NOT a licensed "
                 "corpus photograph. Verify before using as a hard pass/fail gate."),
    }
    (gdir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return img_path


def load_generated_masks(object_name: str) -> dict[str, np.ndarray]:
    """Silhouette masks derived from cached LLM-generated references, keyed by
    canonical view ({} when nothing cached)."""
    import cv2

    masks: dict[str, np.ndarray] = {}
    gdir = generated_dir(object_name)
    if not gdir.is_dir():
        return masks
    for png in sorted(gdir.glob("*.png")):
        img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        m = derive_silhouette(img)
        if m is not None and m.any():
            masks[png.stem] = m
    return masks


def _chamfer(a: np.ndarray, b: np.ndarray, size: int = CANVAS) -> float | None:
    """Symmetric mean contour-to-contour distance of two normalized masks,
    divided by canvas size (0 = identical curves)."""
    import cv2

    na, nb = reference.normalize_mask(a, size), reference.normalize_mask(b, size)
    if not na.any() or not nb.any():
        return None

    def contour(m):
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return np.zeros((0, 0, 2), np.int32)
        return max(cs, key=cv2.contourArea)

    ca, cb = contour(na), contour(nb)
    if len(ca) == 0 or len(cb) == 0:
        return None
    ma = np.zeros_like(na)
    mb = np.zeros_like(nb)
    cv2.drawContours(ma, [ca], -1, 1, 1)
    cv2.drawContours(mb, [cb], -1, 1, 1)
    dt_b = cv2.distanceTransform((1 - mb).astype(np.uint8), cv2.DIST_L2, 3)
    dt_a = cv2.distanceTransform((1 - ma).astype(np.uint8), cv2.DIST_L2, 3)
    d = (dt_b[ma.astype(bool)].mean() + dt_a[mb.astype(bool)].mean()) / 2.0
    return float(d) / size


def compare_to_reference(obj, object_name: str, size: int = CANVAS,
                         allow_llm: bool = True) -> dict:
    """Compare projections against reference silhouettes per canonical view.

    Corpus masks (`reference.load_reference_masks`) win; views without corpus
    coverage fall back to cached LLM-generated references under
    ``_generated/``. Returns per-view IoU + contour chamfer + source.
    """
    renders = project_views(obj, views=ORTHO_VIEWS, size=size)
    corpus_masks = reference.load_reference_masks(object_name)
    llm_masks = load_generated_masks(object_name) if allow_llm else {}
    views = sorted(
        {_CORPUS_VIEW_MAP.get(v, v) for v in corpus_masks} | set(llm_masks)
    )
    out: dict[str, dict] = {}
    for view in views:
        entry: dict = {"iou": None, "chamfer": None, "source": None}
        ref_mask, src = None, None
        for cview, cmask in corpus_masks.items():
            if _CORPUS_VIEW_MAP.get(cview, cview) == view:
                ref_mask, src = cmask, "corpus"
                break
        if ref_mask is None and view in llm_masks:
            ref_mask, src = llm_masks[view], "llm_generated"
        if ref_mask is not None:
            ours = renders[view].silhouette
            entry["iou"] = round(reference.silhouette_iou(ours, ref_mask), 4)
            ch = _chamfer(ours, ref_mask, size)
            entry["chamfer"] = round(ch, 4) if ch is not None else None
            entry["source"] = src
        out[view] = entry
    scored = [e["iou"] for e in out.values() if e["iou"] is not None]
    return {
        "object": object_name,
        "views": out,
        "iou_mean": round(sum(scored) / len(scored), 4) if scored else None,
        "n_views_scored": len(scored),
        "sources": sorted({e["source"] for e in out.values() if e["source"]}),
    }


# ---------------------------------------------------------------------------
# aggregate QA report
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHT = {"error": 25, "warning": 8, "info": 2}


def qa_report(obj, object_name: str | None = None, declared_bbox=None,
              size: int = CANVAS, reference_compare: bool = True) -> dict:
    """Full multi-view QA report with per-issue severities and a 0-100 score."""
    spec_input = obj if (hasattr(obj, "primitives") and hasattr(obj, "bbox_size")) else None
    if declared_bbox is None and spec_input is not None:
        declared_bbox = list(getattr(spec_input, "bbox_size", None) or []) or None
    parts = as_parts(obj)
    renders = project_views(parts, views=ALL_VIEWS, size=size)
    issues: list[dict] = []

    # per-view metrics first: geometry severity is coupled to what the views show
    view_metrics: dict[str, dict] = {}
    densities = []
    min_ortho_facing = 1.0
    bad_views: dict[str, list[str]] = {"error": [], "warning": []}
    for view, r in renders.items():
        bc = boundary_completeness(r)
        dd = detail_density(r)
        bc.update(dd)
        view_metrics[view] = bc
        if view in ORTHO_VIEWS:
            densities.append(dd["density"])
            if bc["silhouette_area_px"]:
                min_ortho_facing = min(min_ortho_facing, bc["facing_ratio"])
        note = (f"{view}: closed {bc['completeness']:.2f} (facing "
                f"{bc['facing_ratio']:.2f}, holes {bc['hole_ratio']:.3f})")
        if bc["silhouette_area_px"] == 0 and parts:
            issues.append({"severity": "error", "code": "empty_render", "view": view,
                           "message": f"{view} view renders empty despite geometry"})
        elif bc["facing_ratio"] < 0.5:
            # nearest surface mostly back-facing => looking THROUGH an opening
            # (missing faces). Strict error regardless of holes.
            bad_views["error"].append(note)
        elif bc["completeness"] < COMPLETE_ERR:
            # big enclosed gaps with intact facing: perforation (fences, foliage)
            # is usually intentional -> warning unless facing is also degraded
            bad_views["error" if bc["facing_ratio"] < 0.85 else "warning"].append(note)
        elif bc["completeness"] < COMPLETE_WARN:
            bad_views["warning"].append(note)
    for sev in ("error", "warning"):
        if bad_views[sev]:
            shown = bad_views[sev][:4]
            more = f" (+{len(bad_views[sev]) - 4} more views)" if len(bad_views[sev]) > 4 else ""
            issues.append({"severity": sev, "code": "boundary_incomplete",
                           "views": [n.split(":")[0] for n in bad_views[sev]],
                           "message": "boundary open/incomplete — " + "; ".join(shown) + more})

    geo = geometry_integrity(parts)
    if geo["nan_vertices"]:
        issues.append({"severity": "error", "code": "nan_geometry",
                       "message": f"{geo['nan_vertices']} non-finite vertices — "
                                  "geometry contains NaN/inf"})
    if geo["open_edges_total"]:
        # geometry + views must agree for an error: open edges alone are a
        # warning (legitimately open surfaces exist: planes, leaves, fabric);
        # they escalate only when a view also shows see-through (low facing)
        sev = "error" if min_ortho_facing < 0.85 else "warning"
        worst = sorted(geo["open_parts"],
                       key=lambda p: -p.get("open_edges", 0))[:5]
        issues.append({"severity": sev, "code": "open_boundary",
                       "message": f"{geo['open_edges_total']} open boundary edges on "
                                  f"welded topology (worst: {worst})"})
    if geo["degenerate_faces"]:
        issues.append({"severity": "warning", "code": "degenerate_faces",
                       "message": f"{geo['degenerate_faces']} zero-area faces"})

    if parts and densities and max(densities) > 0:
        mean_density = sum(densities) / len(densities)
        if mean_density < 0.02:
            issues.append({"severity": "warning", "code": "too_simple",
                           "message": f"mean internal detail density {mean_density:.4f} "
                                      "edges/px — model may be too simple"})

    scale = scale_sanity(parts, declared_bbox=declared_bbox, object_name=object_name)
    issues.extend(scale.pop("issues"))

    vis = part_visibility({v: renders[v] for v in ORTHO_VIEWS}, parts)
    for lbl in vis["invisible_parts"]:
        issues.append({"severity": "warning", "code": "part_invisible", "part": lbl,
                       "message": f"part {lbl!r} is not visible from any canonical view"})

    ref_cmp = None
    coverage = None
    if reference_compare and object_name:
        try:
            ref_cmp = compare_to_reference(parts, object_name, size=size)
            for view, e in ref_cmp["views"].items():
                if e["iou"] is not None and e["iou"] < 0.35:
                    issues.append({"severity": "warning", "code": "reference_mismatch",
                                   "view": view,
                                   "message": f"{view} silhouette IoU {e['iou']:.2f} vs "
                                              f"{e['source']} reference"})
        except Exception as e:  # corpus problems must not sink QA
            _log.info("reference compare skipped: %s", e)
    if spec_input is not None and object_name:
        try:
            coverage = reference.parts_coverage(spec_input, object_name)
            if "error" in coverage:
                coverage = None
        except Exception as e:
            _log.info("parts coverage skipped: %s", e)

    score = 100 - sum(_SEVERITY_WEIGHT[i["severity"]] for i in issues)
    return {
        "object": object_name,
        "n_parts": len(parts),
        "n_vertices": int(sum(len(p.vertices) for p in parts)),
        "n_faces": int(sum(len(p.faces) for p in parts)),
        "geometry": geo,
        "views": view_metrics,
        "detail_density_mean": round(
            float(np.mean([m["density"] for m in view_metrics.values()])), 4
        ) if view_metrics else 0.0,
        "scale": scale,
        "parts_visibility": vis,
        "reference_compare": ref_cmp,
        "parts_coverage": coverage,
        "issues": issues,
        "n_errors": sum(1 for i in issues if i["severity"] == "error"),
        "score": max(0, int(score)),
        "pass": not any(i["severity"] == "error" for i in issues),
    }


def worst_issues(report: dict, n: int = 3) -> list[dict]:
    """The n most severe issues of a qa_report (errors first, then warnings)."""
    rank = {"error": 0, "warning": 1, "info": 2}
    return sorted(report["issues"], key=lambda i: rank[i["severity"]])[:n]


# ---------------------------------------------------------------------------
# contact sheet
# ---------------------------------------------------------------------------

_SHEET_COLS = 3


def render_contact_sheet(renders: dict[str, ViewRender], out_path,
                         title: str = "", cell: int = 256) -> Path:
    """Grid PNG: silhouette fill, wireframe overlay, defect tints
    (red = see-through/open boundary, blue = enclosed holes)."""
    import cv2
    from PIL import Image, ImageDraw

    views = list(renders)
    rows = (len(views) + _SHEET_COLS - 1) // _SHEET_COLS
    pad, bar = 8, 34
    W = _SHEET_COLS * (cell + pad) + pad
    H = rows * (cell + bar + pad) + pad + (26 if title else 0)
    sheet = Image.new("RGB", (W, H), (250, 250, 248))
    draw = ImageDraw.Draw(sheet)
    y0 = pad + (26 if title else 0)
    if title:
        draw.text((pad, pad - 2), title, fill=(20, 20, 20))
    for k, view in enumerate(views):
        r = renders[view]
        bc = boundary_completeness(r)
        dd = detail_density(r)
        img = np.full((cell, cell, 3), 255, np.uint8)
        sil = cv2.resize(r.silhouette, (cell, cell), interpolation=cv2.INTER_NEAREST)
        sgn = cv2.resize(r.visible_sign, (cell, cell), interpolation=cv2.INTER_NEAREST)
        wir = cv2.resize(r.wireframe, (cell, cell), interpolation=cv2.INTER_NEAREST)
        img[sil.astype(bool)] = (208, 208, 204)          # body fill
        open_px = sil.astype(bool) & (sgn < 0)
        img[open_px] = (238, 175, 170)                   # see-through / open
        img[hole_mask(sil).astype(bool)] = (170, 195, 238)  # enclosed holes
        img[wir.astype(bool)] = (60, 58, 55)             # internal edges
        x = pad + (k % _SHEET_COLS) * (cell + pad)
        y = y0 + (k // _SHEET_COLS) * (cell + bar + pad)
        sheet.paste(Image.fromarray(img), (x, y + bar))
        draw.text((x + 2, y + 4),
                  f"{view}{' [3/4]' if r.perspective else ''}", fill=(20, 20, 20))
        warn = bc["completeness"] < COMPLETE_WARN
        draw.text((x + 2, y + 18),
                  f"closed {bc['completeness']:.2f}  detail {dd['density']:.3f}",
                  fill=(178, 34, 34) if warn else (90, 90, 88))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
