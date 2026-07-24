"""Projection-similarity validator against the real-object reference corpus.

Renders a GenerationSpec from canonical orthographic views (reusing
`generation.analytic_mesh`, headless, no BonaFide dependency), rasterizes a
silhouette, and compares it with reference silhouette masks of REAL objects
(E:\\SceneEditorAssets\\_reference) via intersection-over-union (IoU). A
parts-coverage check verifies that the spec contains every annotated part of
the object and that part proportions stay near real-world values.

Public API:
    score_spec(spec, object_name) -> dict        # full report
    suggest_repairs(report) -> list[dict]        # proportion repair hints
    render_silhouette(spec, view, size) -> np.ndarray
    silhouette_iou(a, b) -> float
    parts_coverage(spec, object_name) -> dict

The corpus root defaults to E:/SceneEditorAssets/_reference and can be
overridden with the IRONENGINE_REFERENCE_ROOT environment variable. When the
corpus (or an object's masks) is unavailable, silhouette scoring degrades
gracefully to `iou: None` while parts coverage still works from the bundled
reference_data/*.json annotations.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np

from .analytic_mesh import build_spec_meshes

_log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "reference_data"
DEFAULT_CORPUS_ROOT = Path(r"E:\SceneEditorAssets\_reference")

CANVAS = 256          # raster size for silhouette comparison
PROPORTION_TOL = 0.35  # relative tolerance on typical_proportion before flagging

# view -> (horizontal axis, vertical axis, depth axis)
_VIEW_AXES = {
    "front": (0, 1, 2),   # look along -Z: x right, y up
    "back": (0, 1, 2),
    "side": (2, 1, 0),    # look along -X: z right, y up
    "top": (0, 2, 1),     # look along -Y: x right, z up
}


# ---------------------------------------------------------------------------
# corpus access
# ---------------------------------------------------------------------------


def corpus_root() -> Path:
    return Path(os.environ.get("IRONENGINE_REFERENCE_ROOT", str(DEFAULT_CORPUS_ROOT)))


def load_annotations(object_name: str) -> dict | None:
    """Load the bundled part annotations for an object (None if unknown)."""
    path = DATA_DIR / f"{object_name}.json"
    if not path.exists():
        _log.info("no annotations for %r at %s", object_name, path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_reference_masks(object_name: str) -> dict[str, np.ndarray]:
    """Load reference silhouette masks for an object, keyed by canonical view.

    Looks for mask_*.png next to the corpus images; falls back to deriving a
    mask from any image with an alpha channel. Returns {} when unavailable.
    """
    odir = corpus_root() / object_name
    masks: dict[str, np.ndarray] = {}
    if not odir.is_dir():
        return masks
    sources_path = corpus_root() / "SOURCES.json"
    views: dict[str, str] = {}
    if sources_path.exists():
        try:
            src = json.loads(sources_path.read_text(encoding="utf-8"))
            for e in src.get(object_name, []):
                views[e["file"]] = e.get("view", "front")
        except Exception as e:  # pragma: no cover - defensive
            _log.info("SOURCES.json unreadable: %s", e)
    try:
        import cv2
    except ImportError:  # pragma: no cover
        _log.warning("opencv unavailable; cannot load reference masks")
        return masks
    for img_path in sorted(odir.glob("image_*.*")):
        mask_path = odir / f"mask_{img_path.stem}.png"
        mask = None
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3]
                if alpha.min() < 250:  # meaningful transparency
                    mask = (alpha > 30).astype(np.uint8) * 255
        if mask is None or not np.any(mask):
            continue
        view = views.get(img_path.name, "front")
        masks[view] = (mask > 127).astype(np.uint8)
    return masks


# ---------------------------------------------------------------------------
# headless orthographic silhouette renderer
# ---------------------------------------------------------------------------


def render_silhouette(spec, view: str = "front", size: int = CANVAS) -> np.ndarray:
    """Rasterize the spec's analytic meshes into a binary silhouette mask.

    Orthographic projection, painter's algorithm; no GPU / windowing needed.
    Returns a (size, size) uint8 array of 0/1.
    """
    import cv2

    if view not in _VIEW_AXES:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(_VIEW_AXES)}")
    parts = build_spec_meshes(spec)
    canvas = np.zeros((size, size), dtype=np.uint8)
    if not parts:
        return canvas
    ax_h, ax_v, ax_d = _VIEW_AXES[view]
    tris: list[tuple[float, np.ndarray]] = []
    for part in parts:
        v = part.vertices
        f = part.faces
        pts2d = np.stack([v[:, ax_h], v[:, ax_v]], axis=1)
        tri_depth = v[f, ax_d].mean(axis=1)
        for t, d in zip(pts2d[f], tri_depth):
            tris.append((float(d), t))
    all2d = np.concatenate([t[1] for t in tris], axis=0)
    lo = all2d.min(axis=0)
    hi = all2d.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scale = (size * 0.92) / span.max()
    centre = (lo + hi) / 2.0

    def to_px(p: np.ndarray) -> np.ndarray:
        q = (p - centre) * scale + size / 2.0
        q[:, 1] = size - q[:, 1]  # flip y for image space
        return q.astype(np.int32)

    tris.sort(key=lambda td: td[0])  # far first, painter's algorithm
    for _, t in tris:
        cv2.fillConvexPoly(canvas, to_px(t), 1)
    return canvas


# ---------------------------------------------------------------------------
# mask metrics
# ---------------------------------------------------------------------------


def normalize_mask(mask: np.ndarray, size: int = CANVAS) -> np.ndarray:
    """Crop to content bbox, pad square, resize to (size, size) 0/1 mask."""
    import cv2

    m = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    m = m[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = m.shape
    side = max(h, w)
    pad = np.zeros((side, side), dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    pad[y0:y0 + h, x0:x0 + w] = m
    return (cv2.resize(pad, (size, size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)


def silhouette_iou(a: np.ndarray, b: np.ndarray, allow_mirror: bool = True) -> float:
    """IoU of two silhouette masks after aspect-preserving normalization."""
    na, nb = normalize_mask(a), normalize_mask(b)
    inter = np.logical_and(na, nb).sum()
    union = np.logical_or(na, nb).sum()
    best = inter / union if union else 0.0
    if allow_mirror:
        nb_flip = np.fliplr(nb)
        inter = np.logical_and(na, nb_flip).sum()
        union = np.logical_or(na, nb_flip).sum()
        best = max(best, inter / union if union else 0.0)
    return float(best)


# ---------------------------------------------------------------------------
# parts coverage + proportions
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z]+")


def _label_tokens(label: str) -> set[str]:
    return set(_TOKEN_RE.findall(label.lower()))


def _part_keywords(part: dict) -> list[str]:
    words = [part["name"], *part.get("aliases", [])]
    out = []
    for w in words:
        out.extend(_TOKEN_RE.findall(w.lower()))
    return out


def parts_coverage(spec, object_name: str) -> dict:
    """Check that every annotated part exists in the spec and its proportion.

    A part is "present" when at least one primitive label contains one of the
    part's name/alias tokens. Proportion is the matched parts' combined extent
    along the annotation's proportion axis, relative to the whole model.
    """
    ann = load_annotations(object_name)
    if ann is None:
        return {"object": object_name, "error": f"no annotations for {object_name!r}"}
    parts = build_spec_meshes(spec)
    axis = ann.get("proportion_axis", "height")
    if parts:
        all_min = np.min([p.aabb_min for p in parts], axis=0)
        all_max = np.max([p.aabb_max for p in parts], axis=0)
    else:
        all_min = all_max = np.zeros(3)

    def axis_index(name: str) -> int:
        """'height'->y, 'length'->longest horizontal, x/y/z direct."""
        if name in ("x", "y", "z"):
            return "xyz".index(name)
        if name == "length":
            return int(max((0, 2), key=lambda i: all_max[i] - all_min[i]))
        return 1  # height default

    total = {i: float(all_max[i] - all_min[i]) or 1.0 for i in range(3)}
    report: dict[str, dict] = {}
    # Assign each primitive to the single best-matching part (max keyword
    # overlap), so shared tokens like "leg" don't pollute front/hind parts.
    part_kw = {part["name"]: set(_part_keywords(part)) for part in ann["parts"]}
    assignment: dict[str, list] = {part["name"]: [] for part in ann["parts"]}
    for p in parts:
        toks = _label_tokens(p.label)
        norm = p.label.lower().replace("_", " ").replace("-", " ")
        best_name, best_score = None, 0
        for part in ann["parts"]:
            kw = part_kw[part["name"]]
            score = len(toks & kw)
            for w in [part["name"], *part.get("aliases", [])]:
                if len(w) >= 4 and w.lower() in norm:
                    score += 2
            if score > best_score:
                best_name, best_score = part["name"], score
        if best_name is not None and best_score > 0:
            assignment[best_name].append(p)
    for part in ann["parts"]:
        matched = assignment[part["name"]]
        entry: dict = {
            "present": bool(matched),
            "matched_labels": [p.label for p in matched],
            "expected_proportion": part["typical_proportion"],
            "notes": part.get("notes", ""),
        }
        if matched:
            lo = np.min([p.aabb_min for p in matched], axis=0)
            hi = np.max([p.aabb_max for p in matched], axis=0)
            ax_i = axis_index(part.get("measure_axis", axis))
            measured = float(hi[ax_i] - lo[ax_i]) / total[ax_i]
            exp = part["typical_proportion"]
            entry["measure_axis"] = part.get("measure_axis", axis)
            entry["measured_proportion"] = round(measured, 4)
            entry["proportion_delta"] = round(measured - exp, 4)
            entry["proportion_ok"] = (
                abs(measured - exp) <= PROPORTION_TOL * max(exp, 0.05)
            )
        report[part["name"]] = entry
    n_present = sum(1 for e in report.values() if e["present"])
    return {
        "object": object_name,
        "canonical_view": ann["canonical_view"],
        "n_parts": len(report),
        "n_present": n_present,
        "coverage": round(n_present / len(report), 4) if report else 0.0,
        "parts": report,
    }


# ---------------------------------------------------------------------------
# top-level scoring + repairs
# ---------------------------------------------------------------------------


def score_spec(spec, object_name: str, size: int = CANVAS) -> dict:
    """Full validation report for a spec against the object's references."""
    coverage = parts_coverage(spec, object_name)
    ann = load_annotations(object_name)
    ref_masks = load_reference_masks(object_name)
    views = sorted(set(ref_masks) | ({ann["canonical_view"]} if ann else {"front"}))
    iou: dict[str, float | None] = {}
    for view in views:
        if view not in ref_masks:
            iou[view] = None
            continue
        rendered = render_silhouette(spec, view=view, size=size)
        iou[view] = round(silhouette_iou(rendered, ref_masks[view]), 4)
    iou_vals = [v for v in iou.values() if v is not None]
    cov_ok = coverage.get("coverage", 0.0) >= 0.999
    prop_bad = [
        name for name, e in coverage.get("parts", {}).items()
        if e.get("present") and e.get("proportion_ok") is False
    ]
    report = {
        "object": object_name,
        "iou": iou,
        "iou_mean": round(sum(iou_vals) / len(iou_vals), 4) if iou_vals else None,
        "iou_available": bool(iou_vals),
        "parts": coverage,
        "proportion_violations": prop_bad,
        "pass": bool(cov_ok and not prop_bad and (not iou_vals or min(iou_vals) >= 0.5)),
    }
    return report


def suggest_repairs(report: dict) -> list[dict]:
    """Turn a score report into concrete proportion repair hints."""
    repairs: list[dict] = []
    parts = (report.get("parts") or {}).get("parts") or {}
    for name, e in parts.items():
        if not e.get("present"):
            repairs.append({
                "part": name,
                "issue": "missing",
                "suggestion": f"add a primitive labelled for part {name!r} ({e.get('notes', '')})",
            })
            continue
        if e.get("proportion_ok") is False:
            measured = e.get("measured_proportion", 0.0) or 1e-6
            expected = e.get("expected_proportion", 0.0)
            scale = expected / measured if measured else 1.0
            direction = "enlarge" if scale > 1.0 else "shrink"
            repairs.append({
                "part": name,
                "issue": "proportion",
                "measured": measured,
                "expected": expected,
                "suggestion": (
                    f"{direction} {name!r} by ~{abs(scale):.2f}x along the "
                    f"{(report.get('parts') or {}).get('canonical_view', 'main')} axis "
                    f"(measured {measured:.3f} vs expected {expected:.3f})"
                ),
                "scale_hint": round(scale, 3),
            })
    iou = report.get("iou") or {}
    low = {v: s for v, s in iou.items() if s is not None and s < 0.5}
    for view, s in low.items():
        repairs.append({
            "part": "*silhouette*",
            "issue": "silhouette_mismatch",
            "suggestion": (
                f"{view} silhouette IoU {s:.2f} < 0.5 — reshape overall outline "
                f"to match the reference {view} view"
            ),
        })
    return repairs
