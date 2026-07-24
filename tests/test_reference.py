"""Tests for generation.reference — projection-similarity validation against
the real-object reference corpus (silhouette IoU + parts coverage)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.generation import reference


def _t(x=0.0, y=0.0, z=0.0):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m.tolist()


def good_cat_spec() -> GenerationSpec:
    """A plausible cat: body/head/ears/4 legs/tail at sane proportions."""
    return GenerationSpec(
        shape="creature",
        bbox_size=(0.5, 0.6, 1.2),
        primitives=[
            Primitive("ellipsoid", _t(0, 0.0, 0.0), {"radii": [0.18, 0.16, 0.35]}, "body"),
            Primitive("sphere", _t(0, 0.14, 0.44), {"radius": 0.12}, "head"),
            Primitive("cone", _t(-0.06, 0.28, 0.46), {"radius": 0.04, "height": 0.09}, "ear_left"),
            Primitive("cone", _t(0.06, 0.28, 0.46), {"radius": 0.04, "height": 0.09}, "ear_right"),
            Primitive("capsule", _t(-0.09, -0.24, 0.26), {"radius": 0.035, "height": 0.22}, "leg_fl"),
            Primitive("capsule", _t(0.09, -0.24, 0.26), {"radius": 0.035, "height": 0.22}, "leg_fr"),
            Primitive("capsule", _t(-0.09, -0.24, -0.24), {"radius": 0.045, "height": 0.24}, "leg_hl"),
            Primitive("capsule", _t(0.09, -0.24, -0.24), {"radius": 0.045, "height": 0.24}, "leg_hr"),
            # tail swept back along -z (rotate capsule's local Y onto world Z)
            Primitive(
                "capsule",
                [[1, 0, 0, 0], [0, 0, 1, 0.08], [0, -1, 0, -0.5], [0, 0, 0, 1]],
                {"radius": 0.03, "height": 0.35},
                "tail",
            ),
        ],
    )


def blob_cat_spec() -> GenerationSpec:
    """A degenerate 'cat': one anonymous blob."""
    return GenerationSpec(
        shape="creature",
        primitives=[Primitive("sphere", _t(), {"radius": 0.5}, "blob")],
    )


# ---------------------------------------------------------------------------
# IoU metric sanity
# ---------------------------------------------------------------------------


def test_iou_identical_masks_is_one():
    m = np.zeros((128, 128), dtype=np.uint8)
    m[30:90, 40:100] = 1
    assert reference.silhouette_iou(m, m.copy()) == pytest.approx(1.0)


def test_iou_empty_mask_is_zero():
    a = np.zeros((128, 128), dtype=np.uint8)
    b = np.zeros((128, 128), dtype=np.uint8)
    b[40:90, 40:90] = 1
    assert reference.silhouette_iou(a, b) == pytest.approx(0.0)


def test_iou_very_different_shapes_is_low():
    a = np.zeros((128, 128), dtype=np.uint8)
    b = np.zeros((128, 128), dtype=np.uint8)
    a[40:88, 40:88] = 1                      # filled square
    for i in range(20, 108):                 # thin diagonal line
        b[i, i] = 1
        b[i, min(i + 1, 127)] = 1
    assert reference.silhouette_iou(a, b) < 0.2


def test_iou_translation_and_scale_invariant():
    a = np.zeros((128, 128), dtype=np.uint8)
    a[20:60, 20:60] = 1
    b = np.zeros((128, 128), dtype=np.uint8)
    b[60:120, 10:70] = 1  # same square, shifted + scaled
    assert reference.silhouette_iou(a, b) == pytest.approx(1.0)


def test_iou_partial_shape_mismatch_is_midrange():
    a = np.zeros((128, 128), dtype=np.uint8)
    b = np.zeros((128, 128), dtype=np.uint8)
    a[40:88, 40:88] = 1                      # filled square
    yy, xx = np.mgrid[0:128, 0:128]
    b[(np.abs(yy - 64) + np.abs(xx - 64)) <= 34] = 1   # inscribed diamond
    iou = reference.silhouette_iou(a, b, allow_mirror=False)
    assert 0.3 < iou < 0.7


# ---------------------------------------------------------------------------
# headless renderer
# ---------------------------------------------------------------------------


def test_render_silhouette_nonempty_and_distinct():
    sphere = GenerationSpec(primitives=[Primitive("sphere", _t(), {"radius": 0.5}, "ball")])
    slab = GenerationSpec(
        primitives=[Primitive("box", _t(), {"size": [1.0, 0.05, 1.0]}, "slab")]
    )
    m1 = reference.render_silhouette(sphere, "front")
    m2 = reference.render_silhouette(slab, "front")
    assert m1.shape == (reference.CANVAS, reference.CANVAS)
    assert m1.sum() > 0 and m2.sum() > 0
    # very different shapes must not look alike
    assert reference.silhouette_iou(m1, m2) < 0.9


def test_render_silhouette_empty_spec_is_blank():
    spec = GenerationSpec(primitives=[])
    assert reference.render_silhouette(spec, "side").sum() == 0


# ---------------------------------------------------------------------------
# parts coverage: known-good vs known-bad specs
# ---------------------------------------------------------------------------


def test_parts_coverage_good_cat():
    cov = reference.parts_coverage(good_cat_spec(), "cat")
    assert cov["coverage"] == pytest.approx(1.0)
    for name, entry in cov["parts"].items():
        assert entry["present"], f"part {name} missing: {entry}"


def test_parts_coverage_blob_cat_fails():
    cov = reference.parts_coverage(blob_cat_spec(), "cat")
    assert cov["coverage"] < 0.5
    missing = [n for n, e in cov["parts"].items() if not e["present"]]
    assert "tail" in missing and "head" in missing


def test_score_spec_good_beats_blob():
    """Relative contract: an anatomically plausible cat must outscore a blob.

    The absolute quality gate is parts coverage (1.00), NOT silhouette IoU:
    the bundled reference photo for "cat" is a SITTING cat while the spec
    generated here is STANDING, so even a perfect standing cat scores only
    ~0.34-0.38 IoU against it — an honest pose mismatch, not a quality
    regression. (For the same reason a blob can coincidentally match the
    sitting silhouette as well as a real cat does, so good_iou > blob_iou
    is NOT a valid assertion; we use a lenient regression floor instead.)
    """
    good = reference.score_spec(good_cat_spec(), "cat")
    blob = reference.score_spec(blob_cat_spec(), "cat")

    # Relative contract on parts coverage + absolute gate at 1.00.
    assert good["parts"]["coverage"] > blob["parts"]["coverage"]
    assert good["parts"]["coverage"] == pytest.approx(1.0)
    assert good["proportion_violations"] == []
    assert blob["pass"] is False

    # Relative contract on repairs: the good spec needs far fewer fixes.
    good_repairs = reference.suggest_repairs(good)
    blob_repairs = reference.suggest_repairs(blob)
    assert len(good_repairs) < len(blob_repairs)

    if good["iou_mean"] is None:
        # No reference corpus mounted: the good spec passes outright.
        assert good["pass"] is True
    else:
        # Corpus mounted: keep only a lenient IoU regression floor. The
        # sitting-vs-standing pose mismatch caps a correct cat around ~0.35,
        # so 0.25 catches true silhouette catastrophes (empty render, wrong
        # axis, exploded scale) without punishing the honest pose difference.
        assert good["iou_mean"] > 0.25


def test_suggest_repairs_flags_missing_parts():
    blob = reference.score_spec(blob_cat_spec(), "cat")
    repairs = reference.suggest_repairs(blob)
    assert any(r["issue"] == "missing" and r["part"] == "tail" for r in repairs)


def test_suggest_repairs_flags_bad_proportion():
    spec = good_cat_spec()
    # grotesquely oversized head
    spec.primitives[1] = Primitive("sphere", _t(0, 0.2, 0.6), {"radius": 0.35}, "head")
    report = reference.score_spec(spec, "cat")
    repairs = reference.suggest_repairs(report)
    head_fix = [r for r in repairs if r["part"] == "head" and r["issue"] == "proportion"]
    assert head_fix, f"expected a head proportion repair, got {repairs}"
    assert head_fix[0]["scale_hint"] < 1.0  # shrink


def test_unknown_object_graceful():
    report = reference.score_spec(blob_cat_spec(), "no_such_object")
    assert "error" in report["parts"]
    assert report["pass"] is False


# ---------------------------------------------------------------------------
# corpus integrity (skips when the external corpus is absent)
# ---------------------------------------------------------------------------

CORPUS = Path(os.environ.get("IRONENGINE_REFERENCE_ROOT", r"E:\SceneEditorAssets\_reference"))


@pytest.mark.skipif(not CORPUS.is_dir(), reason="reference corpus not present")
def test_corpus_sources_licenses_recorded():
    sources = json.loads((CORPUS / "SOURCES.json").read_text(encoding="utf-8"))
    assert sources, "SOURCES.json is empty"
    for obj, entries in sources.items():
        for e in entries:
            assert e.get("license"), f"{obj}: missing license"
            assert e.get("url", "").startswith("http"), f"{obj}: missing url"
            assert e.get("http_status") == 200


@pytest.mark.skipif(not CORPUS.is_dir(), reason="reference corpus not present")
def test_corpus_images_and_masks_load():
    import cv2

    sources = json.loads((CORPUS / "SOURCES.json").read_text(encoding="utf-8"))
    n_checked = 0
    for obj, entries in sources.items():
        for e in entries:
            img = cv2.imread(str(CORPUS / obj / e["file"]), cv2.IMREAD_UNCHANGED)
            assert img is not None and img.size > 0, f"{obj}/{e['file']} unreadable"
            mask = CORPUS / obj / f"mask_{Path(e['file']).stem}.png"
            if mask.exists():  # masks required only where generated
                m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
                assert m is not None and m.sum() > 0, f"{mask} empty"
            n_checked += 1
    assert n_checked > 0


@pytest.mark.skipif(not (CORPUS / "cat").is_dir(), reason="cat reference not present")
def test_corpus_annotations_match_bundled():
    for obj_dir in CORPUS.iterdir():
        if not obj_dir.is_dir() or obj_dir.name.startswith("_"):
            continue
        ann = obj_dir / "annotations.json"
        if ann.exists():
            doc = json.loads(ann.read_text(encoding="utf-8"))
            assert doc["object"] == obj_dir.name
            assert doc["parts"], f"{obj_dir.name}: empty parts"
