"""Tests for wearable garments (generation.clothing).

Covers: each garment builds as distinct named parts bound to body regions,
shirt collar + buttons, per-garment world AABBs, dress skirt, jacket open
front panels, and garment swapping that keeps the skeleton/face/hair parts
identical. No network.
"""
from __future__ import annotations

import numpy as np
import pytest

from ironengine_3d_creator.generation.clothing import GARMENTS, swap_garments
from ironengine_3d_creator.generation.human_anatomy import (
    SIM_BONE_NAMES, build_human)


def _garment_parts(spec, garment):
    return {n: node for n, node in spec.graph.nodes.items()
            if node.metadata.get("garment") == garment}


@pytest.mark.parametrize("garment", GARMENTS)
def test_each_garment_builds(garment):
    spec = build_human(clothes=(garment,))
    r = spec.build()
    parts = _garment_parts(spec, garment)
    assert parts, f"{garment}: no garment parts"
    # root part named exactly after the garment, with its own world AABB
    assert garment in parts
    aabbs = r.aabbs()
    lo, hi = aabbs[garment]
    assert np.all(hi > lo)
    # every garment part is bound to a Sim bone region
    for n, node in parts.items():
        bind = node.metadata.get("bind_bone")
        assert bind in SIM_BONE_NAMES, f"{n}: bad bind_bone {bind!r}"
        assert node.parent is not None


def test_shirt_has_collar_and_buttons():
    spec = build_human(clothes=("shirt",))
    names = set(spec.graph.nodes)
    assert "shirt_collar" in names
    buttons = [n for n in names if n.startswith("shirt_button_")]
    assert len(buttons) >= 5
    assert "shirt_placket" in names


def test_dress_has_skirt_and_bodice():
    spec = build_human(clothes=("dress",))
    parts = _garment_parts(spec, "dress")
    assert {"dress", "dress_skirt"} <= set(parts)
    r = spec.build().aabbs()
    skirt_lo, skirt_hi = r["dress_skirt"]
    # skirt flares wider than the bodice waist and hems just above the knee
    # (knee at 0.291·H ≈ 0.509 m; hem at 0.33·H ≈ 0.577 m)
    assert float(skirt_hi[0] - skirt_lo[0]) > 0.20
    assert float(skirt_lo[1]) == pytest.approx(0.33 * 1.75, abs=0.03)


def test_jacket_open_front_panels():
    spec = build_human(clothes=("jacket",))
    names = set(spec.graph.nodes)
    assert {"jacket_front_l", "jacket_front_r"} <= names
    assert "jacket_collar" in names
    # the two panels are split apart at the centre front (the opening)
    r = spec.build().aabbs()
    _, lhi = r["jacket_front_l"]
    _, rhi = r["jacket_front_r"]
    l_lo, _ = r["jacket_front_l"]
    r_lo, _ = r["jacket_front_r"]
    assert float(lhi[0]) < 0.0 < float(r_lo[0])


def test_swap_garments_keeps_skeleton_identical():
    base = build_human(clothes=("tshirt", "pants"))
    swapped = swap_garments(base, ("dress",))
    a = base.build().aabbs()
    b = swapped.build().aabbs()
    for bone in SIM_BONE_NAMES:
        assert bone in a and bone in b
        assert np.allclose(a[bone][0], b[bone][0]), bone
        assert np.allclose(a[bone][1], b[bone][1]), bone
    # face + hair parts survive the swap untouched
    face_a = {n for n in base.graph.nodes if n.startswith(("eye", "hair"))}
    face_b = {n for n in swapped.graph.nodes if n.startswith(("eye", "hair"))}
    assert face_a == face_b
    # garment sets actually differ
    assert set(base.extras["clothes"]["garments"]) == {"tshirt", "pants"}
    assert set(swapped.extras["clothes"]["garments"]) == {"dress"}


def test_no_clothes_option():
    spec = build_human(clothes=())
    assert spec.extras["clothes"]["garments"] == []
    r = spec.build()
    assert all(SIM_BONE_NAMES[i] in r.aabbs() for i in range(19))


def test_unknown_garment_rejected():
    with pytest.raises(ValueError):
        build_human(clothes=("spacesuit",))


def test_cloth_color_override():
    spec = build_human(clothes=("tshirt",), cloth_colors={"tshirt": (0.8, 0.1, 0.1)})
    node = spec.graph.nodes["tshirt"]
    assert node.metadata["albedo"] == pytest.approx((0.8, 0.1, 0.1))
    assert node.material == "fabric"
