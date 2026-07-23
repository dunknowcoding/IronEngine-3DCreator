"""Tests for panel-native emission in generation.style_families.

Desktops / tabletops / seats used to be emitted as panels with box-semantics
3-element `size` triples, which the validator had to repair via a compat
shim. The grammar now emits proper panel params: 2-element in-plane `size`,
a separate `thickness`, and an explicit rotation (flat rx=π/2 for horizontal
tops; none for upright back slats). The validator compat shim stays for
legacy LLM output.

Note: `StyleEngine.generate` fits the whole object to the bbox with a
uniform scale, so rotation assertions first strip the global scale from the
transform's 3x3 block.
"""
from __future__ import annotations

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.alignment.validator import normalize
from ironengine_3d_creator.generation.style_engine import StyleEngine

_SMALL = 6_000


def _furniture(seed: int) -> GenerationSpec:
    return StyleEngine(seed=seed).generate(family="furniture", n_points=_SMALL)


def _find(spec: GenerationSpec, *label_prefixes: str) -> list[Primitive]:
    return [
        p for p in spec.primitives
        if any((p.label or "").startswith(pre) for pre in label_prefixes)
    ]


def _rotation_of(prim: Primitive) -> np.ndarray:
    """Rotation part of the primitive transform (uniform scale stripped)."""
    M = np.asarray(prim.transform, dtype=np.float64)[:3, :3]
    norms = np.linalg.norm(M, axis=0)
    assert norms.min() > 1e-9
    assert norms.max() / norms.min() == pytest.approx(1.0, rel=1e-3)
    return M / norms


class TestPanelNativeEmission:
    def test_tabletops_are_panels_with_native_params(self):
        flat = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        for seed in range(1, 11):
            tops = _find(_furniture(seed), "seat", "top")
            assert len(tops) == 1, f"seed {seed}: expected exactly one seat/top"
            top = tops[0]
            assert top.kind == "panel", f"seed {seed}: top should be panel-native"
            # Native params: 2-element in-plane size + separate thickness.
            assert len(top.params["size"]) == 2
            assert top.params["thickness"] > 0
            # Laid flat with rx=π/2 (panel in-plane Y → world Z, thickness → Y).
            assert _rotation_of(top) == pytest.approx(flat, abs=1e-4)

    def test_chair_back_slats_are_upright_panels(self):
        slats = []
        for seed in range(1, 30):
            slats = _find(_furniture(seed), "back_slat")
            if slats:
                break
        assert slats, "no chair with back slats found in 29 seeds"
        for slat in slats:
            assert slat.kind == "panel"
            assert len(slat.params["size"]) == 2
            assert slat.params["thickness"] > 0
            # Upright is a panel's native orientation: rotation ≈ identity.
            assert _rotation_of(slat) == pytest.approx(np.eye(3), abs=1e-4)

    def test_no_box_semantics_compat_warning_from_grammar(self):
        for seed in range(1, 11):
            _, warns = normalize(_furniture(seed))
            assert not [w for w in warns if "interpreted as box slab" in w], (
                seed, warns,
            )


class TestLegacyBoxSemanticsCompat:
    """The validator shim stays for legacy LLM / saved-spec output."""

    def test_three_element_panel_size_still_normalized(self):
        legacy = GenerationSpec(
            shape="table", n_points=_SMALL, bbox_size=(1.0, 1.0, 1.0),
            primitives=[Primitive(
                kind="panel",
                transform=Primitive.identity_transform(),
                params={"size": [0.8, 0.04, 0.6]},  # box slab [w, t, d]
                label="top",
            )],
            features=[], color=None, seed=1,
        )
        clean, warns = normalize(legacy)
        assert any("interpreted as box slab" in w for w in warns)
        panel = clean.primitives[0]
        assert panel.params["size"] == [0.8, 0.6]
        assert panel.params["thickness"] == pytest.approx(0.04)
        # Compat rotation lays it flat (rx=π/2 applied to the transform).
        flat = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        assert _rotation_of(panel) == pytest.approx(flat, abs=1e-4)
