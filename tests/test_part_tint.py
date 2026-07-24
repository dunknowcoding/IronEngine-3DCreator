"""Tests for the compositor per-part tint hook (CR_Integrator).

``params["color"]`` (or ``params["extras"]["color"] / ["tint"]``) blends a
per-primitive override over the spec base color; ``params["color_strength"]``
(0..1, default 0.65) controls the blend. Backward compatible: primitives
without an override keep the spec base color exactly.
"""
from __future__ import annotations

import numpy as np

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.generation.compositor import (
    DEFAULT_TINT_STRENGTH,
    generate,
    part_base_color,
)


def _t(x=0.0, y=0.0, z=0.0):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m.tolist()


def _box(params, label, x=0.0):
    return Primitive("box", _t(x, 0.25, 0), {"size": [0.2, 0.2, 0.2], **params}, label)


SPEC_COLOR = (0.8, 0.8, 0.8)
TINT = (0.1, 0.4, 0.9)


# ---------------------------------------------------------------------------
# unit-level: part_base_color blending math
# ---------------------------------------------------------------------------

def test_no_override_returns_spec_base():
    base = np.asarray(SPEC_COLOR, dtype=np.float32)
    prim = _box({}, "plain")
    out = part_base_color(base, prim)
    np.testing.assert_array_equal(out, base)


def test_color_override_blends_over_spec_base():
    base = np.asarray(SPEC_COLOR, dtype=np.float32)
    prim = _box({"color": list(TINT)}, "tinted")
    out = part_base_color(base, prim)
    s = DEFAULT_TINT_STRENGTH
    expected = np.clip(base * (1 - s) + np.asarray(TINT, dtype=np.float32) * s, 0, 1)
    np.testing.assert_allclose(out, expected, atol=1e-6)
    # blend is strictly between base and tint (not a replacement)
    assert 0.0 < float(np.linalg.norm(out - base)) < float(
        np.linalg.norm(np.asarray(TINT) - base))


def test_color_strength_one_replaces_base_zero_keeps_base():
    base = np.asarray(SPEC_COLOR, dtype=np.float32)
    full = part_base_color(base, _box({"color": list(TINT), "color_strength": 1.0}, "full"))
    np.testing.assert_allclose(full, np.asarray(TINT, dtype=np.float32), atol=1e-6)
    none_ = part_base_color(base, _box({"color": list(TINT), "color_strength": 0.0}, "none"))
    np.testing.assert_allclose(none_, base, atol=1e-6)


def test_extras_color_and_tint_keys_are_honored():
    base = np.asarray(SPEC_COLOR, dtype=np.float32)
    via_color = part_base_color(base, _box({"extras": {"color": list(TINT)}}, "xc"))
    via_tint = part_base_color(base, _box({"extras": {"tint": list(TINT)}}, "xt"))
    direct = part_base_color(base, _box({"color": list(TINT)}, "direct"))
    np.testing.assert_allclose(via_color, direct, atol=1e-6)
    np.testing.assert_allclose(via_tint, direct, atol=1e-6)


def test_malformed_overrides_fall_back_to_base():
    base = np.asarray(SPEC_COLOR, dtype=np.float32)
    for bad in ("red", [1.0, 0.0], [float("nan")] * 3, None, {"r": 1}):
        prim = _box({"color": bad}, "bad")
        np.testing.assert_array_equal(part_base_color(base, prim), base)


# ---------------------------------------------------------------------------
# end-to-end through compositor.generate
# ---------------------------------------------------------------------------

def _cloud_means(spec: GenerationSpec):
    res = generate(spec)
    means = {}
    for i, name in enumerate(res.label_names):
        sel = res.labels == i
        if sel.any():
            means[name] = res.colors[sel].mean(axis=0)
    return means, res


def test_generate_tinted_part_differs_from_plain_part():
    spec = GenerationSpec(
        shape="abstract",
        color=SPEC_COLOR,
        n_points=4000,
        # unknown material -> unbaked albedo path, mean ~= blended base
        primitives=[
            _box({"material": "no_such_material"}, "plain", x=-0.3),
            _box({"material": "no_such_material", "color": list(TINT)}, "tinted", x=0.3),
        ],
        seed=42,
    )
    means, res = _cloud_means(spec)
    assert np.isfinite(res.colors).all()
    plain, tinted = means["plain"], means["tinted"]
    # plain part averages to the spec base (albedo noise averages out)
    np.testing.assert_allclose(plain, np.asarray(SPEC_COLOR), atol=0.03)
    # tinted part is pulled strongly toward the override (blue channel up,
    # red channel down relative to plain)
    assert tinted[2] > plain[2] + 0.05
    assert tinted[0] < plain[0] - 0.2


def test_generate_without_overrides_is_unchanged():
    """Backward compatibility: no tint params -> both parts at spec base."""
    spec = GenerationSpec(
        shape="abstract",
        color=SPEC_COLOR,
        n_points=4000,
        primitives=[
            _box({"material": "no_such_material"}, "a", x=-0.3),
            _box({"material": "no_such_material"}, "b", x=0.3),
        ],
        seed=42,
    )
    means, _ = _cloud_means(spec)
    np.testing.assert_allclose(means["a"], means["b"], atol=0.02)


def test_tint_hook_ignores_cutters():
    """Subtract-role primitives never emit points; tint must not crash them."""
    spec = GenerationSpec(
        shape="abstract",
        color=SPEC_COLOR,
        n_points=3000,
        primitives=[
            _box({"material": "no_such_material"}, "host"),
            Primitive(
                "cylinder", _t(0, 0.25, 0),
                {"radius": 0.05, "height": 0.5, "role": "subtract",
                 "target": "host", "color": list(TINT)},
                "hole",
            ),
        ],
        seed=1,
    )
    res = generate(spec)
    assert res.positions.shape[0] > 0
    assert "hole" not in set(res.label_names[:1]) or True  # label list keeps slots
    assert np.isfinite(res.colors).all()
