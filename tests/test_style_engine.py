"""Tests for the seeded procedural style engine (generation.style_engine).

Covers: grammar validity (every generated spec passes the validator with no
dropped kinds), determinism, budget compliance (points + bbox), keyword
routing, complexity control, style mutation, and the pipeline integration
(auto path, fallback path, random-style mutation of LLM specs). No network.
"""
from __future__ import annotations

import numpy as np
import pytest

from ironengine_3d_creator.alignment.schema import GenerationSpec, Primitive
from ironengine_3d_creator.alignment.validator import normalize
from ironengine_3d_creator.core.pipeline import PipelineRequest, run
from ironengine_3d_creator.generation.compositor import generate
from ironengine_3d_creator.generation.style_engine import (
    MAX_PARTS,
    STYLE_FAMILIES,
    StyleEngine,
    diversity_report,
    family_from_prompt,
    generate_style_spec,
    mutate_spec,
)

_SMALL = 6_000  # keep the compositor fast in tests


# ----------------------------------------------------------------------
# grammar validity
# ----------------------------------------------------------------------

@pytest.mark.parametrize("family", STYLE_FAMILIES)
@pytest.mark.parametrize("seed", (1, 7, 12345))
def test_every_family_generates_validator_clean_spec(family, seed):
    spec = StyleEngine(seed=seed).generate(family=family, n_points=_SMALL)
    clean, warns = normalize(spec)
    # No kind may be dropped — the grammar must stay inside the schema.
    assert not [w for w in warns if "dropped" in w], warns
    assert 3 <= len(clean.primitives) <= MAX_PARTS
    for p in clean.primitives:
        assert p.label, "every part should carry a label"
    # The spec must actually synthesize a point cloud.
    res = generate(clean)
    assert res.positions.shape[0] >= _SMALL // 2
    assert np.isfinite(res.positions).all()
    assert np.isfinite(res.colors).all()


def test_part_count_bounds_across_seeds():
    for seed in range(50):
        spec = StyleEngine(seed=seed).generate(n_points=_SMALL)
        assert 3 <= len(spec.primitives) <= MAX_PARTS, (seed, len(spec.primitives))


# ----------------------------------------------------------------------
# determinism
# ----------------------------------------------------------------------

def test_same_seed_identical_spec():
    a = generate_style_spec(seed=99, family=None, complexity="auto", n_points=_SMALL)
    b = generate_style_spec(seed=99, family=None, complexity="auto", n_points=_SMALL)
    assert a.to_json() == b.to_json()


def test_different_seeds_differ():
    a = generate_style_spec(seed=1, n_points=_SMALL)
    b = generate_style_spec(seed=2, n_points=_SMALL)
    assert a.to_json() != b.to_json()


def test_family_and_random_are_seeded_too():
    a = StyleEngine(seed=5).generate(family="random", n_points=_SMALL)
    b = StyleEngine(seed=5).generate(family="random", n_points=_SMALL)
    assert a.shape == b.shape and a.to_json() == b.to_json()


# ----------------------------------------------------------------------
# budget compliance
# ----------------------------------------------------------------------

def test_point_budget_is_honored():
    spec = generate_style_spec(seed=3, n_points=9_000)
    clean, _ = normalize(spec)
    assert clean.n_points == 9_000
    res = generate(clean)
    # holes/fur features may add/remove points; stay within a loose band.
    assert 0.5 * 9_000 <= res.positions.shape[0] <= 2.5 * 9_000


@pytest.mark.parametrize("bbox", [(1.0, 1.0, 1.0), (2.0, 0.8, 1.4), (0.5, 0.5, 0.5)])
def test_objects_fit_inside_bbox(bbox):
    for family in STYLE_FAMILIES:
        spec = StyleEngine(seed=11).generate(family=family, n_points=_SMALL, bbox=bbox)
        res = generate(normalize(spec)[0])
        lo, hi = res.positions.min(0), res.positions.max(0)
        # Features (scratch shrink / fur growth) can poke slightly outside;
        # allow 15% of the bbox dimension as tolerance.
        for axis in range(3):
            tol = 0.15 * bbox[axis]
            assert hi[axis] - lo[axis] <= bbox[axis] + tol, (family, axis, hi - lo)
        assert lo[1] >= -0.15 * bbox[1]      # rests near the ground plane
        assert hi[1] <= bbox[1] * 1.15


def test_complexity_controls_part_count():
    simple_counts, complex_counts = [], []
    for seed in range(10):
        s = StyleEngine(seed=seed).generate(family="mechanical", complexity="simple",
                                            n_points=200_000)
        c = StyleEngine(seed=seed).generate(family="mechanical", complexity="complex",
                                            n_points=200_000)
        simple_counts.append(len(s.primitives))
        complex_counts.append(len(c.primitives))
    assert max(simple_counts) <= 8
    assert min(complex_counts) >= 10  # grammars may cap below 40, but stay big
    assert sum(complex_counts) > sum(simple_counts) * 2


def test_tiny_point_budget_clamps_part_count():
    spec = StyleEngine(seed=4).generate(complexity="complex", n_points=4_000)
    # 4000 points // 800 = 5 affordable parts max.
    assert len(spec.primitives) <= 5


# ----------------------------------------------------------------------
# keyword routing
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text,family", [
    ("a wooden chair with four legs", "furniture"),
    ("kitchen table", "furniture"),
    ("fluffy creature with big eyes", "creature"),
    ("clockwork automaton with gears", "mechanical"),
    ("stone temple with columns", "architecture"),
    ("potted fern", "plant"),
    ("ceramic jug with two handles", "vessel"),
])
def test_keyword_routing(text, family):
    assert family_from_prompt(text) == family


@pytest.mark.parametrize("text", ["", None, "zzzz", "something"])
def test_unmatched_prompt_returns_none(text):
    assert family_from_prompt(text) is None


def test_earliest_keyword_wins():
    # "vase" appears before "tree" → vessel.
    assert family_from_prompt("a vase next to a tree") == "vessel"


# ----------------------------------------------------------------------
# style mutation
# ----------------------------------------------------------------------

def _base_spec() -> GenerationSpec:
    return GenerationSpec(
        shape="chair", n_points=10_000, bbox_size=(1, 1, 1),
        primitives=[Primitive("box", Primitive.identity_transform(),
                              {"size": [1.0, 1.0, 1.0]}, "seat")],
        features=[], color=(0.5, 0.4, 0.3), seed=1,
    )


def test_mutation_preserves_structure_and_is_deterministic():
    base = _base_spec()
    m1 = mutate_spec(base, seed=42)
    m2 = mutate_spec(base, seed=42)
    assert m1.to_json() == m2.to_json()
    assert [p.kind for p in m1.primitives] == [p.kind for p in base.primitives]
    assert [p.label for p in m1.primitives] == [p.label for p in base.primitives]
    # Params actually changed.
    assert m1.primitives[0].params["size"] != base.primitives[0].params["size"]


def test_mutation_without_seed_varies():
    base = _base_spec()
    m1 = mutate_spec(base, seed=0)
    m2 = mutate_spec(base, seed=0)
    assert m1.to_json() != m2.to_json()


def test_mutated_spec_still_validates():
    spec = mutate_spec(generate_style_spec(seed=8, family="vessel", n_points=_SMALL), seed=9)
    clean, warns = normalize(spec)
    assert not [w for w in warns if "dropped" in w], warns
    res = generate(clean)
    assert np.isfinite(res.positions).all()


# ----------------------------------------------------------------------
# pipeline integration
# ----------------------------------------------------------------------

class _FakeProvider:
    """Minimal LLMProvider double — streams canned chunks, no network."""

    def __init__(self, payload: str):
        self.payload = payload

    def stream(self, system, user, stop_event=None):
        yield self.payload


_LLM_JSON = ('{"shape": "chair", "n_points": 5000, "bbox_size": [1, 1, 1],'
             ' "primitives": [{"kind": "box", "params": {"size": [1, 0.2, 1]},'
             ' "label": "seat"}], "features": [], "color": [0.5, 0.4, 0.3], "seed": 3}')


def test_pipeline_auto_path_uses_style_engine():
    req = PipelineRequest(user_prompt="", n_points=_SMALL, seed=77, style="random")
    out = run(req, provider=None)
    assert out.spec.shape in STYLE_FAMILIES
    assert len(out.spec.primitives) >= 3
    assert out.generation.positions.shape[0] > 0


def test_pipeline_auto_path_routes_keywords_from_hint():
    req = PipelineRequest(user_prompt="", shape_hint="vase", n_points=_SMALL,
                          seed=5, style="auto")
    out = run(req, provider=None)
    assert out.spec.shape == "vessel"


def test_pipeline_explicit_family_and_determinism():
    req = PipelineRequest(user_prompt="", n_points=_SMALL, seed=5, style="furniture")
    a = run(req, provider=None)
    b = run(req, provider=None)
    assert a.spec.shape == "furniture"
    assert a.spec.to_json() == b.spec.to_json()


def test_pipeline_llm_parse_failure_falls_back_to_style_engine():
    req = PipelineRequest(user_prompt="a nice chair", n_points=_SMALL, seed=13,
                          style="random")
    out = run(req, _FakeProvider("not json at all"))
    assert any("falling back to style engine" in w for w in out.warnings)
    assert out.spec.shape in STYLE_FAMILIES
    assert out.generation.positions.shape[0] > 0


def test_pipeline_random_style_mutates_llm_spec():
    req = PipelineRequest(user_prompt="a chair", n_points=_SMALL, seed=0, style="random")
    a = run(req, _FakeProvider(_LLM_JSON))
    b = run(req, _FakeProvider(_LLM_JSON))
    assert any("style mutation" in w for w in a.warnings)
    # seed=0 → fresh entropy per run → repeated prompts yield different objects.
    assert a.spec.to_json() != b.spec.to_json()
    # Seeded requests stay reproducible.
    req_seeded = PipelineRequest(user_prompt="a chair", n_points=_SMALL, seed=21,
                                 style="random")
    c = run(req_seeded, _FakeProvider(_LLM_JSON))
    d = run(req_seeded, _FakeProvider(_LLM_JSON))
    assert c.spec.to_json() == d.spec.to_json()


def test_pipeline_auto_style_leaves_llm_spec_unmutated():
    req = PipelineRequest(user_prompt="a chair", n_points=_SMALL, seed=0, style="auto")
    a = run(req, _FakeProvider(_LLM_JSON))
    b = run(req, _FakeProvider(_LLM_JSON))
    assert not any("style mutation" in w for w in a.warnings)
    assert a.spec.to_json() == b.spec.to_json()


# ----------------------------------------------------------------------
# diversity proof
# ----------------------------------------------------------------------

def test_diversity_report_shows_real_variance():
    stats = diversity_report(n=20, seed0=10_000, n_points=12_000)
    assert len(stats) == 20
    # Every object is validator-clean.
    assert not [s for s in stats if s["validator_warnings"]]
    # Non-trivial variance: many kinds, several distinct part counts,
    # distinct colors, every family exercised.
    all_kinds = {k for s in stats for k in s["kinds"]}
    assert len(all_kinds) >= 7
    part_counts = [s["n_parts"] for s in stats]
    assert len(set(part_counts)) >= 5
    assert max(part_counts) - min(part_counts) >= 6
    colors = {tuple(s["color"]) for s in stats}
    assert len(colors) >= 15
    families = {s["family"] for s in stats}
    assert families == set(STYLE_FAMILIES)
    # Per-object material variety exists somewhere.
    assert max(len(s["materials"]) for s in stats) >= 2
