"""Tests for the self-repair wiring in core.pipeline (llm.repair integration).

Providers are fakes serving canned answers in order — no network. Asserts:
a broken-then-valid LLM stream is repaired and accepted, a twice-broken
stream falls back to the seeded deterministic style engine, a valid first
answer never triggers a repair round, and the offline (no provider) path
is untouched.
"""
from __future__ import annotations

from ironengine_3d_creator.core.pipeline import PipelineRequest, run
from ironengine_3d_creator.generation.style_engine import STYLE_FAMILIES

_SMALL = 6_000  # keep the compositor fast in tests

_GOOD_CHAIR = """{
  "shape": "chair", "n_points": 5000, "bbox_size": [0.5, 0.9, 0.5],
  "primitives": [
    {"kind": "box", "label": "seat", "params": {"size": [0.45, 0.04, 0.45]},
     "transform": [[1,0,0,0],[0,1,0,0.45],[0,0,1,0],[0,0,0,1]]},
    {"kind": "cylinder", "label": "leg_0", "params": {"radius": 0.03, "height": 0.45},
     "transform": [[1,0,0,-0.2],[0,1,0,0.225],[0,0,1,-0.2],[0,0,0,1]]},
    {"kind": "cylinder", "label": "leg_1", "params": {"radius": 0.03, "height": 0.45},
     "transform": [[1,0,0,0.2],[0,1,0,0.225],[0,0,1,-0.2],[0,0,0,1]]},
    {"kind": "cylinder", "label": "leg_2", "params": {"radius": 0.03, "height": 0.45},
     "transform": [[1,0,0,-0.2],[0,1,0,0.225],[0,0,1,0.2],[0,0,0,1]]},
    {"kind": "cylinder", "label": "leg_3", "params": {"radius": 0.03, "height": 0.45},
     "transform": [[1,0,0,0.2],[0,1,0,0.225],[0,0,1,0.2],[0,0,0,1]]}
  ]
}"""


class ScriptedProvider:
    """Serves canned answers in order; repeats the last one when exhausted."""

    name = "scripted"

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self._last = ""
        self.calls: list[tuple[str, str]] = []

    def stream(self, system, user, stop_event=None):
        self.calls.append((system, user))
        if self._answers:
            self._last = self._answers.pop(0)
        yield self._last


def _req(**kw) -> PipelineRequest:
    kw.setdefault("user_prompt", "a wooden chair")
    kw.setdefault("n_points", _SMALL)
    kw.setdefault("seed", 7)
    return PipelineRequest(**kw)


class TestPipelineRepairWiring:
    def test_broken_then_valid_spec_is_repaired_and_accepted(self):
        provider = ScriptedProvider(["this is not json at all", _GOOD_CHAIR])
        out = run(_req(), provider)
        # The repaired (second) answer was accepted as the spec.
        assert len(out.spec.primitives) == 5
        assert out.spec.shape == "chair"
        assert out.raw_llm == _GOOD_CHAIR
        assert len(provider.calls) == 2
        assert any("self-repaired" in w for w in out.warnings)
        assert not any("falling back to style engine" in w for w in out.warnings)
        # The repair prompt carries the validator errors and the bad answer.
        assert "did not parse" in provider.calls[1][1]
        assert "this is not json at all" in provider.calls[1][1]
        # And the pipeline still produced a point cloud.
        assert out.generation.positions.shape[0] > 0

    def test_valid_first_answer_skips_repair(self):
        provider = ScriptedProvider([_GOOD_CHAIR])
        out = run(_req(), provider)
        assert len(provider.calls) == 1
        assert not any("self-repaired" in w for w in out.warnings)
        assert len(out.spec.primitives) == 5

    def test_twice_broken_falls_back_to_seeded_style_engine(self):
        provider_a = ScriptedProvider(["garbage-1", "garbage-2"])
        out = run(_req(), provider_a)
        assert len(provider_a.calls) == 2  # one repair round, never a third
        assert any("falling back to style engine" in w for w in out.warnings)
        assert any("2 attempt(s)" in w for w in out.warnings)
        assert out.spec.shape in STYLE_FAMILIES
        assert out.generation.positions.shape[0] > 0
        # Deterministic fallback: same seed → identical object.
        provider_b = ScriptedProvider(["garbage-1", "garbage-2"])
        out_b = run(_req(), provider_b)
        assert out.spec.to_json() == out_b.spec.to_json()

    def test_offline_path_still_uses_style_engine(self):
        out = run(_req(user_prompt=""), provider=None)
        assert out.spec.shape in STYLE_FAMILIES
        assert len(out.spec.primitives) >= 3

    def test_on_token_streams_both_rounds(self):
        provider = ScriptedProvider(["bad", _GOOD_CHAIR])
        tokens: list[str] = []
        out = run(_req(), provider, on_token=tokens.append)
        assert "".join(tokens) == "bad" + _GOOD_CHAIR
        assert out.raw_llm == _GOOD_CHAIR
