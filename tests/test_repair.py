"""Tests for the spec self-repair loop (llm.repair).

Providers are fakes — no HTTP. The validator integration tests use the real
parser / normalizer / integrity repair on in-memory spec JSON.
"""
from __future__ import annotations

import threading

from ironengine_3d_creator.llm.repair import (
    RepairResult,
    make_spec_validator,
    stream_with_repair,
)


class FakeProvider:
    """Serves canned answers in order; records every (system, user) call."""

    name = "fake"

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def stream(self, system, user, stop_event=None):
        self.calls.append((system, user))
        answer = self._answers.pop(0) if self._answers else "{}"
        yield answer


# ------------------------------------------------------------------ loop mechanics
class TestStreamWithRepair:
    def test_valid_first_answer_no_repair(self):
        provider = FakeProvider(['{"shape":"rock"}'])
        result = stream_with_repair(provider, "sys", "usr", lambda raw: [])
        assert result.ok and not result.repaired and result.attempts == 1
        assert result.text == '{"shape":"rock"}'
        assert len(provider.calls) == 1

    def test_failed_first_answer_triggers_one_repair_round(self):
        provider = FakeProvider(["garbage", '{"shape":"rock"}'])
        calls: list[str] = []

        def validator(raw: str) -> list[str]:
            return [] if raw.startswith("{") else ["not JSON"]

        result = stream_with_repair(provider, "sys", "usr", validator)
        assert result.ok and result.repaired and result.attempts == 2
        assert result.text == '{"shape":"rock"}'
        assert len(provider.calls) == 2
        # The repair request carries the errors and the rejected answer.
        repair_prompt = provider.calls[1][1]
        assert "usr" in repair_prompt
        assert "not JSON" in repair_prompt
        assert "garbage" in repair_prompt

    def test_second_failure_gives_up_without_third_attempt(self):
        provider = FakeProvider(["bad-1", "bad-2", "bad-3"])
        result = stream_with_repair(
            provider, "sys", "usr", lambda raw: ["still broken"]
        )
        assert not result.ok and result.repaired and result.attempts == 2
        assert result.errors == ["still broken"]
        assert result.text == "bad-2"
        assert len(provider.calls) == 2  # never a third round

    def test_stop_event_and_token_callback_forwarded(self):
        provider = FakeProvider(['{"a":1}'])
        tokens: list[str] = []
        result = stream_with_repair(
            provider, "sys", "usr", lambda raw: [],
            stop_event=threading.Event(), on_token=tokens.append,
        )
        assert result.ok
        assert tokens == ['{"a":1}']

    def test_validator_none_treated_as_valid(self):
        provider = FakeProvider(["x"])
        result = stream_with_repair(provider, "s", "u", lambda raw: None)  # type: ignore[return-value]
        assert result.ok and result.attempts == 1


# ------------------------------------------------------------------ spec validator
_GOOD_CHAIR = """{
  "shape": "chair", "n_points": 50000, "bbox_size": [0.5, 0.9, 0.5],
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

# Same chair but every part floats at y=5 — integrity must move all of them.
_FLOATING_CHAIR = _GOOD_CHAIR.replace("0.45],[0,0,1,0]", "5.45],[0,0,1,0]").replace(
    "0.225],[0,0,1,", "5.225],[0,0,1,"
)


class TestMakeSpecValidator:
    def test_unparseable_output(self):
        validate = make_spec_validator()
        errors = validate("this is not json at all")
        assert len(errors) == 1
        assert "did not parse" in errors[0]

    def test_empty_primitives_flagged(self):
        validate = make_spec_validator()
        errors = validate('{"shape": "abstract", "primitives": []}')
        assert any("no primitives" in e for e in errors)

    def test_good_spec_passes(self):
        validate = make_spec_validator()
        assert validate(_GOOD_CHAIR) == []

    def test_structurally_incoherent_spec_flagged(self):
        validate = make_spec_validator()
        errors = validate(_FLOATING_CHAIR)
        assert any("integrity repair" in e for e in errors), errors

    def test_threshold_is_configurable(self):
        # A 100% threshold tolerates the fully-floating chair.
        validate = make_spec_validator(integrity_threshold=1.0)
        assert validate(_FLOATING_CHAIR) == []

    def test_full_loop_with_spec_validator(self):
        provider = FakeProvider(["not json", _GOOD_CHAIR])
        result = stream_with_repair(provider, "sys", "usr", make_spec_validator())
        assert result.ok and result.repaired and result.attempts == 2
