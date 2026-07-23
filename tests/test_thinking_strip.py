"""Tests for thinking-tag stripping, incl. the MiniMax-M3 misplaced-tag quirk."""
from __future__ import annotations

import json

from ironengine_3d_creator.llm.thinking import strip


class TestStrip:
    def test_no_tags_is_identity(self):
        text = '{"shape": "chair", "primitives": []}'
        assert strip(text) == text

    def test_normal_paired_block_removed(self):
        text = '<think>let me think about chairs</think>{"shape":"chair"}'
        assert strip(text) == '{"shape":"chair"}'

    def test_reasoning_with_brace_example_removed(self):
        # A balanced JSON example inside the reasoning must NOT be mistaken
        # for the answer.
        text = '<think>use {"radius":0.03} for legs</think>{"shape":"chair"}'
        assert json.loads(strip(text))["shape"] == "chair"

    def test_alias_tags(self):
        text = '<thinking>hmm</thinking>{"a":1}'
        assert json.loads(strip(text)) == {"a": 1}

    def test_misplaced_close_tag_inside_answer(self):
        # Observed from MiniMax-M3 (api.minimax.io): the model starts the
        # JSON answer and emits </think> a few tokens later, mid-object.
        text = (
            '<think>The user wants a chair. Seat at y=0.45...'
            '{"shape":"chair</think>\n\n","n_points":50000,"primitives":[]}'
        )
        out = strip(text)
        data = json.loads(out)
        assert data["shape"] == "chair"
        assert data["n_points"] == 50000

    def test_stray_close_tag_without_open(self):
        text = '{"shape":"chai</think>r"}'
        # Tag token removed; surrounding content preserved.
        assert strip(text) == '{"shape":"chair"}'

    def test_unclosed_open_tag_with_answer(self):
        text = '<think>reasoning without end {"shape":"chair"}'
        assert json.loads(strip(text))["shape"] == "chair"

    def test_prose_around_answer_dropped_to_json(self):
        # With tags present, strip returns from the answer's first '{' —
        # leading prose (tagged or not) is reasoning noise for the parser.
        text = '<think>x</think>Here is the spec: {"shape":"rock"}'
        assert json.loads(strip(text))["shape"] == "rock"
