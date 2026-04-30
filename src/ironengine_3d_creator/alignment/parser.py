"""Forgiving JSON → GenerationSpec parser.

Tolerates:
- Extra whitespace, code fences, prose surrounding the JSON object.
- Missing fields (filled by defaults in `validator.normalize`).
- `transform` given as a list of 16 floats instead of a 4x4 list of lists.
- Unknown `kind` values (left as-is so the validator can clamp/reject).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .schema import GenerationSpec, Primitive

_log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _find_json_object(text: str) -> str:
    """Best-effort extraction of the first balanced { ... } block."""
    text = _strip_fences(text)
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _normalize_transform(t: Any) -> list[list[float]]:
    if t is None:
        return Primitive.identity_transform()
    if isinstance(t, list) and t and isinstance(t[0], (int, float)) and len(t) == 16:
        flat = [float(x) for x in t]
        return [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]
    if isinstance(t, list) and len(t) == 4 and all(isinstance(r, list) and len(r) == 4 for r in t):
        return [[float(x) for x in r] for r in t]
    _log.debug("invalid transform %r → identity", t)
    return Primitive.identity_transform()


def parse_spec(raw: str) -> GenerationSpec:
    """Parse an LLM string response into a GenerationSpec.

    Reasoning models often wrap chain-of-thought in <think>…</think> blocks
    around the JSON. We strip those first so the JSON scan isn't fooled by
    braces inside the thoughts.
    """
    from ..llm.thinking import strip as strip_thinking
    blob = _find_json_object(strip_thinking(raw))
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e.msg}") from e

    # Normalize each primitive's transform.
    for p in data.get("primitives", []):
        p["transform"] = _normalize_transform(p.get("transform"))
        p.setdefault("params", {})

    return GenerationSpec.from_json(data)
