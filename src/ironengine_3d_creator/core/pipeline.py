"""End-to-end orchestrator: requirements → spec → point cloud.

Runs synchronously inside a worker thread (the UI wraps each call in a QThread
so the main thread stays responsive).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

from ..alignment.defaults import auto_template
from ..alignment.integrity import check_and_fix as integrity_fix
from ..alignment.parser import parse_spec
from ..alignment.schema import GenerationSpec
from ..alignment.validator import normalize
from ..generation.code_sandbox import run_sandbox
from ..generation.colorize import base_color, shaded_colors
from ..generation.compositor import GenerationResult, generate
from ..llm.base import LLMProvider
from ..llm.prompts import CODE_SYSTEM_PROMPT, SPEC_SYSTEM_PROMPT

_log = logging.getLogger(__name__)


@dataclass
class PipelineRequest:
    user_prompt: str            # free-form description; "" → auto mode
    shape_hint: str | None = None   # combo selection from UI
    n_points: int = 50_000
    bbox: tuple[float, float, float] = (1.0, 1.0, 1.0)
    legs: int = 0               # numeric hint folded into the user prompt
    details: str = ""
    seed: int = 0
    code_mode: bool = False     # advanced: LLM emits Python instead of JSON


@dataclass
class PipelineResult:
    spec: GenerationSpec
    generation: GenerationResult
    warnings: list[str]
    raw_llm: str = ""


def build_user_prompt(req: PipelineRequest) -> str:
    pieces = []
    if req.user_prompt:
        pieces.append(req.user_prompt.strip())
    if req.shape_hint:
        pieces.append(f"Shape style: {req.shape_hint}.")
    pieces.append(f"Approximate point budget: {req.n_points}.")
    pieces.append(f"Bounding box (x, y, z) ~ ({req.bbox[0]:.2f}, {req.bbox[1]:.2f}, {req.bbox[2]:.2f}).")
    if req.legs:
        pieces.append(f"Number of legs / supports: {req.legs}.")
    if req.details.strip():
        pieces.append(f"Surface details: {req.details.strip()}.")
    return " ".join(pieces)


def run(
    req: PipelineRequest,
    provider: LLMProvider | None,
    *,
    on_token: Callable[[str], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    stop_event: Optional[threading.Event] = None,
) -> PipelineResult:
    """Execute the full pipeline. `on_token` is called for every streaming chunk;
    `on_stage` for stage transitions ('aligning', 'sampling', 'finalizing').

    `stop_event` is forwarded to the provider's stream method — set it to make
    the LLM streaming bail out promptly and close its socket."""
    warnings: list[str] = []
    raw = ""

    # ---- Step 1: build a spec ----------------------------------------------
    if not req.user_prompt.strip() or provider is None:
        if on_stage:
            on_stage("auto")
        spec = auto_template(req.shape_hint)
        # Honor the user's point budget / bbox / seed even in auto mode.
        spec.n_points = req.n_points
        spec.bbox_size = req.bbox
        if req.seed:
            spec.seed = req.seed
    elif req.code_mode:
        if on_stage:
            on_stage("code")
        chunks = []
        for tok in provider.stream(CODE_SYSTEM_PROMPT, build_user_prompt(req), stop_event=stop_event):
            chunks.append(tok)
            if on_token:
                on_token(tok)
        raw = "".join(chunks)
        # Code mode skips the spec route entirely — we run the sandbox and wrap.
        if on_stage:
            on_stage("sandbox")
        positions, colors = run_sandbox(raw, n_points=req.n_points)
        if colors is None:
            rng = np.random.default_rng(req.seed or None)
            colors = shaded_colors(positions, base_color(req.shape_hint or "abstract", None), rng)
        result = GenerationResult(
            positions=positions,
            colors=colors,
            labels=np.zeros(positions.shape[0], dtype=np.int32),
            label_names=["code_mode"],
        )
        spec = GenerationSpec(
            shape=req.shape_hint or "abstract",
            n_points=positions.shape[0],
            bbox_size=req.bbox,
            primitives=[],
            features=[],
            color=None,
            seed=req.seed,
        )
        return PipelineResult(spec=spec, generation=result, warnings=warnings, raw_llm=raw)
    else:
        if on_stage:
            on_stage("aligning")
        chunks = []
        for tok in provider.stream(SPEC_SYSTEM_PROMPT, build_user_prompt(req), stop_event=stop_event):
            chunks.append(tok)
            if on_token:
                on_token(tok)
        raw = "".join(chunks)
        try:
            spec = parse_spec(raw)
        except Exception as e:
            warnings.append(f"could not parse LLM JSON ({e}); falling back to auto")
            spec = auto_template(req.shape_hint)
        if req.seed:
            spec.seed = req.seed
        if req.n_points:
            spec.n_points = req.n_points

    # ---- Step 2: validate / normalize --------------------------------------
    if on_stage:
        on_stage("validating")
    spec, warns = normalize(spec)
    warnings.extend(warns)

    # ---- Step 2b: structural integrity (snap floating parts together) -----
    if on_stage:
        on_stage("integrity")
    spec, integrity_warns = integrity_fix(spec)
    warnings.extend(integrity_warns)

    # ---- Step 3: synthesize the point cloud -------------------------------
    if on_stage:
        on_stage("sampling")
    result = generate(spec)

    if on_stage:
        on_stage("done")
    return PipelineResult(spec=spec, generation=result, warnings=warnings, raw_llm=raw)


def replay_spec(spec: GenerationSpec) -> PipelineResult:
    """Re-run the deterministic generator from an existing spec (no LLM)."""
    spec, warns = normalize(spec)
    spec, ifix = integrity_fix(spec)
    return PipelineResult(spec=spec, generation=generate(spec), warnings=warns + ifix)


def stream_tokens(provider: LLMProvider, system: str, user: str) -> Iterator[str]:
    """Convenience pass-through for the UI's token stream widget."""
    yield from provider.stream(system, user)
