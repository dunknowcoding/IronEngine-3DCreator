"""End-to-end orchestrator: requirements → spec → point cloud.

Runs synchronously inside a worker thread (the UI wraps each call in a QThread
so the main thread stays responsive).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from typing import Callable, Iterator, Optional

import numpy as np

from ..alignment.integrity import check_and_fix as integrity_fix
from ..alignment.parser import parse_spec
from ..alignment.schema import GenerationSpec
from ..alignment.validator import normalize
from ..generation.code_sandbox import run_sandbox
from ..generation.colorize import base_color, shaded_colors
from ..generation.compositor import GenerationResult, generate
from ..generation.style_engine import (
    STYLE_FAMILIES,
    StyleEngine,
    family_from_prompt,
    mutate_spec,
)
from ..llm.base import LLMProvider
from ..llm.prompts import CODE_SYSTEM_PROMPT, SPEC_SYSTEM_PROMPT
from ..llm.repair import make_spec_validator, stream_with_repair

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
    ram_cap_mb: int = 0         # 0 → no cap; otherwise n_points is clamped to fit
    style: str = "auto"         # 'auto' | 'random' | style-family name
    complexity: str = "auto"    # 'auto' | 'simple' | 'complex' (style engine only)


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


def _enforce_ram_cap(req: PipelineRequest, warnings: list[str]) -> PipelineRequest:
    """Clamp n_points so the estimated generation footprint fits the RAM cap.

    The Resources panel slider used to be saved but never enforced; now the
    cap reaches the pipeline via PipelineRequest.ram_cap_mb (0 = unlimited).
    """
    from .resources import estimate_generation_ram_mb

    cap = int(req.ram_cap_mb or 0)
    if cap <= 0 or req.n_points <= 0:
        return req
    est = estimate_generation_ram_mb(req.n_points)
    if est <= cap:
        return req
    # Inverse of estimate_generation_ram_mb: points = mb * 2**20 / (24 * 4).
    clamped = max(1000, int(cap * 1024 * 1024 / (24 * 4)))
    warnings.append(
        f"point budget clamped {req.n_points:,} → {clamped:,} to fit the "
        f"{cap} MB RAM cap (estimated {est:,.0f} MB)"
    )
    _log.info("n_points clamped %d → %d by RAM cap %d MB", req.n_points, clamped, cap)
    return replace(req, n_points=clamped)


def _style_spec_for_request(req: PipelineRequest) -> GenerationSpec:
    """Procedural style-engine spec for auto / no-LLM / fallback paths.

    Family selection: an explicit style-family name wins; 'auto' routes on
    prompt/shape-hint keywords; 'random' (or an auto route with no keyword
    hit) draws a weighted-random family from the request's seed.
    """
    style = (req.style or "auto").strip().lower()
    family = None
    if style in STYLE_FAMILIES:
        family = style
    elif style == "auto":
        family = family_from_prompt(
            " ".join(p for p in (req.user_prompt, req.shape_hint or "", req.details) if p)
        )
    engine = StyleEngine(seed=req.seed or 0)
    return engine.generate(family=family, complexity=req.complexity,
                           n_points=req.n_points, bbox=req.bbox)


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
    req = _enforce_ram_cap(req, warnings)

    # ---- Step 1: build a spec ----------------------------------------------
    if not req.user_prompt.strip() or provider is None:
        if on_stage:
            on_stage("auto")
        spec = _style_spec_for_request(req)
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
        # Self-repair loop: the first answer is validated (parseable JSON,
        # non-empty primitives, <30% integrity churn); on failure the model
        # gets exactly ONE repair round with the error list before we fall
        # back to the deterministic style engine.
        outcome = stream_with_repair(
            provider, SPEC_SYSTEM_PROMPT, build_user_prompt(req),
            make_spec_validator(),
            stop_event=stop_event, on_token=on_token,
        )
        raw = outcome.text
        spec: GenerationSpec | None = None
        if outcome.ok:
            if outcome.repaired:
                warnings.append(
                    "LLM spec self-repaired after one validator-feedback round"
                )
            try:
                spec = parse_spec(raw)
            except Exception as e:  # defensive: validator already parsed it
                warnings.append(f"could not parse LLM JSON ({e}); falling back to style engine")
        else:
            warnings.append(
                f"LLM spec still invalid after {outcome.attempts} attempt(s) "
                f"({'; '.join(outcome.errors)}); falling back to style engine"
            )
        if spec is None:
            spec = _style_spec_for_request(req)
        else:
            # Style 'random' on the LLM path: seeded style mutation so repeated
            # identical prompts don't yield identical objects.
            if (req.style or "").strip().lower() == "random":
                spec = mutate_spec(spec, seed=req.seed or 0)
                warnings.append("style 'random': applied seeded style mutation")
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
