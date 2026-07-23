"""One-shot self-repair loop for LLM spec generation.

When the streamed LLM answer fails validation — unparseable JSON, a spec
with no primitives, or one so structurally incoherent that the integrity
repair has to rewrite more than ~30% of its primitives — the model usually
*can* fix it if we show it the errors. This module streams the first answer,
validates it, and on failure sends the original answer plus the error list
back for exactly ONE repair round before giving up.

`stream_with_repair` is provider-agnostic: the caller supplies
`validator_fn(raw_text) -> list[str]` (empty = valid). `make_spec_validator`
builds the standard GenerationSpec validator the pipeline can adopt later —
the pipeline itself is intentionally untouched.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from .base import LLMProvider

_log = logging.getLogger(__name__)

# validator_fn(raw_text) -> list of human-readable problems; empty = valid.
ValidatorFn = Callable[[str], list[str]]

#: Default churn budget: if deterministic integrity repair has to modify or
#: drop more than this fraction of primitives, the spec is treated as a
#: failed generation and sent back for one repair round.
DEFAULT_INTEGRITY_THRESHOLD = 0.30

_REPAIR_INSTRUCTION = """Your previous answer was rejected by the spec validator.

Problems found:
{errors}

Previous answer:
```json
{answer}
```

Return a corrected answer that fixes every problem listed. Follow the same
JSON schema as before. Return JSON only — no prose, no code fences."""


@dataclass
class RepairResult:
    """Outcome of `stream_with_repair`.

    `text` is the final answer (the repair round's answer when one ran).
    `errors` is empty when the final answer passed validation.
    """

    text: str
    repaired: bool
    attempts: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _collect(
    provider: LLMProvider,
    system: str,
    user: str,
    stop_event: Optional[threading.Event],
    on_token: Optional[Callable[[str], None]],
) -> str:
    chunks: list[str] = []
    for tok in provider.stream(system, user, stop_event=stop_event):
        chunks.append(tok)
        if on_token is not None:
            on_token(tok)
    return "".join(chunks)


def stream_with_repair(
    provider: LLMProvider,
    system: str,
    user: str,
    validator_fn: ValidatorFn,
    *,
    stop_event: Optional[threading.Event] = None,
    on_token: Optional[Callable[[str], None]] = None,
) -> RepairResult:
    """Stream a generation and give the model ONE chance to repair it.

    Returns the first answer unchanged when it validates. Otherwise issues a
    single follow-up request containing the errors and the rejected answer,
    and returns that second answer with its own validation outcome. No third
    attempt is ever made — a model that fails twice falls through to the
    caller's normal fallback path.
    """
    first = _collect(provider, system, user, stop_event, on_token)
    errors = list(validator_fn(first) or [])
    if not errors:
        return RepairResult(text=first, repaired=False, attempts=1, errors=[])

    _log.info("spec validation failed (%d problem(s)); requesting one repair round", len(errors))
    repair_user = user + "\n\n---\n\n" + _REPAIR_INSTRUCTION.format(
        errors="\n".join(f"- {e}" for e in errors),
        answer=first,
    )
    second = _collect(provider, system, repair_user, stop_event, on_token)
    errors2 = list(validator_fn(second) or [])
    if errors2:
        _log.warning("repair round still invalid (%d problem(s)); giving up", len(errors2))
    return RepairResult(text=second, repaired=True, attempts=2, errors=errors2)


# ---------------------------------------------------------------------- spec validator
def _fraction_changed(before, after) -> float:
    """Fraction of primitives the integrity repair materially altered.

    Counts primitives whose kind/params changed, whose translation moved by
    more than a tolerance (5% of the largest bbox dimension, floor 1 cm —
    the repair routinely micro-snaps well-placed parts by ~2 cm and that
    must not count as churn), plus any primitives added or dropped,
    relative to the pre-repair count.
    """
    old = list(before.primitives)
    new = list(after.primitives)
    tol = max(0.01, 0.05 * max(getattr(before, "bbox_size", (1.0, 1.0, 1.0))))
    changed = abs(len(new) - len(old))
    for p, q in zip(old, new):
        if p.kind != q.kind or dict(p.params or {}) != dict(q.params or {}):
            changed += 1
            continue
        t_old = [x for row in (p.transform or []) for x in row]
        t_new = [x for row in (q.transform or []) for x in row]
        if len(t_old) != len(t_new):
            changed += 1
            continue
        # Row-major 4x4: translation lives at indices 3, 7, 11.
        shift = max(
            (abs(t_old[i] - t_new[i]) for i in (3, 7, 11) if i < len(t_old)),
            default=0.0,
        )
        basis = max(
            (abs(t_old[i] - t_new[i]) for i in range(len(t_old)) if i not in (3, 7, 11)),
            default=0.0,
        )
        if shift > tol or basis > 1e-3:
            changed += 1
    return changed / max(1, len(old))


def make_spec_validator(
    *,
    integrity_threshold: float = DEFAULT_INTEGRITY_THRESHOLD,
) -> ValidatorFn:
    """Build the standard GenerationSpec validator.

    Reports (as repairable error strings):
    - unparseable / non-JSON output,
    - specs with zero primitives,
    - specs where deterministic integrity repair churns more than
      `integrity_threshold` of primitives (default 30%) — a sign the model
      emitted structurally incoherent geometry rather than a near-miss.

    Imports are lazy so `llm` never hard-depends on `alignment` at import
    time (alignment already imports `llm.thinking`).
    """

    def validate(raw: str) -> list[str]:
        import copy

        from ..alignment.integrity import check_and_fix
        from ..alignment.parser import parse_spec
        from ..alignment.validator import normalize

        try:
            spec = parse_spec(raw)
        except Exception as e:
            return [f"output did not parse as a spec: {e}"]

        errors: list[str] = []
        if not spec.primitives:
            errors.append("spec contains no primitives — emit at least one")
        spec, _ = normalize(spec)
        snapshot = copy.deepcopy(spec)
        fixed, _ = check_and_fix(spec)
        churn = _fraction_changed(snapshot, fixed)
        if churn > integrity_threshold:
            errors.append(
                f"integrity repair had to modify {churn:.0%} of primitives "
                f"(limit {integrity_threshold:.0%}) — parts are floating or "
                f"disconnected; re-place them so every part touches its "
                f"neighbour and rests on the floor"
            )
        return errors

    return validate
