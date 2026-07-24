"""Provider fallback chain — transparent retry across cloud providers.

User policy: MiniMax M3 is the PRIMARY cloud provider and DeepSeek the
automatic FALLBACK. When the primary provider fails — auth error, timeout,
rate limit, connection failure, or a spec that stays invalid even after the
self-repair round — the pipeline retries the same request with the next
provider in the chain instead of dropping straight to the style engine.

The chain is an ordered list of `ChainLink`s (name + constructed provider).
`generate_spec_with_fallback` runs the standard `stream_with_repair` loop per
link, records every switch as a `FallbackEvent` (surfaced in the pipeline
warnings and the UI), and annotates the winning provider name on the outcome
so callers can mark the spec's source.

Chain *configuration* (order, per-provider enable/disable) lives in
`llm.registry` (`default_chain_config` / `normalize_chain_config` /
`chain_status`); this module is the runtime side. `probe_endpoint` is the
lightweight reachability check the config panel uses for its per-provider
status column — plain urllib, no SDK, short timeout.
"""
from __future__ import annotations

import logging
import socket
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

from .base import LLMProvider
from .repair import RepairResult, ValidatorFn, stream_with_repair

_log = logging.getLogger(__name__)

#: Default fallback order when nothing is configured: MiniMax M3 primary,
#: DeepSeek (V4 Pro class) as the automatic fallback.
DEFAULT_CHAIN_ORDER = ("minimax", "deepseek")


# ---------------------------------------------------------------------- model
@dataclass
class ChainLink:
    """One runnable chain member: a registry name plus its provider instance."""

    name: str
    provider: LLMProvider


@dataclass
class FallbackEvent:
    """A provider switch. `to_provider` is None when the chain ran out and the
    caller dropped to its own (deterministic) fallback."""

    from_provider: str
    to_provider: str | None
    reason: str


@dataclass
class ChainOutcome:
    """Result of `generate_spec_with_fallback`.

    Mirrors `RepairResult` (text / ok / repaired / attempts / errors) and adds
    the provenance the pipeline annotates onto the spec: which provider
    produced the final text and which switches happened along the way.
    """

    text: str
    provider_name: str
    repaired: bool
    attempts: int
    errors: list[str] = field(default_factory=list)
    fallbacks: list[FallbackEvent] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def fell_back(self) -> bool:
        return any(ev.to_provider for ev in self.fallbacks)


class ProviderChain:
    """Ordered, iterable container of chain links.

    Accepts `(name, provider)` pairs or `ChainLink`s; links with a None
    provider are dropped so a partially-constructible chain stays runnable.
    """

    def __init__(self, links: Iterable[ChainLink | tuple[str, LLMProvider]]) -> None:
        self.links: list[ChainLink] = []
        for link in links:
            if isinstance(link, ChainLink):
                if link.provider is not None:
                    self.links.append(link)
            else:
                name, provider = link
                if provider is not None:
                    self.links.append(ChainLink(name, provider))

    def __iter__(self) -> Iterator[ChainLink]:
        return iter(self.links)

    def __len__(self) -> int:
        return len(self.links)

    def __bool__(self) -> bool:
        return bool(self.links)

    @property
    def names(self) -> list[str]:
        return [l.name for l in self.links]


# ---------------------------------------------------------- failure taxonomy
def classify_failure(exc: BaseException) -> str:
    """Human-readable failure category for a provider exception.

    Drives both the log line and the FallbackEvent reason. Covers the SDK
    styles we see (openai's `status_code` attribute, requests/urllib
    errors) without importing any SDK: status codes first, then exception
    class names, then message keywords.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    cls = type(exc).__name__.lower()
    msg = str(exc).lower()

    if status in (401, 403) or "authenticationerror" in cls or "permissiondenied" in cls:
        return f"auth error (HTTP {status})" if status else "auth error"
    if status == 429 or "ratelimit" in cls.replace("_", ""):
        return "rate limit (HTTP 429)"
    if (
        status in (408, 504)
        or isinstance(exc, (TimeoutError, socket.timeout))
        or "timeout" in cls
        or "timed out" in msg
    ):
        return "timeout"
    if status is not None and 500 <= status < 600:
        return f"server error (HTTP {status})"
    if (
        isinstance(exc, (ConnectionError, urllib.error.URLError))
        or "connection" in cls
        or "getaddrinfo" in msg
        or "name or service not known" in msg
    ):
        return f"connection error ({exc})" if str(exc) else "connection error"
    # Message-level fallbacks for SDKs that only put the status in the text.
    if "401" in msg or "invalid api key" in msg or "unauthorized" in msg:
        return "auth error"
    if "429" in msg or "rate limit" in msg:
        return "rate limit"
    return f"{type(exc).__name__}: {exc}"


# ------------------------------------------------------------- fallback loop
def generate_spec_with_fallback(
    chain: Iterable[ChainLink],
    system: str,
    user: str,
    validator_fn: ValidatorFn,
    *,
    stop_event: Optional[threading.Event] = None,
    on_token: Optional[Callable[[str], None]] = None,
) -> ChainOutcome:
    """Run the repair-validated spec generation across the chain, in order.

    Per link this is exactly `stream_with_repair` (one validator-feedback
    repair round, never a third attempt). A link is abandoned for the next
    one when:

    - streaming raises (auth / timeout / rate limit / connection / other —
      the reason is classified for the log), or
    - the answer is still invalid after the repair round.

    Every switch is logged and recorded as a FallbackEvent. When the last
    link fails by *raising*, the last exception propagates (same contract as
    a single provider). When the last link merely stays invalid, an
    unsuccessful ChainOutcome is returned so the caller can take its
    deterministic fallback path — again matching single-provider semantics.
    """
    links = [l for l in chain if l.provider is not None]
    if not links:
        raise ValueError("provider chain is empty")

    fallbacks: list[FallbackEvent] = []

    for i, link in enumerate(links):
        if i > 0 and stop_event is not None and stop_event.is_set():
            _log.info("stop_event set; not starting fallback provider %s", link.name)
            break
        nxt = links[i + 1].name if i + 1 < len(links) else None
        try:
            outcome: RepairResult = stream_with_repair(
                link.provider, system, user, validator_fn,
                stop_event=stop_event, on_token=on_token,
            )
        except Exception as e:
            reason = classify_failure(e)
            fallbacks.append(FallbackEvent(link.name, nxt, reason))
            if nxt:
                _log.warning("provider %s failed (%s); falling back to %s",
                             link.name, reason, nxt)
                continue
            _log.error("provider %s failed (%s); chain exhausted", link.name, reason)
            raise

        if outcome.ok:
            if fallbacks:
                _log.warning("provider %s succeeded after %d fallback(s)",
                             link.name, len(fallbacks))
            return ChainOutcome(
                text=outcome.text, provider_name=link.name,
                repaired=outcome.repaired, attempts=outcome.attempts,
                errors=[], fallbacks=fallbacks,
            )

        reason = "invalid spec after self-repair: " + "; ".join(outcome.errors)
        fallbacks.append(FallbackEvent(link.name, nxt, reason))
        if nxt:
            _log.warning("provider %s spec invalid after repair; falling back to %s",
                         link.name, nxt)
            continue
        _log.warning("provider %s spec invalid after repair; chain exhausted", link.name)
        return ChainOutcome(
            text=outcome.text, provider_name=link.name,
            repaired=outcome.repaired, attempts=outcome.attempts,
            errors=list(outcome.errors), fallbacks=fallbacks,
        )

    # Only reachable when stop_event interrupted between links. Cancellation
    # wins over the recorded error: return an unsuccessful outcome so the
    # caller takes its deterministic fallback instead of surfacing an error
    # for a run the user deliberately stopped.
    return ChainOutcome(
        text="", provider_name=links[-1].name, repaired=False, attempts=0,
        errors=["cancelled before a fallback provider could run"],
        fallbacks=fallbacks,
    )


# ------------------------------------------------------------------ builder
def build_chain(
    chain_cfg: Iterable[dict],
    *,
    model_for: Callable[[str], str | None],
    endpoint_for: Callable[[str], str | None],
    key_for: Callable[[str], str | None],
    think_mode: bool = False,
    json_mode: bool = True,
) -> ProviderChain:
    """Construct a ProviderChain from a normalized chain config.

    `chain_cfg` is the output of `registry.normalize_chain_config` (ordered
    {name, enabled} dicts); disabled entries are skipped. Providers are
    constructed lazily via `registry.make_provider`, and a provider that
    fails to construct (missing SDK, …) is logged and skipped so one broken
    install never blocks the rest of the chain.
    """
    from .registry import make_provider

    links: list[ChainLink] = []
    for entry in chain_cfg:
        if not entry.get("enabled", True):
            continue
        name = str(entry.get("name", "")).lower()
        if not name:
            continue
        try:
            provider = make_provider(
                name,
                model=model_for(name) or "",
                endpoint=endpoint_for(name),
                api_key=key_for(name),
                think_mode=think_mode,
                json_mode=json_mode,
            )
        except Exception as e:
            _log.warning("skipping chain provider %s: %s: %s", name, type(e).__name__, e)
            continue
        links.append(ChainLink(name, provider))
    return ProviderChain(links)


def chain_from_settings(settings, *, think_mode: bool = False, json_mode: bool = True) -> ProviderChain:
    """Build the runnable chain straight from a core.settings.Settings object.

    Reads `llm.chain` (order/enabled), `llm.models` / `llm.endpoints`
    overrides, and resolves keys through `core.secrets.get_api_key`. This is
    the integration point for callers that own a Settings instance.
    """
    from ..core.secrets import get_api_key
    from .registry import default_endpoint, known_models_fallback, normalize_chain_config

    cfg = normalize_chain_config(settings.get("llm", "chain", default=None))
    models = settings.get("llm", "models", default={}) or {}
    endpoints = settings.get("llm", "endpoints", default={}) or {}
    return build_chain(
        cfg,
        model_for=lambda n: models.get(n) or known_models_fallback(n),
        endpoint_for=lambda n: endpoints.get(n) or default_endpoint(n) or None,
        key_for=get_api_key,
        think_mode=think_mode,
        json_mode=json_mode,
    )


# ------------------------------------------------------------------ probing
def probe_endpoint(
    name: str,
    endpoint: str | None,
    api_key: str | None = None,
    *,
    timeout: float = 8.0,
) -> tuple[bool, str]:
    """Lightweight reachability probe for a provider's `/models` listing.

    Plain urllib (no SDK), short timeout — meant for the config panel's
    per-provider status column. A 401/403 means the host is reachable but
    the key was rejected; a 404 means reachable without a models listing.
    """
    if not endpoint:
        return True, "SDK-native endpoint (nothing to probe)"
    url = endpoint.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, f"reachable (HTTP {resp.status})"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, f"reachable but key rejected (HTTP {e.code})"
        if e.code == 404:
            return True, "reachable (no /models listing)"
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
