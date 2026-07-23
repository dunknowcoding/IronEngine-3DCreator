"""Stream-safe <think>…</think> handling.

Many local models (DeepSeek-R1, Qwen3 reasoning, gemma3-thinking, …) emit a
chain-of-thought block wrapped in a tag before the actual answer. The block can
arrive split across chunk boundaries — we cannot just `text.split("<think>")`
on each chunk because the opening or closing tag may straddle two chunks.

`ThinkingFilter` consumes streaming chunks and emits `(segment, is_thinking)`
pieces. The UI uses this to style thinking content differently; the JSON
parser uses `strip(text)` to drop thinking before parsing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Tag aliases we accept. First-of-tuple is the canonical form.
_OPEN_TAGS = ("<think>", "<thinking>", "<|thinking|>", "<reasoning>")
_CLOSE_TAGS = ("</think>", "</thinking>", "<|/thinking|>", "</reasoning>")


@dataclass
class Segment:
    text: str
    is_thinking: bool


class ThinkingFilter:
    """Stateful chunk-stream parser.

    feed(chunk) → list[Segment] (may be empty if the chunk only contained part
    of a tag). flush() returns any buffered tail.
    """

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def feed(self, chunk: str) -> list[Segment]:
        if not chunk:
            return []
        self._buf += chunk
        out: list[Segment] = []
        while self._buf:
            tags = _CLOSE_TAGS if self._in_think else _OPEN_TAGS
            # Find the earliest tag in the buffer.
            best_idx, best_tag = -1, ""
            for tag in tags:
                idx = self._buf.find(tag)
                if idx != -1 and (best_idx == -1 or idx < best_idx):
                    best_idx, best_tag = idx, tag
            if best_idx == -1:
                # No complete tag in buffer. Only emit what we're sure isn't
                # the prefix of a tag — keep up to (max_tag_len - 1) tail bytes.
                keep = max(len(t) for t in tags) - 1
                if len(self._buf) > keep:
                    head = self._buf[:-keep]
                    out.append(Segment(head, self._in_think))
                    self._buf = self._buf[-keep:]
                break
            # Emit text before the tag, swap state, drop the tag.
            if best_idx > 0:
                out.append(Segment(self._buf[:best_idx], self._in_think))
            self._buf = self._buf[best_idx + len(best_tag):]
            self._in_think = not self._in_think
        return out

    def flush(self) -> list[Segment]:
        if not self._buf:
            return []
        out = [Segment(self._buf, self._in_think)]
        self._buf = ""
        return out


_PAIRED_RE = re.compile(
    r"(?:" + "|".join(re.escape(t) for t in _OPEN_TAGS) + r")"
    r".*?"
    r"(?:" + "|".join(re.escape(t) for t in _CLOSE_TAGS) + r")",
    flags=re.DOTALL,
)


def _first_index(text: str, tags: tuple[str, ...]) -> int:
    """Earliest occurrence of any tag, or -1."""
    best = -1
    for tag in tags:
        idx = text.find(tag)
        if idx != -1 and (best == -1 or idx < best):
            best = idx
    return best


def _balanced_object_end(text: str, start: int) -> int:
    """Index just past the balanced {...} object starting at `start`, or -1."""
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
                return i + 1
    return -1


def _remove_tag_tokens(text: str) -> str:
    """Remove stray tag tokens plus whitespace glued to them.

    Models that misplace a tag inside their JSON answer usually pad it with
    newlines (``"chair</think>\n\n","n_points"``); leaving the newlines
    behind would produce invalid JSON (raw control chars inside a string),
    so the glued whitespace goes with the tag.
    """
    for tag in _OPEN_TAGS + _CLOSE_TAGS:
        text = re.sub(r"\s*" + re.escape(tag) + r"\s*", "", text)
    return text


def strip(text: str) -> str:
    """One-shot removal of all <think>…</think> blocks (any alias).

    Also handles two real-world failure modes the naive paired-block regex
    cannot:

    - **Misplaced close tag** (observed from MiniMax-M3 via api.minimax.io):
      the model starts its JSON answer and *then* emits ``</think>`` a few
      tokens later, e.g. ``…reasoning {"shape":"chair</think>","n_points":…``.
      Removing open→close would behead the JSON. When the first ``{`` opens
      an object that is *still unbalanced* when the close tag arrives, the
      tag is embedded in the answer: we drop only the reasoning prose before
      that ``{`` and remove the stray tag tokens, leaving the answer intact.
      (A brace group that closes *before* the close tag is treated as an
      example inside the reasoning and removed with it — normal path.)
    - **Stray/unmatched tags**: any remaining open/close tag tokens are
      removed after the paired-block pass so a truncated stream can never
      leak a tag into the parser.
    """
    i_open = _first_index(text, _OPEN_TAGS)
    i_close = _first_index(text, _CLOSE_TAGS)
    if i_open == -1 and i_close == -1:
        return text

    i_brace = text.find("{")
    if i_open != -1 and i_brace != -1 and i_open < i_brace:
        obj_end = _balanced_object_end(text, i_brace)
        # The brace group is the answer when it is unbalanced (truncated
        # stream), when there is no close tag at all, or when it is still
        # open when the close tag arrives (tag embedded in the answer).
        answer_overruns_close = (
            obj_end == -1 or i_close == -1 or obj_end > i_close
        )
        if answer_overruns_close:
            # Misplaced/absent close tag: reasoning prose precedes the first
            # '{' which begins the answer. Keep from the brace on, minus tags.
            text = text[i_brace:]
            return _remove_tag_tokens(text)

    text = _PAIRED_RE.sub("", text)
    # Stray tags (unmatched open, or close tags left over inside the answer).
    return _remove_tag_tokens(text)
