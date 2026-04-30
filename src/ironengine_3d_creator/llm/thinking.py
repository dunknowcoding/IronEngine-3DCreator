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


def strip(text: str) -> str:
    """One-shot removal of all <think>…</think> blocks (any alias)."""
    pattern = re.compile(
        r"(?:" + "|".join(re.escape(t) for t in _OPEN_TAGS) + r")"
        r".*?"
        r"(?:" + "|".join(re.escape(t) for t in _CLOSE_TAGS) + r")",
        flags=re.DOTALL,
    )
    return pattern.sub("", text)
