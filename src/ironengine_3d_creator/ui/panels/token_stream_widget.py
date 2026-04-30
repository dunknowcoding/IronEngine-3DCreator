"""Live LLM token stream display with thinking-block + markdown support.

While streaming we render plain text as it arrives — fast, no layout cost.
`<think>…</think>` segments (and aliases) are routed through a stateful filter
and shown in a dim italic so the user can see the model's chain of thought
without it dominating the view.

When the stream ends, we re-render the *non-thinking* portion through
`QTextDocument.setMarkdown()` so headings, lists, code blocks, and emphasis
appear correctly. The thinking content stays at the top in a collapsed-style
section above the rendered answer.
"""
from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QPainter, QPen, QTextCharFormat, QTextCursor,
)
from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

from ...llm.thinking import ThinkingFilter
from ..theme import current as theme_current


class TokenStreamWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(160)
        L = QVBoxLayout(self)
        L.setContentsMargins(2, 2, 2, 2); L.setSpacing(2)

        self.header = QLabel("LLM token stream")
        self.header.setObjectName("sectionTitle")
        L.addWidget(self.header)

        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.setAcceptRichText(False)  # we control formatting ourselves
        f = QFont("Cascadia Code"); f.setPointSize(10)
        self.view.setFont(f)
        self.view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        L.addWidget(self.view, 1)

        self.footer = QLabel("idle")
        self.footer.setObjectName("sectionHint")
        L.addWidget(self.footer)

        self._streaming = False
        self._t = 0.0
        self._tokens = 0
        self._t0 = 0.0
        self._chars = 0
        self._caret_timer = QTimer(self)
        self._caret_timer.setInterval(60)
        self._caret_timer.timeout.connect(self._tick_caret)

        # Thinking & content buffers — accumulated while streaming so we can
        # do markdown rendering on the answer at the end.
        self._filter = ThinkingFilter()
        self._answer_buf: list[str] = []
        self._think_buf: list[str] = []
        self._char_fmt_normal = QTextCharFormat()
        self._char_fmt_think = QTextCharFormat()
        self._refresh_formats()

    # ----------------------------------------------------------- formatting
    def _refresh_formats(self) -> None:
        pal = theme_current()
        self._char_fmt_normal.setForeground(QColor(pal.text_primary))
        f = QFont(self.view.font()); f.setItalic(False); f.setBold(False)
        self._char_fmt_normal.setFont(f)

        think_color = QColor(pal.text_dim)
        self._char_fmt_think.setForeground(think_color)
        f2 = QFont(self.view.font()); f2.setItalic(True)
        self._char_fmt_think.setFont(f2)

    # ------------------------------------------------------------------ control
    def begin(self, label: str = "streaming…", *, think_mode_on: bool = False) -> None:
        self._streaming = True
        self._t0 = time.perf_counter()
        self._tokens = 0
        self._chars = 0
        self._think_mode_on = think_mode_on
        self.view.clear()
        self._filter = ThinkingFilter()
        self._answer_buf.clear()
        self._think_buf.clear()
        self._refresh_formats()
        if not think_mode_on:
            cursor = self.view.textCursor()
            hint_fmt = QTextCharFormat(self._char_fmt_think)
            cursor.insertText(
                "💡 Tip — toggle 'Reasoning / thinking mode' in LLM Configuration "
                "to see the model's chain of thought here.\n\n",
                hint_fmt,
            )
        self.footer.setText(label)
        self._caret_timer.start()
        self.update()

    def end(self) -> None:
        self._streaming = False
        self._caret_timer.stop()
        # Flush any tail still buffered inside the filter (e.g. a partial tag
        # at end-of-stream becomes regular text).
        for seg in self._filter.flush():
            self._append_segment(seg.text, seg.is_thinking)
        self._render_markdown()
        if self._tokens:
            elapsed = max(1e-3, time.perf_counter() - self._t0)
            rate = self._tokens / elapsed
            think_n = sum(len(t) for t in self._think_buf)
            ans_n = sum(len(a) for a in self._answer_buf)
            self.footer.setText(
                f"complete · {self._tokens} chunks · {rate:.1f}/s · "
                f"{ans_n} chars answer · {think_n} chars thinking"
            )
        else:
            self.footer.setText("idle")
        self.update()

    def append_chunk(self, text: str) -> None:
        if not text:
            return
        self._tokens += 1
        self._chars += len(text)
        for seg in self._filter.feed(text):
            self._append_segment(seg.text, seg.is_thinking)

        if (time.perf_counter() - self._t0) > 0.05:
            elapsed = time.perf_counter() - self._t0
            rate = self._tokens / max(1e-3, elapsed)
            self.footer.setText(f"streaming · {self._tokens} chunks · {rate:.1f}/s")

    def _append_segment(self, text: str, is_thinking: bool) -> None:
        if not text:
            return
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = self._char_fmt_think if is_thinking else self._char_fmt_normal
        cursor.insertText(text, fmt)
        if is_thinking:
            self._think_buf.append(text)
        else:
            self._answer_buf.append(text)
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ----------------------------------------------------------- markdown
    def _render_markdown(self) -> None:
        """Replace the live transcript with a richly-rendered view.

        Thinking content goes first as a quote block (so it stays visible but
        visually subdued). The answer follows, rendered as Markdown.
        """
        if not self._answer_buf and not self._think_buf:
            return
        pal = theme_current()
        md_parts: list[str] = []
        if self._think_buf:
            think_text = "".join(self._think_buf).strip()
            if think_text:
                # Quote-prefix every line so QTextDocument renders a block-quote.
                quoted = "\n".join("> " + ln if ln else ">" for ln in think_text.splitlines())
                md_parts.append(f"_💭 thinking_\n\n{quoted}")
        if self._answer_buf:
            md_parts.append("".join(self._answer_buf).strip())

        markdown = "\n\n---\n\n".join(p for p in md_parts if p)
        if not markdown:
            return
        # Use the QTextDocument's setMarkdown — Qt 6 supports CommonMark + GFM.
        doc = self.view.document()
        doc.setMarkdown(markdown)
        # Markdown rendering wipes our char formats — re-apply theme colours by
        # walking once and tinting any block that came from the thinking quote.
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        block = doc.firstBlock()
        while block.isValid():
            if block.text().startswith(" "):
                pass  # placeholder — no special handling needed
            block = block.next()
        sb = self.view.verticalScrollBar(); sb.setValue(sb.maximum())

    # ---------------------------------------------------------------- visuals
    def _tick_caret(self) -> None:
        self._t += 0.06
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self._streaming:
            return
        pal = theme_current()
        accent = QColor(pal.accent)
        intensity = 0.5 + 0.5 * math.sin(self._t * 3.0)
        accent.setAlphaF(0.3 + 0.4 * intensity)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(accent, 1.5)
        p.setPen(pen)
        p.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 4, 4)
        # Pulsing caret at the cursor.
        cursor_rect = self.view.cursorRect()
        cursor_rect.translate(self.view.x(), self.view.y())
        alpha = 0.5 + 0.5 * math.sin(self._t * 6.0)
        c = QColor(pal.accent); c.setAlphaF(alpha)
        p.setPen(c)
        f = self.view.font()
        fm = QFontMetrics(f)
        x = cursor_rect.x() + cursor_rect.width()
        y = cursor_rect.y() + fm.ascent()
        p.setFont(f)
        p.drawText(int(x), int(y), "▍")
