"""Undo/redo via shallow numpy snapshots.

We snapshot positions+colors before every op. For 100k points each snapshot is
~2.4 MB — we cap at 32 entries to keep memory bounded around 80 MB.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

MAX_STEPS = 32


@dataclass
class Snapshot:
    positions: np.ndarray
    colors: np.ndarray
    op_name: str


class History:
    def __init__(self, max_steps: int = MAX_STEPS) -> None:
        self._undo: deque[Snapshot] = deque(maxlen=max_steps)
        self._redo: deque[Snapshot] = deque(maxlen=max_steps)

    def push(self, positions: np.ndarray, colors: np.ndarray, op_name: str) -> None:
        self._undo.append(Snapshot(positions.copy(), colors.copy(), op_name))
        self._redo.clear()

    def can_undo(self) -> bool:
        return len(self._undo) > 0

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def undo(self, current_positions: np.ndarray, current_colors: np.ndarray) -> Snapshot | None:
        if not self._undo:
            return None
        snap = self._undo.pop()
        # Save current state for redo.
        self._redo.append(Snapshot(current_positions.copy(), current_colors.copy(), snap.op_name))
        return snap

    def redo(self, current_positions: np.ndarray, current_colors: np.ndarray) -> Snapshot | None:
        if not self._redo:
            return None
        snap = self._redo.pop()
        self._undo.append(Snapshot(current_positions.copy(), current_colors.copy(), snap.op_name))
        return snap

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
