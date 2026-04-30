"""Per-primitive point budget allocation."""
from __future__ import annotations

import numpy as np

from ..alignment.schema import Primitive
from .primitives import primitive_area


def allocate_budget(primitives: list[Primitive], total: int) -> list[int]:
    """Distribute `total` points across primitives, weighted by world-space area.

    World-space area approximation: scale local area by max(|det(rotation*scale)|^(2/3))
    so that a 2x scale gives ~4x area.
    """
    if not primitives:
        return []
    weights = []
    for p in primitives:
        T = p.transform_matrix()[:3, :3]
        det = abs(np.linalg.det(T))
        scale_factor = det ** (2.0 / 3.0) if det > 0 else 1.0
        weights.append(max(1e-6, primitive_area(p.kind, p.params) * scale_factor))
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    counts = np.floor(weights * total).astype(int).tolist()
    # Distribute the remainder to the largest primitives.
    remainder = total - sum(counts)
    if remainder > 0:
        order = np.argsort(weights)[::-1]
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
    return counts
