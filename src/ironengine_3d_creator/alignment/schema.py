"""GenerationSpec — the canonical structure consumed by generation/.

The LLM emits JSON in this shape; the alignment layer validates and normalizes
it; the procedural generator turns it into points. This dataclass is also the
authoritative list of legal `Primitive.kind` and `Feature.kind` values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

PRIMITIVE_KINDS = (
    "box",
    "sphere",
    "cylinder",
    "capsule",
    "cone",
    "torus",
    "ellipsoid",
    "prism",
    "helix",
    "plane",
)

FEATURE_KINDS = (
    "scratch",
    "curve_pattern",
    "bump_field",
    "dent",
    "erosion",
    "ridges",
    "holes",
    "fur",
)

SHAPE_KINDS = (
    "chair",
    "table",
    "vase",
    "lamp",
    "creature",
    "tree",
    "rock",
    "vehicle",
    "abstract",
)


@dataclass
class Primitive:
    kind: str
    transform: list[list[float]]   # 4x4 row-major; identity by default
    params: dict[str, Any]
    label: str | None = None

    @staticmethod
    def identity_transform() -> list[list[float]]:
        return np.eye(4, dtype=np.float32).tolist()

    def transform_matrix(self) -> np.ndarray:
        return np.asarray(self.transform, dtype=np.float32)


@dataclass
class Feature:
    kind: str
    region: Any = "all"            # "all" | label string | dict region descriptor
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationSpec:
    shape: str = "abstract"
    n_points: int = 50_000
    bbox_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    primitives: list[Primitive] = field(default_factory=list)
    features: list[Feature] = field(default_factory=list)
    color: tuple[float, float, float] | None = None
    seed: int = 0

    # ------------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "n_points": int(self.n_points),
            "bbox_size": list(self.bbox_size),
            "primitives": [asdict(p) for p in self.primitives],
            "features": [asdict(f) for f in self.features],
            "color": list(self.color) if self.color else None,
            "seed": int(self.seed),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "GenerationSpec":
        prims = [
            Primitive(
                kind=p["kind"],
                transform=p.get("transform") or Primitive.identity_transform(),
                params=p.get("params") or {},
                label=p.get("label"),
            )
            for p in data.get("primitives", [])
        ]
        feats = [
            Feature(
                kind=f["kind"],
                region=f.get("region", "all"),
                params=f.get("params") or {},
            )
            for f in data.get("features", [])
        ]
        bbox = tuple(data.get("bbox_size", (1.0, 1.0, 1.0)))
        color = data.get("color")
        return cls(
            shape=data.get("shape", "abstract"),
            n_points=int(data.get("n_points", 50_000)),
            bbox_size=(float(bbox[0]), float(bbox[1]), float(bbox[2])),
            primitives=prims,
            features=feats,
            color=tuple(color) if color else None,
            seed=int(data.get("seed", 0)),
        )
