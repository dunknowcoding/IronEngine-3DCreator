"""Save / load .iecreator.json sessions.

A session captures everything needed to deterministically replay a generation:
the user's requirements, the resolved spec, the seed, plus any post-generation
edit operations in order.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "iecreator/1"


@dataclass
class SessionRequirements:
    prompt: str = ""
    shape: str = "abstract"
    n_points: int = 50_000
    bbox: tuple[float, float, float] = (1.0, 1.0, 1.0)
    legs: int = 0
    details: str = ""


@dataclass
class Session:
    version: str = "0.1.0"
    requirements: SessionRequirements = field(default_factory=SessionRequirements)
    spec: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    edit_history: list[dict[str, Any]] = field(default_factory=list)
    export: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "version": self.version,
            "requirements": asdict(self.requirements),
            "spec": self.spec,
            "seed": self.seed,
            "edit_history": self.edit_history,
            "export": self.export,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "Session":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported session schema: {data.get('schema')!r}")
        req = SessionRequirements(**data.get("requirements", {}))
        return cls(
            version=data.get("version", "0.1.0"),
            requirements=req,
            spec=data.get("spec", {}),
            seed=data.get("seed", 0),
            edit_history=data.get("edit_history", []),
            export=data.get("export", {}),
        )
