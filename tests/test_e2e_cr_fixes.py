"""Regression tests for the E2E CR fixes (B4 + B1-CR).

B4: the LLM spec-mode system prompt must document params for *every*
primitive kind in the schema (including superellipsoid / tube / sweep /
arch / panel), and every few-shot example embedded in the prompt must
survive the real spec validator.

B1-CR: cloth manifests must carry world dimensions in extras so the Sim
side does not have to guess them: ``extras["cloth"] = {"width_m", "height_m",
"resolution"}``.
"""
from __future__ import annotations

import json
import re

import pytest

from ironengine_3d_creator.alignment.parser import _find_json_object, parse_spec
from ironengine_3d_creator.alignment.schema import PRIMITIVE_KINDS
from ironengine_3d_creator.alignment.validator import normalize
from ironengine_3d_creator.core.manifest import write_manifest
from ironengine_3d_creator.generation import soft_author as sa
from ironengine_3d_creator.llm.prompts import SPEC_SYSTEM_PROMPT


# ------------------------------------------------------------- B4: prompt docs

def test_prompt_documents_params_for_every_primitive_kind():
    """Every kind in the schema appears with a param-doc line in the prompt."""
    for kind in PRIMITIVE_KINDS:
        assert re.search(rf"^- {kind}: ", SPEC_SYSTEM_PROMPT, re.MULTILINE), \
            f"SPEC_SYSTEM_PROMPT lacks param docs for {kind!r}"


def _fewshot_outputs() -> list[str]:
    """Raw JSON of every few-shot 'Output:' block in the prompt."""
    return [
        _find_json_object(chunk)
        for chunk in SPEC_SYSTEM_PROMPT.split("Output:")[1:]
    ]


def test_prompt_fewshot_examples_validate_against_validator():
    """Each few-shot Output parses and normalizes without dropping a part."""
    outputs = _fewshot_outputs()
    assert len(outputs) >= 5  # stool / footbridge / mug / curved seat / gate
    for blob in outputs:
        spec = parse_spec(blob)
        clean, warnings = normalize(spec)
        dropped = [w for w in warnings if "unknown primitive kind" in w]
        assert not dropped, f"few-shot dropped parts: {dropped}"
        assert len(clean.primitives) == len(spec.primitives)
        assert clean.n_points >= 100


def test_prompt_fewshot_uses_tube_handle_and_arch():
    """The complex-geometry few-shot demonstrates tube + arch (not torus hacks)."""
    seen: set[str] = set()
    for blob in _fewshot_outputs():
        for prim in parse_spec(blob).primitives:
            seen.add(prim.kind)
    assert "tube" in seen
    assert "arch" in seen

    gate = [b for b in _fewshot_outputs() if '"garden gate' in b
            or ('"tube"' in b and '"arch"' in b)]
    assert gate, "no few-shot spec combines a tube handle and an arch"
    clean, warnings = normalize(parse_spec(gate[0]))
    assert not [w for w in warnings if "malformed" in w]
    kinds = {p.kind for p in clean.primitives}
    assert {"tube", "arch"} <= kinds
    # The arch kept its semicircle; the tube kept its bent path.
    arch = next(p for p in clean.primitives if p.kind == "arch")
    assert arch.params["minor_radius"] < arch.params["major_radius"]
    tube = next(p for p in clean.primitives if p.kind == "tube")
    assert len(tube.params["path"]) >= 2


# ------------------------------------------------------------- B1-CR: cloth extras

def test_cloth_manifest_extras_round_trip(tmp_path):
    """cloth block survives build_manifest → write → read with exact keys."""
    result = sa.author_cloth(
        material="cotton", width=0.5, depth=0.35, resolution=(20, 12), seed=3)

    cloth = result.extras["cloth"]
    assert cloth["width_m"] == pytest.approx(0.5)
    assert cloth["height_m"] == pytest.approx(0.35)
    assert cloth["resolution"] == [20, 12]
    assert isinstance(cloth["width_m"], float)
    assert isinstance(cloth["height_m"], float)

    manifest = result.build_manifest(mesh_path="towel.glb")
    assert manifest["cloth"] == cloth

    out = tmp_path / "towel.iemodel.json"
    write_manifest(out, manifest)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["cloth"]["width_m"] == pytest.approx(0.5)
    assert loaded["cloth"]["height_m"] == pytest.approx(0.35)
    assert loaded["cloth"]["resolution"] == [20, 12]
    # Dimensions are consistent with the soft-body grid and measured mass.
    assert loaded["soft_body"]["resolution"] == loaded["cloth"]["resolution"]
    area = loaded["cloth"]["width_m"] * loaded["cloth"]["height_m"]
    density = sa.CLOTH_FABRICS["cotton"]["area_density_kg_m2"]
    assert loaded["physics"]["mass_kg"] == pytest.approx(density * area)


def test_cloth_extras_match_default_towel_dimensions():
    result = sa.author_cloth(resolution=(8, 6), seed=1)
    assert result.extras["cloth"] == {
        "width_m": 0.6, "height_m": 0.4, "resolution": [8, 6],
    }
