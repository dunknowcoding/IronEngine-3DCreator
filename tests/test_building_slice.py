"""Slice-view tests: the '3D-printing slice' schematic per floor — SVG/PNG
render data must contain every room label, wall distribution, door swings,
furniture footprints and stair runs."""
from __future__ import annotations

from pathlib import Path

import pytest

from ironengine_3d_creator.generation import building_arch as ba


@pytest.fixture(scope="module")
def built():
    return ba.build_building({"seed": 7, "floors": 2, "style": "neoclassical"})


def test_slice_contains_every_room_label_svg(built, tmp_path):
    svg_path = ba.slice_to_svg(built["slices"], tmp_path / "slice_sheet.svg")
    svg = Path(svg_path).read_text(encoding="utf-8")
    for floor in built["slices"].floors:
        assert f"floor {floor['floor']}" in svg
        for room in floor["rooms"]:
            assert f">{room['name']}<" in svg, \
                f"floor {floor['floor']}: room label {room['name']} missing from SVG"
            assert f"{room['area']:.1f} m²" in svg


def test_slice_render_data_complete(built):
    for floor in built["slices"].floors:
        assert floor["wall_segments"], "no wall distribution in slice"
        assert floor["doors"], "no door swings in slice"
        assert floor["rooms"], "no rooms in slice"
        assert floor["windows"], "no windows in slice"
        # door swings carry the arc metadata needed to draw the sweep
        hinged = [d for d in floor["doors"] if "hinge" in d]
        assert hinged, "no hinged door swing arcs in slice"
        for d in hinged:
            assert d["radius"] > 0.3 and d["rom_deg"] == 110.0
    # furniture markers on at least one floor
    assert any(fl["furniture"] for fl in built["slices"].floors)
    # stair run on floor 0 of a 2-storey build
    assert built["slices"].floors[0]["stairs"], "no staircase in floor-0 slice"
    st = built["slices"].floors[0]["stairs"][0]
    assert 0.15 <= st["rise"] <= 0.185 and 0.25 <= st["going"] <= 0.32


def test_slice_wall_segments_cover_walls(built):
    """Per floor, sliced wall segments + opening gaps reconstruct each wall's
    full length (no wall material lost, no overlaps)."""
    plan = built["built"].plan
    for floor in built["slices"].floors:
        f = floor["floor"]
        cut = floor["cut_height"] - plan.floor_base(f)
        for wall in plan.walls:
            if wall.floor != f:
                continue
            segs = sorted((s for s in floor["wall_segments"] if s["wall"] == wall.label),
                          key=lambda s: s["u"][0])
            gaps = sorted((o for o in wall.openings
                           if o.sill - 1e-6 < cut < o.sill + o.height + 1e-6),
                          key=lambda o: o.offset)
            expected = wall.length - sum(o.width for o in gaps)
            total = sum(s["u"][1] - s["u"][0] for s in segs)
            assert abs(total - expected) < 1e-6, \
                f"{wall.label}: segments {total} + gaps != length {wall.length}"
            # segments don't overlap
            for a, b in zip(segs, segs[1:]):
                assert a["u"][1] <= b["u"][0] + 1e-9


def test_slice_png_written(built, tmp_path):
    png = ba.slice_to_png(built["slices"], tmp_path / "slice_sheet.png")
    p = Path(png)
    assert p.exists() and p.stat().st_size > 10_000


def test_slice_respects_custom_heights(built):
    slices = ba.slice_building(built["built"], heights=[1.5, 1.1])
    assert slices.floors[0]["cut_height"] == pytest.approx(1.5)
    assert slices.floors[1]["cut_height"] == pytest.approx(built["built"].plan.floor_height + 1.1)


def test_slice_to_dict_jsonable(built):
    import json
    json.dumps(built["slices"].to_dict())  # must not raise
