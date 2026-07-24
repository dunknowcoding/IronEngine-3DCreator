# TEXTURES — tileable procedural texture maps (CR_Textures)

Real-world surface richness **without geometric cost**: surface micro-detail
(wood grain, weave threads, carved scrollwork, panel grooves) is baked into
tileable PNG maps instead of being modelled, so heavy-looking objects stay
computationally light.

Owner modules:

| File | Role |
| --- | --- |
| `generation/texture_maps.py` | Seeded, tileable map generators (albedo / bump / roughness / alpha) |
| `generation/texture_apply.py` | UV sampling onto analytic meshes, `ietexture/1` manifest block, bake doctrine |
| `generation/textures.py` | Pre-existing per-point cloud texturing + the material→map-kind bridge |
| `tools/demo_textures.py` | Gallery + textured-GLB demo |
| `tests/test_texture_maps.py` | Contract / tileability / determinism / performance tests |

---

## 1. Generating maps

```python
from ironengine_3d_creator.generation import texture_maps as tm

tm.list_texture_kinds()
# ['brick', 'brushed_metal', 'concrete', 'denim', 'granite', 'grass',
#  'leather', 'linen', 'marble', 'rococo_ornament', 'rust', 'scifi_panel',
#  'stone', 'wood_oak', 'wood_walnut']

maps = tm.generate_maps("wood_oak", size=512, seed=7)
# {"albedo": (512,512,3) uint8, "bump": (512,512) uint8}
paths = tm.save_maps(maps, "out/textures", kind="wood_oak", size=512, seed=7)
```

**Contract**

- `size` — integer in `[64, 1024]` (contract range; 256–1024 recommended).
- `seed` — any int; `(kind, size, seed)` is fully deterministic
  (byte-identical arrays across runs and machines).
- Channels: `albedo` (H,W,3) uint8 sRGB, always present; `bump` (H,W) uint8
  height (128 ≈ neutral); `roughness` (H,W) uint8 (0 = mirror, 255 = matte);
  `alpha` (H,W) uint8 coverage, only on `rococo_ornament` (saved packed as an
  RGBA PNG by `save_maps`).
- **Tileable** — every map wraps seamlessly (`wrap: "repeat"`). All noise is
  periodic value noise on integer-frequency lattices; all patterns use
  integer repeats per tile (brick courses, weave threads, panel cells).
- **Fast** — every kind renders a full map set in < 200 ms at 512²
  (measured ≤ ~80 ms warm on a desktop CPU; enforced by tests).

Material hints from the LLM/spec map onto kinds through the bridge in
`generation/textures.py`:

```python
from ironengine_3d_creator.generation import textures
textures.map_kind_for_material("wood")                    # "wood_oak"
textures.maps_for_material("metal", size=512, seed=7)     # channel dict
```

---

## 2. Applying maps to analytic meshes (UV + export wiring)

Analytic parts (`generation.analytic_mesh.AnalyticPart`) already carry
per-primitive UVs (box-projection, cylindrical, spherical/parametric —
conventions unchanged). `texture_apply` samples the maps at those UVs:

```python
from ironengine_3d_creator.generation.analytic_mesh import build_spec_meshes
from ironengine_3d_creator.generation import texture_apply as ta

parts = build_spec_meshes(spec)
colors, generated = ta.apply_maps_to_parts(
    parts,
    {"tabletop": "wood_oak", "*": "brushed_metal"},  # label -> kind, "*" fallback
    size=512, seed=7,
    uv_scale={"tabletop": (2.0, 1.0)},               # tileable repeats per part
)
```

`apply_maps_to_part` also folds the **bump channel into the albedo** as cheap
self-shadowing (`bump_strength`), so micro-relief survives at export vertex
density without a single added triangle.

### Export path (no changes to `core/exporter.py`)

The stock GLB exporter **already supports image textures**:
`core.exporter._write_glb_scene` builds a glTF `baseColorTexture` per part by
baking per-vertex colours onto the part's UV grid, alongside a PBR material
and `COLOR_0`. The wiring is therefore data, not code:

```python
from ironengine_3d_creator.core import exporter
import numpy as np

positions = np.concatenate([p.vertices for p in parts], axis=0)
colors_v  = np.concatenate(colors, axis=0)   # sampled above, same order
exporter.write_glb("model.glb", positions, colors_v, spec=spec, texture_size=512)
```

Passing the concatenated part vertices as `positions` (same count and order
as the parts) takes the exporter's 1:1 colour path, so each part's baked
`baseColorTexture` *is* the sampled texture map. Untouched parts may keep
neutral grey (`apply_maps_to_parts` does this automatically for unassigned
labels).

Because the exporter re-bakes from vertices, the original full-resolution
maps are emitted **alongside** the export and referenced from the manifest
`textures` block (next section) so downstream tools can re-bind them at full
fidelity (bump/roughness/alpha are not representable in the baked albedo).

---

## 3. The `textures` manifest block (`ietexture/1`)

`core.manifest.build_manifest` is unchanged; producers attach the block
before writing (see `tools/demo_textures.py`):

```python
from ironengine_3d_creator.core.manifest import build_manifest, write_manifest

manifest = build_manifest(spec, positions)
manifest["textures"] = ta.textures_manifest_block(
    [
        {"part": "tabletop", "material": "wood", "kind": "wood_oak",
         "channels": ["albedo", "bump"],
         "uv": {"wrap": "repeat", "scale": [2, 1]}},
        {"part": "leg", "material": "metal", "kind": "brushed_metal",
         "channels": ["albedo", "roughness"]},
    ],
    size=512, seed=7, maps_dir="textures",
)
write_manifest("model.iemodel.json", manifest)
```

Schema (versioned `ietexture/1`; consumers must ignore unknown fields):

```jsonc
"textures": {
  "schema": "ietexture/1",
  "maps": {                       // texture map registry, keyed by map id
    "wood_oak_albedo": {
      "file": "textures/wood_oak_albedo_512px_s7.png",  // relative to the manifest
      "kind": "wood_oak",
      "channel": "albedo",        // albedo | bump | roughness | alpha | rgba | normal
      "size": 512,
      "seed": 7,
      "tileable": true,           // safe to sample with wrap = "repeat"
      "format": "png"
    }
  },
  "assignments": [                // texture -> part -> channel mapping
    {
      "part": "tabletop",         // AnalyticPart label / GLB node name
      "material": "wood",         // spec material hint (informational)
      "maps": {"albedo": "wood_oak_albedo", "bump": "wood_oak_bump"},
      "uv": {"wrap": "repeat", "scale": [2, 1]}
    }
  ]
}
```

Rules:

- Every map id referenced by an assignment must exist in `maps`
  (`validate_textures_block` enforces this and the rest of the contract).
- Files are PNG, paths relative to the manifest, canonical name
  `<kind>_<channel>_<size>px_s<seed>.png` (`texture_maps.map_filename`).
- `uv.scale` repeats the tileable map across the part; `uv.wrap` is always
  `"repeat"` for generated maps.
- GLB node names equal part labels, so `part` binds unambiguously to the
  exported scene graph.

---

## 4. Bake doctrine — when a texture beats geometry

`bake_detail_to_texture` converts high-frequency procedural detail into maps
instead of triangles:

```python
from ironengine_3d_creator.generation import texture_apply as ta

# Carved flutes on a column — 12 flutes as a height field, not 100k triangles.
maps = ta.bake_detail_to_texture(
    ta.flute_detail(flutes=12), size=512, base_color=(0.62, 0.58, 0.52),
)
# {"albedo": AO-shaded colour, "bump": height, "normal": tangent-space normals}
```

Ready-made detail fields: `weave_detail(threads)` (cloth), `flute_detail(flutes)`
(columns, frames), `pore_detail(cells)` (stone, glaze, leather). Any callable
`detail(u, v, rng) -> height in [0,1]` on the unit tile works — keep repeats
integer to stay tileable.

**Performance budget guidance**

| Situation | Choose |
| --- | --- |
| Detail wavelength ≪ part size (weave, pores, grain, grooves, scrollwork) | **Texture** — a 512² bump/albedo set costs ~1 MB and ~0 triangles; the same relief modelled would add 100k+ triangles per part |
| Detail must silhouette (a part's outline, large mouldings, legs/handles) | **Geometry** — textures never change a silhouette |
| Close-up hero surface with parallax | Texture + normal map; add geometry only for the silhouette edge |
| Physics contact / collider surface | Geometry (simplified) — textures are visual only |
| Far/mid-field repetition (brick walls, lawns, hull panels) | Texture with `uv.scale` repeats; one 512 tile serves unlimited area |
| Export vertex density (SEG_U×SEG_V ≈ 500–1000 verts/part) | Bake bump→albedo shading (`bump_strength`) so relief reads even at vertex sampling |

Rule of thumb: if a feature is smaller than ~2× the mesh's edge length and
does not affect the silhouette or collisions, it belongs in a map. A 512²
map set generates in < 200 ms once and can be cached on disk keyed by
`(kind, size, seed)`; geometry costs are paid every frame, every physics
step, and every export.

---

## 5. Testing & demo

```bash
python -m pytest tests/test_texture_maps.py -q
python tools/demo_textures.py            # gallery + textured GLB into tools/out/textures/
```

The suite covers: channel/dtype/size contracts, determinism (same seed →
byte-identical), tileability (wrap-seam step never exceeds the pattern's own
maximum single-pixel step), the < 200 ms/512 px budget, UV application onto
analytic parts, `ietexture/1` JSON round-trip + validation, and the bake
helpers.
