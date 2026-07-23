# Changelog

## 2026-07-23

Realism milestone: analytic PBR export, `iemodel/2`, MiniMax provider, async handoff.

- Added a MiniMax LLM provider (`llm/minimax.py`): OpenAI-compatible endpoint `https://api.minimaxi.com/v1`, default model `MiniMax-M3`, API key resolved via `core.secrets` (env `MINIMAX_API_KEY` / Windows Credential Manager, with the legacy Paperfessor entry as fallback).
- Fixed a `stop_event` crash in the cloud providers and added `json_mode` support for OpenAI-compatible chat endpoints.
- Spec-driven GLB exports now use analytic per-primitive meshes (`generation/analytic_mesh.py`) with exact normals and UVs instead of ball-pivot reconstruction; ball-pivot remains the fallback for spec-less clouds.
- GLB exports are now full PBR: named per-part nodes, baked `baseColorTexture`, `COLOR_0` vertex colors, and metallic/roughness factors from the material presets.
- Bumped the export manifest to `iemodel/2`: per-part `parts` summaries and named `materials`, measured `solid_volume_m3`, computed `physics.mass_kg`, and collider kinds (`box` / `convex` / `parts`).
- Added unbaked albedo export alongside the baked textured variant.
- Surface features now displace points along vertex normals, and ellipsoid sampling is area-uniform.
- Fixed PCD exports writing float-packed colors instead of integer RGB.
- Replaced the per-point PLY/PCD writers with fast `numpy.savetxt` implementations and added a binary output option.
- Added OBJ export with a matching MTL material file.
- "Send to SceneEditor" is now asynchronous: a background worker writes the `PLY`/`GLB`/`.iemodel.json` triple plus a `handoff.json` pointer, and the editor is launched with `--import <manifest>` (legacy fallback preserved).
- Enforced a RAM cap during generation/export so large point budgets fail gracefully instead of exhausting memory.
- Trimmed the selectable compute backends to the ones that actually work: `cuda_cupy` and `cpu_numpy` (`auto` picks); the no-op `cuda_torch`/Taichi entries were removed.
- Added `tests/test_analytic_mesh.py`, `tests/test_export_formats.py`, `tests/test_cloud_providers.py`, and `tests/test_minimax_provider.py`.

## Unreleased

- Added `.iemodel.json` export manifest (`iemodel/1`) carrying units, bounds, PBR material, and physics metadata alongside `PLY`/`GLB` exports.
- Added a physical material preset table (`generation/materials.py`) resolving LLM material hints (wood, stone, metal, ceramic, glass, …) to roughness/metallic/density/friction/restitution.
- Fixed GLB/OBJ mesh export: `_reconstruct_to_mesh` now delegates to `generation.reconstruct` (adaptive ball-pivot radii, oriented normals, Poisson fallback) and transfers vertex colors onto mesh vertices.
- GLB writing now prefers trimesh's binary writer when available, working around Open3D 0.19 emitting GLBs its own ASSIMP reader rejects.
- "Send to SceneEditor" now writes a timestamped `creator_model_*.{ply,glb,iemodel.json}` triple; GLB/PLY dialog exports write a sibling manifest.
- Added `tests/test_manifest.py` covering material resolution, manifest round-trip, and GLB export geometry/colors/normals.

## 0.2.0 - 2026-04-29

- Rebuilt the repository documentation for package consumers and contributors.
- Added Conda-first installation assets, including `environment.yml` and `MANIFEST.in`.
- Bundled `SOUL.md` inside the Python package so wheel installs keep structural prompting intact.
- Switched repository licensing and package metadata to Apache 2.0.
- Embedded real local UI screenshots into README.md.
- Removed repository-local test and cache artefacts to prepare the repo for distribution.

