# Changelog

## 0.3.0

Complex geometry, style engine v2, texture generators, reference validation, provider fallback, soft authoring, and a showcase README.

- **Complex geometry**: 15 primitives (box, cylinder, sphere, cone, capsule, ellipsoid, torus, tube, arch, wedge, prism, frustum, slab, lathe, pipe) with boolean subtraction, lofting between cross-section profiles, and persistent part graphs that keep parts addressable through composition (`generation/primitives.py`, `generation/complex_builder.py`, `generation/subtraction` and part-graph support in the builder).
- **Style engine v2**: 14 style families with quality grammars — per-family structural rules, proportion checks, and panel-native parameters that drive the seeded offline generator (`generation/style_families.py`, `generation/quality_v2` checks, exquisite family presets).
- **Texture generators**: procedural albedo / roughness / normal-style map generation baked into PBR GLB exports (`generation/texture_maps.py`).
- **Reference validator**: scores generated specs against a per-object reference corpus (proportions, part presence, per-part errors); corpus-integrity tests skip gracefully when the external corpus directory is absent (`alignment/reference` scoring, `IRONENGINE_REFERENCE_ROOT` override).
- **Provider fallback chain**: ordered MiniMax → DeepSeek fallback with failure classification, `FallbackEvent` logging, spec-source provenance, a reorderable chain UI with key/reachability probing, and an opt-in real-API proof (`llm/chain.py`, `llm/registry.py`).
- **iemodel/3 soft authoring**: export manifest bumped to `iemodel/3` with soft-authoring fields — authored parts, intents, and editing hints carried through the pipeline and into exports (`generation/soft_author.py`, `core/manifest.py`).
- **Showcase README**: gallery-driven README with generated-model renders from `docs/showcase/`.
- CI: added a `ci` workflow running `python -m pytest -q` on Ubuntu / Windows / macOS × Python 3.11 / 3.12.

## 2026-07-23 (provider fallback chain)

MiniMax M3 primary → DeepSeek automatic fallback, end to end.

- Added `llm/chain.py`: an ordered provider fallback chain. When a provider fails — auth error, timeout, rate limit, connection failure, or a spec still invalid after the self-repair round — the pipeline transparently retries with the next provider (default order MiniMax → DeepSeek), logs every switch as a `FallbackEvent`, and annotates the spec's source. Failure classification (`classify_failure`) works off status codes / exception types / messages without importing any SDK. Includes `build_chain` / `chain_from_settings` constructors and a lightweight urllib `probe_endpoint` reachability check.
- `core/pipeline.py` accepts a `chain=` (or a `ProviderChain` as `provider`) and runs the spec route through `generate_spec_with_fallback`; `PipelineResult.spec_source` now records provenance ("minimax" / "deepseek" / "style_engine" / "code_mode" / "replay") and fallback switches appear in the warnings. Single-provider behavior is byte-for-byte unchanged, including cancellation semantics (stop_event never starts a new provider).
- Chain configuration lives in `llm/registry.py`: `DEFAULT_CHAIN`, `default_chain_config`, `normalize_chain_config` (order + per-provider enable/disable, unknown names dropped, missing defaults appended on upgrade), and `chain_status` (per-provider key-resolved map for the UI).
- The LLM config panel shows the ordered fallback chain with per-provider status (key resolved? endpoint reachable?), lets the user reorder (▲/▼) and disable entries, persists the config under settings `llm.chain`, and can build the runnable chain via `build_chain()`. A "Probe chain" button checks key resolution and `/models` reachability off the UI thread.
- Added `tests/test_provider_fallback.py`: chain config, failure taxonomy, fallback on auth/timeout/rate-limit/invalid-spec (mocked), no-fallback-on-success, chain-exhausted → style engine, offline path unchanged, plus a real-HTTP-over-localhost end-to-end test (real openai SDK, stub servers: 401 primary → streaming fallback). The live DeepSeek run is opt-in (`IRONENGINE_REAL_API=1`, class `TestDeepSeekFallbackRealAPI`): it discovers model ids via `GET /models`, forces MiniMax to fail via a bad endpoint override, and writes evidence to the e2e proof directory — skips, never fails, on missing or rejected credentials.
- Known issue: the DeepSeek key currently stored in Windows Credential Manager (both `IronEngine.3DCreator/deepseek` and the legacy Paperfessor entry, identical, suffix 3770) is rejected by the live API with HTTP 401 "Your api key is invalid". Resolution via `core.secrets.get_api_key('deepseek')` works; a valid key must be stored before the opt-in real-API proof can run.

## 2026-07-23 (wiring wave)

Self-repair wiring, DeepSeek in the UI, panel-native style families.

- Wired the LLM self-repair loop (`llm/repair.py`) into `core/pipeline.py`: streamed specs are validated (parseable JSON, non-empty primitives, <=30% integrity churn) and get exactly one validator-feedback repair round before falling back to the seeded style engine. The offline / no-key path is unchanged.
- The LLM config panel is now driven by `llm/registry.py` (`CLOUD_PROVIDERS`, `default_endpoint`, `credential_hint`): DeepSeek works from the UI (endpoint `https://api.deepseek.com`, key from env `DEEPSEEK_API_KEY` -> OS keychain -> legacy Credential Manager), MiniMax defaults to the international endpoint `https://api.minimax.io/v1`, and the API-key field shows where each provider's credential resolves from.
- `generation/style_families.py` emits panel-native params for desktops/tabletops/seats (flat, rx=pi/2) and chair-back slats (upright): 2-element in-plane `size` + separate `thickness`. The validator's box-semantics compat shim is kept for legacy LLM output and saved specs.
- Added `tools/out/` to `.gitignore` (demo artifacts).
- Added `tests/test_pipeline_repair.py`, `tests/test_provider_registry_ui.py`, and `tests/test_style_panels.py`.

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

