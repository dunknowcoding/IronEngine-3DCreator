# Project Analysis

## Architecture snapshot

IronEngine 3D Creator is a desktop application that converts natural-language requirements into a `GenerationSpec`, repairs structure deterministically, then samples a colored point cloud and optionally reconstructs a preview mesh.

## Strengths

- The pipeline is cleanly split into alignment, generation, rendering, UI, and LLM layers.
- The integrity layer provides a strong differentiator by repairing common structural failures.
- The renderer exposes both in-app preview and offscreen APIs, which is ideal for docs and automation.
- The app already supports multiple local and cloud LLM providers and thinking-aware streaming.

## Repo-readiness issues found

- The old `README.md` and root `SOUL.md` were notebook-style JSON documents instead of plain Markdown.
- Prompt rules were repo-root only, which risked being dropped from wheel installs.
- Tests, caches, and editable-install artefacts were present in the repository snapshot.
- Install guidance was not explicit enough about Conda-first setup and optional acceleration backends.

## Improvements applied

- Reworked packaging metadata in `pyproject.toml`.
- Added package data for `SOUL.md` so prompt rules ship with the distribution.
- Added `environment.yml`, `MANIFEST.in`, `CONTRIBUTING.md`, `CHANGELOG.md`, and `docs/INSTALL.md`.
- Removed `tests`, `__pycache__`, `.pytest_cache`, and egg-info artefacts from the working tree.
- Added README screenshots from the real local UI + Ollama flow.

## Validation completed

- Ran the UI against local Ollama using `qwen3.5:0.8b` inside `IronEngineWorld`.
- Captured generated viewport screenshots and embedded them into `README.md`.
- Built both sdist and wheel successfully after the documentation pass.

