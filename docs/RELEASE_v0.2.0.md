# IronEngine 3D Creator v0.2.0

Released: 2026-04-29
Tag: `v0.2.0`

## Summary

`v0.2.0` turns the project into a cleaner, package-ready public repository with:

- a rebuilt README and install documentation
- Conda-first packaging guidance
- bundled `SOUL.md` prompt rules inside the wheel
- Apache 2.0 licensing
- real local UI screenshots in the README
- GitHub Actions for package validation and PyPI publishing

## Highlights

### Repo and documentation refresh

- Rewrote `README.md` into a repository-ready landing page with usage guidance, screenshots, install instructions, and packaging notes.
- Added `docs/INSTALL.md`, `docs/PROJECT_ANALYSIS.md`, `CONTRIBUTING.md`, and `CHANGELOG.md`.
- Added `environment.yml`, `MANIFEST.in`, and `.gitattributes` for cleaner packaging and Git behavior.

### Packaging improvements

- Updated `pyproject.toml` for public distribution.
- Set the package author to `NiusRobotLab`.
- Kept `open3d` as a core dependency.
- Preserved optional extras for cloud providers and GPU acceleration.
- Bundled `src/ironengine_3d_creator/llm/SOUL.md` into the wheel so packaged installs retain prompt behavior.

### Verified local app flow

- Ran the UI locally in the Conda environment `IronEngineWorld`.
- Used the local Ollama model `qwen3.5:0.8b`.
- Generated and rendered a porcelain-vase example.
- Captured real screenshots and embedded them in `README.md`.

### Release automation

- Added `.github/workflows/package-check.yml` to build and validate distributions on pushes and pull requests.
- Added `.github/workflows/publish-pypi.yml` to publish distributions on GitHub release publication.
- Created and pushed the Git tag `v0.2.0`.

## Validation

The release contents were checked locally with the Conda environment `IronEngineWorld`:

- `python -m build --no-isolation`
- `python -m twine check dist/*`
- wheel asset verification for `ironengine_3d_creator/llm/SOUL.md`

## Recommended GitHub release title

`IronEngine 3D Creator v0.2.0`

## Suggested GitHub release body

IronEngine 3D Creator `v0.2.0` is the first polished public release of the prompt-to-3D desktop generator.

This release focuses on repository quality and packaging readiness:

- rebuilt README and install documentation
- Conda-first setup for `IronEngineWorld`
- Apache 2.0 licensing
- packaged `SOUL.md` prompt rules
- embedded real local screenshots
- GitHub Actions for package checks and PyPI publishing

The app remains centered on structured 3D generation:

- prompt -> SOUL rules -> LLM JSON spec -> validator -> integrity repair -> sampler -> mesh reconstruction -> preview/export

## PyPI publish checklist

Before publishing from GitHub Releases, configure a trusted publisher on PyPI for this repository:

- PyPI project: `ironengine-3d-creator`
- owner: `dunknowcoding`
- repository: `IronEngine-3DCreator`
- workflow: `publish-pypi.yml`
- environment: `pypi`

If trusted publishing is not configured yet, add the GitHub Actions secret `PYPI_API_TOKEN` and rerun the publish workflow against `v0.2.0`.

For the complete setup and troubleshooting steps, see [`docs/PYPI_PUBLISHING.md`](docs/PYPI_PUBLISHING.md).
