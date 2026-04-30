# PyPI Publishing

This project supports two ways to publish from GitHub Actions:

1. Trusted Publishing through PyPI OIDC
2. API token fallback through the `PYPI_API_TOKEN` GitHub secret

The workflow file is `.github/workflows/publish-pypi.yml`.

## Recommended: Trusted Publishing

If you want passwordless publishing, configure a trusted publisher on PyPI that matches the GitHub Actions identity exactly.

### Exact values for this repository

- PyPI project name: `ironengine-3d-creator`
- GitHub owner: `dunknowcoding`
- GitHub repository: `IronEngine-3DCreator`
- Workflow filename: `publish-pypi.yml`
- GitHub environment: `pypi`

### If the project already exists on PyPI

1. Go to PyPI and open `Your projects`.
2. Open `ironengine-3d-creator`.
3. Click `Manage`.
4. Open the `Publishing` page.
5. Add a GitHub Actions trusted publisher with the exact values above.

### If the project does not exist on PyPI yet

1. Go to your PyPI account settings.
2. Open the account-level `Publishing` page.
3. Add a pending publisher.
4. Use the exact values above.
5. Set the PyPI project name to `ironengine-3d-creator`.

After the first successful publish, the pending publisher becomes a normal publisher automatically.

## API token fallback

If you do not want to configure trusted publishing immediately, the workflow can publish with a classic PyPI API token.

### Required GitHub secret

- Secret name: `PYPI_API_TOKEN`

### Setup

1. Create a project-scoped PyPI API token for `ironengine-3d-creator`.
2. Go to GitHub repository settings.
3. Open `Settings -> Secrets and variables -> Actions`.
4. Add a new repository secret named `PYPI_API_TOKEN`.
5. Paste the token value.

When that secret exists, the workflow uses token publishing automatically.

## Manual rerun

The workflow supports manual execution.

1. Open the `Publish to PyPI` workflow in GitHub Actions.
2. Click `Run workflow`.
3. Set `ref` to the release tag you want to publish, for example `v0.2.0`.
4. Run the workflow.

## Common failure: `invalid-publisher`

This means GitHub successfully minted an OIDC token, but PyPI could not match it to a trusted publisher.

For this repository, the expected identity includes:

- repository: `dunknowcoding/IronEngine-3DCreator`
- workflow: `.github/workflows/publish-pypi.yml`
- environment: `pypi`
- release ref: `refs/tags/v0.2.0` when publishing the current release tag

Check for typos in the owner, repository, workflow filename, and environment name.

## Workflow behavior summary

- On GitHub release publish, the workflow builds the package and publishes it.
- On manual dispatch, you can choose which ref to publish.
- If `PYPI_API_TOKEN` exists, token auth is used.
- Otherwise, the workflow falls back to trusted publishing.
