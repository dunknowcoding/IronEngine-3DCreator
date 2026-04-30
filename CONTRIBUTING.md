# Contributing

## Development environment

Use Conda and keep the environment name aligned with the project docs:

```powershell
conda env create -f environment.yml
conda activate IronEngineWorld
python -m pip install -e .[all]
```

## Local model setup

The recommended local path uses Ollama with `qwen3.5:0.8b` or `qwen3.5:4b`:

```powershell
ollama serve
ollama pull qwen3.5:0.8b
```

## Common checks

```powershell
python -m build
```

## Notes

- Keep docs and screenshots in sync when user-facing workflows change.
- If you edit the prompt contract, update both the repo-root `SOUL.md` and the packaged copy in `src/ironengine_3d_creator/llm/SOUL.md`.
- Prefer focused changes and avoid committing generated caches or local exports.


