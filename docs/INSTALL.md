# Installation Guide

## Recommended: Conda environment

```powershell
conda env create -f environment.yml
conda activate IronEngineWorld
python -m pip install -e .
```

If you already have the environment and only need the package:

```powershell
conda activate IronEngineWorld
python -m pip install -e .
```

## Base dependency behavior

- `open3d` is included in the main package requirements.
- Mesh preview and mesh export should work immediately after the base install.
- GPU acceleration remains optional and should be installed only when it matches your machine.

## Optional extras

```powershell
python -m pip install -e .[anthropic]
python -m pip install -e .[openai]
python -m pip install -e .[gpu_taichi]
python -m pip install -e .[gpu_cupy]
python -m pip install -e .[gpu_torch]
python -m pip install -e .[nvidia]
python -m pip install -e .[sim]
python -m pip install -e .[all]
```

## Local provider setup with Ollama

```powershell
ollama serve
ollama pull qwen3.5:0.8b
```

In the UI:

1. Set **Provider** to `ollama`.
2. Keep **Endpoint** at `http://localhost:11434`.
3. Pick `qwen3.5:0.8b` or another installed `qwen3.5` variant.
4. Enable **Reasoning / thinking mode** only when you want slower but more deliberate generations.

## Launch

```powershell
conda activate IronEngineWorld
python -m ironengine_3d_creator
```

## Package distribution smoke test

```powershell
conda activate IronEngineWorld
python -m build
python -m pip install dist\ironengine_3d_creator-0.2.0-py3-none-any.whl
```
