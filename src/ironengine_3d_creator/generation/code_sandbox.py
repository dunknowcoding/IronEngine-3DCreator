"""AST-allowlisted Python sandbox for LLM-emitted code-mode scripts.

We compile and execute the code under a strict allowlist of node types and
attribute accesses, and a runtime watchdog. The script must define a
`generate()` function returning `(positions, colors)`. This is a defence in
depth — code mode is opt-in and clearly labelled in the UI.
"""
from __future__ import annotations

import ast
import math
import threading
from typing import Any

import numpy as np

ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Call, ast.Name, ast.Constant,
    ast.Load, ast.Store, ast.Del,
    ast.List, ast.Tuple, ast.Set, ast.Dict,
    ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp,
    ast.comprehension,
    ast.For, ast.If, ast.While, ast.Pass, ast.Break, ast.Continue,
    ast.Subscript, ast.Slice,
    ast.FunctionDef, ast.Return, ast.arguments, ast.arg,
    ast.IfExp,
    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult,
    ast.UAdd, ast.USub, ast.Not, ast.Invert,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
)

# Attribute accesses are restricted to these root names → only their public methods.
ALLOWED_ATTR_ROOTS = {"np", "math"}


class SandboxError(RuntimeError):
    pass


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES) and not isinstance(node, ast.Attribute):
            raise SandboxError(f"disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            # Walk down chained attribute accesses to find the root Name.
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name) or root.id not in ALLOWED_ATTR_ROOTS:
                raise SandboxError(f"attribute access not allowed: {ast.unparse(node)}")
            if node.attr.startswith("_"):
                raise SandboxError(f"private attribute access not allowed: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile", "getattr", "setattr",
                                "delattr", "globals", "locals", "vars", "open",
                                "__import__", "input"}:
                raise SandboxError(f"call to {node.func.id} not allowed")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxError("imports not allowed")


_NUMPY_SAFE = {
    name: getattr(np, name)
    for name in (
        "array", "asarray", "stack", "concatenate", "zeros", "ones", "empty",
        "arange", "linspace", "sin", "cos", "tan", "arctan2", "sqrt", "exp", "log",
        "abs", "minimum", "maximum", "clip", "where", "pi", "newaxis",
        "float32", "float64", "int32", "uint8",
    )
    if hasattr(np, name)
}


_MATH_SAFE = {
    name: getattr(math, name)
    for name in ("pi", "tau", "e", "sin", "cos", "tan", "atan2", "sqrt", "log", "exp", "floor", "ceil")
}


def run_sandbox(source: str, n_points: int, *, timeout_s: float = 5.0) -> tuple[np.ndarray, np.ndarray | None]:
    """Compile, validate, and execute `source`. Returns (positions, colors)."""
    tree = ast.parse(source, mode="exec")
    _validate(tree)
    code = compile(tree, "<sandbox>", "exec")

    # Tightly restricted globals — no builtins beyond a tiny safe subset.
    safe_builtins = {
        "len": len, "range": range, "min": min, "max": max, "sum": sum,
        "abs": abs, "round": round, "int": int, "float": float, "list": list,
        "tuple": tuple, "dict": dict, "enumerate": enumerate, "zip": zip,
    }
    g: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "np": type("NPProxy", (), _NUMPY_SAFE),
        "math": type("MathProxy", (), _MATH_SAFE),
        "n_points": int(n_points),
    }

    # Watchdog: a thread races the executor; if it expires, raise.
    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            exec(code, g)
            gen = g.get("generate")
            if not callable(gen):
                raise SandboxError("script must define `def generate(): ...`")
            r = gen()
            result_box["value"] = r
        except BaseException as e:
            error_box["error"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        # We can't safely kill the thread; mark the result invalid and let it die in the background.
        raise SandboxError(f"sandbox script exceeded timeout of {timeout_s}s")
    if "error" in error_box:
        raise SandboxError(str(error_box["error"]))
    if "value" not in result_box:
        raise SandboxError("sandbox script returned nothing")

    val = result_box["value"]
    if isinstance(val, tuple) and len(val) == 2:
        positions, colors = val
    else:
        positions, colors = val, None

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise SandboxError(f"positions must have shape (N, 3); got {positions.shape}")
    if positions.shape[0] > int(n_points * 1.5) + 1024:
        raise SandboxError(f"positions exceeds 1.5x n_points cap ({positions.shape[0]} > {int(n_points * 1.5)})")

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.shape != positions.shape:
            raise SandboxError(f"colors shape {colors.shape} must match positions {positions.shape}")
        colors = np.clip(colors, 0.0, 1.0)

    return positions, colors
