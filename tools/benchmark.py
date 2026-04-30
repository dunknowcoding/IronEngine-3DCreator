"""End-to-end battery: a grid of (model, prompt, n_points) generations.

Drives the same pipeline the UI uses, records timings + spec quality + any
warnings, and writes a CSV + a thumbnail PLY per cell. Designed to be run from
the command line so the GUI is not a prerequisite.

Usage:
  python tools/benchmark.py
  python tools/benchmark.py --models qwen3.5:0.8b qwen3.5:2b
  python tools/benchmark.py --models qwen3.5:0.8b --prompts 4 --counts 5000 50000
  python tools/benchmark.py --think  # opt into chain-of-thought (slow)
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ironengine_3d_creator.alignment.parser import parse_spec
from ironengine_3d_creator.core.exporter import write_ply
from ironengine_3d_creator.core.pipeline import PipelineRequest, run as pipeline_run
from ironengine_3d_creator.core.resources import set_active_backend
from ironengine_3d_creator.llm.registry import make_provider
from ironengine_3d_creator.llm.thinking import strip as strip_thinking


DEFAULT_MODELS = ("qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b")

DEFAULT_PROMPTS = (
    "a wooden chair with four legs",
    "a tall ceramic vase with curved patterns",
    "a small mushroom with a bumpy cap",
    "a six-legged creature with a long tail",
    "a hexagonal prism gear",
    "a coiled metal spring",
    "a tree with a thick trunk and a leafy crown",
    "a battered rock with deep cracks",
    "a futuristic lamp with a curved arm",
    "a stack of three crates",
    "a torus pendant with engraved ridges",
    "a small wooden boat with a square sail",
)

DEFAULT_COUNTS = (5_000, 25_000, 50_000, 100_000, 250_000)


def run_one(
    *,
    provider,
    prompt: str,
    n_points: int,
    seed: int,
    out_dir: Path,
    cell_budget_s: float = 60.0,
) -> dict:
    """Execute one generation; return a dict with metrics + path to written PLY."""
    import threading
    req = PipelineRequest(user_prompt=prompt, n_points=n_points, seed=seed)
    timings = {"first_token_s": None, "stream_end_s": None, "generate_end_s": None}
    chunks: list[str] = []
    box: dict = {}
    stop_event = threading.Event()

    t0 = time.perf_counter()

    def on_token(c: str) -> None:
        if timings["first_token_s"] is None:
            timings["first_token_s"] = time.perf_counter() - t0
        chunks.append(c)

    def on_stage(stage: str) -> None:
        if stage in ("validating", "sampling"):
            timings["stream_end_s"] = time.perf_counter() - t0

    def runner() -> None:
        try:
            box["result"] = pipeline_run(
                req, provider, on_token=on_token, on_stage=on_stage,
                stop_event=stop_event,
            )
        except Exception as e:
            box["error"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=runner, daemon=True)
    th.start()
    th.join(timeout=cell_budget_s)
    timings["generate_end_s"] = time.perf_counter() - t0
    if th.is_alive():
        # Ask the streaming loop to bail out and close its connection so
        # the next cell does not queue behind a dead request.
        stop_event.set()
        th.join(timeout=5.0)
        return {
            "ok": False, "error": f"cell exceeded {cell_budget_s}s budget",
            "timings": timings, "raw_chars": sum(len(c) for c in chunks),
            "shape": None, "primitives": 0, "features": 0, "points": 0,
            "warnings": [], "ply_path": "",
        }
    if "error" in box:
        return {
            "ok": False, "error": box["error"],
            "timings": timings, "raw_chars": sum(len(c) for c in chunks),
            "shape": None, "primitives": 0, "features": 0, "points": 0,
            "warnings": [], "ply_path": "",
        }
    result = box["result"]
    ok = True
    err = ""

    raw_text = "".join(chunks)
    raw_clean = strip_thinking(raw_text)

    # Extract a "did the LLM emit valid JSON" signal even when validator
    # successfully fell back to auto.
    json_ok = True
    try:
        parse_spec(raw_text)
    except Exception:
        json_ok = False

    # Save the PLY only for non-trivial point counts to keep disk usage modest.
    ply_path = ""
    if n_points >= 5_000:
        slug = "_".join(prompt.lower().split()[:5]).replace("/", "-").replace(",", "")
        out = out_dir / f"{provider.model.replace(':','_')}__{n_points}__{slug}.ply"
        try:
            write_ply(out, result.generation.positions, result.generation.colors)
            ply_path = str(out)
        except Exception as e:
            err = f"PLY write failed: {e}"

    return {
        "ok": ok, "error": err, "timings": timings,
        "raw_chars": len(raw_text), "json_ok": json_ok,
        "shape": result.spec.shape,
        "primitives": len(result.spec.primitives),
        "features": len(result.spec.features),
        "points": int(result.generation.positions.shape[0]),
        "warnings": result.warnings,
        "ply_path": ply_path,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--prompts", type=int, default=len(DEFAULT_PROMPTS),
                   help="Use the first N prompts from the curated list")
    p.add_argument("--counts", nargs="+", type=int, default=list(DEFAULT_COUNTS))
    p.add_argument("--endpoint", default="http://localhost:11434")
    p.add_argument("--think", action="store_true", help="Enable model thinking mode (slower)")
    p.add_argument("--out", type=Path, default=Path("benchmark_out"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    set_active_backend("auto", prefer_gpu=True)

    prompts = list(DEFAULT_PROMPTS)[: args.prompts]
    counts = list(args.counts)

    print(f"models : {args.models}")
    print(f"prompts: {len(prompts)} -> {prompts[:3]}...")
    print(f"counts : {counts}")
    print(f"think  : {args.think}")
    print(f"out    : {args.out}")
    print()

    rows: list[dict] = []
    csv_path = args.out / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = (
        "model", "n_points", "prompt", "ok", "error", "json_ok",
        "first_token_s", "stream_end_s", "generate_end_s",
        "shape", "primitives", "features", "points", "warnings", "raw_chars", "ply_path",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for model in args.models:
            print(f"=== {model} ===")
            try:
                provider = make_provider(
                    "ollama", model=model, endpoint=args.endpoint, think_mode=args.think,
                )
            except Exception as e:
                print(f"  could not build provider: {e}")
                continue

            for prompt in prompts:
                for n in counts:
                    print(f"  * {n:>7,} pts | {prompt[:48]:<48}", end=" ", flush=True)
                    res = run_one(
                        provider=provider, prompt=prompt, n_points=n,
                        seed=args.seed, out_dir=args.out,
                    )
                    label = "ok" if res["ok"] else "FAIL"
                    timings = res["timings"]
                    print(
                        f"{label} "
                        f"first={timings['first_token_s']!s:.5} "
                        f"end={timings['generate_end_s']!s:.5} "
                        f"shape={res['shape']!s:<10} "
                        f"prims={res['primitives']:>2} "
                        f"pts={res['points']:>7,} "
                        f"warn={len(res['warnings'])}"
                    )
                    rows.append({"model": model, "n_points": n, "prompt": prompt, **res})
                    writer.writerow({
                        "model": model, "n_points": n, "prompt": prompt,
                        "ok": res["ok"], "error": res["error"],
                        "json_ok": res.get("json_ok", False),
                        "first_token_s": timings["first_token_s"],
                        "stream_end_s": timings["stream_end_s"],
                        "generate_end_s": timings["generate_end_s"],
                        "shape": res["shape"],
                        "primitives": res["primitives"],
                        "features": res["features"],
                        "points": res["points"],
                        "warnings": "; ".join(res["warnings"])[:200],
                        "raw_chars": res["raw_chars"],
                        "ply_path": res["ply_path"],
                    })
                    fh.flush()

    print()
    print(f"--- summary -> {csv_path} ---")
    if rows:
        ok = sum(1 for r in rows if r["ok"])
        json_ok = sum(1 for r in rows if r.get("json_ok"))
        print(f"  cells: {len(rows)}  ok: {ok}  json_ok: {json_ok}  fails: {len(rows) - ok}")
        for model in args.models:
            mr = [r for r in rows if r["model"] == model]
            if not mr:
                continue
            ts = [r["timings"]["generate_end_s"] for r in mr if r["ok"] and r["timings"]["generate_end_s"]]
            jok = sum(1 for r in mr if r.get("json_ok"))
            print(f"  {model:<18}  ok={sum(1 for r in mr if r['ok']):>3}/{len(mr)}  "
                  f"json_ok={jok:>3}/{len(mr)}  median_total={np.median(ts) if ts else 0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
