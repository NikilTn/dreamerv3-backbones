#!/usr/bin/env python3
"""Read smoke run logdirs, compute steady-state throughput, project sweep wall-clock.

Pipeline:
  1) For each smoke logdir, parse metrics.jsonl and pick the steady-state fps
     (median of last N entries). The trainer's `fps/fps` is the rate measured
     between consecutive Logger.write() calls and is in the same units as
     `trainer.steps`: env-interactions for vector envs, FRAMES for Atari (because
     trainer.py increments step by `action_repeat`).
  2) Compute backbone overhead ratios (vs GRU on POPGym) and subexp overheads.
  3) Project a candidate sweep's GPU-hours and wall-clock at several concurrencies.

Run after `bash scripts/smoke_throughput.sh` finishes:
  python scripts/smoke_analyze.py --smoke-root logdir/smoke
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

# Production env step budgets, in the units `trainer.steps` counts.
# Atari counts FRAMES (after action_repeat); POPGym/BSuite count env interactions.
ENV_BUDGETS: dict[str, int] = {
    "atari100k": 410_000,   # configs/env/atari100k.yaml: steps=4.1e5
    "bsuite":    500_000,   # configs/env/bsuite_*.yaml:  steps=5e5
    "popgym":  1_000_000,   # configs/env/popgym_*.yaml:  steps=1e6
}


def read_steady_sps(logdir: Path, tail_n: int = 10) -> Optional[float]:
    """Median of the last `tail_n` `fps/fps` entries in metrics.jsonl.

    Returns None if the run did not produce enough samples (e.g. crashed early).
    """
    metrics_path = logdir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    fps_values: list[float] = []
    for line in metrics_path.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = d.get("fps/fps")
        if isinstance(v, (int, float)) and v > 0:
            fps_values.append(float(v))
    if len(fps_values) < 3:
        return None
    return statistics.median(fps_values[-tail_n:])


def parse_smoke_dir(smoke_root: Path) -> dict[str, Optional[float]]:
    return {p.name: read_steady_sps(p) for p in sorted(smoke_root.iterdir()) if p.is_dir()}


def env_family(name: str) -> str:
    if name.startswith("atari"):
        return "atari100k"
    if name.startswith("bsuite"):
        return "bsuite"
    if name.startswith("popgym"):
        return "popgym"
    raise ValueError(f"Unknown env family for: {name}")


def require(sps: dict, key: str) -> float:
    v = sps.get(key)
    if v is None:
        sys.exit(
            f"ERROR: smoke run '{key}' missing or has no usable fps samples.\n"
            f"  Check logdir/smoke/{key}/console.log and metrics.jsonl.\n"
            f"  All 10 smoke jobs must complete before running the analyzer."
        )
    return v


def project(
    sps: dict[str, Optional[float]],
    sweep: list[tuple[str, str, str, int]],
) -> None:
    base_popgym = require(sps, "popgym_gru_none")
    base_bsuite = require(sps, "bsuite_gru_none")
    base_atari  = require(sps, "atari_gru_none")
    base = {"popgym": base_popgym, "bsuite": base_bsuite, "atari100k": base_atari}

    bb_overhead = {
        bb: base_popgym / require(sps, f"popgym_{bb}_none")
        for bb in ("gru", "transformer", "mamba", "s4", "s5")
    }
    sub_overhead = {
        "none":    1.0,
        "cpc":     base_popgym / require(sps, "popgym_gru_cpc"),
        "dfs":     base_popgym / require(sps, "popgym_gru_dfs"),
    }
    sub_overhead["cpc_dfs"] = sub_overhead["cpc"] * sub_overhead["dfs"]

    print("\n=== Steady-state fps per smoke run ===")
    for k, v in sorted(sps.items()):
        v_str = f"{v:7.1f}" if v is not None else "  N/A  "
        print(f"  {k:32s} {v_str}")

    print("\n=== Backbone overhead (multiplicative slowdown vs GRU on POPGym) ===")
    for bb, x in bb_overhead.items():
        print(f"  {bb:12s} {x:.2f}x")

    print("\n=== Subexp overhead (multiplicative) ===")
    for s, x in sub_overhead.items():
        print(f"  {s:8s} {x:.2f}x")

    print("\n=== Per-env-family GRU baseline fps ===")
    for e, x in base.items():
        print(f"  {e:10s} {x:7.1f}  (units: {'frames' if e == 'atari100k' else 'env-steps'}/sec)")

    # Cross-suite sanity check
    actual_atari_xfmr = sps.get("atari_transformer_none")
    expected_atari_xfmr = base_atari / bb_overhead["transformer"]
    print("\n=== Cross-suite sanity (transformer overhead on Atari) ===")
    print(f"  Predicted from POPGym ratio : {expected_atari_xfmr:7.1f} fps")
    if actual_atari_xfmr is None:
        print("  Actual on Atari             :   N/A   (smoke run missing)")
    else:
        ratio = actual_atari_xfmr / expected_atari_xfmr
        print(f"  Actual on Atari             : {actual_atari_xfmr:7.1f} fps")
        print(f"  Transfer ratio (1.0 ideal)  : {ratio:.2f}")
        if not (0.85 <= ratio <= 1.15):
            print("  WARNING: backbone overhead does NOT transfer cleanly across suites.")
            print("           Add per-suite backbone smokes for accurate Atari estimates.")

    # Project the candidate sweep
    print(f"\n=== Projected sweep ===")
    print(f"{'backbone':12s} {'env':10s} {'subexp':8s} {'runs':>5s} {'h/run':>8s} {'h_total':>10s}")
    total_secs, total_runs = 0.0, 0
    for bb, env_fam, subexp, n_runs in sweep:
        eff_sps = base[env_fam] / (bb_overhead[bb] * sub_overhead[subexp])
        per_run_sec = ENV_BUDGETS[env_fam] / eff_sps
        total = n_runs * per_run_sec
        total_secs += total
        total_runs += n_runs
        print(
            f"{bb:12s} {env_fam:10s} {subexp:8s} "
            f"{n_runs:>5d} {per_run_sec/3600:>8.2f} {total/3600:>10.1f}"
        )
    total_h = total_secs / 3600
    print(f"\n  TOTAL: {total_runs} runs, {total_h:.1f} GPU-hours")

    print("\n=== Wall-clock at sustained concurrency ===")
    for nconc in (4, 6, 8, 10, 12):
        wall_h = total_h / nconc
        print(f"  {nconc:>2d} GPUs : {wall_h:>6.1f} h  ({wall_h/24:>4.1f} days)")

    print("\n=== Decision rule ===")
    print("  Target: pick the largest matrix where TOTAL <= 0.7 * (deadline_hours * sustained_GPUs).")
    print("  Example: 10 days * 24 h * 8 GPUs = 1920 GPU-h budget; usable = 0.7 * 1920 = 1344 GPU-h.")
    print("  Reserve the 30% slack for: failed-seed re-runs, contention spikes, analysis iteration.")


# Edit this list to define the sweep you want priced. Each tuple:
#   (backbone, env_family, subexp, n_runs)
# n_runs counts every (task_variant, seed) combination, e.g. 3 difficulties * 5 seeds = 15.
DEFAULT_SWEEP: list[tuple[str, str, str, int]] = [
    # Main 75-run plan: POPGym RepeatPrevious x 5 backbones x 5 seeds x 3 difficulties.
    ("gru",         "popgym", "none", 15),
    ("transformer", "popgym", "none", 15),
    ("mamba",       "popgym", "none", 15),
    ("s4",          "popgym", "none", 15),
    ("s5",          "popgym", "none", 15),
]


# Larger ~10% coverage matrix (uncomment in main() to use this instead).
EXPANDED_SWEEP: list[tuple[str, str, str, int]] = [
    # POPGym RepeatPrevious x all backbones x 3 difficulties x 5 seeds = 75
    ("gru",         "popgym", "none", 15),
    ("transformer", "popgym", "none", 15),
    ("mamba",       "popgym", "none", 15),
    ("s4",          "popgym", "none", 15),
    ("s5",          "popgym", "none", 15),
    # POPGym Autoencode x all backbones x 3 difficulties x 3 seeds = 45
    ("gru",         "popgym", "none", 9),
    ("transformer", "popgym", "none", 9),
    ("mamba",       "popgym", "none", 9),
    ("s4",          "popgym", "none", 9),
    ("s5",          "popgym", "none", 9),
    # BSuite memory_len + memory_size x all backbones x 3 ids each x 3 seeds = 90
    ("gru",         "bsuite", "none", 18),
    ("transformer", "bsuite", "none", 18),
    ("mamba",       "bsuite", "none", 18),
    ("s4",          "bsuite", "none", 18),
    ("s5",          "bsuite", "none", 18),
    # Atari100k subset (8 games) x all backbones x 3 seeds = 120
    ("gru",         "atari100k", "none", 24),
    ("transformer", "atari100k", "none", 24),
    ("mamba",       "atari100k", "none", 24),
    ("s4",          "atari100k", "none", 24),
    ("s5",          "atari100k", "none", 24),
    # Sub-experiments on POPGym Hard only x all backbones x 3 envs x 3 seeds.
    # (3 subexps cells: cpc, dfs, cpc_dfs, since `none` is already counted above.)
    ("gru",         "popgym", "cpc",     9),
    ("transformer", "popgym", "cpc",     9),
    ("mamba",       "popgym", "cpc",     9),
    ("s4",          "popgym", "cpc",     9),
    ("s5",          "popgym", "cpc",     9),
    ("gru",         "popgym", "dfs",     9),
    ("transformer", "popgym", "dfs",     9),
    ("mamba",       "popgym", "dfs",     9),
    ("s4",          "popgym", "dfs",     9),
    ("s5",          "popgym", "dfs",     9),
    ("gru",         "popgym", "cpc_dfs", 9),
    ("transformer", "popgym", "cpc_dfs", 9),
    ("mamba",       "popgym", "cpc_dfs", 9),
    ("s4",          "popgym", "cpc_dfs", 9),
    ("s5",          "popgym", "cpc_dfs", 9),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", default=Path("logdir/smoke"), type=Path)
    parser.add_argument(
        "--sweep",
        choices=["main", "expanded"],
        default="main",
        help="'main' = 75-run RepeatPrevious plan; 'expanded' = ~10% coverage plan.",
    )
    args = parser.parse_args()

    if not args.smoke_root.exists():
        sys.exit(f"smoke root not found: {args.smoke_root}")

    sps = parse_smoke_dir(args.smoke_root)
    sweep = DEFAULT_SWEEP if args.sweep == "main" else EXPANDED_SWEEP
    project(sps, sweep)


if __name__ == "__main__":
    main()
