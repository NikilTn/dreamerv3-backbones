"""Aggregate POPGym RepeatPrevious results into paper/PPT-ready tables and plots.

Reads:
- deterministic eval TSVs for GRU/S3M/S5 and Transformer
- ``checkpoints/repeat_previous_reduced_20260504_004958/full_runs/.../metrics.jsonl``
  (training learning curves, throughput, final episode scores)
- ``checkpoints/repeat_previous_reduced_20260504_004958/full_runs/.../run_metadata.json``
  (status / completed flag, backbone, seed, env)

Writes a self-contained report folder under
``checkpoints/repeat_previous_reduced_20260504_004958/figures_for_report_4backbones_20260510/``:

- coverage_table.{csv,md}                — task × backbone seed-completion grid
- final_eval_table.{csv,md}              — deterministic eval (20-ep argmax) per task × backbone
- final_train_table.{csv,md}             — last episode/score from metrics.jsonl per task × backbone
- aggregate_backbone_table.{csv,md}      — macro-average over 3 difficulties
- pairwise_vs_gru.{csv,md}               — non-GRU backbones vs GRU per task (P(>GRU), Δ mean, paired n)
- sample_efficiency_auc.{csv,md}         — area under smoothed learning curve, by task × backbone
- compute_efficiency.{csv,md}            — median steady-state fps/fps per task × backbone
- final_eval_by_task.{png,pdf}           — grouped bar (Easy/Medium/Hard × backbone), eval return
- final_train_by_task.{png,pdf}          — grouped bar (Easy/Medium/Hard × backbone), train return
- learning_curve_<task>.{png,pdf}        — mean ± SEM band over seeds, one PNG/PDF per task
- sample_efficiency_auc.{png,pdf}        — AUC bar plot, grouped by task
- compute_efficiency_fps.{png,pdf}       — median fps bar plot, grouped by task

Excluded: Mamba (teammate handles separately). Transformer is included only for
the completed one-seed-per-difficulty A100 runs.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import sys
import warnings
from collections import defaultdict
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------- paths
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "repeat_previous_reduced_20260504_004958"
EVAL_TSV = CHECKPOINT_ROOT / "eval_20ep_20260505" / "eval_summary.tsv"
EVAL_TSVS = [
    EVAL_TSV,
    CHECKPOINT_ROOT / "eval_20ep_missing_20260510" / "eval_summary.tsv",
    REPO_ROOT
    / "checkpoints"
    / "transformer_a100_20260509"
    / "logdir"
    / "transformer_repeat_previous_a100_bs128_eager_20260506"
    / "eval_20ep_seed0_20260509"
    / "eval_summary.tsv",
]
FULL_RUNS_ROOT = CHECKPOINT_ROOT / "full_runs"
TRANSFORMER_RUNS_ROOT = (
    REPO_ROOT
    / "checkpoints"
    / "transformer_a100_20260509"
    / "logdir"
    / "transformer_repeat_previous_a100_bs128_eager_20260506"
)
FULL_RUNS_ROOTS = [FULL_RUNS_ROOT, TRANSFORMER_RUNS_ROOT]
OUT_DIR = CHECKPOINT_ROOT / "figures_for_report_4backbones_20260510"

# Three difficulties of POPGym RepeatPrevious. Display order = difficulty.
TASKS = [
    "popgym_RepeatPreviousEasy-v0",
    "popgym_RepeatPreviousMedium-v0",
    "popgym_RepeatPreviousHard-v0",
]
TASK_LABELS = {t: t.replace("popgym_RepeatPrevious", "").replace("-v0", "") for t in TASKS}

# Backbones included in the comparison. Mamba is excluded until teammate runs
# land. Transformer has one completed seed per difficulty, unlike the 5-seed
# GRU/S3M/S5 cells.
BACKBONES = ["gru", "s3m", "s5", "transformer"]
BACKBONE_LABELS = {
    "gru": "GRU",
    "s3m": "S3M / S4D",
    "s5": "S5",
    "transformer": "Transformer-XL",
}
BACKBONE_COLORS = {
    "gru": "#222222",
    "s3m": "#1f77b4",
    "s5": "#2ca02c",
    "transformer": "#d62728",
}
EXPECTED_SEEDS = {
    "gru": {0, 1, 2, 3, 4},
    "s3m": {0, 1, 2, 3, 4},
    "s5": {0, 1, 2, 3, 4},
    "transformer": {0},
}

# Numerical knobs.
SMOOTH_WINDOW = 50          # episode-score rolling window (in episode-log entries)
BOOTSTRAP_N = 5_000          # bootstrap resamples for CI
BOOTSTRAP_SEED = 0
TARGET_STEPS = 1_000_000     # sweep was launched with trainer.steps = 1e6
CI_LEVEL = 0.95


# --------------------------------------------------------------- small utils
def _is_finite(value) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def _bootstrap_mean_ci(values: np.ndarray, n: int = BOOTSTRAP_N, level: float = CI_LEVEL,
                       seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """Return (mean, lo, hi) bootstrap CI on the mean.

    Falls back to (mean, nan, nan) when too few finite samples.
    """
    values = np.asarray([v for v in values if _is_finite(v)], dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n, values.size))
    boot_means = values[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return (float(values.mean()), float(lo), float(hi))


def _sem(values: np.ndarray) -> float:
    values = np.asarray([v for v in values if _is_finite(v)], dtype=np.float64)
    if values.size <= 1:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(values.size))


def _iqm(values: np.ndarray) -> float:
    """Interquartile mean: average of values between the 25th and 75th percentiles."""
    values = np.asarray([v for v in values if _is_finite(v)], dtype=np.float64)
    if values.size == 0:
        return float("nan")
    if values.size < 4:
        return float(values.mean())
    q25, q75 = np.quantile(values, [0.25, 0.75])
    keep = values[(values >= q25) & (values <= q75)]
    if keep.size == 0:
        return float(values.mean())
    return float(keep.mean())


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Right-aligned rolling mean. Output length matches input."""
    if x.size == 0:
        return x
    window = max(1, min(window, x.size))
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full_like(x, fill_value=np.nan, dtype=np.float64)
    # Right-aligned means: out[i] = mean(x[max(0, i-window+1) : i+1])
    for i in range(x.size):
        lo = max(0, i - window + 1)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
    return out


# ------------------------------------------------------------------ loaders
def load_eval_summary(path: pathlib.Path) -> list[dict]:
    """Load deterministic-eval rows from the eval_summary.tsv file.

    Returns a list of dicts with at minimum: task, backbone, seed, mean_return.
    """
    if not path.exists():
        print(f"WARNING: eval_summary.tsv not found at {path}; skipping eval tables.")
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("error") or "").strip():
                continue
            try:
                row["seed_int"] = int(row["seed"])
                row["mean_return_f"] = float(row["mean_return"])
                row["std_return_f"] = float(row["std_return"]) if row.get("std_return") else float("nan")
                row["final_train_score_f"] = (
                    float(row["final_train_score"])
                    if (row.get("final_train_score") or "").strip()
                    else float("nan")
                )
            except (TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def load_eval_summaries(paths: Iterable[pathlib.Path]) -> list[dict]:
    """Load multiple eval_summary.tsv files and dedupe by task/backbone/seed.

    Later files win. This lets the supplemental eval file fill the missing
    S3M/S5 rows and lets the A100 Transformer eval join the main checkpoint
    report without rewriting the original eval artifacts.
    """
    rows_by_key = {}
    for path in paths:
        for row in load_eval_summary(path):
            key = (row["task"], row["backbone"], row["seed_int"])
            rows_by_key[key] = row
    return [
        rows_by_key[k]
        for k in sorted(rows_by_key, key=lambda x: (TASKS.index(x[0]) if x[0] in TASKS else 999, x[1], x[2]))
    ]


def discover_completed_runs(root: pathlib.Path) -> list[dict]:
    """Walk full_runs/ for run_metadata.json files where status == 'completed'."""
    runs = []
    for meta_path in root.rglob("run_metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        if meta.get("status") != "completed":
            continue
        runs.append({
            "meta_path": meta_path,
            "run_dir": meta_path.parent,
            "task": meta["env_name"],
            "backbone": meta["model_backbone"],
            "seed": int(meta.get("seed", -1)),
            "elapsed_seconds": meta.get("elapsed_seconds"),
            "trainer_steps": meta.get("trainer_steps"),
        })
    runs.sort(key=lambda r: (r["task"], r["backbone"], r["seed"]))
    return runs


def load_metrics_jsonl(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream metrics.jsonl, return:
    - episode_steps, episode_scores: (N_episode_logs,) arrays
    - fps_steps, fps_values: (N_fps_logs,) arrays
    """
    ep_steps, ep_scores = [], []
    fps_steps, fps_values = [], []
    if not path.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step")
            if step is None:
                continue
            if "episode/score" in rec and _is_finite(rec["episode/score"]):
                ep_steps.append(int(step))
                ep_scores.append(float(rec["episode/score"]))
            if "fps/fps" in rec and _is_finite(rec["fps/fps"]):
                fps_steps.append(int(step))
                fps_values.append(float(rec["fps/fps"]))
    return (
        np.asarray(ep_steps, dtype=np.int64),
        np.asarray(ep_scores, dtype=np.float64),
        np.asarray(fps_steps, dtype=np.int64),
        np.asarray(fps_values, dtype=np.float64),
    )


# ------------------------------------------------------------------ tables
def _fmt(value, digits: int = 4) -> str:
    if not _is_finite(value):
        return "—"
    return f"{value:.{digits}f}"


def _write_csv(path: pathlib.Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _write_md_table(path: pathlib.Path, header: list[str], rows: list[list], title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if title:
            f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(str(h) for h in header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join("" if v is None else str(v) for v in r) + " |\n")


# ------------------------------------------------------------- output blocks
def write_coverage_table(eval_rows: list[dict], runs: list[dict]) -> None:
    """task × backbone seed-completion grid (training + eval)."""
    train_seeds = defaultdict(set)
    for r in runs:
        train_seeds[(r["task"], r["backbone"])].add(r["seed"])
    eval_seeds = defaultdict(set)
    for row in eval_rows:
        eval_seeds[(row["task"], row["backbone"])].add(row["seed_int"])

    header = ["task", "backbone", "completed_train_seeds", "evaluated_seeds", "missing_train", "missing_eval"]
    rows = []
    for task in TASKS:
        for backbone in BACKBONES:
            expected_seeds = EXPECTED_SEEDS.get(backbone, {0, 1, 2, 3, 4})
            t = train_seeds[(task, backbone)]
            e = eval_seeds[(task, backbone)]
            rows.append([
                TASK_LABELS[task],
                BACKBONE_LABELS[backbone],
                f"{len(t)}/{len(expected_seeds)}: {sorted(t)}",
                f"{len(e)}/{len(t) if t else 0}: {sorted(e)}",
                f"{sorted(expected_seeds - t)}" if (expected_seeds - t) else "—",
                f"{sorted(t - e)}" if (t - e) else "—",
            ])
    _write_csv(OUT_DIR / "coverage_table.csv", header, rows)
    _write_md_table(
        OUT_DIR / "coverage_table.md", header, rows,
        title="Coverage: training and deterministic-eval seeds per task × backbone",
    )


def aggregate_eval(eval_rows: list[dict]) -> dict:
    """{(task, backbone): {'returns': [...], 'mean': ..., 'sem': ..., 'lo': ..., 'hi': ..., 'n': ...}}"""
    by_cell = defaultdict(list)
    for row in eval_rows:
        by_cell[(row["task"], row["backbone"])].append(row["mean_return_f"])
    out = {}
    for k, vs in by_cell.items():
        arr = np.asarray(vs, dtype=np.float64)
        mean, lo, hi = _bootstrap_mean_ci(arr)
        out[k] = {
            "returns": arr,
            "mean": mean,
            "sem": _sem(arr),
            "lo": lo,
            "hi": hi,
            "n": int(arr.size),
            "iqm": _iqm(arr),
        }
    return out


def write_final_eval_table(agg: dict) -> None:
    header = ["task", "backbone", "n_seeds", "mean_eval_return", "std", "sem", "ci95_lo", "ci95_hi", "iqm"]
    rows = []
    for task in TASKS:
        for backbone in BACKBONES:
            cell = agg.get((task, backbone))
            if cell is None or cell["n"] == 0:
                rows.append([TASK_LABELS[task], BACKBONE_LABELS[backbone], 0, "—", "—", "—", "—", "—", "—"])
                continue
            std = float(cell["returns"].std(ddof=1)) if cell["n"] > 1 else float("nan")
            rows.append([
                TASK_LABELS[task], BACKBONE_LABELS[backbone], cell["n"],
                _fmt(cell["mean"]), _fmt(std), _fmt(cell["sem"]),
                _fmt(cell["lo"]), _fmt(cell["hi"]), _fmt(cell["iqm"]),
            ])
    _write_csv(OUT_DIR / "final_eval_table.csv", header, rows)
    _write_md_table(
        OUT_DIR / "final_eval_table.md", header, rows,
        title="Final deterministic eval (20 episodes/checkpoint, argmax policy). Higher is better.",
    )


def collect_train_finals(runs: list[dict]) -> dict:
    """{(task, backbone): np.ndarray of last episode/score per seed}."""
    by_cell = defaultdict(list)
    for r in runs:
        ep_steps, ep_scores, _, _ = load_metrics_jsonl(r["run_dir"] / "metrics.jsonl")
        if ep_scores.size == 0:
            continue
        # Take the last log entry as "final train score" (matches eval_summary.tsv's column).
        by_cell[(r["task"], r["backbone"])].append(float(ep_scores[-1]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in by_cell.items()}


def write_final_train_table(train_finals: dict) -> None:
    header = ["task", "backbone", "n_seeds", "mean_final_train_score", "std", "sem", "ci95_lo", "ci95_hi", "iqm"]
    rows = []
    for task in TASKS:
        for backbone in BACKBONES:
            arr = train_finals.get((task, backbone), np.array([]))
            if arr.size == 0:
                rows.append([TASK_LABELS[task], BACKBONE_LABELS[backbone], 0, "—", "—", "—", "—", "—", "—"])
                continue
            mean, lo, hi = _bootstrap_mean_ci(arr)
            std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
            rows.append([
                TASK_LABELS[task], BACKBONE_LABELS[backbone], arr.size,
                _fmt(mean), _fmt(std), _fmt(_sem(arr)),
                _fmt(lo), _fmt(hi), _fmt(_iqm(arr)),
            ])
    _write_csv(OUT_DIR / "final_train_table.csv", header, rows)
    _write_md_table(
        OUT_DIR / "final_train_table.md", header, rows,
        title="Final training episode score (last metrics.jsonl entry). Higher is better.",
    )


def write_aggregate_backbone_table(eval_agg: dict, train_finals: dict, fps_by_cell: dict) -> None:
    """Macro-average across the 3 difficulties for each backbone."""
    header = [
        "backbone", "n_tasks_eval", "macro_mean_eval_return",
        "n_tasks_train", "macro_mean_final_train", "macro_iqm_eval",
        "median_fps_steady_state", "n_completed_seeds_total",
    ]
    rows = []
    for backbone in BACKBONES:
        eval_means = [eval_agg[(t, backbone)]["mean"] for t in TASKS if (t, backbone) in eval_agg and eval_agg[(t, backbone)]["n"] > 0]
        eval_iqms = [eval_agg[(t, backbone)]["iqm"] for t in TASKS if (t, backbone) in eval_agg and eval_agg[(t, backbone)]["n"] > 0]
        train_means = [
            float(train_finals[(t, backbone)].mean())
            for t in TASKS
            if (t, backbone) in train_finals and train_finals[(t, backbone)].size > 0
        ]
        seed_count = sum(
            train_finals[(t, backbone)].size
            for t in TASKS
            if (t, backbone) in train_finals
        )
        # Median fps across all seeds × all tasks for this backbone.
        all_fps = []
        for t in TASKS:
            all_fps.extend(fps_by_cell.get((t, backbone), []))
        rows.append([
            BACKBONE_LABELS[backbone],
            len(eval_means),
            _fmt(float(np.mean(eval_means))) if eval_means else "—",
            len(train_means),
            _fmt(float(np.mean(train_means))) if train_means else "—",
            _fmt(float(np.mean(eval_iqms))) if eval_iqms else "—",
            _fmt(float(np.median(all_fps)), digits=1) if all_fps else "—",
            seed_count,
        ])
    _write_csv(OUT_DIR / "aggregate_backbone_table.csv", header, rows)
    _write_md_table(
        OUT_DIR / "aggregate_backbone_table.md", header, rows,
        title="Per-backbone macro-average across Easy / Medium / Hard. Scores are raw POPGym returns (higher = better).",
    )


def write_pairwise_vs_gru(train_finals: dict) -> None:
    """For each (task, non-GRU backbone) compare final-train scores against GRU.

    P(backbone > GRU) is the empirical fraction over all (s, s') seed pairs.
    Δ mean is the unmatched-mean difference.
    """
    header = ["task", "backbone", "n_paired_seeds", "mean_diff", "P_backbone_gt_gru", "ci95_lo_diff", "ci95_hi_diff"]
    rows = []
    for task in TASKS:
        gru_arr = train_finals.get((task, "gru"), np.array([]))
        for backbone in [b for b in BACKBONES if b != "gru"]:
            arr = train_finals.get((task, backbone), np.array([]))
            if gru_arr.size == 0 or arr.size == 0:
                rows.append([TASK_LABELS[task], BACKBONE_LABELS[backbone], 0, "—", "—", "—", "—"])
                continue
            # Cartesian comparison: for each (gru_i, bb_j) pair, count bb_j > gru_i.
            diff_grid = arr[:, None] - gru_arr[None, :]
            p_better = float((diff_grid > 0).mean())
            mean_diff = float(diff_grid.mean())
            flat = diff_grid.flatten()
            mean_diff2, lo, hi = _bootstrap_mean_ci(flat)
            rows.append([
                TASK_LABELS[task], BACKBONE_LABELS[backbone],
                int(min(arr.size, gru_arr.size)),
                _fmt(mean_diff), _fmt(p_better, digits=3),
                _fmt(lo), _fmt(hi),
            ])
    _write_csv(OUT_DIR / "pairwise_vs_gru.csv", header, rows)
    _write_md_table(
        OUT_DIR / "pairwise_vs_gru.md", header, rows,
        title=("Pairwise comparison vs GRU on final training scores. "
               "P(backbone > GRU) is the empirical fraction over all (gru_seed, backbone_seed) pairs. "
               "Bootstrap CI is over per-pair differences. NOT a significance test."),
    )


# --------------------------------------------------- learning curves & AUC
def aggregate_learning_curves(runs: list[dict],
                              n_grid: int = 200,
                              max_step: int = TARGET_STEPS) -> dict:
    """For each (task, backbone), align per-seed scores onto a common step grid via interpolation,
    then compute mean ± SEM at each grid point. Also return per-seed AUC at TARGET_STEPS.

    Returns:
        curves[(task, backbone)] = {
            'grid': (n_grid,) step grid in env steps,
            'mean': (n_grid,),
            'sem':  (n_grid,),
            'per_seed': dict[seed -> (n_grid,)] smoothed scores,
            'auc_per_seed': dict[seed -> float],
            'n_seeds': int,
        }
    """
    by_cell = defaultdict(list)  # (task, backbone) -> [(steps_arr, smoothed_scores_arr, seed)]
    for r in runs:
        ep_steps, ep_scores, _, _ = load_metrics_jsonl(r["run_dir"] / "metrics.jsonl")
        if ep_scores.size == 0:
            continue
        smoothed = _rolling_mean(ep_scores, SMOOTH_WINDOW)
        by_cell[(r["task"], r["backbone"])].append((ep_steps, smoothed, r["seed"]))

    grid = np.linspace(0, max_step, n_grid)
    out = {}
    for (task, backbone), entries in by_cell.items():
        per_seed = {}
        auc_per_seed = {}
        stack = []
        for steps, smoothed, seed in entries:
            # Some early entries may be NaN due to the rolling window; drop NaNs.
            mask = np.isfinite(smoothed)
            if mask.sum() < 2:
                continue
            interp = np.interp(grid, steps[mask], smoothed[mask])
            per_seed[seed] = interp
            stack.append(interp)
            # AUC via trapezoidal rule over the same grid (constant intervals).
            auc_per_seed[seed] = float(np.trapezoid(interp, x=grid)) if hasattr(np, "trapezoid") else float(np.trapz(interp, x=grid))
        if not stack:
            continue
        m = np.stack(stack, axis=0)
        n = m.shape[0]
        mean = m.mean(axis=0)
        sem = m.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros_like(mean)
        out[(task, backbone)] = {
            "grid": grid,
            "mean": mean,
            "sem": sem,
            "per_seed": per_seed,
            "auc_per_seed": auc_per_seed,
            "n_seeds": n,
        }
    return out


def write_sample_efficiency_auc(curves: dict) -> dict:
    """AUC table + bar plot. Returns AUC dict for reuse."""
    header = ["task", "backbone", "n_seeds", "mean_auc", "sem_auc", "ci95_lo", "ci95_hi"]
    rows = []
    auc_by_cell = {}
    for task in TASKS:
        for backbone in BACKBONES:
            cell = curves.get((task, backbone))
            if cell is None:
                rows.append([TASK_LABELS[task], BACKBONE_LABELS[backbone], 0, "—", "—", "—", "—"])
                continue
            auc_arr = np.asarray(list(cell["auc_per_seed"].values()), dtype=np.float64)
            auc_by_cell[(task, backbone)] = auc_arr
            mean, lo, hi = _bootstrap_mean_ci(auc_arr)
            rows.append([
                TASK_LABELS[task], BACKBONE_LABELS[backbone], auc_arr.size,
                _fmt(mean, digits=2), _fmt(_sem(auc_arr), digits=2),
                _fmt(lo, digits=2), _fmt(hi, digits=2),
            ])
    _write_csv(OUT_DIR / "sample_efficiency_auc.csv", header, rows)
    _write_md_table(
        OUT_DIR / "sample_efficiency_auc.md", header, rows,
        title=(f"Sample efficiency: area under the smoothed (window={SMOOTH_WINDOW}) "
               f"learning curve, integrated over [0, {TARGET_STEPS:,}] env steps. "
               "Higher = learns faster + ends higher."),
    )
    return auc_by_cell


# -------------------------------------------------------------- compute fps
def collect_fps(runs: list[dict]) -> dict:
    """{(task, backbone): list of per-step fps values across all seeds}."""
    by_cell = defaultdict(list)
    for r in runs:
        _, _, fps_steps, fps_values = load_metrics_jsonl(r["run_dir"] / "metrics.jsonl")
        if fps_values.size == 0:
            continue
        # Use the second half of training as "steady state" to skip warm-up.
        n = fps_values.size
        steady = fps_values[n // 2:]
        by_cell[(r["task"], r["backbone"])].extend(steady.tolist())
    return by_cell


def write_compute_efficiency(fps_by_cell: dict) -> None:
    header = ["task", "backbone", "n_obs", "median_fps", "p25_fps", "p75_fps"]
    rows = []
    for task in TASKS:
        for backbone in BACKBONES:
            vals = np.asarray(fps_by_cell.get((task, backbone), []), dtype=np.float64)
            if vals.size == 0:
                rows.append([TASK_LABELS[task], BACKBONE_LABELS[backbone], 0, "—", "—", "—"])
                continue
            q25, q50, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
            rows.append([
                TASK_LABELS[task], BACKBONE_LABELS[backbone], vals.size,
                _fmt(q50, digits=1), _fmt(q25, digits=1), _fmt(q75, digits=1),
            ])
    _write_csv(OUT_DIR / "compute_efficiency.csv", header, rows)
    _write_md_table(
        OUT_DIR / "compute_efficiency.md", header, rows,
        title=("Steady-state throughput (median fps from metrics.jsonl, second half of training). "
               "Higher = faster wall-clock training."),
    )


# ------------------------------------------------------------------ plots
def _save_fig(fig, basename: str) -> list[pathlib.Path]:
    paths = []
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"{basename}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _setup_style():
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 100,
        "savefig.dpi": 300,
    })


def plot_grouped_bars(agg_or_finals: dict,
                      kind: str,
                      title: str,
                      ylabel: str,
                      basename: str) -> None:
    """Grouped bar chart: x = tasks, group = backbones. agg_or_finals is the per-cell dict."""
    n_tasks = len(TASKS)
    n_bb = len(BACKBONES)
    width = 0.78 / n_bb
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for j, backbone in enumerate(BACKBONES):
        means, errs_lo, errs_hi, ns = [], [], [], []
        for task in TASKS:
            if kind == "eval":
                cell = agg_or_finals.get((task, backbone))
                if cell is None or cell["n"] == 0:
                    means.append(np.nan); errs_lo.append(0); errs_hi.append(0); ns.append(0)
                    continue
                m = cell["mean"]
                # Use bootstrap CI half-widths if available; else use SEM.
                if _is_finite(cell["lo"]) and _is_finite(cell["hi"]):
                    lo_err = max(0.0, m - cell["lo"]); hi_err = max(0.0, cell["hi"] - m)
                else:
                    sem = cell["sem"] if _is_finite(cell["sem"]) else 0.0
                    lo_err = sem; hi_err = sem
                means.append(m); errs_lo.append(lo_err); errs_hi.append(hi_err); ns.append(cell["n"])
            elif kind == "train":
                arr = agg_or_finals.get((task, backbone), np.array([]))
                if arr.size == 0:
                    means.append(np.nan); errs_lo.append(0); errs_hi.append(0); ns.append(0); continue
                mean, lo, hi = _bootstrap_mean_ci(arr)
                means.append(mean)
                errs_lo.append(max(0.0, mean - lo) if _is_finite(lo) else 0.0)
                errs_hi.append(max(0.0, hi - mean) if _is_finite(hi) else 0.0)
                ns.append(arr.size)
            else:
                raise ValueError(f"unknown kind: {kind}")
        means = np.asarray(means)
        offsets = (j - (n_bb - 1) / 2.0) * width
        valid = np.isfinite(means)
        bar_x = x + offsets
        ax.bar(
            bar_x[valid], means[valid], width=width,
            color=BACKBONE_COLORS[backbone], label=BACKBONE_LABELS[backbone],
            edgecolor="black", linewidth=0.6,
        )
        # Error bars only on valid bars.
        if valid.any():
            ax.errorbar(
                bar_x[valid], means[valid],
                yerr=[np.asarray(errs_lo)[valid], np.asarray(errs_hi)[valid]],
                fmt="none", ecolor="black", capsize=3, linewidth=1,
            )
        # Annotate bar with n.
        for xi, m, n in zip(bar_x, means, ns):
            if not np.isfinite(m):
                ax.text(xi, 0, "n/a", ha="center", va="bottom", fontsize=9, color="grey")
            else:
                # Place label just above the bar (or below if negative).
                vshift = 0.01 * (max(np.nanmax(np.abs(means)), 1.0))
                y = m + (vshift if m >= 0 else -vshift)
                va = "bottom" if m >= 0 else "top"
                ax.text(xi, y, f"n={n}", ha="center", va=va, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS])
    ax.set_xlabel("RepeatPrevious difficulty")
    ax.set_ylabel(ylabel + "  (higher is better)")
    ax.set_title(title)
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="-")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _save_fig(fig, basename)


def plot_learning_curves(curves: dict) -> None:
    """One PNG/PDF per task. Mean ± SEM band over seeds."""
    for task in TASKS:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        any_data = False
        for backbone in BACKBONES:
            cell = curves.get((task, backbone))
            if cell is None:
                continue
            grid = cell["grid"]
            mean = cell["mean"]
            sem = cell["sem"]
            color = BACKBONE_COLORS[backbone]
            label = f"{BACKBONE_LABELS[backbone]} (n={cell['n_seeds']})"
            ax.plot(grid, mean, color=color, label=label, linewidth=2.0)
            ax.fill_between(grid, mean - sem, mean + sem, color=color, alpha=0.20)
            any_data = True
        if not any_data:
            plt.close(fig); continue
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(f"Episode score (rolling mean, window={SMOOTH_WINDOW})  — higher is better")
        ax.set_title(f"Learning curve: POPGym RepeatPrevious{TASK_LABELS[task]}")
        ax.legend(loc="best", frameon=True)
        ax.set_xlim(0, TARGET_STEPS)
        # Use scientific or formatted x ticks: 0, 250k, 500k, 750k, 1M.
        ax.set_xticks([0, 250_000, 500_000, 750_000, 1_000_000])
        ax.set_xticklabels(["0", "250k", "500k", "750k", "1M"])
        fig.tight_layout()
        _save_fig(fig, f"learning_curve_{TASK_LABELS[task]}")


def plot_sample_efficiency_auc(auc_by_cell: dict) -> None:
    n_tasks = len(TASKS); n_bb = len(BACKBONES)
    width = 0.78 / n_bb
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for j, backbone in enumerate(BACKBONES):
        means, errs_lo, errs_hi, ns = [], [], [], []
        for task in TASKS:
            arr = auc_by_cell.get((task, backbone), np.array([]))
            if arr.size == 0:
                means.append(np.nan); errs_lo.append(0); errs_hi.append(0); ns.append(0); continue
            mean, lo, hi = _bootstrap_mean_ci(arr)
            means.append(mean)
            errs_lo.append(max(0.0, mean - lo) if _is_finite(lo) else 0.0)
            errs_hi.append(max(0.0, hi - mean) if _is_finite(hi) else 0.0)
            ns.append(arr.size)
        means = np.asarray(means)
        offsets = (j - (n_bb - 1) / 2.0) * width
        valid = np.isfinite(means)
        bar_x = x + offsets
        ax.bar(
            bar_x[valid], means[valid], width=width,
            color=BACKBONE_COLORS[backbone], label=BACKBONE_LABELS[backbone],
            edgecolor="black", linewidth=0.6,
        )
        if valid.any():
            ax.errorbar(
                bar_x[valid], means[valid],
                yerr=[np.asarray(errs_lo)[valid], np.asarray(errs_hi)[valid]],
                fmt="none", ecolor="black", capsize=3, linewidth=1,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS])
    ax.set_xlabel("RepeatPrevious difficulty")
    ax.set_ylabel(f"AUC of smoothed learning curve over [0, {TARGET_STEPS:,}] env steps  (higher is better)")
    ax.set_title("Sample efficiency: AUC of learning curve")
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="-")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _save_fig(fig, "sample_efficiency_auc")


def plot_compute_efficiency(fps_by_cell: dict) -> None:
    n_tasks = len(TASKS); n_bb = len(BACKBONES)
    width = 0.78 / n_bb
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for j, backbone in enumerate(BACKBONES):
        med, lo_err, hi_err, ns = [], [], [], []
        for task in TASKS:
            vals = np.asarray(fps_by_cell.get((task, backbone), []), dtype=np.float64)
            if vals.size == 0:
                med.append(np.nan); lo_err.append(0); hi_err.append(0); ns.append(0); continue
            q25, q50, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
            med.append(q50); lo_err.append(q50 - q25); hi_err.append(q75 - q50); ns.append(vals.size)
        med = np.asarray(med)
        offsets = (j - (n_bb - 1) / 2.0) * width
        valid = np.isfinite(med)
        bar_x = x + offsets
        ax.bar(
            bar_x[valid], med[valid], width=width,
            color=BACKBONE_COLORS[backbone], label=BACKBONE_LABELS[backbone],
            edgecolor="black", linewidth=0.6,
        )
        if valid.any():
            ax.errorbar(
                bar_x[valid], med[valid],
                yerr=[np.asarray(lo_err)[valid], np.asarray(hi_err)[valid]],
                fmt="none", ecolor="black", capsize=3, linewidth=1,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS])
    ax.set_xlabel("RepeatPrevious difficulty")
    ax.set_ylabel("Steady-state fps (median over second half of training; IQR error bars)")
    ax.set_title("Compute efficiency: steady-state training throughput  (higher is better)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _save_fig(fig, "compute_efficiency_fps")


# ------------------------------------------------------------------- main
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()

    print(f"# Loading inputs from {CHECKPOINT_ROOT}")
    eval_rows = load_eval_summaries(EVAL_TSVS)
    run_by_key = {}
    for root in FULL_RUNS_ROOTS:
        for run in discover_completed_runs(root):
            if run["backbone"] not in BACKBONES:
                continue
            key = (run["task"], run["backbone"], run["seed"])
            run_by_key[key] = run
    runs = [
        run_by_key[k]
        for k in sorted(run_by_key, key=lambda x: (TASKS.index(x[0]) if x[0] in TASKS else 999, BACKBONES.index(x[1]), x[2]))
    ]
    print(f"#   eval rows: {len(eval_rows)}")
    print(f"#   completed runs ({', '.join(BACKBONES)}): {len(runs)}")

    print("# Building tables...")
    write_coverage_table(eval_rows, runs)
    eval_agg = aggregate_eval(eval_rows)
    write_final_eval_table(eval_agg)

    train_finals = collect_train_finals(runs)
    write_final_train_table(train_finals)

    fps_by_cell = collect_fps(runs)
    write_compute_efficiency(fps_by_cell)

    write_aggregate_backbone_table(eval_agg, train_finals, fps_by_cell)
    write_pairwise_vs_gru(train_finals)

    print("# Aggregating learning curves...")
    curves = aggregate_learning_curves(runs)
    auc_by_cell = write_sample_efficiency_auc(curves)

    print("# Plotting...")
    plot_grouped_bars(
        eval_agg, kind="eval",
        title="Final Deterministic Evaluation on POPGym RepeatPrevious",
        ylabel="Eval return  (20-ep argmax)",
        basename="final_eval_by_task",
    )
    plot_grouped_bars(
        train_finals, kind="train",
        title="Final Training Score on POPGym RepeatPrevious",
        ylabel="Final training score  (last episode/score)",
        basename="final_train_by_task",
    )
    plot_learning_curves(curves)
    plot_sample_efficiency_auc(auc_by_cell)
    plot_compute_efficiency(fps_by_cell)

    print(f"\n# Done. Outputs under: {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size:>8,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
