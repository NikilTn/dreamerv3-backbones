#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ATARI_BASELINES = {
    "alien": (227.8, 7127.7),
    "amidar": (5.8, 1719.5),
    "assault": (222.4, 742.0),
    "asterix": (210.0, 8503.3),
    "bank_heist": (14.2, 753.1),
    "battle_zone": (2360.0, 37187.5),
    "boxing": (0.1, 12.1),
    "breakout": (1.7, 30.5),
    "chopper_command": (811.0, 7387.8),
    "crazy_climber": (10780.5, 35829.4),
    "demon_attack": (152.1, 1971.0),
    "freeway": (0.0, 29.6),
    "frostbite": (65.2, 4334.7),
    "gopher": (257.6, 2412.5),
    "hero": (1027.0, 30826.4),
    "jamesbond": (29.0, 302.8),
    "kangaroo": (52.0, 3035.0),
    "krull": (1598.0, 2665.5),
    "kung_fu_master": (258.5, 22736.3),
    "ms_pacman": (307.3, 6951.6),
    "pong": (-20.7, 14.6),
    "private_eye": (24.9, 69571.3),
    "qbert": (163.9, 13455.0),
    "road_runner": (11.5, 7845.0),
    "seaquest": (68.4, 42054.7),
    "up_n_down": (533.4, 11693.2),
}

FIELD_ORDER = [
    "run_id",
    "status",
    "suite",
    "subexp",
    "task",
    "backbone",
    "seed",
    "final_score_raw",
    "final_score_suite",
    "final_eval_score_raw",
    "final_train_score_raw",
    "elapsed_seconds",
    "trainer_steps",
    "seconds_per_env_step",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate proposal benchmark runs and generate summaries.")
    parser.add_argument("--logdir-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--profile-points", type=int, default=101)
    return parser.parse_args()


def load_json(path: Path, default=None):
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {} if default is None else default


def load_run_config(run_dir: Path):
    config = load_json(run_dir / "resolved_config.json", default=None)
    if config:
        return config
    hydra_path = run_dir / ".hydra" / "config.yaml"
    if hydra_path.exists():
        try:
            from omegaconf import OmegaConf

            return OmegaConf.to_container(OmegaConf.load(hydra_path), resolve=True)
        except Exception:
            return {}
    return {}


def read_metrics(path: Path):
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def infer_suite(task: str) -> str:
    if task.startswith("atari_"):
        return "atari100k"
    if task.startswith("bsuite_"):
        return "bsuite"
    if task.startswith("popgym_"):
        return "popgym"
    return task.split("_", 1)[0]


def canonical_atari_task(task: str) -> str:
    task = task.removeprefix("atari_")
    if task == "james_bond":
        return "jamesbond"
    return task


def suite_score(task: str, raw_score: float):
    if raw_score is None or math.isnan(raw_score):
        return math.nan
    if task.startswith("atari_"):
        random_score, human_score = ATARI_BASELINES[canonical_atari_task(task)]
        return (raw_score - random_score) / (human_score - random_score)
    return raw_score


def last_metric(entries, key):
    values = [entry[key] for entry in entries if key in entry]
    return values[-1] if values else None


def iqm(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    arr = np.sort(arr)
    trim = int(math.floor(0.25 * arr.size))
    if trim * 2 >= arr.size:
        return float(arr.mean())
    return float(arr[trim : arr.size - trim].mean())


def mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def median(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else math.nan


def std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else math.nan
    return float(arr.std(ddof=1))


def bootstrap_ci(values, stat_fn, samples, rng):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    stats = []
    for _ in range(samples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats.append(stat_fn(sample))
    return tuple(np.percentile(stats, [2.5, 97.5]))


def bootstrap_suite_ci(task_to_scores, stat_fn, samples, rng):
    tasks = sorted(task_to_scores.keys())
    if not tasks:
        return math.nan, math.nan
    if len(tasks) == 1 and len(task_to_scores[tasks[0]]) == 1:
        value = stat_fn(task_to_scores[tasks[0]])
        return value, value
    stats = []
    for _ in range(samples):
        sampled = []
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        for task in sampled_tasks:
            scores = np.asarray(task_to_scores[task], dtype=float)
            scores = scores[np.isfinite(scores)]
            if scores.size == 0:
                continue
            sampled.extend(rng.choice(scores, size=scores.size, replace=True).tolist())
        stats.append(stat_fn(sampled))
    return tuple(np.percentile(stats, [2.5, 97.5]))


def probability_of_improvement(scores_a, scores_b):
    common = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if not common:
        return math.nan
    per_task = []
    for task in common:
        a = np.asarray(scores_a[task], dtype=float)
        b = np.asarray(scores_b[task], dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if not a.size or not b.size:
            continue
        per_task.append(float((a[:, None] > b[None, :]).mean()))
    return mean(per_task)


def bootstrap_probability_ci(scores_a, scores_b, samples, rng):
    common = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if not common:
        return math.nan, math.nan
    stats = []
    for _ in range(samples):
        sampled_tasks = rng.choice(common, size=len(common), replace=True)
        per_task = []
        for task in sampled_tasks:
            a = np.asarray(scores_a[task], dtype=float)
            b = np.asarray(scores_b[task], dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if not a.size or not b.size:
                continue
            a = rng.choice(a, size=a.size, replace=True)
            b = rng.choice(b, size=b.size, replace=True)
            per_task.append(float((a[:, None] > b[None, :]).mean()))
        stats.append(mean(per_task))
    return tuple(np.percentile(stats, [2.5, 97.5]))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(curves, output_dir: Path):
    grouped = defaultdict(list)
    for row in curves:
        if row["metric"] != "episode/eval_score":
            continue
        grouped[(row["suite"], row["subexp"], row["task"], row["backbone"], row["step"])].append(row["score_suite"])
    task_groups = defaultdict(lambda: defaultdict(list))
    for (suite, subexp, task, backbone, step), scores in grouped.items():
        task_groups[(suite, subexp, task)][backbone].append(
            {"step": step, "score_suite": float(np.mean(scores)), "score_std": float(np.std(scores))}
        )
    for (suite, subexp, task), by_backbone in task_groups.items():
        plt.figure(figsize=(8, 5))
        for backbone, rows in sorted(by_backbone.items()):
            rows = sorted(rows, key=lambda x: x["step"])
            steps = np.array([row["step"] for row in rows], dtype=float)
            values = np.array([row["score_suite"] for row in rows], dtype=float)
            stds = np.array([row["score_std"] for row in rows], dtype=float)
            plt.plot(steps, values, label=backbone)
            plt.fill_between(steps, values - stds, values + stds, alpha=0.15)
        plt.xlabel("Environment Steps")
        plt.ylabel("Suite Score")
        plt.title(f"{suite} [{subexp}]: {task}")
        plt.legend()
        plt.tight_layout()
        path = output_dir / "plots" / "learning_curves" / f"{suite}_{subexp}_{task.replace('/', '_')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()


def plot_performance_profiles(profiles, output_dir: Path):
    grouped = defaultdict(list)
    for row in profiles:
        grouped[(row["suite"], row["subexp"], row["backbone"])].append(row)
    suite_groups = defaultdict(dict)
    for (suite, subexp, backbone), rows in grouped.items():
        suite_groups[(suite, subexp)][backbone] = sorted(rows, key=lambda x: x["threshold"])
    for (suite, subexp), by_backbone in suite_groups.items():
        plt.figure(figsize=(8, 5))
        for backbone, rows in sorted(by_backbone.items()):
            x = np.array([row["threshold"] for row in rows], dtype=float)
            y = np.array([row["profile"] for row in rows], dtype=float)
            plt.plot(x, y, label=backbone)
        plt.xlabel("Score Threshold")
        plt.ylabel("Probability of Exceeding Threshold")
        plt.title(f"Performance Profile: {suite} [{subexp}]")
        plt.legend()
        plt.tight_layout()
        path = output_dir / "plots" / "performance_profiles" / f"{suite}_{subexp}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()


def write_summary_markdown(path: Path, suite_rows):
    lines = ["# Proposal Results Summary", ""]
    for suite, subexp in sorted({(row["suite"], row["subexp"]) for row in suite_rows}):
        lines.append(f"## {suite} [{subexp}]")
        lines.append("")
        lines.append("| Backbone | Mean | Median | IQM | 95% CI (IQM) | Count |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in sorted(
            [x for x in suite_rows if x["suite"] == suite and x["subexp"] == subexp], key=lambda x: x["backbone"]
        ):
            lines.append(
                f"| {row['backbone']} | {row['mean']:.4f} | {row['median']:.4f} | {row['iqm']:.4f} | "
                f"[{row['iqm_ci_low']:.4f}, {row['iqm_ci_high']:.4f}] | {row['count']} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    root = args.logdir_root
    output_dir = args.output_dir or (root / "analysis")
    rng = np.random.default_rng(0)

    run_rows = []
    curve_rows = []
    metric_files = sorted(root.rglob("metrics.jsonl"))
    if not metric_files:
        raise FileNotFoundError(f"No metrics.jsonl files found under {root}")

    for metrics_file in metric_files:
        run_dir = metrics_file.parent
        config = load_run_config(run_dir)
        metadata = load_json(run_dir / "run_metadata.json")
        task = metadata.get("env_name") or config.get("env", {}).get("task")
        if not task:
            continue
        suite = infer_suite(task)
        backbone = metadata.get("model_backbone") or config.get("model", {}).get("backbone", "unknown")
        seed = metadata.get("seed", config.get("seed"))
        status = metadata.get("status", "unknown")
        subexp = metadata.get("subexp", "none")
        entries = read_metrics(metrics_file)
        final_eval = last_metric(entries, "episode/eval_score")
        final_train = last_metric(entries, "episode/score")
        final_raw = final_eval if final_eval is not None else final_train
        final_suite = suite_score(task, float(final_raw)) if final_raw is not None else math.nan
        elapsed_seconds = metadata.get("elapsed_seconds")
        trainer_steps = metadata.get("trainer_steps")
        seconds_per_step = (
            float(elapsed_seconds) / float(trainer_steps) if elapsed_seconds is not None and trainer_steps else math.nan
        )
        run_id = str(run_dir.relative_to(root))
        run_rows.append(
            {
                "run_id": run_id,
                "status": status,
                "suite": suite,
                "subexp": subexp,
                "task": task,
                "backbone": backbone,
                "seed": seed,
                "final_score_raw": final_raw,
                "final_score_suite": final_suite,
                "final_eval_score_raw": final_eval,
                "final_train_score_raw": final_train,
                "elapsed_seconds": elapsed_seconds,
                "trainer_steps": trainer_steps,
                "seconds_per_env_step": seconds_per_step,
            }
        )
        for entry in entries:
            step = entry.get("step")
            for key in ("episode/eval_score", "episode/score"):
                if key in entry:
                    raw = float(entry[key])
                    curve_rows.append(
                        {
                            "run_id": run_id,
                            "suite": suite,
                            "subexp": subexp,
                            "task": task,
                            "backbone": backbone,
                            "seed": seed,
                            "metric": key,
                            "step": step,
                            "score_raw": raw,
                            "score_suite": suite_score(task, raw),
                        }
                    )

    write_csv(output_dir / "runs.csv", run_rows, FIELD_ORDER)
    write_csv(
        output_dir / "learning_curves.csv",
        curve_rows,
        ["run_id", "suite", "subexp", "task", "backbone", "seed", "metric", "step", "score_raw", "score_suite"],
    )

    task_groups = defaultdict(list)
    suite_groups = defaultdict(lambda: defaultdict(list))
    for row in run_rows:
        if not math.isfinite(row["final_score_suite"]):
            continue
        task_groups[(row["suite"], row["subexp"], row["task"], row["backbone"])].append(row["final_score_suite"])
        suite_groups[(row["suite"], row["subexp"], row["backbone"])][row["task"]].append(row["final_score_suite"])

    per_task_rows = []
    for (suite, subexp, task, backbone), scores in sorted(task_groups.items()):
        mean_ci_low, mean_ci_high = bootstrap_ci(scores, mean, args.bootstrap_samples, rng)
        iqm_ci_low, iqm_ci_high = bootstrap_ci(scores, iqm, args.bootstrap_samples, rng)
        per_task_rows.append(
            {
                "suite": suite,
                "subexp": subexp,
                "task": task,
                "backbone": backbone,
                "count": len(scores),
                "mean": mean(scores),
                "std": std(scores),
                "median": median(scores),
                "iqm": iqm(scores),
                "mean_ci_low": mean_ci_low,
                "mean_ci_high": mean_ci_high,
                "iqm_ci_low": iqm_ci_low,
                "iqm_ci_high": iqm_ci_high,
            }
        )
    write_csv(
        output_dir / "per_task_summary.csv",
        per_task_rows,
        [
            "suite",
            "subexp",
            "task",
            "backbone",
            "count",
            "mean",
            "std",
            "median",
            "iqm",
            "mean_ci_low",
            "mean_ci_high",
            "iqm_ci_low",
            "iqm_ci_high",
        ],
    )

    suite_rows = []
    for (suite, subexp, backbone), task_to_scores in sorted(suite_groups.items()):
        all_scores = [score for scores in task_to_scores.values() for score in scores]
        mean_ci_low, mean_ci_high = bootstrap_suite_ci(task_to_scores, mean, args.bootstrap_samples, rng)
        median_ci_low, median_ci_high = bootstrap_suite_ci(task_to_scores, median, args.bootstrap_samples, rng)
        iqm_ci_low, iqm_ci_high = bootstrap_suite_ci(task_to_scores, iqm, args.bootstrap_samples, rng)
        suite_rows.append(
            {
                "suite": suite,
                "subexp": subexp,
                "backbone": backbone,
                "count": len(all_scores),
                "task_count": len(task_to_scores),
                "mean": mean(all_scores),
                "std": std(all_scores),
                "median": median(all_scores),
                "iqm": iqm(all_scores),
                "mean_ci_low": mean_ci_low,
                "mean_ci_high": mean_ci_high,
                "median_ci_low": median_ci_low,
                "median_ci_high": median_ci_high,
                "iqm_ci_low": iqm_ci_low,
                "iqm_ci_high": iqm_ci_high,
                "seconds_per_env_step": mean(
                    [
                        row["seconds_per_env_step"]
                        for row in run_rows
                        if row["suite"] == suite
                        and row["subexp"] == subexp
                        and row["backbone"] == backbone
                        and math.isfinite(row["seconds_per_env_step"])
                    ]
                ),
            }
        )
    write_csv(
        output_dir / "suite_summary.csv",
        suite_rows,
        [
            "suite",
            "subexp",
            "backbone",
            "count",
            "task_count",
            "mean",
            "std",
            "median",
            "iqm",
            "mean_ci_low",
            "mean_ci_high",
            "median_ci_low",
            "median_ci_high",
            "iqm_ci_low",
            "iqm_ci_high",
            "seconds_per_env_step",
        ],
    )

    pairwise_rows = []
    grouped_suite_backbones = defaultdict(dict)
    for (suite, subexp, backbone), task_to_scores in suite_groups.items():
        grouped_suite_backbones[(suite, subexp)][backbone] = task_to_scores
    for (suite, subexp), backbones in sorted(grouped_suite_backbones.items()):
        names = sorted(backbones.keys())
        for i, name_a in enumerate(names):
            for name_b in names[i + 1 :]:
                poi = probability_of_improvement(backbones[name_a], backbones[name_b])
                ci_low, ci_high = bootstrap_probability_ci(
                    backbones[name_a], backbones[name_b], args.bootstrap_samples, rng
                )
                pairwise_rows.append(
                    {
                        "suite": suite,
                        "subexp": subexp,
                        "backbone_a": name_a,
                        "backbone_b": name_b,
                        "probability_of_improvement": poi,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    write_csv(
        output_dir / "pairwise_probability_of_improvement.csv",
        pairwise_rows,
        ["suite", "subexp", "backbone_a", "backbone_b", "probability_of_improvement", "ci_low", "ci_high"],
    )

    profile_rows = []
    for (suite, subexp), backbones in sorted(grouped_suite_backbones.items()):
        observed_scores = [
            score for task_to_scores in backbones.values() for scores in task_to_scores.values() for score in scores
        ]
        if not observed_scores:
            continue
        observed_scores = np.asarray(observed_scores, dtype=float)
        if suite == "atari100k":
            thresholds = np.linspace(0.0, max(1.5, float(np.nanmax(observed_scores))), args.profile_points)
        elif suite == "bsuite":
            thresholds = np.linspace(0.0, 1.0, args.profile_points)
        else:
            thresholds = np.linspace(float(np.nanmin(observed_scores)), float(np.nanmax(observed_scores)), args.profile_points)
        for backbone, task_to_scores in sorted(backbones.items()):
            all_scores = [score for scores in task_to_scores.values() for score in scores]
            for threshold in thresholds:
                profile_rows.append(
                    {
                        "suite": suite,
                        "subexp": subexp,
                        "backbone": backbone,
                        "threshold": float(threshold),
                        "profile": float(np.mean(np.asarray(all_scores) >= threshold)),
                    }
                )
    write_csv(
        output_dir / "performance_profiles.csv",
        profile_rows,
        ["suite", "subexp", "backbone", "threshold", "profile"],
    )

    plot_learning_curves(curve_rows, output_dir)
    plot_performance_profiles(profile_rows, output_dir)
    write_summary_markdown(output_dir / "summary.md", suite_rows)
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
