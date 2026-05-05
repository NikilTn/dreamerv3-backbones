"""Local checkpoint evaluation for DreamerV3 / R2-Dreamer POPGym checkpoints.

Loads a saved ``latest.pt`` produced by ``train.py``, rebuilds the agent from
the run's ``resolved_config.{json,yaml}``, and runs deterministic evaluation
episodes via the same ``agent.act(..., eval=True)`` path used by
``trainer.eval``.

Designed for **local sanity checks** on Mac (CPU first; MPS optional, never
default). Cluster training is not affected.

Examples::

    # Smoke: 2 episodes on the first manifest entry.
    python scripts/eval_checkpoint.py \\
        --manifest checkpoints/repeat_previous_reduced_20260504_004958/CHECKPOINT_MANIFEST.tsv \\
        --episodes 2 --max-runs 1 --device cpu

    # Dry-run: enumerate all checkpoints without loading.
    python scripts/eval_checkpoint.py \\
        --manifest checkpoints/repeat_previous_reduced_20260504_004958/CHECKPOINT_MANIFEST.tsv \\
        --dry-run

    # Full evaluation, all manifest entries, 20 episodes each.
    python scripts/eval_checkpoint.py \\
        --manifest checkpoints/repeat_previous_reduced_20260504_004958/CHECKPOINT_MANIFEST.tsv \\
        --episodes 20 --device cpu

    # Single run dir.
    python scripts/eval_checkpoint.py \\
        --run-dir checkpoints/.../full_runs/.../gru/seed0 \\
        --episodes 5 --device cpu

Outputs (default under ``<manifest_dir>/eval/``):
- ``eval_results.jsonl``: one JSON object per evaluated checkpoint.
- ``eval_summary.tsv``: TSV summary table sorted by backbone, seed.

Failure modes:
- With ``--manifest``, individual run failures are logged in the result row's
  ``error`` field and the script continues; with ``--run-dir``, errors raise.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import traceback
import warnings
from typing import Any

import torch
from omegaconf import OmegaConf, DictConfig, ListConfig, open_dict

warnings.filterwarnings("ignore")

# Allow ``from dreamer import Dreamer`` etc. when run as ``python scripts/...``.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dreamer import Dreamer  # noqa: E402
from envs import make_envs  # noqa: E402


def _replace_devices(cfg: Any, new_device: str) -> None:
    """In-place: walk an OmegaConf node and rewrite all device fields.

    Replaces values for keys named ``device`` or ``storage_device``.
    The ``resolved_config.yaml`` saved by ``tools.save_resolved_config``
    materializes ``${device}`` interpolations, so we must rewrite each
    occurrence rather than rely on a single top-level override.
    """
    if isinstance(cfg, DictConfig):
        for key in list(cfg.keys()):
            value = cfg[key]
            if key in ("device", "storage_device") and isinstance(value, str):
                cfg[key] = new_device
            elif isinstance(value, (DictConfig, ListConfig)):
                _replace_devices(value, new_device)
    elif isinstance(cfg, ListConfig):
        for i, value in enumerate(cfg):
            if isinstance(value, (DictConfig, ListConfig)):
                _replace_devices(value, new_device)


def load_resolved_config(run_dir: pathlib.Path, device: str) -> DictConfig:
    """Load resolved_config.{json,yaml} from a run directory and apply local-eval overrides.

    - Rewrites every ``device`` / ``storage_device`` field to ``device`` (e.g. "cpu").
    - Forces ``model.compile = False`` (compile is irrelevant + risky for eval).
    - Mirrors ``train.py``: ``model.burn_in = trainer.burn_in``.
    """
    json_path = run_dir / "resolved_config.json"
    yaml_path = run_dir / "resolved_config.yaml"
    if json_path.exists():
        cfg = OmegaConf.load(str(json_path))
    elif yaml_path.exists():
        cfg = OmegaConf.load(str(yaml_path))
    else:
        raise FileNotFoundError(f"No resolved_config.[json|yaml] in {run_dir}")
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {run_dir}; got {type(cfg)}")
    OmegaConf.set_struct(cfg, False)
    _replace_devices(cfg, device)
    if "model" in cfg:
        cfg.model.compile = False
    if "trainer" in cfg and "burn_in" in cfg.trainer and "model" in cfg:
        with open_dict(cfg.model):
            cfg.model.burn_in = int(cfg.trainer.burn_in)
    return cfg


def parse_manifest(manifest_path: pathlib.Path) -> list[dict[str, str]]:
    """Read TSV manifest into a list of row dicts."""
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def read_final_train_score(metrics_path: pathlib.Path) -> float | None:
    """Best-effort: return the last ``episode/score`` value in metrics.jsonl.

    Returns ``None`` if the file is missing or contains no such key.
    """
    if not metrics_path.exists():
        return None
    last_score: float | None = None
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "episode/score" in rec:
                try:
                    last_score = float(rec["episode/score"])
                except (TypeError, ValueError):
                    pass
    return last_score


def _close_envs(envs) -> None:
    """Close each underlying ``Parallel`` worker. ``ParallelEnv`` itself
    has no ``close`` method but exposes ``envs: list[Parallel]``."""
    for env in getattr(envs, "envs", []):
        try:
            env.close()
        except Exception:
            pass


def run_eval_episodes(
    agent: Dreamer,
    eval_envs,
    target_episodes: int,
    env_num: int,
    *,
    max_steps_safety: int = 100_000,
) -> tuple[list[float], list[int]]:
    """Roll out `target_episodes` deterministic evaluation episodes.

    Mirrors ``trainer.OnlineTrainer.eval``: spawn ``env_num`` parallel envs,
    loop until each has produced one episode (``once_done.all()``), record
    per-env return and length. To get more than ``env_num`` episodes, we
    re-run that loop in waves and reset agent state between waves.

    Returns ``(returns, lengths)`` lists of length ``target_episodes``.
    """
    returns: list[float] = []
    lengths: list[int] = []
    device = agent.device

    # Decide wave sizes. Each wave runs the full eval loop over `env_num` envs;
    # a partial last wave keeps only the first `wave_size` envs' results.
    waves: list[int] = []
    remaining = target_episodes
    while remaining > 0:
        waves.append(min(env_num, remaining))
        remaining -= min(env_num, remaining)

    for wave_idx, wave_size in enumerate(waves):
        # Mirror trainer.eval bookkeeping.
        agent.eval()
        done = torch.ones(env_num, dtype=torch.bool, device=device)
        once_done = torch.zeros(env_num, dtype=torch.bool, device=device)
        steps = torch.zeros(env_num, dtype=torch.int32, device=device)
        wave_returns = torch.zeros(env_num, dtype=torch.float32, device=device)
        agent_state = agent.get_initial_state(env_num)
        act = agent_state["prev_action"].clone()

        with torch.no_grad():
            safety = 0
            while not bool(once_done.all().item()):
                steps += (~done & ~once_done).to(steps.dtype)
                act_cpu = act.detach().to("cpu")
                done_cpu = done.detach().to("cpu")
                trans_cpu, done_cpu = eval_envs.step(act_cpu, done_cpu)
                trans = trans_cpu.to(device, non_blocking=False)
                done = done_cpu.to(device)
                trans["action"] = act
                act, agent_state = agent.act(trans, agent_state, eval=True)
                wave_returns += trans["reward"][:, 0] * (~once_done).to(trans["reward"].dtype)
                once_done = once_done | done
                safety += 1
                if safety > max_steps_safety:
                    raise RuntimeError(
                        f"Eval loop exceeded {max_steps_safety} env steps without all envs "
                        f"producing an episode (wave {wave_idx + 1}/{len(waves)}, "
                        f"once_done={once_done.tolist()})."
                    )

        returns.extend(wave_returns[:wave_size].cpu().tolist())
        lengths.extend(steps[:wave_size].cpu().tolist())

    return returns, lengths


def evaluate_run(
    run_dir: pathlib.Path,
    *,
    episodes: int,
    env_num_override: int | None,
    device: str,
) -> dict[str, Any]:
    """Evaluate a single checkpoint. Returns a dict suitable for JSONL/TSV output."""
    run_dir = pathlib.Path(run_dir)
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint_path": None,
        "task": None,
        "backbone": None,
        "seed": None,
        "episodes": episodes,
        "mean_return": None,
        "std_return": None,
        "min_return": None,
        "max_return": None,
        "mean_length": None,
        "episode_returns": None,
        "episode_lengths": None,
        "checkpoint_load_ok": False,
        "error": None,
        "final_train_score": None,
        "eval_seconds": None,
    }
    eval_envs = None
    train_envs = None
    t0 = time.perf_counter()
    try:
        checkpoint_path = run_dir / "latest.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No latest.pt in {run_dir}")
        result["checkpoint_path"] = str(checkpoint_path)

        cfg = load_resolved_config(run_dir, device)
        result["task"] = str(cfg.env.task)
        result["backbone"] = str(cfg.model.backbone)
        result["seed"] = int(cfg.seed) if "seed" in cfg else None
        result["final_train_score"] = read_final_train_score(run_dir / "metrics.jsonl")

        # Pick env parallelism. Fall back to a sensible default if the
        # resolved config has eval_episode_num=0 (training was launched with
        # eval disabled).
        if env_num_override is not None and env_num_override > 0:
            env_num = int(env_num_override)
        else:
            env_num = int(getattr(cfg.env, "eval_episode_num", 0))
            if env_num <= 0:
                env_num = min(episodes, 4)

        with open_dict(cfg.env):
            cfg.env.eval_episode_num = env_num
            # train_envs is built by make_envs but unused for eval. Keep ≥1
            # to satisfy the constructor; we close it immediately below.
            cfg.env.env_num = max(1, env_num)

        train_envs, eval_envs, obs_space, act_space = make_envs(cfg.env)
        _close_envs(train_envs)
        train_envs = None

        agent = Dreamer(cfg.model, obs_space, act_space).to(device)
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        if "agent_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint at {checkpoint_path} is missing 'agent_state_dict'.")
        agent.load_state_dict(ckpt["agent_state_dict"])
        result["checkpoint_load_ok"] = True
        agent.eval()

        returns, lengths = run_eval_episodes(agent, eval_envs, episodes, env_num)
        result["episode_returns"] = [float(x) for x in returns]
        result["episode_lengths"] = [int(x) for x in lengths]
        returns_t = torch.tensor(returns, dtype=torch.float64)
        lengths_t = torch.tensor(lengths, dtype=torch.float64)
        result["mean_return"] = float(returns_t.mean())
        result["std_return"] = float(returns_t.std(unbiased=False))
        result["min_return"] = float(returns_t.min())
        result["max_return"] = float(returns_t.max())
        result["mean_length"] = float(lengths_t.mean())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        result["eval_seconds"] = round(time.perf_counter() - t0, 3)
        if eval_envs is not None:
            _close_envs(eval_envs)
        if train_envs is not None:
            _close_envs(train_envs)
    return result


def write_outputs(
    results: list[dict[str, Any]],
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write JSONL (full results) + TSV (summary) under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "eval_results.jsonl"
    tsv_path = output_dir / "eval_summary.tsv"

    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary_columns = [
        "task", "backbone", "seed", "episodes",
        "mean_return", "std_return", "min_return", "max_return", "mean_length",
        "final_train_score", "checkpoint_load_ok", "eval_seconds",
        "run_dir", "error",
    ]
    sortable = sorted(
        results,
        key=lambda r: (
            r.get("backbone") or "",
            int(r.get("seed") or 0) if r.get("seed") is not None else 0,
            r.get("task") or "",
        ),
    )
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(summary_columns)
        for r in sortable:
            w.writerow([r.get(col, "") if r.get(col) is not None else "" for col in summary_columns])

    return jsonl_path, tsv_path


def collect_run_dirs(args: argparse.Namespace) -> list[pathlib.Path]:
    """Resolve --run-dir or --manifest to a list of run directories."""
    if args.run_dir is not None:
        return [pathlib.Path(args.run_dir)]
    rows = parse_manifest(pathlib.Path(args.manifest))
    run_dirs: list[pathlib.Path] = []
    for row in rows:
        rd = row.get("run_dir") or ""
        if not rd:
            continue
        path = pathlib.Path(rd)
        if not path.is_absolute():
            path = REPO_ROOT / path
        run_dirs.append(path)
    return run_dirs


def default_output_dir(args: argparse.Namespace) -> pathlib.Path:
    """Default eval output location.

    For --manifest: ``<manifest_dir>/eval/``.
    For --run-dir:  ``<run_dir>/eval/``.
    """
    if args.output_dir:
        return pathlib.Path(args.output_dir)
    if args.manifest:
        return pathlib.Path(args.manifest).resolve().parent / "eval"
    return pathlib.Path(args.run_dir) / "eval"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=str, help="Single run directory containing latest.pt")
    src.add_argument("--manifest", type=str, help="Path to CHECKPOINT_MANIFEST.tsv")
    ap.add_argument("--episodes", type=int, default=20, help="Eval episodes per checkpoint (default: 20)")
    ap.add_argument("--env-num", type=int, default=None,
                    help="Parallel envs per wave. Default: cfg.env.eval_episode_num (typically 10).")
    ap.add_argument("--device", type=str, default="cpu",
                    help='Eval device. Default "cpu". "mps"/"cuda" allowed if available.')
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Where to write eval_results.jsonl and eval_summary.tsv. "
                         "Default: <manifest-dir>/eval or <run-dir>/eval.")
    ap.add_argument("--max-runs", type=int, default=None,
                    help="Limit how many manifest rows to evaluate (smoke testing).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the run_dirs that would be evaluated and exit.")
    args = ap.parse_args()

    # Validate device early.
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is not available; falling back to cpu.")
        args.device = "cpu"
    if args.device == "mps" and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        print("WARNING: --device mps requested but MPS is not available; falling back to cpu.")
        args.device = "cpu"

    run_dirs = collect_run_dirs(args)
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]

    if args.dry_run:
        print(f"# Would evaluate {len(run_dirs)} run dir(s):")
        for rd in run_dirs:
            exists = "OK" if (rd / "latest.pt").exists() else "MISSING latest.pt"
            print(f"  {exists:>16}  {rd}")
        return 0

    if not run_dirs:
        print("No run directories to evaluate.")
        return 1

    output_dir = default_output_dir(args)
    print(f"# Eval output dir: {output_dir}")
    print(f"# Device: {args.device}, episodes per checkpoint: {args.episodes}")
    print(f"# {len(run_dirs)} checkpoint(s) to evaluate.")

    results: list[dict[str, Any]] = []
    for i, run_dir in enumerate(run_dirs):
        print(f"\n[{i + 1}/{len(run_dirs)}] {run_dir}")
        try:
            result = evaluate_run(
                run_dir,
                episodes=args.episodes,
                env_num_override=args.env_num,
                device=args.device,
            )
        except Exception as exc:
            if args.run_dir is not None:
                raise
            result = {
                "run_dir": str(run_dir),
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "checkpoint_load_ok": False,
            }
        results.append(result)

        if result.get("error"):
            print(f"  ERROR: {result['error'].splitlines()[0]}")
            if args.run_dir is not None:
                jsonl_path, tsv_path = write_outputs(results, output_dir)
                print(f"\n# Wrote failed result to:\n  {jsonl_path}\n  {tsv_path}")
                return 1
        else:
            print(
                f"  task={result['task']} backbone={result['backbone']} seed={result['seed']}: "
                f"mean_return={result['mean_return']:.3f} ± {result['std_return']:.3f} "
                f"(min={result['min_return']:.3f}, max={result['max_return']:.3f}) "
                f"mean_length={result['mean_length']:.1f} "
                f"in {result['eval_seconds']}s"
            )

    jsonl_path, tsv_path = write_outputs(results, output_dir)
    print(f"\n# Wrote {len(results)} result(s) to:")
    print(f"  {jsonl_path}")
    print(f"  {tsv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
