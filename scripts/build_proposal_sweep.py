#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path


ATARI100K_TASKS = [
    "atari_alien",
    "atari_amidar",
    "atari_assault",
    "atari_asterix",
    "atari_bank_heist",
    "atari_battle_zone",
    "atari_boxing",
    "atari_breakout",
    "atari_chopper_command",
    "atari_crazy_climber",
    "atari_demon_attack",
    "atari_freeway",
    "atari_frostbite",
    "atari_gopher",
    "atari_hero",
    "atari_james_bond",
    "atari_kangaroo",
    "atari_krull",
    "atari_kung_fu_master",
    "atari_ms_pacman",
    "atari_pong",
    "atari_private_eye",
    "atari_qbert",
    "atari_road_runner",
    "atari_seaquest",
    "atari_up_n_down",
]

POPGYM_PRESETS = {
    "popgym_repeat_previous": [
        "popgym_RepeatPreviousEasy-v0",
        "popgym_RepeatPreviousMedium-v0",
        "popgym_RepeatPreviousHard-v0",
    ],
    "popgym_autoencode": [
        "popgym_AutoencodeEasy-v0",
        "popgym_AutoencodeMedium-v0",
        "popgym_AutoencodeHard-v0",
    ],
    "popgym_concentration": [
        "popgym_ConcentrationEasy-v0",
        "popgym_ConcentrationMedium-v0",
        "popgym_ConcentrationHard-v0",
    ],
}

BSUITE_PREFIX = {
    "bsuite_memory_len": "memory_len",
    "bsuite_memory_size": "memory_size",
    "bsuite_discounting_chain": "discounting_chain",
}

DEFAULT_EXPERIMENTS = [
    "atari100k",
    "bsuite_memory_len",
    "bsuite_memory_size",
    "bsuite_discounting_chain",
    "popgym_repeat_previous",
    "popgym_autoencode",
    "popgym_concentration",
]

DEFAULT_BACKBONES = ["gru", "transformer", "mamba2", "s3m", "s5"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_SUBEXPS = ["none"]


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


def discover_bsuite_tasks(prefix: str) -> list[str]:
    fallback = [f"bsuite_{prefix}/0"]
    try:
        from bsuite import sweep
    except ImportError:
        return fallback

    candidates = set()
    for name in dir(sweep):
        value = getattr(sweep, name)
        if isinstance(value, str):
            candidates.add(value)
        elif isinstance(value, (list, tuple, set)):
            candidates.update(x for x in value if isinstance(x, str))
    matched = sorted(f"bsuite_{x}" for x in candidates if x.startswith(f"{prefix}/"))
    return matched or fallback


def experiment_tasks(experiment: str) -> list[str]:
    if experiment == "atari100k":
        return ATARI100K_TASKS
    if experiment in POPGYM_PRESETS:
        return POPGYM_PRESETS[experiment]
    if experiment in BSUITE_PREFIX:
        return discover_bsuite_tasks(BSUITE_PREFIX[experiment])
    raise ValueError(f"Unknown experiment preset: {experiment}")


def build_command(
    python_exec: str,
    experiment: str,
    task: str,
    backbone: str,
    subexp: str,
    seed: int,
    model: str,
    root_logdir: Path,
    extra_overrides: list[str],
) -> str:
    task_slug = slugify(task)
    logdir = root_logdir / experiment / subexp / task_slug / backbone / f"seed{seed}"
    parts = [
        python_exec,
        "train.py",
        f"env={experiment}",
        f"model={model}",
        f"subexp={subexp}",
        f"model.backbone={backbone}",
        f"seed={seed}",
        f"logdir={logdir}",
    ]
    if task:
        parts.append(f"env.task={task}")
    parts.extend(extra_overrides)
    return " ".join(shlex.quote(part) for part in parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Build or run proposal benchmark sweeps.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment presets to include. Default covers the proposal benchmark presets.",
    )
    parser.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    parser.add_argument("--subexps", nargs="+", default=DEFAULT_SUBEXPS, help="Hydra subexperiment presets to use.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--model", default="size12M", help="Hydra model config group to use.")
    parser.add_argument("--python", dest="python_exec", default=sys.executable)
    parser.add_argument("--root-logdir", default="./logdir/proposal")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra overrides. Repeatable, for example --override trainer.steps=1e6",
    )
    parser.add_argument("--write-joblist", type=Path, help="Optional file to write all commands into.")
    parser.add_argument("--run", action="store_true", help="Execute the generated commands sequentially.")
    return parser.parse_args()


def main():
    args = parse_args()
    root_logdir = Path(args.root_logdir)
    commands = []
    for experiment in args.experiments:
        for task in experiment_tasks(experiment):
            for backbone in args.backbones:
                for subexp in args.subexps:
                    for seed in args.seeds:
                        commands.append(
                            build_command(
                                python_exec=args.python_exec,
                                experiment=experiment,
                                task=task,
                                backbone=backbone,
                                subexp=subexp,
                                seed=seed,
                                model=args.model,
                                root_logdir=root_logdir,
                                extra_overrides=args.override,
                            )
                        )

    if args.write_joblist:
        args.write_joblist.parent.mkdir(parents=True, exist_ok=True)
        with args.write_joblist.open("w") as f:
            f.write("\n".join(commands) + ("\n" if commands else ""))

    for command in commands:
        print(command)

    if args.run:
        for index, command in enumerate(commands, start=1):
            print(f"[{index}/{len(commands)}] {command}", flush=True)
            subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    main()
