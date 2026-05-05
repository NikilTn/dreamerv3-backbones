#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run commands from a newline-delimited job list.")
    parser.add_argument("joblist", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="0-based index of the command to run.")
    group.add_argument("--all", action="store_true", help="Run every command in the job list sequentially.")
    return parser.parse_args()


def load_commands(path: Path) -> list[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def completed_from_metadata(command: str) -> bool:
    """Return True when a train.py command already wrote completed metadata."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    logdir = None
    for part in parts:
        if part.startswith("logdir="):
            logdir = Path(part.split("=", 1)[1])
            break
    if logdir is None:
        return False
    metadata_path = logdir / "run_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        with metadata_path.open() as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("status") == "completed"


def run_command(command: str):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        # Python can occasionally return 120 during interpreter shutdown after
        # training has fully completed and run_metadata.json has been written.
        # Treat that cleanup-only case as success so Slurm arrays stay readable.
        if exc.returncode == 120 and completed_from_metadata(command):
            print(
                "Command returned exit code 120 after writing completed metadata; "
                "treating as success.",
                flush=True,
            )
            return
        raise


def main():
    args = parse_args()
    commands = load_commands(args.joblist)
    if args.all:
        for index, command in enumerate(commands, start=1):
            print(f"[{index}/{len(commands)}] {command}", flush=True)
            run_command(command)
        return

    if args.index < 0 or args.index >= len(commands):
        raise IndexError(f"Job index {args.index} is out of range for {len(commands)} commands.")
    command = commands[args.index]
    print(command, flush=True)
    run_command(command)


if __name__ == "__main__":
    main()
