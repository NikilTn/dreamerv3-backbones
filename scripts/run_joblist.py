#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def main():
    args = parse_args()
    commands = load_commands(args.joblist)
    if args.all:
        for index, command in enumerate(commands, start=1):
            print(f"[{index}/{len(commands)}] {command}", flush=True)
            subprocess.run(command, shell=True, check=True)
        return

    if args.index < 0 or args.index >= len(commands):
        raise IndexError(f"Job index {args.index} is out of range for {len(commands)} commands.")
    command = commands[args.index]
    print(command, flush=True)
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    main()
