#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import sys


CHECKS = [
    ("torch", None),
    ("torchrl", None),
    ("hydra", None),
    ("tensordict", None),
    ("gymnasium", None),
    ("popgym", None),
    ("matplotlib", None),
    ("ale_py", "Atari / ALE"),
    ("bsuite", "BSuite (expected on Python 3.11)"),
]


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    failures = []
    for module_name, label in CHECKS:
        try:
            importlib.import_module(module_name)
            print(f"[ok] {label or module_name}")
        except Exception as exc:
            print(f"[missing] {label or module_name}: {exc}")
            failures.append(module_name)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
