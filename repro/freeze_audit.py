#!/usr/bin/env python
"""Fail fast if the installed environment cannot be recorded as a portable freeze.

Run in the (untraced) setup stage, before any paid step, so a run that could only
produce an un-rebuildable lineage stops immediately instead of at verification.

Two failure modes, both seen on real Reproducible AI Campaign rows:

1. **PEP 440 local versions** (`torch==2.7.0+cu128`). A `+local` wheel is not on
   PyPI, so `pip install -r <recorded freeze>` cannot resolve it anywhere and the
   lineage cannot be rebuilt on another host.
2. **PEP 503 name collisions** (`importlib-metadata` and `importlib_metadata`
   installed at different versions). A freeze that lists both is unsatisfiable:
   pip normalises the names, sees one project pinned twice, and fails with
   ResolutionImpossible.
"""

from __future__ import annotations

import importlib.metadata as md
import re
import sys


def normalize(name: str) -> str:
    """PEP 503 normalized project name."""
    return re.sub(r"[-_.]+", "-", name or "").lower()


def main() -> int:
    seen: dict[str, set[str]] = {}
    local_versions: list[str] = []

    for dist in md.distributions():
        name = dist.metadata["Name"]
        version = dist.version or ""
        if not name:
            continue
        if "+" in version:
            local_versions.append(f"{name}=={version}")
        seen.setdefault(normalize(name), set()).add(version)

    collisions = {n: sorted(v) for n, v in seen.items() if len(v) > 1}

    print(f"freeze audit: {len(seen)} distributions")
    if local_versions:
        print("  LOCAL-VERSION PINS (not on PyPI):")
        for pin in sorted(local_versions):
            print(f"    {pin}")
    else:
        print("  local-version pins (`+cuNNN` etc.): none")

    if collisions:
        print("  PEP-503 NAME COLLISIONS AT DIFFERENT VERSIONS:")
        for n, versions in sorted(collisions.items()):
            print(f"    {n}: {versions}")
    else:
        print("  PEP-503 duplicate pins: none")

    if local_versions or collisions:
        print("freeze audit: FAIL — this environment cannot produce a portable lineage")
        return 1
    print("freeze audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
