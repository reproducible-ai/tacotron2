#!/usr/bin/env python
"""Record the interpreter this job INHERITED, before the job changes anything.

WHY
---
Campaign hosts are handed out warm: an instance can have served an earlier job
before this one is scheduled onto it, and campaign setup stages install into the
*shared* system interpreter. So the environment a job starts with is not a
property of the machine image — it is a property of whatever ran here first, and
after the job finishes there is no way to reconstruct it.

That is not a hypothetical. On this row's second attempt the shared interpreter
reported `torch 2.3.0+cu121`; on its first and third, `torch 2.7.0+cu126` — same
target, same AMI, same `/opt/pytorch/bin/python` build string, same afternoon.
Only one of those three runs could have been right about what the image ships,
and nothing in the record said which.

This script is run in the untraced setup stage **before that stage installs
anything**, against the inherited interpreter rather than this row's own venv. It
prints the complete distribution list into the job log, so that if anything about
this run's record looks strange afterwards, the starting state is evidence rather
than a guess.

It changes nothing and never fails the run.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import sys

# Packages worth calling out by name: this row's own stack, plus the ones other
# campaign rows have been observed writing into the shared interpreter.
WATCH = {
    "torch", "torchvision", "torchaudio", "triton",
    "numpy", "scipy", "matplotlib", "pillow", "setuptools", "pip",
    "librosa", "soundfile", "audioread", "numba", "llvmlite", "resampy",
    "inflect", "unidecode", "tensorboard", "tensorflow",
    "huggingface-hub", "trackio", "einops", "pandas", "tqdm", "safetensors",
    "transformers", "datasets", "accelerate", "shap", "roar-cli",
}


def normalize(name: str) -> str:
    return (name or "?").strip().lower().replace("_", "-")


def main() -> int:
    print(f"[inherited] interpreter: {sys.executable}")
    print(f"[inherited] version:     {sys.version.split()[0]} "
          f"({sys.version.split('(')[-1].rstrip(')')})")

    dists: dict[str, str] = {}
    for dist in md.distributions():
        name = (dist.metadata["Name"] if dist.metadata else None) or "?"
        dists[name] = dist.version or "?"

    print(f"[inherited] distributions installed: {len(dists)}")
    watched = {n: v for n, v in dists.items() if normalize(n) in WATCH}
    for name in sorted(watched, key=normalize):
        print(f"[inherited]   {name}=={watched[name]}")
    missing = sorted(WATCH - {normalize(n) for n in dists})
    print(f"[inherited] watched-but-absent: {', '.join(missing) if missing else 'none'}")

    # The full list, one line, so a reader can diff two runs mechanically.
    print("[inherited] FULL=" + json.dumps(dict(sorted(dists.items())),
                                           separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
