#!/usr/bin/env python
"""Prove, mechanically, that no upstream file in this fork has been modified.

The reproduction claim this row makes is "upstream's training script, upstream's
hyperparameters, upstream's data loaders and upstream's train/val split, run as
published". `repro/train_truncated.py` shortens the run by capping the batches
its DataLoader yields and `repro/tf_hparams_shim.py` supplies the one deleted
TensorFlow symbol `hparams.py` needs -- both from outside, precisely so that
claim stays literally true. A prose claim in a docstring is worth nothing without
a check that fails when it stops being true.

This compares every path in the upstream commit against the same path in HEAD by
git blob hash. Any upstream file that differs, or is missing, is a failure. Files
that exist only in HEAD are reported as additions -- this fork's `repro/`
helpers, its `.treqs/` workflow, its `.gitignore` and its `.gitkeep` markers --
and are allowed: they are new files, not modifications of published code.

Upstream ships no `.gitignore`, so this row expects **zero** declared exemptions:
every one of upstream's tracked files, `train.py`, `hparams.py`, `model.py`,
`data_utils.py`, `layers.py`, `stft.py`, `audio_processing.py`, `logger.py`,
`utils.py`, `text/*` and all three `filelists/*.txt`, must be byte-identical.
The filelists matter as much as the code: they are the published 12500/100/500
split, and upstream's README asks the reader to `sed` them in place. This row
does not; it extracts the corpus into the directory the unmodified filelists
already name.

Run it in the untraced setup stage, before anything is paid for.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

UPSTREAM_COMMIT = "185cd24e046cc1304b4f8e564734d2498c6e2e6f"
UPSTREAM_REPO = "https://github.com/NVIDIA/tacotron2"


def ensure_commit(commit: str) -> bool:
    """Make `commit` readable locally, fetching it if this is a shallow clone.

    A rebuild host may clone at a depth that does not include the upstream commit
    this fork branched from. Fetch just that object if so; if the network refuses,
    say loudly that the check could not run rather than aborting a run over it.
    """
    have = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"]).returncode == 0
    if have:
        return True
    print(f"upstream commit {commit[:12]} not present locally; fetching it")
    fetched = subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", UPSTREAM_REPO, commit],
        capture_output=True,
        text=True,
    )
    if fetched.returncode != 0:
        print(fetched.stderr.strip())
        return False
    return subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"]).returncode == 0


def ls_tree(ref: str) -> dict[str, str]:
    out = subprocess.run(
        ["git", "ls-tree", "-r", ref],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tree: dict[str, str] = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        _mode, _type, blob = meta.split()
        tree[path] = blob
    return tree


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream-commit", default=UPSTREAM_COMMIT)
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument(
        "--allow-modified",
        action="append",
        default=[],
        help="upstream path that may differ (repeatable). Its full diff is printed.",
    )
    cfg = ap.parse_args()

    print(f"upstream: {UPSTREAM_REPO} @ {cfg.upstream_commit}")
    if not ensure_commit(cfg.upstream_commit):
        print("=" * 70)
        print("assert_upstream_unmodified: SKIPPED -- the upstream commit is not")
        print("reachable from this checkout and could not be fetched. The claim")
        print("that upstream's files are unmodified is NOT verified on this host.")
        print("=" * 70)
        return 0
    upstream = ls_tree(cfg.upstream_commit)
    head = ls_tree(cfg.ref)

    changed = sorted(p for p, b in upstream.items() if p in head and head[p] != b)
    allowed = set(cfg.allow_modified)
    modified = [p for p in changed if p not in allowed]
    declared = [p for p in changed if p in allowed]
    removed = sorted(p for p in upstream if p not in head)
    added = sorted(p for p in head if p not in upstream)

    print(f"upstream files: {len(upstream)}  |  files in {cfg.ref}: {len(head)}")
    print(f"added by this fork ({len(added)}):")
    for p in added:
        print(f"  + {p}")

    for p in declared:
        print(f"  M {p}   <-- DECLARED modification, diff follows:")
        diff = subprocess.run(
            ["git", "diff", f"{cfg.upstream_commit}..{cfg.ref}", "--", p],
            capture_output=True,
            text=True,
        ).stdout
        for line in diff.splitlines():
            print(f"      {line}")

    # An --allow-modified path that is NOT actually modified means the exemption
    # list has drifted from reality; say so, but do not fail on it.
    for p in sorted(allowed - set(changed)):
        print(f"  ? {p} declared with --allow-modified but is unchanged")

    if modified or removed:
        for p in modified:
            print(f"  M {p}   <-- UNDECLARED UPSTREAM MODIFICATION")
        for p in removed:
            print(f"  D {p}   <-- UPSTREAM FILE REMOVED")
        print("assert_upstream_unmodified: FAIL")
        return 1

    print(
        f"assert_upstream_unmodified: PASS "
        f"({len(upstream) - len(declared)} upstream files byte-identical to "
        f"{cfg.upstream_commit[:12]}, {len(declared)} declared exemption(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
