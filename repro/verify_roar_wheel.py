#!/usr/bin/env python
"""Prove that the provenance CLI on this host is the exact published artefact we
intended to run, identified by wheel sha256 rather than by version string.

Why a script and not `sha256sum -c`: the CLI is no longer installed from a
pre-downloaded file. It is resolved by name and version from a public package
index, so there is no local file to checksum before the install. Verifying the
version string afterwards is not enough -- several distinct builds have shipped
under one version string in this project's history, so the version string is not
an identity.

What this checks, in order:

1. the index publishes **exactly one** linux/x86_64 wheel for this version, so
   name+version resolution on this platform is unambiguous;
2. that wheel's published sha256 equals the expected value passed in;
3. the bytes actually served match that sha256 (index metadata alone is not the
   artefact);
4. every file the verified wheel's RECORD carries a hash for appears in the
   *installed* distribution's RECORD with the identical hash -- i.e. the files on
   disk came from that wheel and not from some other build that happens to share
   the version number. (The two RECORDs are compared entry-wise, not byte-wise:
   the installer legitimately appends the generated console scripts and drops the
   RECORD's own self-entry, so byte equality is the wrong test.)

Exit 0 only if all four hold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

WHEEL_PLATFORM_MARKERS = ("manylinux", "linux_x86_64")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (https, pinned host)
        return resp.read()


def parse_record(data: bytes) -> dict[str, str]:
    """RECORD -> {path: hash}, skipping entries the installer legitimately rewrites.

    A RECORD line is ``path,<algo>=<b64hash>,<size>``. The RECORD's own entry and
    the ``.dist-info/INSTALLER`` / ``direct_url.json`` entries carry no hash or are
    installer-generated, so they are excluded rather than compared.
    """
    out: dict[str, str] = {}
    for line in data.decode("utf-8", "replace").splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) != 3:
            continue
        path, digest, _size = parts
        if not digest:
            continue
        if path.endswith(("/RECORD", "/INSTALLER", "/direct_url.json", "/REQUESTED")):
            continue
        out[path] = digest
    return out


def find_installed_record(roar_exe: Path, version: str) -> Path:
    """Locate <venv>/lib/python*/site-packages/roar_cli-<version>.dist-info/RECORD."""
    exe = Path(os.path.realpath(str(roar_exe)))
    venv = exe.parent.parent  # <venv>/bin/roar -> <venv>
    matches = sorted(venv.glob(f"lib/python*/site-packages/roar_cli-{version}.dist-info/RECORD"))
    if not matches:
        # A non-venv install (system site-packages / dist-packages).
        for root in sys.path + ["/usr/local/lib", "/usr/lib"]:
            matches += sorted(Path(root).glob(f"**/roar_cli-{version}.dist-info/RECORD"))
            if matches:
                break
    if not matches:
        raise SystemExit(f"FAIL: no installed roar_cli-{version}.dist-info/RECORD found near {exe}")
    return matches[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="roar-cli")
    ap.add_argument("--version", required=True)
    ap.add_argument("--sha256", required=True, help="expected sha256 of the linux x86_64 wheel")
    ap.add_argument("--index-json", default="https://test.pypi.org/pypi/{project}/{version}/json")
    ap.add_argument("--exe", required=True, help="path to the installed roar executable")
    cfg = ap.parse_args()

    url = cfg.index_json.format(project=cfg.project, version=cfg.version)
    meta = json.loads(fetch(url))

    wheels = [
        u
        for u in meta["urls"]
        if u["filename"].endswith(".whl")
        and any(m in u["filename"] for m in WHEEL_PLATFORM_MARKERS)
        and "x86_64" in u["filename"]
        and "aarch64" not in u["filename"]
        and "macosx" not in u["filename"]
    ]
    print(f"index: {url}")
    for u in meta["urls"]:
        print(f"  published: {u['filename']}  sha256={u['digests']['sha256']}")

    if len(wheels) != 1:
        print(f"FAIL: expected exactly 1 linux/x86_64 wheel, found {len(wheels)}")
        return 1
    wheel = wheels[0]
    published = wheel["digests"]["sha256"]
    print(f"selected: {wheel['filename']}")
    print(f"  published sha256: {published}")
    print(f"  expected  sha256: {cfg.sha256}")
    if published != cfg.sha256:
        print("FAIL: published sha256 does not match the expected build")
        return 1

    blob = fetch(wheel["url"])
    served = sha256_bytes(blob)
    print(f"  served    sha256: {served}")
    if served != cfg.sha256:
        print("FAIL: bytes served by the index do not match the published digest")
        return 1

    local_wheel = Path(os.environ.get("ROAR_WHEEL_SCRATCH", "/var/tmp")) / wheel["filename"]
    local_wheel.parent.mkdir(parents=True, exist_ok=True)
    local_wheel.write_bytes(blob)

    with zipfile.ZipFile(local_wheel) as zf:
        names = [n for n in zf.namelist() if n.endswith(".dist-info/RECORD")]
        if len(names) != 1:
            print(f"FAIL: wheel contains {len(names)} RECORD files")
            return 1
        wheel_record = zf.read(names[0])

    installed_record = find_installed_record(Path(cfg.exe), cfg.version)
    print(f"installed RECORD: {installed_record}")
    want = parse_record(wheel_record)
    have = parse_record(installed_record.read_bytes())
    missing = sorted(p for p in want if p not in have)
    differing = sorted(p for p in want if p in have and have[p] != want[p])
    print(f"  wheel entries with a hash: {len(want)}")
    print(f"  installed entries:         {len(have)}")
    if missing or differing:
        for p in missing[:10]:
            print(f"  MISSING from install: {p}")
        for p in differing[:10]:
            print(f"  HASH MISMATCH: {p}\n    wheel={want[p]}\n    disk ={have[p]}")
        print(
            f"FAIL: installed distribution did not come from the verified wheel "
            f"({len(missing)} missing, {len(differing)} mismatched)"
        )
        return 1

    print(f"roar wheel verification: PASS ({wheel['filename']} sha256={cfg.sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
