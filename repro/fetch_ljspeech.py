#!/usr/bin/env python
"""Fetch the LJ Speech corpus and lay it out where upstream's filelists expect it.

WHAT THIS REPLACES
------------------
Upstream's README asks the reader to do two things by hand:

    1. Download and extract the LJ Speech dataset
    5. Update .wav paths: `sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt`

Step 5 **edits three published files** -- `filelists/ljs_audio_text_{train,val,
test}_filelist.txt` -- which are upstream's canonical 12500/100/500 split and the
only record of it. This row does not modify upstream files, so it satisfies the
path literally instead: the audio is extracted into a directory named `DUMMY`
next to the filelists, and every `DUMMY/LJxxx-nnnn.wav` line resolves with no
edit at all. Upstream's own instruction is honoured, in the other direction.

WHY THIS IS ITS OWN TRACED STEP
-------------------------------
The dataset is 2.6 GB and 13100 files. Run as a separate step, the training
step's reads of `DUMMY/*.wav` resolve to files *this* step wrote, and the corpus
becomes an edge in the lineage graph instead of 13100 inputs from nowhere. It
also separates a fixed cost (this download) from a variable one (training), which
is what makes an honest full-run estimate possible: multiplying a total that is
mostly download by the truncation factor overstates a rebuild by an order of
magnitude.

INTEGRITY
---------
The archive is verified against a pinned SHA-256 before anything is extracted.
The corpus has been byte-stable at this URL since 2018-02-19 (the server's own
Last-Modified). A mismatch means the reader is not getting the corpus this row
trained on, which is worth stopping for.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import time
import urllib.request

# Upstream's README links the LJ Speech dataset page; this is that page's own
# download URL. SHA-256 measured on 2026-08-11 over all 2,748,572,632 bytes.
DEFAULT_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
EXPECTED_SHA256 = "be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5"
EXPECTED_BYTES = 2748572632
EXPECTED_WAVS = 13100

FILELISTS = (
    "filelists/ljs_audio_text_train_filelist.txt",
    "filelists/ljs_audio_text_val_filelist.txt",
    "filelists/ljs_audio_text_test_filelist.txt",
)


def sha256_of(path: str, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: str) -> None:
    t0 = time.time()
    print(f"[fetch] GET {url}", flush=True)
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        seen = 0
        next_report = 0
        while True:
            block = response.read(8 << 20)
            if not block:
                break
            out.write(block)
            seen += len(block)
            if total and seen >= next_report:
                print(
                    f"[fetch]   {seen / 1e9:.2f} / {total / 1e9:.2f} GB "
                    f"({100 * seen / total:.0f}%)",
                    flush=True,
                )
                next_report += total // 10
    elapsed = time.time() - t0
    size = os.path.getsize(dest)
    print(
        f"[timing] download: {size} bytes in {elapsed:.1f}s "
        f"({size / elapsed / 1e6:.1f} MB/s)",
        flush=True,
    )


def extract(tarball: str, out_dir: str) -> int:
    """Flatten LJSpeech-1.1/wavs/*.wav into out_dir; keep metadata alongside.

    The archive is opened in **stream mode** (`r|bz2`, not `r:bz2`), and that is
    not a stylistic choice. In seekable mode `tarfile` reads a member header,
    advances, and then seeks *backwards* to the member's data when
    `extractfile()` is called. `bz2.BZ2File` implements a backward seek by
    rewinding and re-decompressing from byte zero, so extracting 13100 members
    from a 2.6 GB bzip2 archive that way is quadratic: measured here, it slowed
    from ~800 files/min to ~300 and would have taken well over half an hour.
    Stream mode is strictly forward-only, so each member is decompressed exactly
    once and the whole corpus lands in ~2 minutes.
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    with tarfile.open(tarball, "r|bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if name.endswith(".wav"):
                target = os.path.join(out_dir, os.path.basename(name))
            elif os.path.basename(name) in ("metadata.csv", "README"):
                target = os.path.join(out_dir, os.path.basename(name))
            else:
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out, length=1 << 20)
            written += 1
            if written % 2000 == 0:
                print(f"[fetch]   extracted {written} files", flush=True)
    print(
        f"[timing] extract: {written} files in {time.time() - t0:.1f}s",
        flush=True,
    )
    return written


def check_filelists(repo_root: str) -> None:
    """Every audio path in upstream's three filelists must now resolve."""
    missing = 0
    checked = 0
    for rel in FILELISTS:
        path = os.path.join(repo_root, rel)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                audio = line.split("|")[0]
                checked += 1
                if not os.path.isfile(os.path.join(repo_root, audio)):
                    if missing < 5:
                        print(f"[check] MISSING {audio}", flush=True)
                    missing += 1
    if missing:
        raise SystemExit(
            f"[check] {missing} of {checked} audio paths in upstream's "
            "filelists do not resolve after extraction"
        )
    print(
        f"[check] all {checked} audio paths in upstream's three filelists "
        "resolve, with the filelists unmodified",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        required=True,
        help="directory the filelists name (upstream's placeholder is DUMMY)",
    )
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument(
        "--scratch-dir",
        default="/var/tmp",
        help="where the archive is downloaded before extraction; it is deleted "
        "afterwards and is deliberately NOT under the repository, so a 2.6 GB "
        "intermediate does not enter the lineage as an artefact",
    )
    ap.add_argument(
        "--keep-archive",
        action="store_true",
        help="do not delete the archive after extraction",
    )
    cfg = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = (
        cfg.out_dir
        if os.path.isabs(cfg.out_dir)
        else os.path.join(repo_root, cfg.out_dir)
    )
    os.makedirs(cfg.scratch_dir, exist_ok=True)
    tarball = os.path.join(cfg.scratch_dir, os.path.basename(cfg.url))

    if os.path.exists(tarball) and os.path.getsize(tarball) == EXPECTED_BYTES:
        print(f"[fetch] reusing {tarball} (correct size); verifying digest",
              flush=True)
    else:
        download(cfg.url, tarball)

    t0 = time.time()
    digest = sha256_of(tarball)
    print(f"[timing] sha256: {time.time() - t0:.1f}s", flush=True)
    print(f"[check] sha256 {digest}", flush=True)
    if digest != EXPECTED_SHA256:
        raise SystemExit(
            f"[check] SHA-256 MISMATCH\n  expected {EXPECTED_SHA256}\n"
            f"  got      {digest}\nThe archive is not the corpus this row used."
        )
    print("[check] archive matches the pinned SHA-256", flush=True)

    written = extract(tarball, out_dir)
    wavs = len([n for n in os.listdir(out_dir) if n.endswith(".wav")])
    print(f"[check] {wavs} .wav files in {out_dir} ({written} files written)",
          flush=True)
    if wavs != EXPECTED_WAVS:
        raise SystemExit(
            f"[check] expected {EXPECTED_WAVS} wav files, found {wavs}"
        )

    check_filelists(repo_root)

    if not cfg.keep_archive:
        os.remove(tarball)
        print(f"[fetch] removed {tarball}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
