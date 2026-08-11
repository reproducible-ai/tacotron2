#!/usr/bin/env python
"""Report every native library this workload loads, and who owns it.

WHY
---
A pip freeze describes Python distributions. It cannot describe the shared
objects those distributions load, and this workload loads a lot of them: the
audio stack (`librosa` -> `soundfile` -> libsndfile, `audioread`), the JIT stack
(`numba` -> `llvmlite` -> a vendored LLVM), BLAS under numpy/scipy, and the
CUDA runtime under torch. Some of those `.so` files travel *inside* the wheel, so
the freeze is sufficient for them. Others come from OS packages, which no freeze
can express. And a third kind belongs to no package manager at all -- files
dropped on the image by a driver installer, for instance -- so they are invisible
to `pip`, invisible to `apt`, and invisible to any record built from either.

This script sorts every loaded library into those three buckets by reading
`/proc/self/maps` *after* importing what the workload imports. It reports what is
actually mapped into the process, not what `ldd` says might be needed.

It is a diagnostic. It prints and never fails the run.
"""

from __future__ import annotations

import os
import subprocess
import sys

# The workload's native-backed import surface, in the order train.py reaches it.
IMPORTS = [
    ("numpy", "numpy"),
    ("scipy.io.wavfile", "scipy"),  # utils.load_wav_to_torch reads wavs with this
    ("scipy.signal", "scipy"),  # stft.py / audio_processing.py
    ("librosa", "librosa"),  # layers.py: librosa.filters.mel
    ("librosa.util", "librosa"),  # audio_processing.py, stft.py
    ("soundfile", "soundfile"),  # pulled in by librosa; owns libsndfile
    ("audioread", "audioread"),  # pulled in by librosa
    ("numba", "numba"),  # pulled in by librosa
    ("torch", "torch"),
    ("matplotlib", "matplotlib"),  # plotting_utils.py
    ("unidecode", "Unidecode"),  # text/cleaners.py
    ("inflect", "inflect"),  # text/numbers.py
]


def dpkg_owner(path: str) -> str | None:
    try:
        out = subprocess.run(
            ["dpkg", "-S", os.path.realpath(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:  # noqa: BLE001 - diagnostic
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip().split(":")[0] or None


def loaded_libraries() -> list[str]:
    libs: set[str] = set()
    with open("/proc/self/maps", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(None, 5)
            if len(parts) < 6:
                continue
            path = parts[5].strip()
            if not path.startswith("/"):
                continue
            base = os.path.basename(path)
            if ".so" in base:
                libs.add(path)
    return sorted(libs)


def main() -> int:
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"platform: {' '.join(os.uname())}")
    print("--- imports ---")
    for module_name, dist in IMPORTS:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "?")
            print(f"  {module_name:<20} ok   ({dist} {version})")
        except Exception as exc:  # noqa: BLE001 - diagnostic
            print(f"  {module_name:<20} FAIL {type(exc).__name__}: {exc}")

    # matplotlib's Agg canvas is what plotting_utils.py uses; importing the
    # backend is what actually maps its extension module.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pylab  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - diagnostic
        print(f"  matplotlib Agg backend FAILED: {type(exc).__name__}: {exc}")

    wheel_owned, os_owned, unowned = [], [], []
    for lib in loaded_libraries():
        if "site-packages" in lib or "dist-packages" in lib or ".libs/" in lib:
            wheel_owned.append(lib)
            continue
        owner = dpkg_owner(lib)
        (os_owned if owner else unowned).append((lib, owner))

    print(f"--- native libraries mapped into this process: "
          f"{len(wheel_owned) + len(os_owned) + len(unowned)} ---")
    print(f"[A] inside a Python wheel -- covered by the pip freeze "
          f"({len(wheel_owned)}):")
    for lib in wheel_owned:
        print(f"    {lib}")
    print(f"[B] outside the wheels but OWNED by a dpkg package "
          f"({len(os_owned)}) -- a pip freeze cannot express these:")
    for lib, owner in os_owned:
        print(f"    {owner:<28} {lib}")
    print(f"[C] outside the wheels and owned by NO dpkg package "
          f"({len(unowned)}) -- invisible to pip AND to apt:")
    for lib, _ in unowned:
        print(f"    {lib}")
    if not unowned:
        print("    (none)")

    # libsndfile specifically: the boundary this row was chosen to probe.
    print("--- libsndfile provenance ---")
    try:
        import soundfile

        print(f"  soundfile {soundfile.__version__} reports libsndfile "
              f"{soundfile.__libsndfile_version__}")
        lib = getattr(soundfile, "_snd", None)
        print(f"  loaded object: {lib}")
    except Exception as exc:  # noqa: BLE001 - diagnostic
        print(f"  soundfile unavailable: {type(exc).__name__}: {exc}")

    print("native_deps_probe: done (diagnostic only, never fatal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
