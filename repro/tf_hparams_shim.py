#!/usr/bin/env python
"""A stand-in for the *one* TensorFlow 1.x symbol this repository imports.

WHY THIS FILE EXISTS
--------------------
`hparams.py` -- upstream's configuration file, and the place this row reads its
untruncated defaults from -- begins:

    import tensorflow as tf
    ...
    hparams = tf.contrib.training.HParams(
        epochs=500, ...

`tf.contrib` was **deleted in TensorFlow 2.0** (September 2019) and
`tf.contrib.training.HParams` has no successor anywhere in the TensorFlow API.
`requirements.txt` pins `tensorflow==1.15.2`, the last 1.x release; it publishes
wheels for CPython 3.5-3.7 only and none for the interpreter this row runs on.
So on any interpreter released after 2020 there is no TensorFlow -- old or new --
that makes `hparams.py` import.

There were three ways out and this is the one that leaves upstream alone:

  (a) rewrite `hparams.py` to build a plain namespace. That edits the file that
      *defines the model*, and it is the file a reader most wants to trust is
      upstream's.
  (b) stop using `hparams.py` and hard-code the values in the harness. Then the
      row's published "untruncated default" would be a number this operator
      typed, not a number read from upstream's own config -- exactly the thing
      the campaign refuses to do.
  (c) supply the missing container. `hparams.py` uses `tf` for two things and
      nothing else: `tf.contrib.training.HParams` and `tf.logging.info`. This
      module implements both, and `repro/train_truncated.py` registers it in
      `sys.modules` under the name `tensorflow` *before* importing upstream's
      `hparams`. Upstream's file is then executed byte-for-byte as published,
      and `create_hparams()` returns upstream's own values.

This is (c). It is not a TensorFlow emulation and does not try to be: it is a
typed attribute bag with the `parse()` grammar TF 1.x used for the
`--hparams=name=value,...` command-line flag, which `train.py` exposes and which
this row uses to set `epochs`, `batch_size` and `iters_per_checkpoint`.

The substitution is announced on stdout by the harness, so a reader of the run
log sees it happen rather than discovering it here.

`HParams.parse` semantics reproduced from `tensorflow/contrib/training/python/
training/hparam.py` (TF 1.15):

  * values are comma-separated `name=value` pairs;
  * each value is cast to the *type of the current value* of that hyperparameter
    -- so `epochs=1` yields the int 1 because the default is an int, and
    `fp16_run=True` yields a bool because the default is a bool;
  * `bool` accepts `true/false` case-insensitively and rejects anything else
    (TF 1.x accepted only `true`/`false`, not `1`/`0`);
  * an unknown hyperparameter name is an error, not a silent no-op;
  * list-valued hyperparameters take `name=[a,b,c]`.
"""

from __future__ import annotations

import sys
import types


class HParams:
    """`tf.contrib.training.HParams`, reduced to what `hparams.py` uses."""

    def __init__(self, **kwargs):
        # Bypass __setattr__ bookkeeping for the registry itself.
        object.__setattr__(self, "_hparam_types", {})
        for name, value in kwargs.items():
            self.add_hparam(name, value)

    # -- construction -----------------------------------------------------
    def add_hparam(self, name, value):
        if name in self._hparam_types:
            raise ValueError(f"Hyperparameter name is reserved: {name}")
        if isinstance(value, (list, tuple)):
            value = list(value)
            if not value:
                raise ValueError(
                    f"Multi-valued hyperparameters cannot be empty: {name}"
                )
            self._hparam_types[name] = (type(value[0]), True)
        else:
            self._hparam_types[name] = (type(value), False)
        object.__setattr__(self, name, value)

    # -- the --hparams= grammar -------------------------------------------
    @staticmethod
    def _cast(raw, kind, name):
        raw = raw.strip()
        if kind is bool:
            lowered = raw.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            raise ValueError(f"Could not parse {raw!r} as bool for {name!r}")
        if kind is str:
            # TF stripped a single layer of quotes if present.
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
                return raw[1:-1]
            return raw
        return kind(raw)

    def parse(self, values):
        """Apply a `name=value,name=value` string, in place; return self."""
        if not values:
            return self
        for chunk in self._split_pairs(values):
            name, sep, raw = chunk.partition("=")
            name = name.strip()
            if not sep:
                raise ValueError(f"Malformed hyperparameter value: {chunk!r}")
            if name not in self._hparam_types:
                raise ValueError(f"Unknown hyperparameter: {name}")
            kind, is_list = self._hparam_types[name]
            if is_list:
                raw = raw.strip()
                if not (raw.startswith("[") and raw.endswith("]")):
                    raise ValueError(
                        f"Must pass a list-valued hyperparameter as [a,b,c]: {name}"
                    )
                parsed = [
                    self._cast(item, kind, name)
                    for item in raw[1:-1].split(",")
                    if item.strip()
                ]
            else:
                parsed = self._cast(raw, kind, name)
            object.__setattr__(self, name, parsed)
        return self

    @staticmethod
    def _split_pairs(values):
        """Split on commas that are not inside a `[...]` list literal."""
        out, depth, current = [], 0, []
        for ch in values:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            if ch == "," and depth == 0:
                out.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            out.append("".join(current))
        return [chunk for chunk in out if chunk.strip()]

    # -- accessors --------------------------------------------------------
    def values(self):
        return {name: getattr(self, name) for name in sorted(self._hparam_types)}

    def get(self, name, default=None):
        return getattr(self, name, default)

    def __contains__(self, name):
        return name in self._hparam_types

    def __repr__(self):
        return f"HParams({self.values()})"


class _Logging:
    """`tf.logging`, reduced to the two calls `hparams.py` makes."""

    @staticmethod
    def info(fmt, *args):
        print("[tf.logging] " + (fmt % args if args else str(fmt)), flush=True)

    warn = warning = error = debug = info


def build_module() -> types.ModuleType:
    """A module object that answers `tf.contrib.training.HParams` / `tf.logging`."""
    tensorflow = types.ModuleType("tensorflow")
    contrib = types.ModuleType("tensorflow.contrib")
    training = types.ModuleType("tensorflow.contrib.training")
    training.HParams = HParams
    contrib.training = training
    tensorflow.contrib = contrib
    tensorflow.logging = _Logging()
    tensorflow.__version__ = "shim (repro/tf_hparams_shim.py; not TensorFlow)"
    return tensorflow


def uninstall(verbose: bool = True) -> None:
    """Remove the shim from `sys.modules` again. **This is not optional.**

    `logger.py` builds a `torch.utils.tensorboard.SummaryWriter`, and TensorBoard
    probes for TensorFlow at construction time: if `import tensorflow` succeeds it
    routes all of its file IO through `tf.io.gfile` instead of the local
    filesystem. A shim that answers `import tensorflow` but has no `io` attribute
    therefore turns a working logger into

        AttributeError: module 'tensorflow' has no attribute 'io'

    at the first validation pass -- i.e. minutes into a paid GPU run, after
    training has already succeeded. (Observed, on CPU, before booking anything.)

    The shim exists only for the duration of `create_hparams()`; the object it
    returns is a plain attribute bag with no further need of the module. So it is
    withdrawn immediately afterwards, and every later `import tensorflow` fails
    exactly as it would on a host where TensorFlow is simply not installed --
    which is the truth.
    """
    for name in (
        "tensorflow.contrib.training",
        "tensorflow.contrib",
        "tensorflow",
    ):
        module = sys.modules.get(name)
        if module is not None and getattr(
            sys.modules.get("tensorflow"), "_repro_shim", False
        ):
            del sys.modules[name]
    if verbose:
        print(
            "[shim] withdrawn: `import tensorflow` now fails as it would on any "
            "host without TensorFlow. TensorBoard's SummaryWriter probes for it "
            "and would otherwise route its file IO through tf.io.gfile.",
            flush=True,
        )


def install(verbose: bool = True) -> types.ModuleType:
    """Register the shim as `tensorflow` so upstream's `hparams.py` imports."""
    if "tensorflow" in sys.modules and not hasattr(
        sys.modules["tensorflow"], "_repro_shim"
    ):
        # A real TensorFlow is present; do not shadow it.
        return sys.modules["tensorflow"]
    tensorflow = build_module()
    tensorflow._repro_shim = True
    sys.modules["tensorflow"] = tensorflow
    sys.modules["tensorflow.contrib"] = tensorflow.contrib
    sys.modules["tensorflow.contrib.training"] = tensorflow.contrib.training
    if verbose:
        print(
            "[shim] `import tensorflow` in hparams.py is served by "
            "repro/tf_hparams_shim.py -- it provides "
            "tf.contrib.training.HParams and tf.logging and nothing else. "
            "TensorFlow itself is NOT installed: tf.contrib was removed in "
            "TF 2.0 and the pinned tensorflow==1.15.2 has no wheel for this "
            "interpreter. hparams.py itself is upstream's, unmodified.",
            flush=True,
        )
    return tensorflow


if __name__ == "__main__":
    import os

    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    install()
    import hparams as upstream_hparams  # noqa: E402

    hp = upstream_hparams.create_hparams()
    for key, value in hp.values().items():
        print(f"{key} = {value!r}")
