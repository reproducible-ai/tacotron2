#!/usr/bin/env python
"""Run upstream's `train.py` training loop, stopping after N optimiser steps.

WHY THIS FILE EXISTS AT ALL
---------------------------
Two things stand between a modern host and `python train.py`, and neither can be
fixed by passing a different flag:

1. **`hparams.py` imports TensorFlow 1.x.** It builds its configuration with
   `tf.contrib.training.HParams`, a symbol deleted in TensorFlow 2.0 and never
   replaced; the pinned `tensorflow==1.15.2` publishes no wheel for any
   interpreter after CPython 3.7. `repro/tf_hparams_shim.py` supplies that one
   container so upstream's `hparams.py` -- and therefore upstream's own default
   values, including `epochs=500` -- is read exactly as published. See that
   file's docstring for the full argument.

2. **The training length is not fully configurable.** `epochs` *is* an hparam,
   so `--hparams=epochs=1` is upstream's own knob and is used here. But one
   epoch of the published `filelists/ljs_audio_text_train_filelist.txt` is 12500
   utterances, and there is no upstream flag for "stop after N steps". Upstream's
   loop is

       for epoch in range(epoch_offset, hparams.epochs):        # train.py:206
           for i, batch in enumerate(train_loader):             # train.py:208

   so the inner loop's length is exactly `len(train_loader)`. Capping the number
   of batches that loader yields shortens the run and changes **nothing else**:
   the model, the loss, the optimiser, the gradient clipping, the validation
   pass, the TensorBoard logging and the checkpoint writer are all upstream's,
   untouched, and they see the arguments upstream's own parser produces.

`repro/assert_upstream_unmodified.py` proves the "untouched" claim mechanically
by comparing every tracked file against the upstream commit this fork was taken
from.

WHY DATA-LOADING IS FORCED SINGLE-PROCESS
-----------------------------------------
`train.py:55` hardcodes `num_workers=1`, so the mel-spectrogram work happens in a
forked child. This row is captured under a provenance tracer that attributes
imports per process: a workload whose library imports happen in a worker can
record a package list that is missing most of what it actually used, and that
produces a record which *looks* complete and rebuilds only by luck. Passing
`--dataloader-workers 0` (the default here, and part of the recorded argv) keeps
every import in the traced parent. It changes scheduling, not arithmetic: the
same batches are produced in the same order by the same code.

TRUNCATION IS RECORDED, NOT HIDDEN
----------------------------------
`--train-iters` is part of the recorded command line, so the lineage states the
truncation on its face, and a reader who wants the untruncated run raises that
one number and drops `epochs=1`. Nothing is configured through the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# `python repro/train_truncated.py` puts repro/ on sys.path, not the repo root,
# so upstream's top-level modules (train, hparams, model, data_utils, ...) would
# not resolve. Doing it here, in committed code, keeps the requirement inside the
# recorded command instead of in a PYTHONPATH set around it -- an environment
# variable set outside the recorded command is not part of the lineage and is not
# replayed on a rebuild.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class CappedLoader:
    """Yield at most `limit` batches from a DataLoader; proxy everything else."""

    def __init__(self, loader, limit):
        self._loader = loader
        self._limit = limit

    def __iter__(self):
        for index, batch in enumerate(self._loader):
            if index >= self._limit:
                return
            yield batch

    def __len__(self):
        return min(len(self._loader), self._limit)

    def __getattr__(self, name):
        return getattr(self._loader, name)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False, description=__doc__)
    ap.add_argument(
        "--train-iters",
        type=int,
        required=True,
        help="stop after this many optimiser steps (upstream: epochs x 12500/batch_size)",
    )
    ap.add_argument(
        "--dataloader-workers",
        type=int,
        default=0,
        help="override train.py's hardcoded num_workers=1 (see module docstring)",
    )
    ap.add_argument("--help", action="help")
    cfg, upstream_argv = ap.parse_known_args()

    if cfg.train_iters < 1:
        print("--train-iters must be >= 1")
        return 2

    # Upstream's argparse must see exactly the arguments it declares.
    sys.argv = ["train.py"] + upstream_argv

    # Must precede `import train`, which imports `hparams`, which imports
    # `tensorflow` at module scope.
    import tf_hparams_shim  # noqa: E402  (repro/ is on sys.path)

    tf_hparams_shim.install()

    import torch  # noqa: E402
    import train as upstream  # noqa: E402

    print(
        f"[env] torch {torch.__version__} | cuda available: "
        f"{torch.cuda.is_available()}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"[env] device: {torch.cuda.get_device_name(0)}", flush=True)

    # ---- upstream's own __main__ block, reproduced verbatim ---------------
    # Entering through this wrapper means train.__name__ != "__main__", so
    # train.py's own entry point does not run. These are its lines 259-291.
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = upstream.create_hparams(args.hparams)
    # The shim has done its one job. Withdraw it before anything constructs a
    # TensorBoard SummaryWriter -- see tf_hparams_shim.uninstall().
    tf_hparams_shim.uninstall()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    # ---- end of upstream's __main__ block ---------------------------------

    steps_per_epoch = None  # filled in once the loader exists

    real_dataloader = upstream.DataLoader

    def single_process_dataloader(*a, **kw):
        kw["num_workers"] = cfg.dataloader_workers
        return real_dataloader(*a, **kw)

    upstream.DataLoader = single_process_dataloader

    real_prepare = upstream.prepare_dataloaders

    def capped_prepare(hp):
        nonlocal steps_per_epoch
        t_data = time.time()
        train_loader, valset, collate_fn = real_prepare(hp)
        steps_per_epoch = len(train_loader)
        print(
            f"[timing] dataloaders built in {time.time() - t_data:.1f}s "
            f"({len(train_loader.dataset)} train utterances, "
            f"{len(valset)} validation utterances)",
            flush=True,
        )
        print(
            f"[truncation] upstream would run {hp.epochs} epoch(s) x "
            f"{steps_per_epoch} steps/epoch = {hp.epochs * steps_per_epoch} "
            f"optimiser steps at batch_size={hp.batch_size}; "
            f"running {min(steps_per_epoch, cfg.train_iters)} step(s) "
            f"(--train-iters {cfg.train_iters}).",
            flush=True,
        )
        print(
            f"[dataloader] num_workers forced to {cfg.dataloader_workers} "
            "(upstream hardcodes 1 at train.py:55); see repro/train_truncated.py",
            flush=True,
        )
        return CappedLoader(train_loader, cfg.train_iters), valset, collate_fn

    upstream.prepare_dataloaders = capped_prepare

    t0 = time.time()
    upstream.train(args.output_directory, args.log_directory,
                   args.checkpoint_path, args.warm_start, args.n_gpus,
                   args.rank, args.group_name, hparams)
    elapsed = time.time() - t0
    print(f"[timing] train() returned after {elapsed:.1f}s", flush=True)
    if steps_per_epoch:
        print(
            f"[timing] {elapsed / min(steps_per_epoch, cfg.train_iters):.2f}s "
            "per optimiser step, including the validation passes",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
