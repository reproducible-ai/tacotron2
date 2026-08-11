#!/usr/bin/env python
"""Exercise every 2019-era API in this repository that a modern host can break.

This repository was last touched in June 2024 and pins `tensorflow==1.15.2`,
`librosa==0.6.0`, `numpy==1.13.3`, `scipy==1.0.0` and `matplotlib==2.1.0` --
none of which install on a current interpreter. The row therefore runs upstream's
code against *modern* versions of those libraries, and the interesting question
is not whether they install but whether upstream's calls still resolve against
them. Four of those calls are load-bearing and each has a known breakage:

1. `layers.py:50` calls `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)`
   **positionally**. librosa 0.10 made every one of those keyword-only, so from
   0.10 onwards this raises `TypeError` on the first line of dataset
   construction. This row pins `librosa==0.9.2`, the last release that accepts
   the positional form (with a `FutureWarning`).
2. `audio_processing.py:50` calls `librosa.util.pad_center(win_sq, n_fft)`
   positionally -- same story, and it sits on the `STFT.inverse` path.
3. `plotting_utils.py` calls `fig.canvas.tostring_rgb()`, **removed in
   matplotlib 3.10**, and `np.fromstring`, deprecated since numpy 1.14. Both are
   reached from `logger.log_validation`, which runs at every checkpoint -- so a
   run can train perfectly and then die at the first validation. This row pins
   `matplotlib==3.9.4`.
4. `librosa==0.9.2` imports `pkg_resources` (`librosa/util/files.py:10`), which
   **setuptools 84 removed**. With a current setuptools, `import librosa` fails
   at `ModuleNotFoundError: No module named 'pkg_resources'` and there is no hint
   anywhere that setuptools is the cause. This row pins `setuptools==79.0.1`.

Every one of those is an *environment* pin, not a source edit. This script proves
the pins are right before the row spends a minute of GPU time -- it runs entirely
on CPU, in a few seconds, and is also the check to run in a bare clone.

It does NOT touch `model.py`: `utils.get_mask_from_lengths` builds a
`torch.cuda.LongTensor`, so upstream's model cannot be constructed off a GPU at
all. That part is unavoidably first exercised on the training host.
"""

from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

failures: list[str] = []


def check(name: str):
    def wrap(fn):
        try:
            result = fn()
            print(f"  PASS  {name}: {result}")
        except Exception as exc:  # noqa: BLE001 - this is the point of the script
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        return fn

    return wrap


def main() -> int:
    import numpy
    import torch

    print(f"python {sys.version.split()[0]} | numpy {numpy.__version__} | "
          f"torch {torch.__version__}")

    import tf_hparams_shim

    tf_hparams_shim.install()

    @check("upstream hparams.py (via the TF shim)")
    def _hparams():
        import hparams as upstream_hparams

        hp = upstream_hparams.create_hparams()
        assert hp.epochs == 500, hp.epochs
        return (f"epochs={hp.epochs} batch_size={hp.batch_size} "
                f"iters_per_checkpoint={hp.iters_per_checkpoint} "
                f"n_symbols={hp.n_symbols}")

    @check("--hparams= override grammar")
    def _override():
        import hparams as upstream_hparams

        hp = upstream_hparams.create_hparams(
            "epochs=1,batch_size=16,iters_per_checkpoint=100"
        )
        assert (hp.epochs, hp.batch_size, hp.iters_per_checkpoint) == (1, 16, 100)
        assert hp.fp16_run is False and hp.text_cleaners == ["english_cleaners"]
        return f"epochs={hp.epochs} batch_size={hp.batch_size} types preserved"

    tf_hparams_shim.uninstall()

    @check("librosa positional mel filterbank (layers.TacotronSTFT)")
    def _stft():
        import layers

        stft = layers.TacotronSTFT()
        return f"mel_basis {tuple(stft.mel_basis.shape)}"

    @check("mel spectrogram of a waveform (data_utils.get_mel path)")
    def _mel():
        import layers

        stft = layers.TacotronSTFT()
        wav = torch.sin(
            2 * 3.14159 * 440 * torch.arange(0, 22050 * 2) / 22050
        ).unsqueeze(0) * 0.5
        mel = stft.mel_spectrogram(wav)
        assert mel.shape[1] == 80, mel.shape
        return f"mel {tuple(mel.shape)} range [{mel.min():.2f}, {mel.max():.2f}]"

    @check("STFT.inverse -> librosa.util.pad_center (audio_processing)")
    def _inverse():
        import stft as stft_mod

        fn = stft_mod.STFT(1024, 256, 1024)
        wav = torch.sin(
            2 * 3.14159 * 440 * torch.arange(0, 22050) / 22050
        ).unsqueeze(0) * 0.5
        magnitude, phase = fn.transform(wav)
        recon = fn.inverse(magnitude, phase)
        return f"round trip {tuple(recon.shape)}"

    @check("text_to_sequence with english_cleaners (inflect + unidecode)")
    def _text():
        from text import text_to_sequence

        seq = text_to_sequence(
            "Mr. Smith paid $3.50 for 27 items on the 1st of May.",
            ["english_cleaners"],
        )
        assert len(seq) > 10
        return f"{len(seq)} symbol ids"

    @check("scipy.io.wavfile.read (utils.load_wav_to_torch)")
    def _wavread():
        import numpy as np
        from scipy.io.wavfile import read, write

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "probe.wav")
            data = (np.sin(np.arange(22050) * 0.05) * 12000).astype(np.int16)
            write(path, 22050, data)
            from utils import load_wav_to_torch

            audio, sr = load_wav_to_torch(path)
            assert sr == 22050
            return f"{tuple(audio.shape)} @ {sr} Hz"

    @check("plotting_utils (matplotlib tostring_rgb + np.fromstring)")
    def _plots():
        import numpy as np
        from plotting_utils import (
            plot_alignment_to_numpy,
            plot_gate_outputs_to_numpy,
            plot_spectrogram_to_numpy,
        )

        alignment = plot_alignment_to_numpy(np.random.rand(50, 30))
        spectrogram = plot_spectrogram_to_numpy(np.random.rand(80, 100))
        gate = plot_gate_outputs_to_numpy(
            np.random.rand(100), np.random.rand(100)
        )
        return (f"alignment {alignment.shape} spectrogram {spectrogram.shape} "
                f"gate {gate.shape}")

    @check("import train.py itself, with the shim already withdrawn")
    def _train_module():
        # This is the check that would have caught the ordering bug found in the
        # bare-clone run: train.py -> logger.py -> torch.utils.tensorboard, whose
        # _embedding.py evaluates `hasattr(tf.io.gfile, "join")` at import time
        # against whatever `import tensorflow` returns. If the shim is still
        # installed here, this import raises AttributeError.
        import train as upstream

        assert "tensorflow" not in sys.modules, (
            "the TF shim is still installed; torch.utils.tensorboard will "
            "route its file IO through it"
        )
        return f"train.train() present: {callable(upstream.train)}"

    @check("logger.Tacotron2Logger (torch.utils.tensorboard)")
    def _logger():
        from logger import Tacotron2Logger

        with tempfile.TemporaryDirectory() as tmp:
            log = Tacotron2Logger(os.path.join(tmp, "logdir"))
            log.log_training(1.0, 2.0, 1e-3, 0.5, 0)
            log.flush()
            written = os.listdir(os.path.join(tmp, "logdir"))
            return f"{len(written)} event file(s) written"

    print("preflight_cpu:", "FAIL" if failures else "PASS")
    for line in failures:
        print(f"  {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    print("--- preflight: upstream's 2019 API calls against this environment ---")
    sys.exit(main())
