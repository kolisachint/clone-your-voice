# clone-your-voice

Clone **your own** voice: train once on a free cloud GPU (Colab), then speak in your voice locally
on an Apple Silicon Mac from the terminal.

> **Status: early.** The scaffold is built and tested on Linux. It has not yet been validated on a
> Mac or a real Colab T4. See [docs/DESIGN.md §11](docs/DESIGN.md#11-implementation-status-and-findings-oct-2026-linux-x86-cpu).

## How it works

1. Record 5–10 minutes of clean speech ([recording guide](docs/guides/recording.md)).
2. `cyv dataset prepare ./my-clips --name me` → `dataset.zip`, with a consent attestation.
3. Train on Colab with [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) → `me.cyvpack.zip`.
4. `cyv setup` (one time), then `cyv voices import me.cyvpack.zip`.
5. `cyv say "Hello there" --preset telegram` or `cyv convert take.wav`.

Every output carries an inaudible AudioSeal watermark by default (`--no-watermark` turns it off).
Read the [usage policy](USAGE_POLICY.md): clone only voices you own or have explicit permission to use.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [docs/DESIGN.md](docs/DESIGN.md) and [docs/adr/](docs/adr/).
MIT licensed; third-party licenses are in [docs/licenses.md](docs/licenses.md).
