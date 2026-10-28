# clone-your-voice

Clone **your own** voice: train once on a free cloud GPU (Colab), then speak in your voice locally
on an Apple Silicon Mac from the terminal.

> **Status: early.** The scaffold is built and validated on an Apple Silicon Mac (M1, macOS 27.2).
> Training still needs a real Colab T4 run. See [docs/DESIGN.md §11](docs/DESIGN.md#11-implementation-status-and-findings-oct-2026-linux-x86-cpu-and-m1-mac) and [docs/benchmarks/](docs/benchmarks/).

## How it works

1. Record 5–10 minutes of clean speech ([recording guide](docs/guides/recording.md)).
2. `cyv dataset prepare ./my-clips --name me` → `dataset.zip`, with a consent attestation.
3. Train on Colab with [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) → `me.cyvpack.zip`.
4. `cyv setup` (one time), then `cyv voices import me.cyvpack.zip`.
5. `cyv convert take.wav --preset telegram`. `cyv say` is paused while its speech models change: F5-TTS for English, IndicF5 for Hindi and Marathi ([ADR-008](docs/adr/0008-shortlist-f5-indicf5-rvc.md)).

Every output carries an inaudible AudioSeal watermark by default (`--no-watermark` turns it off).
Read the [usage policy](USAGE_POLICY.md): clone only voices you own or have explicit permission to use.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [docs/DESIGN.md](docs/DESIGN.md) and [docs/adr/](docs/adr/).
MIT licensed; third-party licenses are in [docs/licenses.md](docs/licenses.md).
