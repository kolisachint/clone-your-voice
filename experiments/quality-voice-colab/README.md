# Quality voice on Colab (experiment)

**Not part of cyv v1.** This is the "quality tier" from
[ADR 0005](../../docs/adr/0005-layered-voice-pipeline.md), run on a free Colab T4 GPU
instead of your 8 GB Mac. The Mac takes about 70 s per sentence. The notes expect about 4 s
per sentence on a T4. That number is a target, not a measurement yet.

For each reference clip and each line of text, the notebook:

1. makes speech with **F5-TTS**, which copies tone from your reference clip;
2. runs that speech through **Applio** with your trained voice model (`sachinkoli2`).

Results go to Google Drive, plus a shuffled blind listening set so you can rank them without
knowing which is which.

**License: F5-TTS weights are probably non-commercial (CC-BY-NC).** Use this for personal
listening only. Verify before any wider use.

## What to put in Google Drive

The notebook reads from your Drive. Create these folders and files:

```
MyDrive/clone-your-voice/
├── sachinkoli2/
│   └── checkpoints/
│       ├── sachinkoli2_<epoch>e_<step>s.pth   the trained model (train_colab.ipynb saves it here)
│       └── sachinkoli2.index                  the matching index
├── refs/
│   ├── casual.wav    8 to 12 s of you, in the tone you want
│   └── casual.txt    the exact words spoken in casual.wav
└── lines.txt         one sentence per line
```

- **Model:** `train_colab.ipynb` saves it in `<name>/checkpoints/` with these names. The notebook
  also looks in `<name>/` itself. It picks the highest-epoch `.pth`.
- **Reference clips:** record them in the tone you want. The clip sets the tone. Make one pair
  per tone, for example `calm` and `casual`. The `.txt` must match the words exactly, or the
  pacing comes out stiff. Use only letters, digits, `_` and `-` in the names.
- **`lines.txt`:** one sentence per line. Blank lines are skipped.
- **Output:** the notebook creates `MyDrive/clone-your-voice/quality-out/`.

Your recordings and transcripts are personal data. Keep these files out of the GitHub repo.
Only the notebook itself belongs in git.

## Open it in Colab

Open this link (branch `main`):

https://colab.research.google.com/github/kolisachint/clone-your-voice/blob/main/experiments/quality-voice-colab/quality_voice_colab.ipynb

Then:

1. **Runtime > Change runtime type > T4 GPU**.
2. **Runtime > Run all.** Allow the Google Drive popup when it appears.
3. The installs take several minutes. The first F5 run also downloads its weights.

If Colab disconnects, run it again. Finished clips are skipped.

## What comes out

```
quality-out/
├── casual/
│   ├── 01_f5.wav        F5 speech (no voice model)
│   └── 01_f5_rvc.wav    F5 speech with your voice
├── calm/ ...
└── blind/
    ├── L01_A.wav        shuffled clips, grouped by line
    ├── ...
    └── key.txt          which file is which. Open it last.
```

Each `.wav` is one line. The notebook prints file names and timings, never the text.

## Before you save the notebook back to GitHub

**Edit > Clear all outputs.** Otherwise Colab saves your audio, the printed file names and
any transcript text into the repository. This happened once before with the other test
notebook (see `experiments/SESSION-NOTES.md`).

## Known gaps

- Not run end to end yet. The F5 call matches the `f5-tts==1.1.22` package source. The Applio
  `core.py infer` and `batch-infer` options match Applio at the pinned commit. Both still need
  a real Colab run.
- The notebook does not check the output audio. Listen to it.
- ADR 0005 calls the second TTS model "ADR-007". That is
  [docs/adr/0007-quality-tier-second-tts.md](../../docs/adr/0007-quality-tier-second-tts.md),
  status Proposed, and it was not committed when this README was written.
- The F5 venv does not use `notebooks/constraints.txt`: its `pillow==13.0.0` pin conflicts with
  the `gradio` that F5-TTS needs.
