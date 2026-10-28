# ADR-005: Layered voice pipeline

Status: Proposed

## Context

`cyv say` turns typed text into a voice that sounds like you. Today it has two steps:

1. **Kokoro** (a plain text-to-speech voice) speaks the text. This sets the rhythm, the emphasis and the tone.
2. **Applio RVC** (a voice conversion model trained on your recordings) changes the timbre, so the voice sounds like you.

This works fast, but the result still sounds like Kokoro's delivery with your voice on top. In the October 2026 tests, pitch shifts did not fix this. The best setting (pitch -6) still did not sound like the user.

The B7 experiment (see `docs/benchmarks/B7-clone-tts-m1.md`) tested a different first step: a voice-cloning text-to-speech model (F5-TTS) that copies delivery from a short reference clip, followed by the same RVC step. Measured on an Apple M1, 8 GB, macOS 27.2, cyv 0.0.1, with one listener:

- In a blind test on a Colab T4, F5 followed by `cyv convert` ranked first on all three sentences. Kokoro followed by convert ranked last on every sentence.
- Our own RVC model (`sachinkoli2`) sounded as good on a recording of the user as it did in the Colab self-test, so training and import lose nothing.
- The remaining gap is in **delivery**, not timbre. The tone of F5's output follows the reference clip. A calm reference gives calm output on every line.

So the loss happens in the layer that sets rhythm, intonation and tone. RVC cannot fix that layer. It only changes the timbre.

This ADR splits the pipeline into layers so each one can be swapped, measured and explained on its own.

## Decision

The voice pipeline has four layers. Each layer has one job.

| Layer | Job | Today | Code |
|---|---|---|---|
| 1. Speech | Text becomes guide audio | Kokoro | `src/cyvd/speech/kokoro/` |
| 2. Prosody | Sets delivery: rhythm, intonation, tone | Kokoro's own delivery, or none (the user's recording) | `src/cyvd/prosody/` |
| 3. Timbre | Makes the audio sound like you | Applio RVC | `src/cyvd/timbre/applio/` (holds `pin.py`) |
| 4. Watermark | Marks the audio as made by cyv | AudioSeal (ADR-004) | `src/cyvd/watermark/` |

Layer 2 gets its delivery from one of three sources:

- **The speech engine** (Kokoro today): the guide voice sets the delivery.
- **A reference clip**: a short recording of you in the tone you want. The speech engine copies its delivery (for example F5-TTS, see ADR-007).
- **The user's own recording** in `cyv convert`: the delivery is already yours, so layers 1 and 2 are skipped.

On top of the layers there are three **tiers**. A tier is a choice of which layers to run.

| Tier | Layers | Use | Cost (measured) |
|---|---|---|---|
| **Everyday fast voice** | Kokoro (1) → RVC (3) → watermark (4). This is today's `cyv say`. | Quick lines, many lines | About 2 s per 10 s line on an M1 (B3, Apple M1, 8 GB, macOS 27.2, cyv 0.0.1 @ `3e5c009`: TTS 1.4 s, f0 0.4 s, embed 0.2 s, before the voice network runs) |
| **Quality voice** | Voice-cloning TTS conditioned on a reference clip (1 + 2) → RVC (3) → watermark (4) | Lines that should sound close to you | Apple M1, 8 GB, macOS 27.2: 67 to 78 s for 4.7 to 6.2 s of audio at 32 steps (B7). Colab T4: about 4 s per sentence (B7) |
| **Best: convert** | Your recording → RVC (3) → watermark (4). This is today's `cyv convert`. | Anything that matters | Convert takes 3 to 5 s per line on the same M1 with the daemon warm (B7, sachinkoli2) |

The best result is always the user's own performance, because it keeps their rhythm and tone. The quality tier is the closest we can get from typed text.

### Why the tiers are not one pipeline

- The everyday tier stays on by default. It is fast, small and runs offline.
- The quality tier needs a second text-to-speech model (ADR-007) and a reference clip in the voice pack (ADR-006). Both are opt-in.
- Each tier reuses layers 3 and 4, so the voice model and the watermark work the same way in every tier.

### Interfaces

The existing interfaces in `src/cyvd/interfaces.py` stay as they are:

- `TTS` (layer 1): `synth(text, voice, lang) -> Audio`. A second implementation (ADR-007) fits this interface.
- `Engine` (layer 3): `load`, `convert`, `device`.
- `Watermarker` (layer 4): `embed`, `detect`.

A future `ProsodySource` (layer 2) is **described here, not built**. It would return a reference clip and its transcript for the speech engine to copy. Nothing in the code depends on it yet. We will add it when a second speech engine needs it.

## Consequences

- **Good:** each layer can be measured and changed on its own. A slow or weak layer can be replaced without touching the others.
- **Good:** the everyday tier does not change. Existing users keep the fast path.
- **Good:** the docs can say honestly where the loss happens (layer 2), not "the voice model is not good enough".
- **Cost:** `src/cyvd/` gets more folders, and `pin.py` moves to `src/cyvd/timbre/applio/`. The Applio pin rule in CLAUDE.md still applies: `pin.py` and `notebooks/train_colab.ipynb` must match in the same commit.
- **Cost:** the quality tier is slow on 8 GB Macs (about 70 s per sentence at 32 steps, per B7). It is for "make a clip, come back later" work until a faster option is measured. Running it on Colab is a manual notebook path (ADR-007).
- **Limit:** all B7 numbers come from one Mac, one reference clip, three to four sentences and one listener. They are single data points, not a distribution.
- **Limit:** the tier names are proposals. Nothing in the README should promise the quality tier until ADR-006 and ADR-007 are accepted and the benchmark is repeated on a quiet machine.
