# ADR-008: Shortlist F5-TTS, IndicF5 and Applio RVC, and drop the rest

Status: Accepted (maintainer decision, 28 October 2026)

## Context

The goal is to clone one person's voice in three languages (English, Hindi and Marathi) and in several styles: conversation, reading, acting (happy, sad, angry) and singing.

Earlier experiments compared several models. Their results stay in the repository:

- **Kokoro** speaks clearly, but its delivery is not yours. In the B7 blind test, Kokoro followed by `cyv convert` ranked last on every sentence. Pitch shifts did not fix it (`docs/guides/sounding-like-you.md`).
- **F5-TTS** copies delivery from a reference clip. In B7 it ranked first on every English sentence. It has no Devanagari in its vocabulary, so it cannot speak Hindi or Marathi.
- **Chatterbox** needs 8.4 GB of GPU memory on an 8 GB M1, so it swaps (B7). It was ruled out.
- **IndicF5** (AI4Bharat) is F5-TTS trained on 1417 hours of speech in 11 Indian languages, including Hindi and Marathi. In a quick listening test, the Marathi sample sounded good. The English sample was bad, which is expected because English is not one of its languages (`docs/benchmarks/B8-indicf5-samples.md`).
- **Applio RVC** (`sachinkoli2`) already makes a recording of the user sound like the user. Training and import lose nothing (B1, B7).

## Decision

1. **Three models, and nothing else:**

   | Layer | Model | Learns from |
   |---|---|---|
   | Speech, English | F5-TTS | English clips |
   | Speech, Hindi and Marathi | IndicF5 | Hindi and Marathi clips |
   | Timbre (sounds like you) | Applio RVC | **every** clip |

2. **One clip trains one speech model, picked by its language.** It always also trains the voice model. Clips with the style `sing` train the voice model only, because neither speech model can sing. The routing table lives in one place, `src/cyv/languages.py`.

3. **Recording batches.** The user works through two steps only:
   1. The user asks for a batch. A batch is a tab-separated file (`file`, `lang`, `style`, `text`). The file name is `<lang>_<style>_<NNN>`, for example `hi_conv_003`. Batches live in `docs/guides/batches/`. The format is checked by `src/cyv/dataset/batch.py`.
   2. The user records the batch and uploads it to Google Drive. The preferred way is one long recording per language (`batch-001_hi.m4a`), with each line's number spoken in English before it. The training notebook cuts it into clips of 4 to 12 s, using the known script, so no speech-to-text transcript is needed. Speech recognition is used only to check that each clip matches its line. One file per line, named after the line, also works. Training, testing and the write-up of results are then done for them.

4. **Kokoro is removed** from the daemon, `cyv setup`, the model list and the dependencies. **Chatterbox is removed** from the experiments. Their benchmark results stay.

5. **The voice lab notebook** (`notebooks/voice_lab_colab.ipynb`) runs a batch on Colab, reading from and writing to Google Drive. Recordings and results stay in Drive. Only code lives in the repository.

6. **Until F5-TTS and IndicF5 are wired into the daemon, `cyv say` returns `tts_unavailable`.** `cyv convert` and `cyv watermark detect` work as before.

## What this changes

- **DESIGN D2 is superseded.** Kokoro is no longer the default text-to-speech.
- **ADR-007 is superseded.** F5 is no longer an optional "quality tier". It is the English speech model.
- **ADR-005** (layers) and **ADR-006** (reference clips in the pack) still apply as proposed.
- **ADR-002** (pack format) is unchanged for now. The `tts_base` field in `voice.json` is still written and validated, but nothing reads it. A pack v2 replaces it with reference clips (ADR-006).

## Licenses

- **F5-TTS weights are CC-BY-NC-4.0**, so they are non-commercial. **IndicF5 weights** are listed as MIT, but they are gated on Hugging Face and are fine-tuned from F5-TTS (verify). This project is for personal use. Before any commercial use, both must be checked again.
- DESIGN §4.6 says non-commercial parts are never in the core install. That still holds: neither model is a dependency of `cyv` or `cyvd`. Today they run only in the training notebook on a cloud GPU.

## Not decided here

- **How the daemon runs F5-TTS and IndicF5 on the Mac.** B7 measured F5 at about 70 s per sentence on an 8 GB M1. That is a later ADR, with a benchmark.
- **Fine-tuning versus reference clips.** With minutes of audio per language, cloning from reference clips may beat fine-tuning. The first batches decide this, measured in a benchmark.
- **Hinglish and other mixed lines.** A line goes to one model by its `lang`. Hindi with English words, written in Devanagari, counts as `hi`.

## Consequences

- **Good:** one clear model per job, and every recording helps the voice model.
- **Good:** Kokoro's GPL espeak-ng dependency is gone from the install.
- **Bad:** `cyv say` does not work until the speech layer is wired in.
- **Bad:** both speech models are slow on an 8 GB Mac and need a GPU for training.
