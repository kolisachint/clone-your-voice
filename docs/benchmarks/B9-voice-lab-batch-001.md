# B9: Voice lab on batch 001 (does RVC help cloned speech, and does tone carry over?)

**Status: planned, not run.** This page fixes the questions, the method and the pass bars *before*
any result exists, so the results cannot be explained away afterwards. Results go at the end.

## Questions

- **B9a.** F5-TTS (English) and IndicF5 (Hindi, Marathi) clone a line from one of your other clips.
  Does running that clone through your voice model (Applio RVC) make it sound more like you, or
  does it only add artifacts? And is protect 0.33 better than the daemon's 0.5?
- **B9b.** Does the speech model copy the tone of the clip it clones from? If a calm line is spoken
  from your angry clip, does it come out angry?

B9a decides whether the voice model belongs in `cyv say` at all (ADR-008). B9b feeds ADR-006
(which tones a voice pack's reference clips need).

## Not tested here

No training of any kind: not the voice model, not the speech models. Mixed Hindi and English
(Hinglish), index rate and pitch-method sweeps, and speed on the Mac are left for later.

## Method

- **Data:** batch 001 as written (`docs/guides/batches/batch-001.tsv`, 27 lines in English, Hindi
  and Marathi). One clip per emotion per language, and two singing lines. The voice model has never
  heard these recordings.
- **Voice model:** `sachinkoli2`, unchanged (B1).
- **Notebook:** `notebooks/voice_lab_colab.ipynb` with `notebooks/cyv_lab.py`.
- **B9a:** every line except singing gets four versions: your real recording, the clone
  (`tts`), the clone through the voice model with protect 0.5 (`rvc`, the daemon's setting, which
  turns protection off) and with protect 0.33 (`rvc33`). The index rate is 0.75 in both, the same
  as the daemon. Applio's own defaults (0.3 and 0.33) are not used.
- **B9b:** per language, one calm line is spoken from a calm clip and from each emotion clip,
  with the speech model only.
- **Scores (B9a):**
  - Speaker similarity (0 to 1) of each clone to your real recording of the same line, from
    `microsoft/wavlm-base-plus-sv`.
  - Character error rate (CER) of each clone against its line, from faster-whisper large-v3.
    Character error rate suits Hindi and Marathi better than word error rate. Treat Marathi CER as
    a sanity check only, because Whisper is weak in Marathi.
- **Listening:** blind. Files are shuffled under letters and the keys sit outside the listening
  folders. You fill in each `sheet.md` before opening a key. One listener (the maintainer), which
  is a bias, and is reported as such.

## Pass bars (fixed before running)

| | The result counts as a pass if... | If it fails |
|---|---|---|
| **B9a: keep RVC** | Per language, a voice model version is ranked above the plain clone on at least two thirds of the lines, and its mean similarity is not lower and its mean CER is no more than about 0.02 higher than the plain clone's | `cyv say` drops the voice model. `cyv convert` keeps it |
| **B9a: protect** | 0.33 beats 0.5 on the breathy and emotion lines and does not lose on the calm ones | Protect stays 0.5 |
| **B9b: tone carries over** | Most emotion files are named correctly in the blind sheet | Expressive `say` needs more than reference clips |

With one clip per emotion, B9b is a **first listen**. It does not decide ADR-006 alone. A pass on
B9a is also limited to batch 001; a second batch should confirm it before the daemon changes.

## Hardware

- **Generation and scoring:** Google Colab, GPU: _to fill in from the notebook's first cell_.
- **Listening:** the maintainer's Mac. Chip, RAM, macOS version: _to fill in_. cyv version: _to
  fill in_.

## Results

Not run yet.
