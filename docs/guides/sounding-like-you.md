# Sounding like you: what works, what doesn't

A plain summary of what we learned in October 2026 about getting output that
really sounds like you. Numbers are from one Apple M1 (8 GB, macOS 27.2) and
one listener, so read them as single data points.

Details: [B7 benchmark](../benchmarks/B7-clone-tts-m1.md) and the
[f5say experiment](../../experiments/f5-say/README.md).

## The short version

| You want | Use | How good |
|---|---|---|
| A recording of you, in your voice | `cyv convert` | **Best.** Sounds like you, because it keeps your own delivery |
| Typed text, quickly | `cyv say` | Your timbre, someone else's delivery. Fast (about 2 s) |
| Typed text that sounds like you | `f5say` (experiment) | **Close to you.** About 70 s per sentence on an 8 GB M1 |

## What we can do

- **Convert your own recordings.** `cyv convert` on your voice sounds as good
  as the Colab self-test. Nothing is lost between training and your Mac.
- **Speak typed English close to your voice**, with the `f5say` experiment
  (F5-TTS, then `cyv convert`). In a blind test it beat `cyv say` on every
  sentence.
- **Run it all on an 8 GB M1.** F5 needs about 3.3 GB, plus about 1.1 GB for
  the cyv daemon.
- **Change the tone** by changing the reference clip (`f5say --ref`). F5
  copies the tone of the clip you give it.

## What we can't do (yet)

- **Make `cyv say` sound like you.** Kokoro sets the rhythm and emphasis;
  the voice model only changes the timbre. No setting fixes that.
- **Set the tone with a switch.** F5 has no "say it warmly" control. Only
  the reference clip changes tone.
- **Hindi or Marathi with F5.** This F5 model cannot read Devanagari.
- **Fast typed-text-in-your-voice on an 8 GB Mac.** About 70 s per sentence.
  A Colab T4 does it in about 4 s.
- **Ship F5 inside cyv.** The F5 weights are probably non-commercial, and
  adding a second TTS needs design decisions (ADRs) first.

## Do

- **Use `cyv convert` for anything that matters.** Read the line into Voice
  Memos, then convert it.
- **Use 32 steps** in `f5say` (the default). 16 steps is twice as fast and
  you can hear the loss.
- **Record reference clips in the tone you want**, 8 to 12 s, and write down
  the exact words.
- **Close big apps before running F5 on 8 GB.** Swapping makes it much slower.
- **Listen blind** when you compare settings: shuffle the file names, rank,
  then check.
- **Change one setting at a time** and keep the seed fixed. See
  [TUNING.md](../../experiments/f5-say/TUNING.md).
- **Before saving a notebook from Colab to GitHub, clear all outputs**
  (Edit > Clear all outputs). Otherwise your audio gets committed.

## Don't

- **Don't expect pitch alone to fix `cyv say`.** Pitch -6 was the best
  setting we found and it still wasn't you.
- **Don't use 8-bit F5 weights on an M1.** They were slower, not faster.
- **Don't trust F5's own length guess.** It cut every clip to about 2.8 s.
  `f5say` uses a text-length estimate instead.
- **Don't use Chatterbox on an 8 GB Mac.** It needed 8.4 GB of GPU memory and
  swapped heavily.
- **Don't commit voice audio, reference clips or their transcripts.** Not
  even inside notebook outputs.
- **Don't use F5 output beyond personal use** until its weights license is
  checked.

## Known unknowns

Things we know we have not answered yet:

- **Tone:** does a reference clip in a different mood fix the tone? Untested.
- **More knobs:** seed, cfg, speed and more steps. Untested; see
  [TUNING.md](../../experiments/f5-say/TUNING.md) section 5.
- **The reference transcript:** reused from Colab and never checked word for
  word against the 10.6 s clip. A mismatch can make the pacing stiff.
- **A quiet machine:** every Mac number was measured with other apps open.
- **Bigger Macs:** speed on an M-series Pro or Max is a guess, not measured.
- **Licenses:** F5-TTS weights (likely CC-BY-NC), IndicF5 and Chatterbox
  weights. All marked "verify".
- **IndicF5** for Hindi and Marathi: supports both, gated on Hugging Face,
  not tested.
- **Watermark:** `f5say` output gets the cyv watermark during convert. We did
  not run `cyv watermark detect` on it.
