# ADR-005: Typed text in your voice: a two-phase cloned-TTS pipeline

Status: Proposed (not accepted, nothing built yet)

This is a design proposal from a discussion in October 2026. It changes D6
("cloud training + local inference") for one new path, so it needs to be
accepted before any code is written. Every model, licence and number marked
"verify" or "estimate" is unchecked.

## Context

### What we learned

A voice has three parts:

| Part | What it is | Who sets it today |
|---|---|---|
| Words | What is said | The text |
| Prosody (delivery) | Rhythm, melody, emphasis, pauses, tone | Whatever makes the speech first |
| Timbre | The sound of the voice itself | RVC (Applio) |

- **RVC only swaps timbre.** It keeps the pitch shape and timing of its input
  exactly. `--pitch` only moves the pitch up or down.
- **So prosody is fixed by the text-to-speech (TTS) step.** In `cyv say`,
  Kokoro sets it, which is why `say` never sounds like you
  ([sounding-like-you guide](../guides/sounding-like-you.md)).
- **F5 → RVC won the blind test** on every English sentence (B7). F5 copies
  prosody, accent and tone from a short reference clip of you.
- **F5 is too slow on an 8 GB M1:** about 70 s per sentence at 32 steps, against
  about 4 s on a Colab T4 (B7). Fewer steps lose audible quality. ONNX would
  not help: it changes the file format, not the amount of work.
- **No model handles Hinglish out of the box.** F5 has no Devanagari. IndicF5
  reads Hindi and Marathi but its English is weak or unknown.

### Our input

The text is **written by an AI** that we prompt ourselves. It mixes Devanagari
and Latin script, for example "कल lunch पे मिलते हैं". Because we control the
prompt, we can ask for a format that is easy to route (see "Text contract").

### Goal

Typed English, Hindi, Marathi and Hinglish that sounds like you. That means your
prosody and accent as well as your timbre. First at the best quality we can get,
then fast enough to run on the Mac.

## Decision (proposed)

Two phases. They share one text front end.

### Phase 1: quality pipeline on Colab (F5 + IndicF5 → RVC)

Generation runs on Colab, as a **batch**: one text file in, one folder of
watermarked WAVs out. The Mac only applies the output presets (ffmpeg), as
today.

```
AI text (Devanagari + Latin, one tagged line per sentence)
  ① normalise   numbers, ₹, times, short forms → spoken words
  ② split       into sentences
  ③ route       by script (and the language tag) → F5 or IndicF5
  ④ adjust      rewrite words into the chosen model's script
  ⑤ speak       chosen model + matching reference clip (en / hi / mr)
  ⑥ join        short pauses between sentences, one file per line
  ⑦ RVC → watermark → WAV to Drive
Mac: download, then cyv presets (telegram, whatsapp, ...)
```

#### Routing rules, per sentence

| Sentence contains | Model | Text change | Reference clip |
|---|---|---|---|
| Latin only (pure English) | F5 | none | English |
| Devanagari only | IndicF5 | none | Hindi or Marathi, from the tag |
| Both (Hinglish) | IndicF5 | Latin words rewritten in Devanagari ("lunch" → "लंच") | Hindi or Marathi, from the tag |

- **A sentence is never split across two models.** Joining F5 and IndicF5 audio
  mid-sentence would change the voice and melody at the join.
- **Mixed sentences go to IndicF5** because F5 cannot read Devanagari at all, and
  English words in Devanagari come out with an Indian accent, which is what
  Hinglish should sound like.
- **Hindi vs Marathi** comes from the language tag. If a tag is missing, fall back
  to marker words (आहे, नाही, -च्या for Marathi) and say so in the report.
- **Fail loudly.** A character the pipeline cannot handle (emoji, another script)
  flags the line. It never produces silence or nonsense quietly.

#### Text contract (what we ask the AI to write)

- One sentence per line, starting with a tag: `[en]`, `[hi]` or `[mr]`.
- Hindi and Marathi in Devanagari, **including English words used inside them**.
- Latin script only for pure English sentences.
- Numbers, money, dates and times written out as words in that language.
- No emoji, no markup.

The router still handles text that breaks the contract (steps ① to ④). The
contract just makes mistakes rare.

#### Reference clips

- Three clips: English, Hindi, Marathi. 8 to 12 s each, clean and dry.
- Each with an **exact** transcript in its own script.
- Record them in the tone you want. Tone follows the clip (B7). More tones can be
  added later as extra clips.
- Never committed to git, like all voice data.

#### Routing report

For each line: the tag, the model used, the text after steps ① to ④, and
any warnings. It stays with the output on your Drive and is never committed.
Its purpose is to explain why a line sounds wrong.

#### Expected speed

About 4 to 5 s per sentence on a T4 for F5 or IndicF5 (F5 measured in B7,
IndicF5 assumed similar), plus RVC (about 1 s on a T4, estimate) and the GPU
start-up once per batch (1 to 3 min).

### Phase 2: fast pipeline on the Mac (Piper/VITS → RVC)

The same front end (steps ① to ③) feeds one fast model:

```
same normalise / split / tag
  → per-word language tag → phonemes (espeak: en, hi, mr, word by word)
  → one Piper/VITS model fine-tuned on you (ONNX, on the Mac)
  → RVC (optional; skip if it already sounds like you)
  → watermark → presets
```

| | Phase 1 | Phase 2 |
|---|---|---|
| The router chooses | the model | the phonemiser, word by word |
| Models | F5 + IndicF5 | one model for all three languages |
| Your voice | copied from a reference clip | trained on your recordings |
| Runs on | Colab (batch) | Mac, ONNX |
| Speed per sentence | about 4 to 5 s on T4 | about 0.2 to 1 s on M1 (estimate), plus 3 to 5 s if RVC is used |
| Quality | higher | lower (expected) |

- **Training:** fine-tune from a pretrained Piper/VITS checkpoint on Colab.
  Start with about 30 min of you (English, Hindi, Marathi and Hinglish, several
  moods, exact transcripts in the text-contract format). Record 1 to 3 hours only
  if the 30-minute spike wins.
- **Why phonemes word by word:** espeak handles one language at a time. Hindi words
  in Latin script would be read as English, which is why the text contract keeps
  Hindi and Marathi in Devanagari.
- **Phase 1 feeds Phase 2:** its best outputs are the quality bar to beat, and its
  routing reports show which kinds of text are hard.
- Once Phase 2 wins its blind test, generation moves back to the Mac and Colab is
  only for training again (D6 holds again).

### What stays the same

- RVC through Applio stays the timbre layer for every path.
- `cyv convert` (record it yourself) stays the best-quality path in every
  language.
- `cyv say` (Kokoro → RVC) stays the quick local path until Phase 2 replaces it.
- The watermark is on by default everywhere, including in the Colab notebook.
- Consent and the usage policy are unchanged: your own voice, or one you have
  explicit permission for.

### Dropped

- **Chatterbox.** It needs 8.4 GB of GPU memory on an 8 GB Mac (B7).

## Gates

Each phase only goes ahead if the one before it passes.

| Gate | Check | Pass means |
|---|---|---|
| P1-a | Blind test for English, Hindi, Marathi and Hinglish: Phase 1 output vs Kokoro → RVC (where Kokoro supports the language) | Phase 1 wins in each language |
| P1-b | Hinglish: is IndicF5 good with English words in Devanagari? | Sounds natural to you |
| P1-c | F5 and IndicF5 weight licences | Personal use only until verified |
| P2-a | Stock Piper voice timed on the 8 GB M1 | Confirms or replaces the 0.2 to 1 s estimate |
| P2-b | 30-minute fine-tune, blind test against Phase 1 | Close enough to justify recording 1 to 3 hours |
| P2-c | Piper licence | The GPL build only as a separate opt-in package (DESIGN §4.6) |

## Consequences

- **D6 changes for this path:** generation may run on Colab in batches until
  Phase 2 passes. `cyv say` stays local.
- **Colab rules:** Colab forbids "creating deepfakes". Making your own voice, with
  consent and a watermark, is defensible, but generating on Colab is more exposed
  than training there. The notebook must keep the consent record and watermark.
- **Privacy:** your text and reference clips are stored on your Google Drive, as
  the training dataset already is.
- **No instant typed-text-in-your-voice** until Phase 2. Phase 1 is a batch tool.
- **New dependencies** each need a row in `docs/licenses.md`: IndicF5, the
  Latin → Devanagari transliteration tool (not chosen yet), Piper. Mark each
  "verify".
- **Benchmarks:** one entry per phase in `docs/benchmarks/`, with hardware
  labelled.

## Open questions

1. Which transliteration tool rewrites English words in Devanagari, and how good is
   it on brand names and acronyms?
2. How does IndicF5 do with an English reference clip vs a Hindi one for
   Hinglish?
3. Can one Piper/VITS model learn three languages from 30 minutes, or does each
   language need its own data share?
4. Should Phase 2 skip RVC when its output is already close enough? Answered by
   the P2-b blind test.
5. Commercial use: F5 and XTTS weights are likely non-commercial. If this is
   ever shared beyond personal use, the Phase 1 models may need replacing.
