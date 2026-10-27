# Tuning log: voice-cloning TTS (F5-TTS + cyv convert)

Everything we ran, with the exact weights and settings, so the next session
can pick up and tune without redoing it. Results in short: B7
(`docs/benchmarks/B7-clone-tts-m1.md`). How to run: [README.md](README.md).

All numbers: Apple M1, 8 GB, macOS 27.2, cyv 0.0.1 (repo `69bbf84`), one
listener, three or fewer sentences. Treat them as single data points.

## 1. What we used

### Voice model (convert step)

| | |
|---|---|
| cyv voice | `sachinkoli2` (Applio RVC v2, 40 kHz, HiFi-GAN, contentvec, rmvpe) |
| Training | 200 epochs, 922.85 s of dataset audio, Applio `c817f1f8` |
| model.pth sha256 | `e076d15d140ef12a…` (full hash in its `voice.json`) |
| Fixed convert settings | index rate 0.75, protect 0.5, volume envelope 1.0, pitch 0 |
| Preset | `master` (48 kHz WAV) |

### F5-TTS (the winner)

| Piece | Value |
|---|---|
| Package | `f5-tts-mlx` 0.2.6 (MLX port of F5-TTS) |
| Runtime | mlx 0.32.4, mlx-metal 0.32.4, vocos-mlx 0.0.7, Python 3.11.17 |
| HF repo | `lucasnewman/f5-tts-mlx` @ `2d719cadec8fd3887c8599475a5f65924d523654` |
| Main weights | `model.safetensors` (fp16, 1,348,433,688 bytes, sha256 `284cf8e33cb9c197…`) |
| Duration model | `duration_v2.safetensors` (86 MB, `25c4701cf765a917…`). Not used by f5say (see 3.3) |
| Vocab | `vocab.txt` (2,545 tokens: Latin + Chinese, **no Devanagari**; `4e173934be56219e…`) |
| Vocoder | `lucasnewman/vocos-mel-24khz` @ `ee89d00fac94b61e1235e0869f3e8d49711b7484`, `model.safetensors` (54 MB, `7fd11dcafe62b373…`) |
| Tried, rejected | `model_8b.safetensors` (8-bit, 398 MB, `72d9d059c6d93294…`): slower, see 3.2 |
| Colab version | `f5-tts` (PyTorch, official), default model, default `infer()` settings |
| Local patch | `cfm.py` line 373: `dur` -> `int(dur)` (mlx 0.32 crash). Original kept as `cfm.py.orig` |

### Chatterbox (ruled out on the Mac)

| Piece | Value |
|---|---|
| Package | `chatterbox-tts` 0.1.7, torch 2.6.0, torchaudio 2.6.0, transformers 5.2.0 |
| HF repo | `ResembleAI/chatterbox` @ `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` |
| Weights | `t3_cfg.safetensors` (2.1 GB, `914cb1696f47527f…`), `s3gen.safetensors` (1.06 GB, `2b78103c65420739…`), `ve.safetensors` (5.7 MB, `f0921cab452fa278…`), `tokenizer.json`, `conds.pt` |
| Settings | defaults (exaggeration 0.5, cfg_weight 0.5), device `mps` |
| Watermark | `resemble-perth` 1.0.1 fails to import on macOS and Colab; we used a mock (no Perth watermark) |

### Reference clip (what F5 copies voice and tone from)

| | |
|---|---|
| Source | Voice Memo "New Recording 33" (46.1 s, me explaining the quality problem) |
| Colab | whole 46 s, mono 24 kHz, no trimming; F5 transcribed it itself (`ref_text=''`) |
| Mac | `~/.cache/tts-spike/ref/ref_11s.wav`: 0.85 s to 11.45 s of that memo (10.6 s), mono 24 kHz, no loudness change. sha256 `3810dfe3f19bcdf8…` |
| Mac transcript | `ref_11s.txt`: Colab's auto-transcript, reused. **Not checked word for word against the 10.6 s cut** |
| Tone | calm, explanatory. This is the tone every output gets |

Never commit the clip or its transcript.

### Test sentences

- **A.** "Hello, this is a test of my cloned voice. How does this sound to you?"
- **B.** "I was thinking we could grab lunch tomorrow, maybe around one, if that works for you."
- **C.** "Honestly, I did not expect the results to be this good. Let me know what you think."
- **D.** "Sorry I missed your call earlier. I was in a meeting, but I am free now if you want to talk."

## 2. Settings in f5say (current best)

```text
--steps 32  --method euler  --cfg 2.0  --sway -1.0  --speed 1.0  --seed 42
duration = ref_frames + ref_frames * len(text bytes) / len(ref_text bytes) / speed
MLX cache limit 256 MB, then cyv convert --voice sachinkoli2 --preset master
```

## 3. Every run, in order

### 3.1 Colab blind test (quality, T4 GPU)

Script: `experiments/tts-clone-test/` (branch `tts-clone-test`). Sentences A, B, C.
F5 about 4 s per sentence, Chatterbox about 5.4 s.

5 versions per sentence, shuffled names, ranked by ear before reading the key:

| Rank | File | Version |
|---|---|---|
| 1 | s2_C | F5 -> convert |
| 2 | s2_A | Chatterbox -> convert |
| 3 | s3_E | F5 -> convert |
| first cut, all 3 sentences | s1_E, s2_B, s3_B | **Kokoro say (pitch -6)** |

"All very close." Kokoro was cut first every time, blind.

### 3.2 F5 on the Mac: finding settings that work

| Run | Bits | Method / steps | Duration | Sentence: gen s / audio s | MLX peak | Verdict |
|---|---|---|---|---|---|---|
| smoke | 8 | rk4 / 8 | predictor | A 136.1/2.79, B 123.5/2.94, C 94.0/2.95 | 1.7 GB | clips cut to ~2.8 s; very slow |
| smoke2 | 8 | rk4 / 8 | predictor | A 116.0, then 96.2 (2.79 s) | 1.7 GB | 8-bit slower than fp16 |
| smoke3 | fp16 | rk4 / 8 | predictor | A 60.1/2.78 | 2.7 GB | still cut to ~2.8 s |
| euler | fp16 | euler / 16 | predictor | B 34.3, 35.2 (2.93 s) | 2.8 GB | faster; still cut |
| est | fp16 | euler / 16 | **text ratio** | A 38.2/4.67, B 44.2/5.75, C 35.7/5.61 | 2.6 GB | right lengths. Listened (B): "close but not that good" |
| s32 | fp16 | euler / 32 | text ratio | B 73.1/5.75 | 2.6 GB | **"very close to what I wanted"** |
| en32 | fp16 | euler / 32 | text ratio | A 67.5/4.67, C 71.6/5.61, D 77.7/6.22 | 2.6 GB | "good, but tone is not right" |
| f5say test | fp16 | euler / 32 | text ratio | "Good night, talk to you tomorrow." 2.2 s in 96 s (incl. daemon start) | — | wrapper works end to end |

Notes:
- rk4 does 4 model passes per step, euler 1. So rk4/8 is about the same
  work as euler/32, but in our runs it came with the broken duration.
- Swap was 2.3 GB at the first runs (apps open), about 1 GB later. The
  early timings are inflated.
- Process peak footprint was about 3.3 GB for fp16.

### 3.3 Why the text-ratio duration

F5's duration model gave about 2.8 s for every sentence, whatever its
length, so speech was cut off. f5say instead sets the total length from my
speaking rate in the reference clip (bytes of text per frame). Side effect:
my pace in the reference sets the pace of the output; `--speed` scales it.

### 3.4 Chatterbox on the Mac (MPS)

Sentences A, B, C x 2 reps. Load 23.4 s. Rep 2: 66.1 / 65.5 / 62.4 s for
3.84 / 4.96 / 5.00 s of audio. MPS driver memory peaked at **8.4 GB**,
footprint 11.2 GB, swap grew from 1 GB to 3 GB. Not listened to. Ruled out
for 8 GB Macs.

### 3.5 cyv convert step

First call about 25 to 27 s (daemon cold start), then 3 to 5 s per line.
Daemon RSS about 1.1 GB.

### 3.6 Kokoro say baseline (before all this)

`cyv say` with `sachinkoli2`, guide voice `am_adam`, pitch -6 / -3 / 0 / +3
/ +6: -6 was best, still "not what I want". Fixed in v1: guide voice, index
rate, protect.

## 4. What changed the sound (as heard)

| Change | Effect |
|---|---|
| Kokoro -> F5 (+ convert) | big win: delivery starts to sound like me |
| + cyv convert after F5 | preferred over raw F5 in the blind test (all 3 top picks were converted) |
| 16 -> 32 euler steps | "sense of loss" -> "very close". Biggest single knob after the model |
| Reference clip tone | sets the tone of every line. Calm clip = calm output |
| 8-bit weights | slower on M1, not listened to |

## 5. Knobs not tried yet (in order of expected value)

1. **Reference clip** (`--ref`): 8 to 12 s, in the tone you want, with an
   exact transcript. Expect the biggest tone change. Also check the current
   transcript word for word.
2. **Seed** (`--seed 1`, `2`, `3`): free variation in delivery, same cost.
   Make 3 takes, keep the best.
3. **Steps** (`--steps 48` or `64`): may add detail; about 105 or 140 s per
   line. Diminishing returns are likely.
4. **cfg** (`--cfg 2.5`, `3.0`): follows the reference voice and text harder.
   Too high can sound strained.
5. **Speed** (`--speed 1.1`): faster, livelier pace.
6. **Method** (`--method midpoint` at 16 steps): same work as euler 32,
   maybe smoother.
7. **Pitch on convert**: f5say uses the voice's fixed pitch (0). Not exposed yet.

Tune one knob at a time on the same sentence (use B: it has a baseline at
16 and 32 steps), keep the seed fixed, and listen blind where you can.

## 6. Not tested

- Hindi and Marathi. This F5 vocab cannot read Devanagari. IndicF5
  (`ai4bharat/IndicF5`, gated, MIT per card, trained on research data:
  verify) supports both.
- A quiet machine (nothing else open). Timings would likely improve.
- F5 on Colab with the Mac settings (euler 32, text-ratio duration).
- The F5 weights license (upstream F5-TTS weights are CC-BY-NC). Verify
  before anything leaves personal use.
