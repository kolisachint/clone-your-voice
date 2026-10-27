# B7: Voice-cloning TTS (F5-TTS on MLX, Chatterbox on MPS) on an 8 GB M1

## Question
Can a voice-cloning text-to-speech model (F5-TTS or Chatterbox) run on an
8 GB M1 Mac, and is it fast enough to use instead of Kokoro (B3: about
1.4 s per 10 s line)?

## Method
Three short English sentences, the same for both models, each cloned from one
10.6 s reference clip (24 kHz, mono). No transcript of the reference is
included in this note.

1. "Hello, this is a test of my cloned voice. How does this sound to you?"
2. "I was thinking we could grab lunch tomorrow, maybe around one, if that
   works for you."
3. "Honestly, I did not expect the results to be this good. Let me know what
   you think."

- **F5:** `f5-tts-mlx` (Apple MLX), fp16 weights. One run per sentence.
- **Chatterbox:** `chatterbox-tts` on PyTorch MPS (`ChatterboxTTS`,
  default exaggeration and cfg). Script: `~/.cache/tts-spike/bench_cb.py`,
  3 sentences x 2 reps in one process, run under `/usr/bin/time -l`. WAVs are
  from the second rep.
- After synthesis, each Chatterbox WAV went through
  `cyv convert --voice sachinkoli2 --preset master` (cyv master, daemon cold
  start included in the first call).
- One heavy process at a time. Weights were already in the local cache
  (`HF_HUB_OFFLINE=1`), so no download time is included.
- Sanity check on every output: `ffprobe` duration and `ffmpeg volumedetect`.
  **Nobody listened to the Chatterbox files** (see Notes and limits).

## Environment
Apple M1, 8 GB, macOS 27.2, `cyv --version` 0.0.1, repo commit `5acb5b4`
(branch `agent/mac-spike`). f5-tts-mlx 0.2.6, mlx 0.32.4. Chatterbox venv:
Python 3.11.17, PyTorch MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
Wall-clock timings. The machine was **not idle**: other apps were open and
the system was swapping (see below).

## Results

### F5-TTS (MLX, fp16)

One local patch was needed: in the venv's `cfm.py` line 373, `dur` became
`int(dur)`, because the code does not work with mlx 0.32 otherwise.

Defaults (rk4 solver, 8 steps, built-in duration predictor):

| Check | Result |
|---|---|
| Time per sentence | 60 to 136 s |
| Clip length | about 2.8 s for every sentence, whatever its length |
| 8-bit weights | slower than fp16 (96 to 136 s vs about 60 s) |

The duration predictor output looks wrong, so the clips were cut short. This
default setup is **not usable**.

Working setup (euler solver, 16 steps, duration from a text-length ratio):

| Sentence | Gen time (s) | Audio (s) | RTF (gen / audio) |
|---|---|---|---|
| 1 | 38.2 | 4.67 | 8.2 |
| 2 | 44.2 | 5.75 | 7.7 |
| 3 | 35.7 | 5.61 | 6.4 |

- Model load: 0.5 s.
- MLX peak memory about 2.6 to 2.8 GB; process peak footprint about 3.3 GB.
- Swap was 2.3 GB used at the start of the first runs and about 1.0 GB later.
  The early numbers may be inflated by swapping.

Through the cyv daemon (`cyv convert`, sachinkoli2, master): first call 26.8 s
(daemon cold start), warm calls 5.3 s and 3.7 s. Daemon RSS about 1.1 GB.

### Chatterbox (PyTorch MPS)

Device: **MPS worked** (no fallback to CPU was needed). Model load: **23.4 s**.

| Sentence | Rep | Gen time (s) | Audio (s) | RTF |
|---|---|---|---|---|
| 1 | 1 | 98.7 | 4.84 | 20.4 |
| 2 | 1 | 44.5 | 4.80 | 9.3 |
| 3 | 1 | 68.3 | 4.80 | 14.2 |
| 1 | 2 | 66.1 | 3.84 | 17.2 |
| 2 | 2 | 65.5 | 4.96 | 13.2 |
| 3 | 2 | 62.4 | 5.00 | 12.5 |

Second-rep range: 62 to 66 s per sentence, RTF 12.5 to 17.2. The reps are not
faster than the first, so there is no useful "warm" speed-up here. Per-step
speed inside one generation swung between about 8 and under 1 steps per
second (one run stalled for 8 to 10 s mid-way), which is the pattern you get
when memory is under pressure.

Memory (whole 460 s process, `/usr/bin/time -l`):

| Measure | Value |
|---|---|
| Max resident set size | 3.18 GB |
| Peak memory footprint | **11.2 GB** (more than the 8 GB of RAM) |
| MPS driver allocated memory (peak) | **8.4 GB** (`torch.mps.driver_allocated_memory`) |

Swap (`vm.swapusage`, total size shown by macOS):

| When | Used | Total |
|---|---|---|
| Before (daemon stopped) | 957 MB | 2048 MB |
| After the Chatterbox run | 3013 MB | 6144 MB |
| After the three converts and `cyv stop` | 1081 MB | 2048 MB |

macOS grew its swap file from 2 GB to 6 GB during the run, then shrank it
again.

`cyv convert` of the Chatterbox WAVs (master preset):

| Sentence | Convert time (s) |
|---|---|
| 1 (daemon cold start) | 24.8 |
| 2 | 5.1 |
| 3 | 3.2 |

### Output sanity check (no listening)

| File | Duration (s) | Mean volume (dB) | Max volume (dB) |
|---|---|---|---|
| raw s1 | 3.84 | -28.3 | -6.7 |
| raw s2 | 4.96 | -28.3 | -6.5 |
| raw s3 | 5.00 | -27.3 | -5.9 |
| converted s1 | 3.82 | -20.5 | -1.6 |
| converted s2 | 4.94 | -20.9 | -1.3 |
| converted s3 | 4.98 | -19.6 | -1.1 |

All six are 3.8 to 5.0 s with clear signal (mean far above -60 dB). **No
2.8 s truncation** like the F5 default had. These checks cannot tell whether
the speech is correct or sounds like the speaker.

### Watermark (Chatterbox's own Perth)

`import perth; perth.PerthImplicitWatermarker()` **failed on macOS**
(`TypeError: 'NoneType' object is not callable`; the class is `None` after
perth's internal import fails). The benchmark therefore used a **mock**
watermarker that returns the audio unchanged. These Chatterbox WAVs carry
**no Perth watermark**. cyv's own watermark is applied later, in the convert
step.

### Comparison

| | Kokoro (B3) | F5-MLX (working setup) | Chatterbox (MPS) | Colab T4 |
|---|---|---|---|---|
| Time per sentence | about 1.4 s per 10 s line | 36 to 44 s for 4.7 to 5.8 s | 62 to 66 s for 3.8 to 5.0 s | about 4 s (F5), about 5.4 s (Chatterbox) |
| RTF | about 0.13 | 6.4 to 8.2 | 12.5 to 17.2 | below 1 |
| Peak memory | not part of this note | about 3.3 GB footprint | 8.4 GB MPS, 11.2 GB footprint | n/a |

## Licenses
Not yet added to `docs/licenses.md`.

| Piece | Code | Weights |
|---|---|---|
| F5-TTS / f5-tts-mlx | MIT | verify (upstream F5-TTS weights are CC-BY-NC, which would block use in a product) |
| Chatterbox | MIT | MIT per the Hugging Face model card (verify) |

## Notes and limits
- One machine, one reference clip, three sentences, one listener on Colab.
  Treat every number as a single data point, not a distribution.
- The machine was under memory pressure the whole time (other apps open, swap
  already in use). F5's early numbers and all Chatterbox numbers are probably
  slower than a clean 8 GB M1 would give. Chatterbox's 11.2 GB footprint means
  that on 8 GB it can only run by swapping, so a quiet machine would not
  remove that.
- Chatterbox rep 1 vs rep 2 differ a lot (for example sentence 1: 98.7 s vs
  66.1 s), so the spread is large. I did not repeat enough to give a median.
- The F5 default setup is broken (wrong duration, clips cut short). The
  working numbers rely on a hand-made duration estimate and a local patch to
  a third-party file, neither of which would ship as is.
- Quality: in a Colab blind test (3 sentences, one listener, "very close"),
  F5 followed by `cyv convert` ranked first. The Mac-made F5 files and **all
  Chatterbox Mac files have not been listened to**. The Chatterbox files were
  only checked for length and loudness.
- Kokoro is a plain TTS voice that is then converted. The F5 and Chatterbox
  models clone directly, so they could skip or add to the convert step. This
  note does not decide that; it only measures speed and memory.
- The mock watermark means Chatterbox's own watermarking was not measured.

## Verdict
- **Does it fit on an 8 GB M1?**
  - **F5-MLX: yes**, with room to spare (about 3.3 GB footprint).
  - **Chatterbox on MPS: only barely, and badly.** It ran to the end, but it
    asked for 8.4 GB of GPU memory and an 11.2 GB footprint, which pushed macOS
    to 3 GB of swap. It also cannot sit next to the cyv daemon (about 1.1 GB)
    and normal apps without more swapping.
- **Is the speed usable?** **No, for either.** Kokoro makes a 10 s line in
  about 1.4 s. F5-MLX takes about 36 to 44 s for a 5 s line (RTF 6 to 8), so
  roughly 50 times slower than Kokoro per second of audio. Chatterbox takes
  about 62 to 66 s for a 4 to 5 s line (RTF 12 to 17), roughly 100 times slower.
  A Colab T4 does either in about 4 to 5 s.
- **Where this leaves the options:** F5-MLX is the only one that fits
  comfortably, and it is usable only for slow, offline "make a clip, come back
  later" work, and only after fixing its duration handling. Chatterbox is not
  a good fit for 8 GB Macs. Neither should replace Kokoro as the default
  `say` voice. Cloned TTS stays a Colab or bigger-Mac option for now.
- **Before relying on this:** listen to the Mac-made F5 and Chatterbox
  files, rerun with a quiet machine (nothing else open), and check the F5
  weight license before shipping anything.
