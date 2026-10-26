# AudioSeal cost: CPU vs MPS on the M1

## Question
What do AudioSeal embed + detect cost per 10 s on Apple Silicon, on CPU vs
MPS — and which device should the daemon's watermarker use? (DESIGN §6.3
ships the watermark in M2; B4 chose AudioSeal over Perth on Linux CPU but
explicitly deferred the Apple Silicon cost question.)

## Method
`AudioSealWatermarker` from the repo (`src/cyvd/watermark/audioseal_wm.py`)
against the sha256-pinned `generator_base.pth` / `detector_base.pth` in
`~/.cyv`-layout scratch models. Fixed 10.0 s clip: real Kokoro output
(`af_heart`, 24 kHz) resampled to the watermarker's native 16 kHz inside
`embed`/`detect`. 3 timed reps per device; rep 0 includes model load and is
reported separately. Detection threshold 0.5, `CYV_MESSAGE` payload,
round-trip verified (`msg == CYV_MESSAGE`). For B4-method comparison only:
same "seconds per 10 s audio" metric as `scripts/bench/b4_watermark_bench.py`.

## Environment
Apple M1, 8 GB, macOS 27.2, torch 2.11.0, audioseal 0.2.0,
`PYTORCH_ENABLE_MPS_FALLBACK=1`. cyv 0.0.1 @ `3e5c009`.

## Results

| Stage (per 10 s) | CPU | MPS |
|---|---|---|
| embed, rep 0 (incl. model load) | 4.46 s | 0.95 s |
| embed, warm reps | **0.28, 0.26 s** | 0.26, 0.29 s |
| detect, warm reps | **0.14, 0.14, 0.14 s** | 0.69, 0.62, 0.64 s |
| detect prob on watermarked clip | 1.0 | 1.0 |
| message round-trip (`msg == CYV_MESSAGE`) | yes | yes |

- Embed is a tie warm (~0.27 s either device). Detect is **~4.5× faster on
  CPU** (0.14 s vs ~0.64 s).
- The MPS path emits `torch._dynamo hit config.recompile_limit` warnings
  (dispatch-key mismatch CPU↔MPS in AudioSeal's `seanet`/`conv` modules):
  the model recompiles when it moves device. Harmless here, but it is
  overhead the CPU path never pays.
- Against B4 (Linux x86, contended 4-vCPU box, torch 2.14.1+cpu): B4
  measured AudioSeal at 4.4 s embed / 6.1 s detect per 10 s CPU. This M1 does
  0.27 / 0.14 s — ~16× / ~40× faster. The boxes, torch builds, and load
  differ so the ratio is not a clean benchmark, but the direction is
  unambiguous: on the M1 the watermark is ~0.4 s of a ~2 s pipeline, not the
  dominant cost B4 feared ("re-measure on Apple Silicon before committing a
  latency budget" — done, and the budget holds).

M1-made audio through the real preset (E2E smoke, `/tmp/cyv-heavy/bench_e2e_smoke.py`):
Kokoro synth → `embed` → `master`/`telegram`/`mp3` encodes all succeed, and
the watermark reads back after the Opus 32k voice-note preset with
**detect prob 1.0 and exact message match** — consistent with B4's 20/20.

## Conclusion
- **Policy: run the AudioSeal watermarker on CPU** (both embed and detect;
  no reason to split devices for a 0.4 s stage). This also keeps Metal
  memory for TTS/RVC and avoids the dynamo recompile noise.
- Audibility is unchanged from B4 (same models, same 16-bit payload, mean
  SNR 28.5 dB on LibriSpeech); no new listening test was needed or done.
- No M2 latency-budget blocker from watermarking.

## Caveats
- One clip (Kokoro TTS output, not LibriSpeech), n=3 reps. B4's robustness
  matrix (attacks, FPR on 50 clean clips) is not repeated here — this doc is
  cost-only; robustness still rests on B4.
- Perth was not re-run on the M1 (B4 already settled default-vs-fallback and
  nothing here reopens it).
- CPU/MPS numbers are warm; a cold daemon pays one ~4.5 s CPU-side model
  load (or ~1 s MPS-side) on first watermark — amortize at daemon warmup.
