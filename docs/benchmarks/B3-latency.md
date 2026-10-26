# B3: Warm latency for a 10 s line + MPS policy

## Question
How slow is a warm 10 s line, and which device should each component run on?
(DESIGN §10: "M1 MPS bug on macOS 26" → per-component CPU fallback.)

## Method
Same single-process script as B2 (`/tmp/cyv-heavy/bench_spikes_m1.py`), plus
`/tmp/cyv-heavy/bench_hubert.py` (ContentVec feature-extract leg, exactly as
`VoiceConverter.voice_conversion` calls it: `model(feats)["last_hidden_state"]`)
and `/tmp/cyv-heavy/bench_rmvpe.py` / `bench_cpu.py` (RMVPE + HuBERT on both
devices). The 10 s line is real Kokoro output: 178 chars of English →
10.83 s @ 24 kHz (`af_heart`). "Warm" = third-or-later rep; first reps are
reported separately because MPS kernel compilation dominates them.

**No trained voice pack exists**, so `net_g.infer` + FAISS lookup (the middle
of every convert) are unmeasured. Component costs around them are real.

## Environment
Apple M1, 8 GB, macOS 27.2, torch 2.11.0, `PYTORCH_ENABLE_MPS_FALLBACK=1`.
cyv 0.0.1 @ `3e5c009`. Timings are wall clock, uncontended machine.

## Results

SDPA probe (the macOS 26 question):

| Check | Result |
|---|---|
| `F.scaled_dot_product_attention` on MPS fp32 tensors | **passes**, correct shape, no error |
| ContentVec `config._attn_implementation` after load to MPS | `sdpa` — SDPA is on the real inference path and works |
| MPS failure / fallback triggered during any bench | none; `engine._MPS_MARKERS` retry never fired |

No MPS SDPA failure on torch 2.11.0 + macOS 27.2. The macOS 26 bug from
DESIGN §10 does not reproduce here; the engine's CPU-retry stays as a safety
net, not as the normal path.

Warm latency per 10 s of audio (seconds wall):

| Component | Device | Cold (1st) | Warm (reps) |
|---|---|---|---|
| Kokoro synth (10.8 s line) | MPS | 6.8–16.0 (kernel compile; varies run to run) | **1.38, 1.42, 1.43** |
| ContentVec feature extract | MPS | — | **0.20, 0.20, 0.21** |
| ContentVec feature extract | CPU | — | 0.33, 0.34, 0.34 |
| RMVPE f0 | MPS | 2.65 (kernel compile) | **0.33, 0.35, 0.45** |
| RMVPE f0 | CPU | — | 0.74, 0.76, 0.78 |
| AudioSeal embed | CPU ≈ MPS | 4.5 / 1.0 (model load) | 0.26–0.29 either device |
| AudioSeal detect | CPU | — | **0.14** (MPS: 0.62–0.69, see `audioseal-mps.md`) |

Preset encodes (ffmpeg, after watermark): master 0.12 s, telegram/Opus 32k
0.16 s, mp3 0.10 s — negligible next to synthesis.

What a warm `say` of a 10 s line costs (measured parts in **bold**):

TTS **1.4** + f0 **0.35** + hubert **0.2** + net_g/FAISS unknown +
watermark **0.27** + encode ~0.15 ≈ **~2.1 s + net_g/FAISS**.

## Conclusion
- **Bad news, stated plainly:** the provisional "warm ≤1.5 s" target
  (DESIGN §3) is already missed by TTS + f0 + embed alone (~2.0 s for a
  10 s line on this M1), before the voice network runs. Either the target
  moves or the pipeline gets faster (streaming/chunked synth is the obvious
  lever — Kokoro is ~7.6× real-time when warm, so time-to-first-audio can
  still feel instant). Do not put the 1.5 s number in the README.
- **Policy** (matches `engine.py`: MPS first, `_MPS_MARKERS` CPU retry):

| Component | Policy | Evidence |
|---|---|---|
| Kokoro TTS | MPS | 1.4 s/10 s warm; bundled espeak dylib loads, no fallback needed |
| ContentVec embedder | MPS | 0.20 s vs 0.34 s CPU; SDPA path healthy |
| RMVPE f0 | MPS | warm 0.33–0.45 s vs 0.74–0.78 s CPU. First call costs ~2.6 s (kernel compile) — pay it at daemon warmup, not on the first request |
| `net_g` + FAISS | MPS (unchanged default) | unmeasured — no voice pack; retry covers Metal errors |
| AudioSeal embed + detect | **CPU** | embed ties, detect is 4.5× faster on CPU; MPS path also triggers `torch._dynamo` recompile warnings (see `audioseal-mps.md`) |
| CREPE path (not default) | CPU on MPS | unchanged Applio behavior (`pitch_device` override); unmeasured here |

- The pipeline comment claiming "FCPE/RMVPE keep their FFT frontend on CPU
  and neural model on MPS" is aspirational for RMVPE: the code only forces
  CPU for CREPE. It happens to be the fast choice for RMVPE too (MPS wins
  warm), so no change needed — but the comment overstates what the code does.

## Caveats
- No voice pack: the central convert cost (`net_g.infer`, FAISS search over a
  real index) is missing. A 10 s `say`/`convert` total cannot be quoted yet.
- Kokoro cold latency varies 2× run to run (6.8 vs 16.0 s) — MPS kernel
  cache state. Warm reps are stable (±0.05 s).
- One M1, no thermal-throttle testing, no concurrent-load testing. The daemon
  serializes GPU work (§4.3), so these single-stream numbers are the right
  shape, but queueing under load is unmeasured.
- RMVPE MPS reps (0.33–0.45 s) have more spread than HuBERT; possibly
  thermal or allocator noise. Median ~0.35 s.
