# B2: Resident memory on the 8 GB M1

## Question
Can the warm daemon (HuBERT/ContentVec + RMVPE + Kokoro + 1 voice) live
comfortably on an 8 GB M1 Mac (DESIGN §10 risk "8 GB memory pressure")?

## Method
Script: `/tmp/cyv-heavy/bench_spikes_m1.py` (scratch, not committed).
One process loads the full warm set in order — Kokoro (mps), AudioSeal
gen+detector (cpu and mps), Applio converter + ContentVec (mps), RMVPE (mps) —
then reports `resource.getrusage(RUSAGE_SELF)`. The whole run is wrapped in
`/usr/bin/time -l`, whose `maximum resident set size` is the authoritative
peak. Every downloaded weight was sha256-verified by `cyv setup`
(`src/cyv/models.toml`).

**No trained voice pack exists on this machine** (training is Colab-only;
B1 stays Colab-T4), so the voice-conditioned convert — `net_g` (~55 MB
`.pth`) + FAISS index (tens of MB) + convert activations — is **not**
included. The gap is estimated, not measured (§Results).

## Environment
Apple M1, 8 GB unified memory, arm64, macOS 27.2, cyv 0.0.1
(worktree `agent/spikes-m1` @ `3e5c009`). Daemon venv
(`/tmp/cyv-heavy/venv`, Python 3.12.15, built with `uv` per `src/cyv/setup.py`
plan): torch 2.11.0, torchaudio 2.11.0, numpy 2.4.6, transformers 5.13.1,
kokoro 0.9.4, audioseal 0.2.0, spacy 3.8.16, preshed 3.0.9, faiss-cpu 1.14.3,
soxr 1.1.0, en_core_web_sm 3.8.0. `PYTORCH_ENABLE_MPS_FALLBACK=1`.
Nothing else heavy was running; no swap observed (`time -l`: 0 swaps).

## Results

Peak RSS, everything loaded except the voice (`/usr/bin/time -l`):

| Measure | Peak RSS |
|---|---|
| `maximum resident set size` | **1,351,122,944 bytes (~1.35 GB)** |
| self `ru_maxrss` at end of run | 1288.5 MB |

RSS after each load phase (self `ru_maxrss`, MB):

| Phase | RSS (MB) |
|---|---|
| torch import + SDPA probe | ~210 |
| + Kokoro (mps, incl. first synth) | ~1282 |
| + AudioSeal cpu and mps | ~1282 (no visible step) |
| + Applio converter import + ContentVec (mps) | ~1282 (no visible step) |
| + RMVPE (mps) + f0 on 10 s clip | ~1288 |

Almost the entire footprint appears at Kokoro load: the MPS caching
allocator reserves a large block up front, and everything after Kokoro
(ContentVec 378 MB of weights, RMVPE 181 MB, AudioSeal ~93 MB) fits inside
the noise (+6 MB self-reported). This is allocator behavior, not proof the
later models are free — but the `/usr/bin/time -l` peak is what the OS saw.

Missing voice, estimated (not measured):

| Piece | Size on disk / estimate |
|---|---|
| `model.pth` (RVC v2) | ~55 MB weights (DESIGN §3) |
| `model.index` (FAISS) | tens of MB (DESIGN §3: "tens of MB or more") |
| convert transient (f0 buffers, feats, `net_g.infer` activations) | unknown, small vs. model weights |

For scale: the Linux CPU E2E run *with* a voice (`docs/benchmarks/linux-cpu-e2e.md`)
peaked at 2.9 GB RSS. On this M1 the measured base is 1.35 GB, so a warm
daemon with one voice should land near ~1.5–1.8 GB — comfortably inside 8 GB
with room for the OS, a browser, and the ffmpeg encode subprocess. That is an
extrapolation, not a measurement: B2 is only fully closed once a real pack is
imported and a convert runs under `time -l`.

## Conclusion
- **Good news:** 1.35 GB measured for the full warm set minus the voice.
  The 8 GB pressure risk looks manageable; nothing here suggests a need for
  the two-venv fallback or weight quantization in v1.
- **Open:** the voice delta. Re-run this script's tail (load pack +
  `engine.convert`) under `time -l` as soon as a Colab-trained pack exists.
- The provisional ≤2.5 GB target (DESIGN §3) is consistent with these numbers
  but stays provisional — and stays out of the README — until the with-voice
  measurement exists.

## Caveats
- One machine, one run (plus a partial rerun); MPS allocator behavior may vary
  with OS version and what else is using the GPU.
- `ru_maxrss` is the process high-water mark; `time -l` additionally counts
  `peak memory footprint` (4.3 GB incl. shared) — quoted above is resident.
- No voice pack, no convert: see above. The FAISS-index memoization in
  `engine.py` (`_IndexCache`) exists precisely to keep repeat converts from
  re-reading tens of MB per call; its effect is unmeasured here.
