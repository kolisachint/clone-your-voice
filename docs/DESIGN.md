# clone-your-voice — Design (v0.4, signed off for M1 spikes)

Status: **design accepted. The Linux-testable scaffold is implemented (§11). B4–B6 are done on Linux CPU. B0–B3 need an Apple Silicon Mac.**

## 0. Decisions log

| # | Decision |
|---|---|
| D1 | License: **MIT** (matches Applio) |
| D2 | Default text-to-speech: **offline Kokoro**. Edge-TTS comes later as an opt-in. |
| D3 | Watermark: **on by default, user can turn it off**. SynthID-style invisible neural audio watermark (ADR-004). |
| D4 | Interface: **CLI only**. The daemon is internal and has no network port. |
| D5 | v1 platform: **macOS on Apple Silicon only**. Training is in the cloud. |
| D6 | "Hybrid" means **cloud training + local inference** |
| ADR-001 | Python, no Rust: the bottleneck is GPU math and cold start, not CPU |
| ADR-002 | Voice pack v1 format (§4.2) |
| ADR-003 | The daemon speaks length-prefixed JSON over a unix socket using plain `asyncio`, **not FastAPI/HTTP** (one local client; faster import, less surface) |
| ADR-004 | Watermarking sits behind a `Watermarker` interface. **Default backend: Meta AudioSeal (MIT)**, the open, offline equivalent of SynthID audio. **SynthID** becomes a backend if Google ever releases an embedding SDK. Perth is the fallback (§6.3). |

## 1. Goal and scope

Let anyone clone **their own** voice cheaply: train once on Colab, then run `cyv say "hello" --preset telegram` locally on an M1 Mac.

**v1 includes:** `setup`, `doctor`, `dataset prepare`, the Colab notebook, `voices import/list/remove`, `say`, `convert`, 4 output presets, the watermark, and a usage policy.

**Not in v1:** live mic, Windows/Linux, an HTTP API, cloning third-party voices, a GUI, a plug-in registry, multiple warm voices, launchd integration, a voice-tuning UI, and watermark detection (that moves to M3).

**Expectation to state in the README:** `say` = Kokoro's delivery with your timbre. Your *timbre*, not your natural rhythm or intonation. For the most "you" result, use `convert` on a recording of yourself performing the line.

## 2. End-to-end user journey

```
① record ─► ② cyv dataset prepare ─► ③ Colab notebook ─► ④ download pack ─► ⑤ cyv voices import ─► ⑥ cyv say / convert
  (Mac)       (Mac, CPU, seconds)      (cloud T4)          (browser, manual)    (Mac, sandboxed)        (Mac, warm daemon)
```

1. **Record** 5–10 min of clean, dry speech, following `docs/guides/recording.md` (benchmark B1 confirms the minimum).
2. **`cyv dataset prepare <dir>`**:
   - Converts to mono, resamples, trims silence, normalizes loudness.
   - Rejects clipping, music, too-short sets.
   - Asks for a one-line consent attestation.
   - Writes `dataset.zip`.
3. **Colab notebook**, using Applio's **NoUI/CLI** path (never the Gradio UI):
   - Install with a **pinned constraints file**.
   - Mount Drive and upload `dataset.zip`.
   - Run preprocess → extract → train → index, with checkpoints saved to Drive so a disconnect can resume.
   - **Self-test:** render a fixed sentence, show a loss summary, PASS/FAIL.
   - Export `<name>.cyvpack.zip` to Drive.
   - Clear failure messages for: no GPU, quota exhausted, Drive auth failed.
4. **Download** the pack from Drive in the browser. v1 has no Drive API integration.
5. **`cyv voices import <path>`**: hardened import (§6.1).
6. **`cyv say` / `cyv convert`**: the CLI starts the daemon on demand, and the daemon exits after it has been idle for a while.

## 3. Facts and benchmarks

**Verified (Applio source/docs, Kokoro repos, Oct 2026)**

- **Applio**
  - MIT. `core.py` exposes importable inference functions.
  - The converter and model are cached in-process, so a warm daemon is feasible.
  - On macOS it uses MPS in **FP32** (not FP16), with `PYTORCH_ENABLE_MPS_FALLBACK=1`, some CPU frontends, and a CPU retry after Metal errors.
- **File sizes:** `.pth` ≈ 55 MB. `.index` is **tens of MB or more** (it scales with data), not 2 MB.
- **Kokoro**
  - Apache-2.0 code and weights.
  - Languages: en-US/GB, es, fr, hi, it, ja, pt-BR, zh.
  - MPS works only with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
  - It **hard-depends on espeak-ng (GPL-3.0)** via `misaki[en]` → `espeakng-loader`, for out-of-dictionary words.
- **Voice bubbles:** WhatsApp and Telegram both need **OGG/Opus mono**.
- **Colab:** forbids "creating deepfakes", and its free tier restricts web-UI-driven notebooks. Hence the NoUI path.

**Spikes (milestone M1), which gate everything else**

| ID | Question |
|---|---|
| **B0** | **Clean-Mac install** of Applio@pin + Kokoro: can they share one venv (Python/torch/numpy constraints)? Which deps need Xcode CLT or lack arm64 wheels? How long does it take? |
| B1 | T4 training time and quality vs epochs (100/200/400) and data length (2/5/10 min) |
| B2 | Resident memory on an 8 GB M1 (HuBERT + RMVPE + Kokoro + 1 voice) |
| B3 | Warm latency for a 10 s line. Also the **M1 MPS SDPA bug on macOS 26** and the cost of CPU fallback. Decide the MPS-vs-CPU policy per component. |
| B4 | AudioSeal vs Perth: detection rate and bit accuracy after an ffmpeg libopus 32 kbps round trip, MP3 192 kbps and resampling. Also audibility, CPU/MPS cost per 10 s, and false-positive rate on 50 clean clips. The winner becomes the default. **Done (Linux CPU): AudioSeal stays default** (§11). |
| B5 | Does `torch.load(weights_only=True)` load Applio checkpoints? (needed for §6.1) |
| B6 | Re-check SynthID: is there an embedding SDK or API for audio that third parties can run offline? (As of Oct 2026, no.) |

Performance targets (warm ≤1.5 s, ≤2.5 GB) are **provisional** and stay out of the README until B2/B3 report.

**Bugs in the original sample code (do not copy):**
- `speak.sh`:
  - Its `&& ||` chain puts a newline in the `ogg` filename (reproduced).
  - It has no `curl -f`.
  - It overwrites a fixed output path every run.
  - It never sets 24-bit.
- `server.py`:
  - It races on fixed `/tmp` paths.
  - It uses the deprecated `on_event`.
  - Its open localhost port is exposed to browser CSRF.

## 4. Architecture

```
┌──────── CLOUD ────────┐           ┌────────────────────────── MAC ───────────────────────────────┐
│ train_colab.ipynb     │ .cyvpack  │ cyv  (thin CLI, no torch) ──unix socket──► cyvd (daemon venv) │
│ Applio @ pin X        │ ────────► │   writes output files              ├─ kokoro  (text → guide)  │
│ + constraints.txt     │  (manual) │   runs ffmpeg presets              ├─ applio-rvc @ pin X      │
└───────────────────────┘           │                                    └─ watermark               │
                                    └──────────────────────────────────────────────────────────────┘
```

### 4.1 Packaging and install

- **Two layers:**
  - `cyv`: a tiny pure-Python package (`pipx`/`uv tool install`) that never imports torch, so `cyv --help` is instant.
  - `cyv setup`: builds the **daemon venv** with `uv` from a committed, macOS-arm64-tested **`uv.lock`**. That venv holds Applio@pin, Kokoro, torch and AudioSeal.
- **Prerequisites:** Xcode Command Line Tools if B0 finds any compiled deps, plus **ffmpeg from Homebrew**. `doctor` checks both and prints the exact fix.
- **Model downloads:** HuBERT/ContentVec, RMVPE, Kokoro weights. Fetched once and verified by sha256.
- **If B0 shows Applio and Kokoro conflict,** they get two daemon venvs. Kokoro then runs as a second worker process behind the same daemon. That's decided after B0, not now.
- **No prebuilt wheelhouse** is distributed. It would bundle GPL espeak-ng, and that changes our license obligations. Users install the dependencies themselves from PyPI.

### 4.2 Voice pack v1 (ADR-002)

```
<name>.cyvpack.zip
├── model.pth        (sanitized on import, §6.1)
├── model.index
├── voice.json       schema_version, name (^[a-z0-9_-]{1,32}$), engine="applio-rvc",
│                    engine_version=<commit>, rvc_version, sample_rate, embedder,
│                    f0_method, vocoder, epochs, dataset_seconds,
│                    tts_base {voice, lang}, pitch_shift, sha256{…}
└── consent.json     attestation text + version, owner_name, created_at
```

- **Unknown values fail closed:** an unknown `schema_version`, `embedder`, `vocoder` or `rvc_version` causes import to refuse with a clear message. A pack must never silently sound wrong.
- `tts_base` is chosen at `dataset prepare` (`--base-voice`, with a sensible default per pitch range).

### 4.3 Daemon `cyvd`

- **Protocol:** `asyncio` unix socket with length-prefixed JSON frames. Requests: `health`, `say`, `convert`, `reload`, `shutdown`. Audio comes back as raw bytes. **The daemon never takes a filesystem path for output.** The CLI writes the file and runs ffmpeg.
- **Socket:** `~/.cyv/run/cyvd.sock`. The directory is created 0700 *before* bind. If the path would exceed macOS's ~104-byte limit, fall back to `$TMPDIR/cyv-<uid>/`. The daemon checks the peer uid (`LOCAL_PEERCRED`).
- **Spawn and idle-exit, race-free:**
  - The CLI takes an `flock` on `~/.cyv/run/cyvd.lock`, connects, and on refused or stale it unlinks and spawns under the lock, then waits for `health`.
  - The daemon exits after `idle_timeout` (default 15 min) only when the queue is empty, checked under a mutex.
  - The CLI retries once on a connection reset.
  - SIGTERM shuts down cleanly and removes the socket.
- **Work:** one GPU worker queue (serialized), one warm voice, and a per-request temp dir. Kokoro handles sentence splitting. Chunks are concatenated.

### 4.4 CLI surface (v1)

```
cyv setup | doctor
cyv dataset prepare <dir> [-o dataset.zip] [--base-voice am_adam]
cyv voices import <pack> | list | remove <name> | default <name>
cyv say "<text>" [--voice] [--preset] [--pitch] [-o file|-] [--no-watermark]
cyv convert <in.wav> [--voice] [--preset] [--pitch] [-o file|-] [--no-watermark]
cyv serve | stop | status
```

`-o -` writes to stdout, which is how editors and scripts integrate without a network API.

### 4.5 Presets (one data file, `presets.toml`)

| Preset | Format |
|---|---|
| `master` | WAV, 48 kHz, `pcm_s24le` |
| `telegram`, `whatsapp` | OGG Opus, mono, 48 kHz, 32 kbps VBR |
| `mp3` | 192 kbps |

### 4.6 Extensibility, without a framework yet

Clean internal interfaces (`Engine`, `TTS`, `Trainer`) are kept as module boundaries. **Entry-point plug-ins arrive when a second implementation exists** (likely edge-tts, or a second engine). Until then, community contributions are presets, recording guides, languages, benchmarks, notebook fixes and docs.

License rule (for later): GPL or non-commercial components (piper1-gpl, F5-TTS weights, XTTS-v2) are only ever separate opt-in packages.

## 5. Platform

- Apple Silicon M1+ on a macOS version floor set by B0/B3. Intel is unsupported.
- Internals stay device-agnostic, so Linux/NVIDIA can be added later without redesign.
- CI runs on GitHub macOS arm64 runners. They have no MPS, so CI is CPU-only and uses a tiny fixture voice. MPS performance is tracked manually in `docs/benchmarks/`.

## 6. Security, safety, privacy

### 6.1 Pack import (the real attack surface)

A `.pth` file is a pickle, and loading an untrusted one can execute code. A sha256 inside the pack proves nothing about who made it. Import therefore:

1. **Zip hygiene:**
   - Never use `extractall`.
   - Allowlist exactly the 4 member names. Reject absolute paths, `..`, symlinks, and extra members.
   - Cap the total decompressed size (1 GB) and per-member size.
   - Extract to a temp dir.
2. **Validation:** check the JSON schema of `voice.json`, the name regex, sha256 checksums, and the engine/embedder/vocoder allowlist.
3. **Sanitize weights:**
   - In a **separate subprocess**, `torch.load(..., weights_only=True)`. Refuse on failure.
   - Re-save only the tensors and the plain config to a fresh `.pth`. From then on the daemon only ever loads sanitized files.
   - Depends on B5. If Applio checkpoints need non-tensor objects, add an explicit allowlist (`torch.serialization.add_safe_globals`) rather than ever using `weights_only=False`.
4. **Install:** move atomically into `~/.cyv/voices/<name>/`.
5. **Warning:** "Packs from other people are like software. Import only from people you trust." CI greps for `torch.load` calls without `weights_only=True`.

### 6.2 Local daemon threat model (stated honestly)

The socket permissions and peer-uid check keep out *other users*. They do not stop malware already running as you. If you are compromised, an attacker could synthesize your voice. Mitigations: the daemon exits when idle, there is no network exposure, and there are no path-taking requests.

### 6.3 Consent and watermark

- **Consent is an attestation record, not enforcement.** `dataset prepare` asks for a one-line attestation and stores it in the pack. `import` only validates the schema; it doesn't gate on consent, because a self-authored file is trivially forged. `USAGE_POLICY.md`: only your own voice or one you have explicit permission for. No impersonation, fraud or harassment.
- **Watermark (ADR-004):** **on by default, user can turn it off** (D3). It ships in **M2**.
  - **What we ship:** a SynthID-style approach, meaning an imperceptible watermark from a neural network, embedded in the audio itself (not in metadata), and detectable after compression and editing.
  - **Why not SynthID itself:** as of Oct 2026, Google offers no SynthID *audio* embedding SDK or API for third parties. It only runs inside Google's products and their partners'. Google's SynthID Detector only detects Google's own watermark, and checking a file means uploading audio to Google. Only SynthID *Text* is open source. A fully offline MIT tool therefore can't use it today. We say "SynthID-style", never "SynthID", to avoid misrepresenting the product.
  - **Default backend: Meta AudioSeal** (MIT code and weights, runs offline, PyTorch).
    - It embeds a 16-bit message. We use it as `cyv` format version + flags, never anything identifying the user.
    - It gives per-frame detection, so partially edited clips still detect.
    - Published benchmarks (AudioMarkBench) rank it most robust among open watermarks.
  - **Fallback backend:** Resemble Perth (MIT), if B4 shows it beats AudioSeal on Opus.
  - **Order of operations:** watermark the 24/48 kHz WAV *before* the preset encode. The pipeline is resample → embed → upsample (if needed) → ffmpeg.
  - **Limits:** it's provenance, not security. B4 (Linux CPU, 20 clips): both AudioSeal and Perth survived the Opus 32 kbps voice-note preset 20/20 with 0/50 false positives. AudioSeal fails at Opus 8 kbps, so detection is not guaranteed after heavier re-encoding, and the docs say so.
- **Not compliance:** none of this makes a use acceptable under Colab's terms. Users are responsible.

### 6.4 Privacy

- There is no telemetry. Voice data leaves the Mac only through the user's own Drive/Colab session.
- The docs state that deleting from Drive goes to **Trash**, and include instructions to empty it.

## 7. Repository layout

```
clone-your-voice/
├── README.md  LICENSE (MIT)  USAGE_POLICY.md  CONTRIBUTING.md  CODE_OF_CONDUCT.md  SECURITY.md  CLAUDE.md
├── docs/  DESIGN.md  adr/  benchmarks/  guides/ (recording, colab, troubleshooting-mac)
├── notebooks/  train_colab.ipynb  constraints.txt
├── src/cyv/        cli/  client/  packs/  dataset/  presets.toml        (thin, no torch)
├── src/cyvd/       server/  engine_applio/  tts_kokoro/  watermark/     (daemon venv)
├── tests/          unit, pack-import fuzz (zip-slip, bad pickles), CPU smoke test
├── uv.lock
└── .github/        workflows/ (ruff, pytest macos-14, notebook lint)  ISSUE_TEMPLATE/  PULL_REQUEST_TEMPLATE.md
```

Weights and voices are never committed.

## 8. Contribution model

- `CLAUDE.md` documents the conventions, so agent-assisted and human PRs look alike.
- Every PR runs ruff and tests. PRs that touch the engine or TTS include a benchmark entry. A new dependency needs its license recorded in `docs/licenses.md`.
- Labels: `good-first-issue`, `preset`, `docs`, `benchmark`, `notebook`, `security`.

## 9. Milestones

- **M0:** design sign-off and ADR-001…003.
- **M1:** spikes B0–B6. Then revise this doc with the measured numbers.
- **M2:** MVP covering the full journey, plus hardened import, watermark and usage policy.
- **M3:** `cyv watermark detect <file>` (local, AudioSeal detector), Drive download helper, more presets and languages, and a SynthID backend if B6 ever flips.
- **M4:** plug-in entry points once a second TTS or engine exists (edge-tts opt-in is the likely first).

## 10. Risks

| Risk | Mitigation |
|---|---|
| Applio + Kokoro can't be installed together on a clean Mac | B0 first; two-venv fallback |
| Colab free-tier or policy changes | NoUI path, pinned constraints, `Trainer` boundary for Kaggle/RunPod later |
| Applio upstream churn (maintenance mode) | Pinned commit; MIT allows vendoring |
| M1 MPS bug on macOS 26 | B3; per-component CPU fallback; `doctor` detection |
| A malicious community pack | §6.1 sandboxed sanitize + zip hygiene |
| espeak-ng GPL via Kokoro | Runtime pip dependency only; no bundled wheelhouse; documented in `docs/licenses.md` |
| 8 GB memory pressure | One warm voice, idle exit, B2 |
| Poor datasets lead to bad clones and churn | `dataset prepare` checks, notebook self-test, recording guide |

## 11. Implementation status and findings (Oct 2026, Linux x86 CPU)

The Linux-testable parts of M2 are in the repo: the thin CLI, the daemon, hardened import, the components, the Colab notebook, and docs. 218 tests pass. An independent code review led to these hardening changes:

- `fullmatch` for every name and hash regex.
- The FAISS index is validated and rewritten inside the sanitize subprocess (type allowlist, d = 768, size caps).
- The checkpoint is cross-checked against `voice.json`. A torch ≥ 2.6 floor is enforced (because of the `weights_only` RCE in older versions).
- The daemon reloads a voice when its files change, reads timeouts are bounded, and error replies survive large uploads.
- On an upgrade, a stale-version daemon is replaced.
- ffmpeg input goes through `pipe:0` with a protocol whitelist.
- Downloads enforce sha256 and size. The spaCy wheel is pinned by hash.
- The notebook verifies the sha256 of its downloaded helpers.

Findings that refine the design:

- **End to end works (Linux CPU):** Kokoro → Applio convert → AudioSeal embed/detect (prob 1.0, exact message; clean audio 0.02–0.05). Peak RSS was **2.9 GB**, which is above the 2.5 GB target, so B2 on an 8 GB M1 is now the key memory question. Timings were measured on a contended CPU and aren't quotable (`docs/benchmarks/linux-cpu-e2e.md`).
- **B4:** AudioSeal is the default and Perth is the fallback. Both survive the voice-note preset. Perth is more robust to Opus 8 kbps and noise, but quieter (SNR 14.9 dB vs 28.5 dB), carries no payload, and is ~7× cheaper on CPU (`docs/benchmarks/B4-watermark.md`).
- **B5:** 7 real Applio and RVC-WebUI checkpoints load with plain `weights_only=True`, with no safe-globals needed, so the §6.1 sanitize plan holds. The sanitizer tolerates `sr` as str or int and `f0` as int or bool. Applio's `model_information.py` and `inspect_checkpoint.py` fall back to unsafe loading, so cyv never calls them, and CI enforces that.
- **B6:** still no third-party SynthID audio embedding as of Oct 2026.
- **Applio @ pin quirks:**
  - `convert_audio` is file-only, so the engine calls `get_vc`/`load_hubert`/`vc.pipeline` in memory and memoizes the FAISS index.
  - Applio resolves base models relative to its cwd, so the engine symlinks them from `~/.cyv/models` and runs Applio calls under a locked chdir.
  - `core.py` exits 0 even on failure, so the notebook checks for output files instead.
  - At this commit the index is `logs/<name>/<name>.index`.
- **Install quirks for `cyv setup` (B0 must confirm on macOS):**
  - `preshed==3.0.9` is pinned in `src/cyv/daemon-constraints.txt`.
  - `en_core_web_sm` is pre-installed, because otherwise Kokoro pip-installs it from GitHub at first use.
  - The bundled espeak-ng library aborted on this Linux box, so `CYV_ESPEAK_LIBRARY` can point at a system library.
  - `soxr` has been added to the daemon dependencies.
  - AudioSeal's own loader uses unsafe unpickling. That's acceptable only because its checkpoints are sha256-pinned in `src/cyv/models.toml`.
- **voice.json `sha256`** records the uploaded `model.pth`. The installed file is the sanitized re-save.
- **CI** is parked at `docs/ci/ci.yml` until a maintainer with GitHub `workflow` permission moves it into `.github/workflows/`.

**Next (needs an Apple Silicon Mac):** B0 clean install via `cyv setup`, B2 memory, B3 latency and the MPS/SDPA fallback, B1 on a real Colab T4, the macOS `LOCAL_PEERCRED` path, and AudioSeal cost on MPS.

