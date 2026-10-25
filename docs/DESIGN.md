# clone-your-voice — Design (DRAFT v0.1)

Status: **proposal, not yet agreed**. No code until the open questions at the bottom are settled.

## 1. Goal

Let anyone clone **their own** voice cheaply:

- **Train in the cloud** on a free or cheap GPU (Colab first), so laptops never take sustained heat.
- **Infer locally** through a warm, resident daemon, with sub-second to ~1.5 s turnaround.
- **Community-driven**: engines, TTS backends, output presets and platforms are plug-ins, not forks.

Non-goals for v1: real-time voice changing (live mic), cloning third-party voices, hosted SaaS, and mobile apps.

## 2. Decisions carried over from the original plan

| Decision | Status | Note |
|---|---|---|
| No Rust; stay in Python | **Accept** (ADR-001) | The bottleneck is GPU matmul plus import/load cold start, not CPU. |
| Hybrid: cloud training, local inference | **Accept** | This is the core architecture. |
| Persistent daemon to kill cold start | **Accept** | Applio already caches the converter (`lru_cache`) and the voice model (`get_vc`). |
| Applio/RVC as the first engine | **Accept** | Applio is MIT (code and weights). Use `core.py`/`VoiceConverter` in-process. |
| FP16 on M1 for speed | **Reject** | Applio's MPS path is FP32 by design. Plan on FP32 and benchmark later. |
| Edge-TTS as the text base voice | **Optional only** | It is an unofficial use of Microsoft's online service: it needs internet and carries ToS risk. The default must be offline. |
| `.m4a` for WhatsApp voice bubbles | **Reject** | WhatsApp and Telegram voice bubbles both need OGG/Opus mono. M4A sends as a plain audio file. |
| "1 s sleep between paragraphs" thermal trick | **Replace** | Use a configurable duty-cycle or queue throttle, backed by measured data, not folklore. |
| Bash `speak.sh` as the main client | **Replace** | Use a cross-platform Python CLI. A shell wrapper is fine as sugar. |

## 3. Claims to fix or benchmark before we rely on them

Verified against Applio's source and docs, Oct 2026:

- **Index size of ~2 MB: likely wrong.** A faiss IVF index stores every feature frame, so it is tens of MB for 2–5 min of audio and hundreds of MB for large datasets. The ~55 MB `.pth` figure is right.
- **Training in 15–25 min on a T4: unverified.** No authoritative numbers exist, and community reports cite 500–800 epochs. We will benchmark this as **B1**.
- **RAM of 1.6–2.1 GB: plausible but unmeasured** (**B2**).
- **M1 + macOS 26 risk:** there is a reported MPS attention (SDPA) kernel bug on A14/M1-class chips. Applio sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and retries on CPU. Our setup command must detect and report this (**B3**).
- **Colab policy:** Colab disallows "creating deepfakes", and free runtimes may not drive things primarily through a web UI. **Use Applio's NoUI notebook / `core.py` CLI path, not the Gradio UI.** The consent gate in §6 helps here too.

Bugs in the plan's sample code, recorded so we don't copy them:

- `speak.sh`: the chain `[ ] && echo || [ ] && echo || echo` produces `output_ogg.ogg\nm4a` for `ogg`, a filename containing a newline (reproduced). The script also has no `curl -f`, so a server error becomes the output "audio". It overwrites the same file on every run, and "24-bit" is claimed but never set (`pcm_s24le` is missing).
- `server.py`: fixed `/tmp/input_voice.wav` paths race under concurrent requests. `@app.on_event` is deprecated (use `lifespan`). There is no auth: a browser page can POST a form to `127.0.0.1` (CSRF), and so can any local process.

## 4. Architecture

```
            ┌────────────── CLOUD (train) ──────────────┐
 clips ──►  │ notebooks/train.ipynb  (pinned Applio, NoUI) │ ──► voice pack (.zip)
            │  preprocess → extract → train → index        │      model.pth
            └──────────────────────────────────────────────┘      model.index
                                                                   voice.json  (manifest)
                                     download / import             consent.json
                                                │
            ┌────────────── LOCAL (infer) ──────▼────────────────────────────┐
            │  cyv CLI ──► cyvd daemon (unix socket / localhost + token)     │
            │                 ├─ TTS backend   (kokoro | piper | edge*)      │
            │                 ├─ Engine        (applio-rvc | future …)       │
            │                 ├─ Device        (cuda | mps | cpu, auto)      │
            │                 ├─ Watermark     (on by default)               │
            │                 └─ Output preset (master | telegram | whatsapp)│
            └────────────────────────────────────────────────────────────────┘
```

### 4.1 Components

1. **Training notebook** (`notebooks/`): a thin wrapper over a **pinned** Applio commit. It runs the NoUI flow, validates the dataset (mono, sample rate, length, clipping, silence ratio), and emits a **voice pack**. Later backends (Kaggle, RunPod, local CUDA) reuse the same pack format.
2. **Voice pack** (the contract between cloud and local), stored at `~/.cyv/voices/<name>/`:
   - `model.pth`, `model.index`
   - `voice.json`: engine, engine version, sample rate, f0 method, base pitch, training metadata, sha256 of each file, license
   - `consent.json`: who the voice belongs to, a self-attestation, and the date (see §6)
3. **Daemon `cyvd`**: FastAPI + uvicorn. It loads the engine once and keeps the N most recent voices warm (LRU, memory-capped). It serializes GPU work through one queue, so concurrency stays safe and thermals stay predictable. Per-request temp files go in a private temp dir.
4. **CLI `cyv`** (Python): `cyv setup`, `cyv doctor`, `cyv voices import|list`, `cyv say "text" --voice me --preset telegram`, `cyv convert in.wav --voice me`, and `cyv serve`. It auto-starts the daemon if it isn't running.
5. **Output presets** (ffmpeg): `master` (WAV 48 kHz, 24-bit), `telegram` / `whatsapp` (OGG Opus, mono, 48 kHz, ~32 kbps), and `mp3`. Presets are data files, so contributors can add their own.

### 4.2 API (localhost only)

- Transport: **unix socket by default** on macOS/Linux. TCP on `127.0.0.1` is opt-in and requires a bearer token from `~/.cyv/token`. TCP mode rejects requests whose `Origin` header is set.
- `GET /health` and `GET /voices`
- `POST /tts` with `{text, voice, pitch, preset}` returns audio
- `POST /convert` takes multipart audio + voice + pitch + preset and returns audio
- `POST /voices/reload`

### 4.3 Plug-in points (where the community contributes)

Python entry points; each is a small interface:

| Group | Interface | v1 implementations | Candidates later |
|---|---|---|---|
| `cyv.engines` | `load(pack)`, `convert(wav, pitch) -> wav` | applio-rvc | GPT-SoVITS (MIT), OpenVoice v2 (MIT), Chatterbox (MIT, zero-shot) |
| `cyv.tts` | `synth(text, lang) -> wav` | kokoro (Apache-2.0), edge (opt-in) | piper (now GPL-3.0, so it stays an optional plug-in) |
| `cyv.presets` | data (ffmpeg args) | master, telegram, whatsapp, mp3 | podcast, video-editor |
| `cyv.trainers` | produce a voice pack | colab-applio | kaggle, local-cuda, runpod |

**License rule:** core stays permissive. Non-commercial weights (F5-TTS, XTTS-v2) are only allowed as separately installed plug-ins with a visible license flag.

## 5. Platform and performance targets

- Tier 1: Apple Silicon (M1/8 GB is the floor) and Linux + NVIDIA. Tier 2: Windows + NVIDIA. Tier 3: CPU-only (works, but slow).
- Targets, to be confirmed by benchmarks:
  - Warm `cyv say` for a 10 s line: ≤1.5 s on M1.
  - Daemon resident memory: ≤2.5 GB with 1 voice warm.
  - Idle CPU: ~0%.
- Thermals: a batch job (audiobook) goes through the queue with an optional `--duty-cycle` setting. `cyv doctor` reports the device, whether MPS fallback is active, and memory use.

## 6. Safety and consent (required for a "clone anyone's own voice" tool)

Without this the project risks Colab/GitHub takedowns and erodes community trust.

- `USAGE_POLICY.md`: clone only voices you own or have explicit permission for. No impersonation, fraud, or harassment.
- A **consent gate** at training and import: the user writes a self-attestation into `consent.json`. Optionally, they record a short spoken consent phrase that becomes part of the dataset.
- A **watermark on every output by default** (e.g. Resemble's Perth, MIT). This needs verification (**B4**: does it survive Opus at 32 kbps?).
- No telemetry. Voice data never leaves the user's machines or their own Colab session.

## 7. Repository layout (proposed)

```
clone-your-voice/
├── README.md  LICENSE (Apache-2.0 or MIT)  USAGE_POLICY.md
├── CONTRIBUTING.md  CODE_OF_CONDUCT.md  CLAUDE.md  SECURITY.md
├── docs/  DESIGN.md  adr/  benchmarks/  guides/(recording-tips, colab, mac)
├── notebooks/train_colab.ipynb
├── src/cyv/  cli/  daemon/  engines/  tts/  presets/  packs/  watermark/
├── tests/  (unit tests, plus a golden-audio smoke test on CPU)
└── .github/  workflows/ (lint, test, notebook-lint)  ISSUE_TEMPLATE/  PULL_REQUEST_TEMPLATE.md
```

Weights and voices are **never** committed. Default pretrains download from Hugging Face, pinned by sha256.

## 8. Contribution model (hybrid human + AI agents)

- `CLAUDE.md` / `AGENTS.md` describe the conventions, so agent-assisted PRs (Haiku for triage, docs and lint fixes; Sonnet for features and review) match human ones.
- Plug-ins live in-tree under `contrib/` until they are stable, then they can move to their own repos.
- Labels: `good-first-issue`, `engine`, `tts`, `platform:mac`, `platform:win`, `benchmark`.
- Every engine or TTS PR must ship a license declaration and a CPU smoke test.

## 9. Milestones

- **M0 – Design sign-off** (this doc plus ADR-001 for no Rust and ADR-002 for the voice pack format).
- **M1 – Benchmarks B1–B4** on Colab T4 and an M1 Air; record the results in `docs/benchmarks/`.
- **M2 – MVP:** the Colab notebook, the voice pack, `cyvd` with applio-rvc and kokoro, the `cyv` CLI, and the 3 presets.
- **M3 – Safety:** the consent gate, the watermark, and the usage policy.
- **M4 – Community:** the plug-in API frozen at v0, the contribution docs, and the first external engine.

## 10. Open questions (need your call)

1. **License**: Apache-2.0 (explicit patent grant) or MIT (matches Applio)?
2. **Default text-to-speech backend**: offline Kokoro, with Edge-TTS opt-in? Or Edge-TTS default for quality/languages?
3. **Watermark on by default**, with an opt-out flag or with no opt-out?
4. **Transport**: is a unix socket by default OK? Or do you want plain HTTP so video editors/Shortcuts can call it?
5. **Scope of v1 platforms**: Mac-only first, or Mac + Linux/NVIDIA from day one?
6. **"Hybrid"**: does this mean cloud + local (as assumed), human + AI contributors, or both?
