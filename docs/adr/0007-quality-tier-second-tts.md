# ADR-007: F5-TTS as an opt-in second text-to-speech for the quality tier

Status: Superseded by [ADR-008](0008-shortlist-f5-indicf5-rvc.md). F5-TTS is now the English speech model, not an optional tier.

## Context

The quality tier in ADR-005 needs a text-to-speech model that copies delivery from a reference clip (ADR-006). Kokoro, our default (DESIGN D2), does not do that. It sets the delivery itself, and that is where the loss comes from.

B7 tested F5-TTS and Chatterbox against Kokoro, on an Apple M1, 8 GB, macOS 27.2, cyv 0.0.1, with one listener and one 10.6 s reference clip (`docs/benchmarks/B7-clone-tts-m1.md`):

- **Quality:** in a blind Colab test (T4 GPU), F5 followed by `cyv convert` ranked first on all three sentences. Kokoro followed by convert ranked last every time.
- **F5 on the Mac, working setup** (euler solver, 32 steps, text-based duration): 67.5 s for 4.67 s of audio, 71.6 s for 5.61 s, 77.7 s for 6.22 s. Listened to: "good, but the tone is not right". The tone follows the reference clip.
- **F5 on the Mac, 16 steps:** 36 to 44 s for 4.7 to 5.8 s of audio. Listened to: "close but not that good", with a clear sense of loss.
- **F5 memory:** MLX peak 2.6 to 2.8 GB, process footprint about 3.3 GB. It fits an 8 GB Mac.
- **F5 on Colab (T4):** about 4 s per sentence (B7).
- **Chatterbox on the Mac (MPS):** 62 to 66 s for 3.8 to 5.0 s of audio. MPS driver memory peaked at 8.4 GB, and the process footprint at 11.2 GB, which pushed macOS into 3 GB of swap. The Mac files were not listened to.
- **F5's own duration predictor is broken in the tested setup.** Every clip came out about 2.8 s long, whatever its text. The working setup replaces it with an estimate from the reference clip's speaking rate. The working setup also needs a one-line patch to a third-party file (`cfm.py`, `dur` to `int(dur)`) for mlx 0.32 (`experiments/f5-say/README.md`).

Licenses (from `docs/licenses.md`, all marked "verify"):

- f5-tts-mlx code: MIT.
- F5-TTS weights: the Hugging Face card for the MLX build says MIT. The upstream F5-TTS weights are CC-BY-NC, which would block any product use.
- Chatterbox code and weights: MIT per the cards, not verified.

DESIGN §4.6 says GPL or non-commercial components are only ever separate opt-in packages.

## Options considered

| Option | What it means | Speed (measured) | Memory | Verdict |
|---|---|---|---|---|
| **A. F5-MLX, local** | F5 runs on the user's Mac through MLX | 67 to 78 s per 4.7 to 6.2 s line at 32 steps (M1, 8 GB) | About 3.3 GB footprint (M1, 8 GB) | Fits. Slow, but usable for "make a clip, come back later". Chosen as the local path |
| **B. Colab / remote** | F5 runs on a cloud GPU, as a manual notebook step | About 4 s per sentence (Colab T4) | Not measured here | Fast and high quality. Breaks the local-first design (DESIGN §1, D6), so it is allowed only as a manual notebook path, like training |
| **C. Chatterbox** | Another voice-cloning TTS | 62 to 66 s per 4 to 5 s line (M1, MPS) | 8.4 GB GPU memory, 11.2 GB footprint (M1, 8 GB) | **Ruled out** for 8 GB Macs. It swaps and cannot run next to the daemon |
| **D. Stay Kokoro-only** | No second TTS. Quality tier not offered | About 2 s per 10 s line (B3) | Smallest | Simplest and fully local. Keeps the known gap in delivery. Remains the default |

## Decision

Propose **option A plus option B**, with the rules below:

1. **F5 is opt-in.** It is not part of the default install. Kokoro stays the default TTS (DESIGN D2).
2. **F5 ships as a separate opt-in package**, not inside the core `cyv` or `cyvd` install. The package name and install command are decided in the implementation. A plausible shape is a package such as `cyv-tts-f5`, installed by an explicit command. `cyv setup` does not install it. This follows DESIGN §4.6 for non-commercial components.
3. **The local path is option A.** It runs on the user's Mac with the defaults that B7 measured: 32 steps, euler solver, reference-based duration, and the cyv watermark applied by `cyv convert` afterwards.
4. **The Colab path is option B,** as a separate notebook, run by hand, like the training notebook. It is not called from the daemon, and it is not a default.
5. **Chatterbox is not adopted.** Its memory use is over the limit for 8 GB Macs (B7).
6. **Upstream patches are not shipped.** The `cfm.py` patch is a local workaround. Before this package ships, we need either a fixed upstream release or a pinned version with a documented patch step. A patched copy of a third-party file is not committed.

### Gates before any non-personal use

- **License verified.** The F5 weights license must be verified (`docs/licenses.md` says "verify") before any use beyond personal use. If the upstream CC-BY-NC terms apply, the package cannot be used in a commercial product, and the docs must say so. The ADR is not accepted for distribution until this is checked.
- **Language: English only.** This F5 model has no Devanagari characters in its vocabulary (B7). Hindi and Marathi are not supported by this option. IndicF5 (AI4Bharat, gated on Hugging Face, license to verify) is a separate question and is not tested here.
- **Benchmark on a quiet machine.** B7 numbers were measured with other apps open and the system swapping. A new benchmark entry in `docs/benchmarks/` is required before the quality tier is promised anywhere (CLAUDE.md, benchmarks rule). It must label the hardware and the cyv version, and report bad results too.
- **Watermark still applies.** Output from F5 goes through `cyv convert` and the watermark (ADR-004). `cyv watermark detect` has not yet been run on F5 output, and it should be before release.

### Entry-point plug-ins (M4)

DESIGN §4.6 and §9 say entry-point plug-ins arrive when a second text-to-speech or engine implementation exists. F5 would be that second TTS implementation, so this ADR brings M4 forward for TTS. The `TTS` interface in `src/cyvd/interfaces.py` already fits F5 (`synth(text, voice, lang) -> Audio`). If this ADR is accepted, the plug-in mechanism (an entry-point group for TTS) must be decided with it, and that decision is a separate ADR. Until then, F5 would be wired in by a direct import from the opt-in package, not by a plug-in registry.

## Consequences

- **Good:** the quality tier becomes possible on an 8 GB Mac, with measured numbers, at the cost of speed.
- **Good:** the default install stays Kokoro-only, so the license risk stays out of the core package.
- **Good:** the Colab path gives faster results for people who want them, without changing the local-first design.
- **Cost:** a second TTS adds a second model, a second install path and a second benchmark to keep current.
- **Cost:** a 67 to 78 s wait per line (M1, 8 GB) is a real limit. The docs must say so, and the quality tier should be described as "make a clip, come back later" until faster numbers exist.
- **Cost:** the duration workaround is a heuristic, not a fix. The package must test it against real lines before shipping.
- **Limit:** every number here comes from one Mac, one reference clip, a few sentences and one listener.
- **Not decided here:** whether the quality tier is shown in the default `cyv say` help, or behind a flag. That is a CLI decision for a later change.
