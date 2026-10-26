# Licenses

This file lists the main components and their licenses. Rows marked **verify** have not been checked against the component's own license file yet. Check them before a release.

Every new dependency needs a row here in the same pull request.

## Components

| Component | Used for | License | Status |
|---|---|---|---|
| Applio (RVC voice conversion) | Voice conversion, training code | MIT | Stated in DESIGN.md |
| Kokoro (code and weights) | Default text-to-speech | Apache-2.0 | Stated in DESIGN.md |
| misaki (`misaki[en]`) | English text front end for Kokoro | Apache-2.0 | verify |
| espeak-ng | Pronunciation of out-of-dictionary words | GPL-3.0 | Stated in DESIGN.md. Runtime dependency only (see below) |
| espeakng-loader | Loads espeak-ng for misaki | verify | Not stated in DESIGN.md |
| soxr | Audio resampling | verify | Daemon dependency (`soxr>=0.5` in pyproject), not stated in DESIGN.md |
| spaCy | English text pipeline used by the Kokoro/misaki front end | verify | Installed by `cyv setup` via `misaki[en]`, not stated in DESIGN.md |
| preshed | Hash tables backing spaCy | verify | Transitive dependency via misaki → spaCy (pinned in `src/cyv/daemon-constraints.txt`), not stated in DESIGN.md |
| en_core_web_sm | English model for the spaCy pipeline | verify | Fetched at setup (see DESIGN.md §11), not stated in DESIGN.md |
| AudioSeal (Meta) | Default watermark | MIT | Stated in DESIGN.md (ADR-004) |
| Resemble Perth | Fallback watermark | MIT | Stated in DESIGN.md (§6.3) |
| PyTorch (`torch`) | Model runtime | BSD-3-Clause | verify |
| NumPy | Array math | BSD-3-Clause | verify |
| soundfile | Reading and writing audio | verify | Not stated in DESIGN.md |
| ffmpeg | Output presets (Opus, MP3, WAV) | LGPL or GPL, depends on build | verify. Installed by the user with Homebrew, not bundled |
| HuBERT / ContentVec weights | Speaker embedding | verify | Not stated in DESIGN.md. Downloaded at setup |
| RMVPE weights | Pitch estimation | verify | Not stated in DESIGN.md. Downloaded at setup |
| Kokoro voice and model files | Default TTS model | Apache-2.0 | Stated in DESIGN.md, downloaded at setup |

Dev tools (`pytest`, `ruff`) and build tools (`hatchling`) are not shipped to users. Their licenses have not been recorded yet. **verify**

## Why espeak-ng is a runtime dependency only

espeak-ng is licensed under GPL-3.0. Kokoro depends on it through `misaki[en]` and `espeakng-loader`, for words that are not in the dictionary.

We do **not** bundle a prebuilt wheelhouse (a folder of ready-made packages). A bundle would redistribute espeak-ng, and that would change our obligations under the GPL. Instead, `cyv setup` installs the dependencies from PyPI on the user's Mac, the same way they would install them by hand.

This project's own code is MIT. The GPL component is not copied into this repository.

## Installed by the user

- **ffmpeg** is installed with Homebrew (`brew install ffmpeg`). Its license depends on how it was built. Some builds are LGPL and some are GPL. We do not redistribute ffmpeg.

## Opt-in components (not in v1)

Some components are GPL or non-commercial and must never be in the core package. If they are added later, they go in separate opt-in packages (DESIGN.md §4.6). Examples named in the design: piper1-gpl, F5-TTS weights, XTTS-v2.
