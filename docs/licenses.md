# Licenses

This file lists the main components and their licenses. Rows marked **verify** have not been checked against the component's own license file yet. Check them before a release.

Every new dependency needs a row here in the same pull request.

## Components

| Component | Used for | License | Status |
|---|---|---|---|
| Applio (RVC voice conversion) | Voice conversion, training code | MIT | Stated in DESIGN.md |
| soxr | Audio resampling | verify | Daemon dependency (`soxr>=0.5` in pyproject), not stated in DESIGN.md |
| AudioSeal (Meta) | Default watermark | MIT | Stated in DESIGN.md (ADR-004) |
| Resemble Perth | Fallback watermark | MIT | Stated in DESIGN.md (§6.3) |
| PyTorch (`torch`) | Model runtime | BSD-3-Clause | verify |
| NumPy | Array math | BSD-3-Clause | verify |
| soundfile | Reading and writing audio | verify | Not stated in DESIGN.md |
| ffmpeg | Output presets (Opus, MP3, WAV) | LGPL or GPL, depends on build | verify. Installed by the user with Homebrew, not bundled |
| HuBERT / ContentVec weights | Speaker embedding | verify | Not stated in DESIGN.md. Downloaded at setup |
| RMVPE weights | Pitch estimation | verify | Not stated in DESIGN.md. Downloaded at setup |

Dev tools (`pytest`, `ruff`) and build tools (`hatchling`) are not shipped to users. Their licenses have not been recorded yet. **verify**

## Removed components

Kokoro was removed by ADR-008. Its front end came with it: misaki, espeak-ng (GPL-3.0), espeakng-loader, spaCy, preshed and en_core_web_sm. `cyv setup` no longer installs any of them, so the install no longer pulls in a GPL component.

## Speech models (ADR-008)

These run only in the training notebook on a cloud GPU. They are not dependencies of `cyv` or `cyvd`, and `cyv setup` does not install them. This project is for personal use. Check both again before any commercial use.

| Component | Used for | License | Status |
|---|---|---|---|
| F5-TTS weights (`SWivid/F5-TTS`) | English speech model | CC-BY-NC-4.0 | Verified (see the experiments table below). Non-commercial |
| IndicF5 weights (`ai4bharat/IndicF5`) | Hindi and Marathi speech model | MIT per the model card. Gated on Hugging Face (you accept terms to download) | verify. It is fine-tuned from F5-TTS, whose weights are non-commercial |
| IndicF5 package (`github.com/ai4bharat/IndicF5`) | Runs IndicF5 in the voice lab notebook. Installed unpinned from GitHub, as in AI4Bharat's demo Space | verify | verify. Pin it to a commit once it can be looked up |
| faster-whisper | Cutting and checking clips in the voice lab notebook | MIT | PyPI metadata (1.2.1). verify |
| Whisper large-v3 weights (`Systran/faster-whisper-large-v3`) | Speech recognition in the voice lab | MIT | Hugging Face card. verify |
| WavLM speaker verification (`microsoft/wavlm-base-plus-sv`) | Speaker similarity scores in the voice lab | Card points to the UniSpeech repository license | verify |
| espeak-ng | Local test of the clip cutter only (B8 notes). Not installed by cyv | GPL-3.0 | Test tool only, never shipped |
| Google FLEURS | Test reference voices in benchmark B8 only. Not stored in the repository | CC-BY-4.0 | verify |

## Installed by the user

- **ffmpeg** is installed with Homebrew (`brew install ffmpeg`). Its license depends on how it was built. Some builds are LGPL and some are GPL. We do not redistribute ffmpeg.

## Opt-in components (not in v1)

Some components are GPL or non-commercial and must never be in the core package. If they are added later, they go in separate opt-in packages (DESIGN.md §4.6). Examples named in the design: piper1-gpl, F5-TTS weights, XTTS-v2.

## Experiments only (not shipped, not installed by cyv)

Used by `experiments/` and benchmark B7. The user installs them by hand into `~/.cache/tts-spike/`. None of these is a cyv dependency.

| Component | Used for | License | Status |
|---|---|---|---|
| f5-tts-mlx (code) | F5-TTS on Apple MLX (`experiments/f5-say`) | MIT | Verified (LICENSE, PyPI 0.2.6) |
| F5-TTS weights, MLX (`lucasnewman/f5-tts-mlx`) | Voice-cloning TTS | CC-BY-NC-4.0 upstream (`SWivid/F5-TTS`). The MLX card is tagged MIT, which conflicts and is not explained | Upstream verified. Treat as non-commercial. Which upstream checkpoint was converted is not stated (verify) |
| F5-TTS weights, PyTorch (`SWivid/F5-TTS`, Colab) | Voice-cloning TTS on PyTorch | CC-BY-NC-4.0 | Verified (HF card and README). Treat as non-commercial |
| Emilia dataset | Training data for the F5-TTS weights. Not shipped | Conflict: HF metadata says CC-BY-4.0, but the dataset text says the Emilia part is CC BY-NC-4.0 | Conflict verified. Treat as non-commercial |
| vocos-mlx (code) | Vocoder runtime for F5 | MIT | Verified (LICENSE). PyPI page not checked |
| vocos-mel-24khz weights (`lucasnewman/vocos-mel-24khz`, upstream `charactr/vocos-mel-24khz`) | Vocoder for F5 | MIT (HF tag; the MLX card has no license text) | verify. The card does not name its upstream checkpoint |
| Vocos code (`gemelo-ai/vocos`) | Upstream of the vocoder | MIT | Verified (LICENSE, Charactr Inc.) |
| MLX (`mlx`, `mlx-metal`) | Apple array runtime | MIT | Verified for `mlx` (LICENSE, Apple Inc.). `mlx-metal`: verify |
| f5-tts (code, Colab notebook) | F5-TTS on PyTorch | MIT | Verified (PyPI 1.1.22, LICENSE). Its weights are the SWivid row above |
| chatterbox-tts (code) | Voice-cloning TTS, Colab and B7 | MIT | verify |
| Chatterbox weights (`ResembleAI/chatterbox`) | Voice-cloning TTS | MIT per HF card | verify |
| IndicF5 (`ai4bharat/IndicF5`) | Hindi and Marathi TTS (not tested) | MIT per HF card; trained on research datasets | verify. Gated on Hugging Face |

### Sources checked for the F5 quality path (2026-10-28)

Code licenses were read from each project's LICENSE file. Weights licenses were read from the Hugging Face model card (the license tag and the card text).

- F5-TTS code, `SWivid/F5-TTS`: https://raw.githubusercontent.com/SWivid/F5-TTS/main/LICENSE (MIT, Copyright (c) 2024 Yushen CHEN). The README at https://github.com/SWivid/F5-TTS says: "The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia."
- F5-TTS weights, `SWivid/F5-TTS`: https://huggingface.co/SWivid/F5-TTS (tag `cc-by-nc-4.0`; dataset `amphion/Emilia-Dataset`).
- F5-TTS MLX code: https://raw.githubusercontent.com/lucasnewman/f5-tts-mlx/main/LICENSE (MIT, Copyright (c) 2024 Lucas Newman). PyPI: https://pypi.org/project/f5-tts-mlx/ (0.2.6, MIT).
- F5-TTS MLX weights: https://huggingface.co/lucasnewman/f5-tts-mlx (tag `mit`; no license text; "reshaped for MLX from the original weights", linking to `SWivid/F5-TTS`).
- Emilia: https://huggingface.co/datasets/amphion/Emilia-Dataset (metadata `cc-by-4.0`; the terms text says "Users are permitted to use Emilia dataset only for non-commercial purposes under the `CC BY-NC-4.0` license.").
- Vocoder code: https://raw.githubusercontent.com/gemelo-ai/vocos/main/LICENSE (MIT, Copyright (c) 2023 Charactr Inc.).
- Vocoder, MLX: https://raw.githubusercontent.com/lucasnewman/vocos-mlx/main/LICENSE (MIT, Copyright (c) 2024 Lucas Newman).
- Vocoder weights: https://huggingface.co/lucasnewman/vocos-mel-24khz (tag `mit`; the card has no license text) and https://huggingface.co/charactr/vocos-mel-24khz (tag `mit`).
- MLX: https://raw.githubusercontent.com/ml-explore/mlx/main/LICENSE (MIT, Copyright (c) 2023 Apple Inc.).
- f5-tts (PyTorch): https://pypi.org/project/f5-tts/ (1.1.22, MIT).

Not checked: the PyPI page for `vocos-mlx` (it failed to load), the `mlx-metal` license, and which F5-TTS checkpoint (base or v1) the MLX weights came from. The Colab notebook's default checkpoint was not checked either. All of these are still **verify**.

## Can the F5 quality tier ship in the default install?

**No.** The code is MIT, but the F5-TTS weights are CC-BY-NC-4.0 because they were trained on Emilia, and Emilia is non-commercial. The MLX conversion's MIT tag does not change that. The card gives no reason for it, and a conversion does not change the terms of the original weights. This is our reading, not legal advice.

What this means:

- The F5 weights may be used for personal, non-commercial use only.
- They must not go in the default MIT install. If the quality tier is added later, it goes in a separate opt-in package (DESIGN.md §4.6), with a clear non-commercial notice.
- The weights are downloaded by the user by hand. We do not redistribute them.
