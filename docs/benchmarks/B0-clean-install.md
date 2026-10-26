# B0: clean install on Apple Silicon (`cyv setup`)

## Question
Can the full daemon stack (Applio + Kokoro + watermark) install into one venv on a clean Mac, or is the two-venv fallback from DESIGN §10 needed?

## Method
Single daemon venv via `cyv setup` (no two-venv fallback attempted — none was needed):
- Dependency resolve dry-run first, then install in two steps (torch stack, then the rest).

## Environment
Apple M1, macOS 27.2, arm64; Python 3.12; Xcode CLT present (no compilation was needed — every package had an arm64 wheel). Cold `import torch` 7.8 s; MPS available.

Python 3.12 is required: kokoro, misaki and preshed reject 3.14, so `cyv setup` must not run under 3.14.

## Results
- Resolve dry-run: 13.9 s, zero conflicts.
- Install: torch stack 21.5 s, everything else 59.5 s — about 81 s total.
- Footprint: 157 packages, 1.6 GB.
- Zero missing arm64 wheels; nothing compiled from source.
- Installed versions: torch 2.11.0, torchaudio 2.11.0, numpy 2.4.6, transformers 5.13.1, kokoro 0.9.4, audioseal 0.2.0, spacy 3.8.14, preshed 3.0.9, faiss-cpu 1.14.3.
- `en_core_web_sm` 3.8.0 auto-fetched at setup.
- The bundled espeak-ng dylib loads as-is; no `CYV_ESPEAK_LIBRARY` needed on macOS (differs from the Linux result in DESIGN §11, where the bundled library aborted and the env var is the escape hatch).

## Conclusion
- **One venv suffices on Apple Silicon; the two-venv fallback is not needed.** The §10 risk row ("Applio + Kokoro can't be installed together on a clean Mac") did not materialize here.
- `cyv setup` must pin/require Python 3.12 until kokoro/misaki/preshed accept newer versions.

## Caveats
- Single machine, single run; timings will vary with network and disk speed.
- Cold `import torch` (7.8 s) is import cost only, not inference latency; MPS inference cost is B3's job.
- Linux behavior differs (espeak-ng needs the `CYV_ESPEAK_LIBRARY` escape hatch there); this report covers macOS only.
