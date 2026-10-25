# ADR-004: Watermarking behind a `Watermarker` interface

Status: Accepted

## Context

Generated audio should carry a signal that shows it came from this tool. Google's SynthID is the model we want to copy, but as of October 2026 there is no SynthID audio embedding SDK or API that third parties can run offline. SynthID Detector only checks Google's own watermarks and needs uploads to Google. Only SynthID Text is open source.

We also cannot claim more than the watermark does. It is provenance, not security.

## Decision

- Watermarking sits behind a `Watermarker` interface, so the backend can change.
- **Default backend: Meta AudioSeal** (MIT code and weights, runs offline with PyTorch). It embeds a 16-bit message, which we use only for a `cyv` format version and flags, never for anything that identifies the user. It detects per frame, so partly edited clips can still be detected.
- **Fallback backend: Resemble Perth** (MIT), if benchmark B4 shows it beats AudioSeal on the voice-note presets.
- **SynthID** becomes a backend only if Google releases an embedding SDK that third parties can run (benchmark B6).
- Watermark the 24 or 48 kHz WAV **before** the preset encode (resample, embed, upsample if needed, then ffmpeg).
- The watermark is on by default. Users can turn it off (D3).
- We describe it as "SynthID-style", never as "SynthID".

## Consequences

- Users get an offline, open watermark that is not tied to Google.
- Opus at 32 kbps is a known weak spot. If B4 shows detection does not survive the voice-note presets, the docs say so plainly.
- Switching backends later is a code change behind the interface, not a redesign.
- Watermark detection itself moves to M3 (`cyv watermark detect`).
- **Open point:** the default is AudioSeal, but DESIGN.md also says "the winner becomes the default" after B4. This ADR is accepted with AudioSeal as the default, and the choice is confirmed or changed by the B4 result.
