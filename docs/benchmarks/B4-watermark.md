# B4: AudioSeal vs Resemble Perth, watermark robustness

## Question
Which open, offline neural audio watermark should be the default (ADR-004): Meta AudioSeal or Resemble Perth? Does the watermark survive the Telegram/WhatsApp voice-note preset (OGG Opus 32 kbps mono, voip)?

## Method
Scripts: `scripts/bench/b4_prepare_clips.py`, `scripts/bench/b4_watermark_bench.py` (both standalone).
- Speech: real LibriSpeech dev-clean utterances from HF `hf-internal-testing/librispeech_asr_dummy` (73 clips, 16 kHz). 20 clips (187.7 s total) are watermarked; a disjoint 50 clips are the unwatermarked set for false positives. (Dataset has only 73 clips, so FPR is over 50, not 100 as in the B4 row of DESIGN; 0/50 gives a 95% upper bound of about 6%.)
- Pipeline (mirrors DESIGN §6.3): embed at the method's native SR (AudioSeal 16 kHz; Perth 32 kHz) -> resample to 48 kHz "delivered" WAV -> attack -> resample back to native SR -> detect.
- Attacks: none; resample 48k->16k->48k; MP3 192k; MP3 64k; OGG Opus 32 kbps mono `-application voip`; Opus 16 kbps mono voip. All via ffmpeg (libopus/libmp3lame). Supplementary harsher run (10 wm + 20 clean clips): Opus 8 kbps voip, MP3 32k, 48k->8k->48k, white noise at 20 dB SNR.
- Detection: AudioSeal `detect_watermark` prob > 0.5, plus 16-bit message bit accuracy (fixed random message); Perth `get_watermark(round=False)` confidence > 0.5 (Perth carries no payload and returns one clip-level score).
- SNR(original, watermarked) as a perceptual proxy (no listening test done). CPU time = wall time, 4 torch threads.

## Environment
Linux x86 (4 vCPU, 15 GB RAM) **CPU only, no MPS/Apple Silicon**; Python 3.12, torch 2.14.1+cpu, `audioseal` (latest on PyPI at run time) and `resemble-perth` 1.0.1 from PyPI (both installed fine, no problems), ffmpeg (/usr/bin/ffmpeg). The box was shared with other benchmark jobs running concurrently, so timings are noisy and probably inflated.

## Results

Main run (20 watermarked clips, 50 clean clips). Cells: detection rate / bit accuracy (AudioSeal) / false-positive rate on clean clips passed through the same attack.

| Attack | AudioSeal detect | AudioSeal bit acc | AudioSeal FPR | Perth detect | Perth FPR |
|---|---|---|---|---|---|
| none | 20/20 | 1.000 | 0/50 | 20/20 | 0/50 |
| resample 48k->16k->48k | 20/20 | 1.000 | 0/50 | 20/20 | 0/50 |
| MP3 192k | 20/20 | 1.000 | 0/50 | 20/20 | 0/50 |
| MP3 64k | 20/20 | 1.000 | 0/50 | 20/20 | 0/50 |
| **Opus 32k voip mono** | 20/20 | 1.000 | 0/50 | 20/20 | 0/50 |
| Opus 16k voip mono | 20/20 | 1.000 | 0/50 | 20/20 | 0/50 |

Supplementary harsher attacks (10 watermarked, 20 clean):

| Attack | AudioSeal detect | AudioSeal bit acc | AudioSeal FPR | Perth detect | Perth FPR |
|---|---|---|---|---|---|
| Opus 8k voip mono | **0/10** | 0.79 | 0/20 | 10/10 | 0/20 |
| MP3 32k | 10/10 | 1.000 | 0/20 | 10/10 | 0/20 |
| 48k->8k->48k | 10/10 | 1.000 | 0/20 | 10/10 | 0/20 |
| white noise, 20 dB SNR | 8/10 | 1.000 (of detected clips) | 0/20 | 10/10 | 0/20 |

Quality and cost:

| | AudioSeal | Perth |
|---|---|---|
| SNR orig vs watermarked (mean / min) | 28.5 / 25.5 dB | 14.9 / 14.1 dB |
| Embed, CPU s per 10 s audio | 4.4 | 0.58 |
| Detect, CPU s per 10 s audio | 6.1 | 0.95 |
| Payload | 16 bits, per-frame localization | none (detect only), clip-level score |
| Native SR | 16 kHz | 32 kHz |

## Conclusion
- **Default: AudioSeal** (keep ADR-004). It survived every preset including the voice-note preset (Opus 32k voip mono) with 100% detection and 100% bit accuracy, 0 false positives, and has a much higher SNR (28.5 vs 14.9 dB), a usable 16-bit payload (cyv version/flags) and frame-level localization.
- **Watermarks survive the voice-note preset** in this test: AudioSeal and Perth both 20/20 at Opus 32k and 16k. The "Opus is a known weak spot" language in §6.3 is not borne out at 32/16 kbps; AudioSeal breaks only at Opus 8 kbps (0/10) and is degraded by 20 dB white noise (8/10).
- Perth is strictly more robust in the harsh cases (10/10 everywhere), ~8x cheaper on CPU, but costs 13.6 dB more embedding distortion by SNR, has no payload and no localization. Keep it as the fallback backend behind the `Watermarker` interface; the trigger in §6.3 ("if B4 shows it beats AudioSeal on Opus") is not met at the voice-note preset.
- Cost: AudioSeal embed at ~4.4 s CPU per 10 s audio on this loaded x86 box is not negligible versus TTS; M2 must re-measure on Apple Silicon/MPS before committing a latency budget.

## Caveats
- Ceiling effect: most attacks yield 100%, so the main run cannot rank the methods; only the supplementary run differentiates them, with n=10.
- 20 clips (read English speech from LibriSpeech dev-clean, 16 kHz source; speaker count not checked) is a small sample; 0/50 false positives does not bound FPR below ~6%. Perth/AudioSeal both could have rarer false positives on music, noise, or TTS output (not tested). Watermarked-vs-clean was compared on different clips.
- SNR is a poor perceptual proxy; Perth shapes noise via masking and may sound better than 14.9 dB suggests. No listening test or PESQ/ViSQOL was run.
- Attacks were single-pass; no cascades (e.g. Opus then resample then re-encode), no time stretch, pitch shift, cropping, or adversarial removal. The watermark is provenance, not security.
- Threshold 0.5 for both detectors is the library default, not tuned.
- Timings are CPU wall time on a shared 4-vCPU box, not MPS.
- Only mono, speech at 16 kHz upsampled; real pipeline audio from TTS/RVC (24-48 kHz, synthetic) may behave differently.
