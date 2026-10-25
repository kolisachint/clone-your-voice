# Linux x86 CPU end-to-end run (NOT Apple Silicon / M1)

Purpose: prove the real components chain together (Kokoro, Applio RVC, AudioSeal). These numbers
say nothing about M1/MPS performance; spike B2/B3 on a Mac still decides targets.

## Setup

- Box: 4 vCPU Intel Xeon @ 2.1 GHz, 15 GB RAM, Linux, **CPU only**, py3.12, torch 2.11.0+cpu.
- **Heavily contended**: another benchmark job ran on the same 4 cores during every run
  (load average 4 to 7). Treat absolute times as pessimistic and noisy (run-to-run swings of 2x).
- Applio at `c817f1f` (the pin), engine requirements from `requirements-engine.txt`, Kokoro 0.9.4, AudioSeal 0.2.0.
- Voice: `maotianchen/maobailing_rvc` (Apache-2.0, RVC v2, 40 kHz, trained only on synthetic speech), with the
  89 MB `.index`. Used only as a test fixture and not redistributed.
- Input: Kokoro `af_heart`, "Hello, this is a short test of the clone your voice pipeline, ..." = 5.8 s of audio.
- Applio defaults: rmvpe, contentvec, index_rate 0.75, protect 0.5.

## Results (5.8 s line, two runs)

| Stage | Run A (s) | Run B (s) | Notes |
|---|---|---|---|
| Kokoro synth, cold (incl. model + spaCy load) | 21.8 | 18.2 | |
| Kokoro synth, warm | 23.7 | 13.7 | contention dominates |
| ApplioEngine.load (voice + HuBERT) | 3.5 | 3.6 | |
| Convert, cold (loads RMVPE, reads index) | 33.6 | 48.4 | |
| Convert, warm | 18.2 / 29.5 | 54.3 / 25.8 | |
| AudioSeal embed, cold (incl. load) | 72.3 | 34.4 | |
| AudioSeal embed, warm | 9.0 | 11.0 | |
| AudioSeal detect, warm | 6.1 | 6.0 | |

Peak RSS of the single process (Kokoro + Applio + AudioSeal resident): **2.9 GB** (1.4 GB after Kokoro,
2.9 GB after the first conversion).

None of these are representative of an idle machine. Rerun on a quiet box before quoting any figure.

## Watermark check

- Embedded `CYV_MESSAGE` (0x1177) into the 40 kHz converted audio.
- Detect on watermarked: prob 1.00, decoded message 0x1177 (match, CRC ok).
- Detect on clean converted audio: prob 0.02 to 0.05, message None.
- Residual level: SNR about 26 dB relative to the converted signal (AudioSeal's own envelope fitting).
- Approach: downsample to 16 kHz, compute the AudioSeal residual, upsample the residual to the original
  rate and add it, so the original bandwidth is untouched. Detection after this works as above.
- Not covered here: Opus/MP3 round trips and false-positive rate (spike B4).

## Not measured / caveats

- No MPS, no fp16. Applio ran fp32 on CPU.
- Output was written as 24-bit WAV (not shipped); no listening test beyond length, finite values and level checks.
