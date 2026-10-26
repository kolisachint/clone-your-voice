# Detect latency: `cyv watermark detect` (M3 stub)

## Question
How long does watermark detection take per second of audio, per backend?
This feeds the M3 latency budget for `cyv watermark detect <file>`.

## Method
Provisional stub: no model-backed measurement has been run for this note.
When measured, run `Pipeline.detect` over the B4 clip set (20 watermarked +
50 clean, 16 kHz mono) on Apple Silicon via MPS and record wall time per
10 s of audio for the `audioseal` and `perth` backends, plus the
daemon round-trip overhead (`Client.detect` over the unix socket).

## Environment
Apple M1, macOS 27.2, cyv (M3 worktree). No numbers collected yet.

## Results
Provisional — not measured on a model backend. For reference, B4 (see `B4-watermark.md`)
measured detect on a loaded Linux x86 CPU box at 6.1 s per 10 s of audio
for AudioSeal and 0.95 s per 10 s for Perth; Apple Silicon/MPS numbers are
expected to be much lower but must be re-measured, not assumed.

## Smoke measurement (plumbing only, no model)
`Pipeline.detect` with the `FakeWatermarker` over a 10 s 16 kHz mono clip
takes ~0.1 ms/call on this machine (20 calls, wall time). This covers WAV
decode plus the detect plumbing only — it says nothing about AudioSeal or
Perth inference cost, which still needs measuring per the Method above.

## Conclusion
Do not quote a detect-latency target until this note has real Apple
Silicon numbers. The CLI, protocol, and daemon plumbing for `detect` are
in place; only the measurement is missing.
