# Session notes: why `cyv say` doesn't sound like me (2026-10-27)

Handoff for the next session. Short version first.

## What we found

1. **The trained voice is fine.** `cyv convert` on my own recording sounds
   as good as the Colab self-test. Export, import and sanitize lose nothing.
2. **`cyv say` is the weak part.** It speaks with Kokoro's delivery and only
   borrows my timbre. Pitch -6 was the best setting; still not me.
3. **Voice-cloning TTS + my model wins.** Blind test on Colab, 3 sentences:
   F5-TTS -> `cyv convert` ranked first, Chatterbox -> convert second,
   Kokoro last in every sentence.
4. **It runs on my 8 GB M1** with F5 on MLX: about 70 s per sentence at
   32 steps, about 3.3 GB. 16 steps is faster but audibly worse.
   Chatterbox needs 8.4 GB of GPU memory and swaps: ruled out.
5. **Tone is still off.** F5 copies tone from the reference clip, and the
   clip is calm and explanatory.

Numbers: [docs/benchmarks/B7-clone-tts-m1.md](../docs/benchmarks/B7-clone-tts-m1.md).

## How to use it now

```bash
f5say "Text to speak." -o ~/Downloads/out.wav
```

Full guide, options and setup: [f5-say/README.md](f5-say/README.md).
Every run, exact weights and hashes, and which knob to try next:
[f5-say/TUNING.md](f5-say/TUNING.md).

## Where things are

| Thing | Where |
|---|---|
| This note, B7, `f5say` | branch `agent/mac-spike` (not pushed) |
| Colab test notebook | branch `tts-clone-test` (pushed; open from GitHub in Colab) |
| F5 install, ref clip, test audio | `~/.cache/tts-spike/` (never in git) |
| Listening files | `~/Downloads/tts-clone-test/` |
| `f5say` shell function | `~/.config/zsh/.zshrc.local` |

## Watch out

- **Colab saves outputs into GitHub.** Before saving a notebook from Colab:
  Edit -> Clear all outputs. (Once, audio and a transcript of my voice were
  pushed; the branch was cleaned and force-pushed, but GitHub may still
  serve old commit `5568959` by its hash until GitHub Support purges it.)
- **PyPI is very slow on this network.** Use the hash-locked mirror install
  in the f5-say README.
- **f5-tts-mlx needs a one-line patch** for mlx 0.32 (README).
- **F5 weights license is probably non-commercial.** Verify before shipping.

## Next steps (pick one)

1. **Tone:** record 2 or 3 Voice Memos (about 10 s each) in the tone I want,
   with exact transcripts. Run `f5say --ref <name>` and compare.
2. **Where to run:** this Mac (~70 s/line), Colab (~4 s/line), or a bigger Mac.
3. **Ship it in cyv:** ADRs for a second TTS engine and a reference clip in
   the voice pack, a license check, and `docs/licenses.md` entries.

## Cleanup when done

- Chatterbox can go: `~/.cache/tts-spike/cb` and
  `~/.cache/huggingface/hub/models--ResembleAI--chatterbox` (about 8.5 GB).
- Unused F5 variants in the HF cache (about 2 GB).
