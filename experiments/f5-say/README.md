# f5say: typed English in your voice (experiment)

**Not part of cyv.** This is a spike. It is the best-sounding way we have
found to turn typed English into your voice:

1. **F5-TTS** (Apple MLX build) speaks the text, copying your voice and
   delivery from a short reference clip.
2. **`cyv convert`** then puts your trained voice model (`sachinkoli2`) on
   top, which also adds the cyv watermark.

Results and numbers: [docs/benchmarks/B7-clone-tts-m1.md](../../docs/benchmarks/B7-clone-tts-m1.md).
Every run, the exact weights, and what to tune next: [TUNING.md](TUNING.md).

## Quick use

```bash
f5say "Sorry I missed your call. I am free now if you want to talk." -o ~/Downloads/call.wav
```

Several lines, one file each (`line1.wav`, `line2.wav`, ...):

```bash
f5say "First line." "Second line." -o ~/Downloads/lines/
```

### Options

| Option | What it does | Default |
|---|---|---|
| `-o` | Output `.wav`, or a folder when you give several lines | required |
| `--voice` | Which cyv voice to convert into | `sachinkoli2` |
| `--ref` | Which reference clip to copy tone from | `ref_11s` |
| `--steps` | Quality vs speed. 32 sounded right; 16 is twice as fast but audibly worse | `32` |
| `--seed` | Change it for a different take of the same line | `42` |
| `--speed` | Speaking rate vs the reference clip (1.1 = 10% faster) | `1.0` |
| `--cfg` | How hard to follow the text and voice | `2.0` |
| `--sway` | Sway sampling coefficient | `-1.0` |
| `--method` | Solver: `euler`, `midpoint` (2x work), `rk4` (4x work) | `euler` |
| `--raw` | Skip `cyv convert` (F5 output only) | off |
| `--force` | Overwrite existing files | off |

### What to expect on an 8 GB M1

- About **70 s per sentence** at 32 steps, plus about 4 s for the convert.
  The first convert of a session takes about 25 s while the daemon starts.
- About 3.3 GB of memory for F5, plus about 1.1 GB for the cyv daemon.
  Close big apps first; swapping makes it much slower.
- **Tone follows the reference clip.** The default clip is calm and
  explanatory, so every line comes out that way. To get another tone,
  record a clip in that tone (next section).

## Use a different reference clip

The clip sets the tone. Put two files in `~/.cache/tts-spike/ref/`:

- `<name>.wav`: 8 to 12 s of you speaking, mono, 24 kHz.
- `<name>.txt`: the **exact** words you said in it. A wrong transcript makes
  the pacing stiff.

Make the `.wav` from a Voice Memo:

```bash
ffmpeg -i "New Recording.m4a" -ac 1 -ar 24000 ~/.cache/tts-spike/ref/casual.wav
```

Then: `f5say --ref casual "Hey, how are you?" -o hi.wav`

## Setup (already done on this Mac)

Everything lives in `~/.cache/tts-spike/` (outside the repo; voice data never
goes in git).

| Piece | Where |
|---|---|
| Python 3.11 venv with f5-tts-mlx 0.2.6, mlx 0.32.4 | `~/.cache/tts-spike/f5/` |
| Locked package list (with hashes) | `~/.cache/tts-spike/req_f5.txt` |
| Model weights | `~/.cache/huggingface/hub/models--lucasnewman--f5-tts-mlx` and `models--lucasnewman--vocos-mel-24khz` |
| The script f5say runs (copy of `experiments/f5-say/f5say.py`) | `~/.cache/tts-spike/f5say.py` |
| Reference clip + transcript | `~/.cache/tts-spike/ref/ref_11s.wav`, `ref_11s.txt` |

If you edit `f5say.py` in the repo, copy it again to `~/.cache/tts-spike/`.

The `f5say` command is a shell function in `~/.config/zsh/.zshrc.local`:

```bash
f5say() { ~/.cache/tts-spike/f5/bin/python ~/.cache/tts-spike/f5say.py "$@"; }
```

### To rebuild from scratch

```bash
uv venv ~/.cache/tts-spike/f5 --python 3.11
uv pip install --python ~/.cache/tts-spike/f5/bin/python f5-tts-mlx
uvx --from huggingface_hub hf download lucasnewman/f5-tts-mlx \
    model.safetensors duration_v2.safetensors vocab.txt
uvx --from huggingface_hub hf download lucasnewman/vocos-mel-24khz
```

Then apply the patch below. Download only the files listed: the full repo is
about 4 GB of variants you do not need.

**Slow PyPI?** On this network files.pythonhosted.org ran at about 20 KB/s.
What worked: lock with hashes from pypi.org, then install from a mirror. The
hashes still guarantee the same files.

```bash
uv pip compile req_f5.in --generate-hashes -o req_f5.txt --python ~/.cache/tts-spike/f5/bin/python
uv pip install --python ~/.cache/tts-spike/f5/bin/python --require-hashes -r req_f5.txt \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Required patch (mlx 0.32)

f5-tts-mlx 0.2.6 crashes with `TypeError: normal(): incompatible function
arguments` on mlx 0.32. In `~/.cache/tts-spike/f5/lib/python3.11/site-packages/f5_tts_mlx/cfm.py`,
line 373, change `(self.num_channels, dur)` to `(self.num_channels, int(dur))`.
The original is kept next to it as `cfm.py.orig`. Reinstalling the package
removes the patch.

## Known problems

- **No tone control.** Tone comes only from the reference clip.
- **English only.** This F5 model has no Devanagari characters. Hindi and
  Marathi would need IndicF5 (AI4Bharat; gated on Hugging Face, not tested).
- **F5's own duration predictor is broken here** (every clip ~2.8 s). f5say
  works around it with a length estimate from your speaking rate.
- **License: verify.** f5-tts-mlx code is MIT; upstream F5-TTS weights are
  CC-BY-NC (non-commercial). Personal use only until checked.
- Not in the cyv design. Shipping it needs ADRs (second TTS engine; reference
  clip in the voice pack).

## Next session: open questions

1. Tone: record 2 or 3 reference clips in different moods and compare.
2. Where to run it: this Mac (~70 s/line), Colab (~4 s/line), or a bigger Mac.
3. Whether to propose it for cyv (ADRs, license check, `docs/licenses.md`).
