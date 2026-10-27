# Voice-cloning TTS test (experiment)

Not part of cyv v1. A one-off test of whether a voice-cloning TTS
(F5-TTS or Chatterbox), optionally run through `cyv convert`, sounds
more like you than today's `cyv say`.

Your reference recording (`ref.wav`) is never committed. Upload it in Colab.

## 1. Colab

Open the notebook:
https://colab.research.google.com/github/kolisachint/clone-your-voice/blob/tts-clone-test/experiments/tts-clone-test/tts-clone-test.ipynb

Pick a T4 GPU, run all, upload `ref.wav` when asked.
It downloads `tts_out.zip` at the end.

## 2. Mac

Put `tts_out.zip` next to `run_local.py`, then:

```bash
python3 run_local.py
```

Listen to `blind/` per sentence, rank, then open `key.txt`.
