# B1: T4 training time vs epochs × data

Environment: Colab Tesla T4, 15360 MiB GPU RAM,
Applio @ c817f1f83f689ef72e0dfeb1c17ae990b9248506,
notebook + cyv @ 0623698 (MPLBACKEND fix live).

| Run | Data | Epochs | Wall clock | Self-test | Listening rank | Pack |
|---|---|---|---|---|---|---|
| Phase 1 (`sachinkoli`) | 1.2 min usable (73 s, 1 clip, -25.9 dBFS, 0% clipping) | 100 | not recorded | PASS (exported) | Robotic/thin, but recognisably the speaker. Expected: smoke test. | 62 MB |
| Phase 2 (`sachinkoli2`, planned) | 15.4 min usable (16.5 min raw, 2 clips, -22.3 dBFS, 0% clipping, 33% silence) | 200 (TBD) | pending | pending | pending | pending |

Notes:

- Phase 1 pack (`sachinkoli.cyvpack.zip`, voice.json epochs 100) verified
  end-to-end on the Mac: `voices import` OK, `say` and `convert` both
  produce audio, watermark detects `prob=1.00 (crc ok)` on both.
- Phase 1 trained on the pre-fix notebook era params but exported with the
  fixed helper; the MPLBACKEND crash and the Voice-Memos `.m4a` decode bug
  were both fixed in-repo afterwards (not training-quality issues).
- Phase 2 dataset (`dataset-p2.zip`): part1 (1.3 min, mono) + one 15.1 min
  stereo Voice Memo, same evening. Phases were mixed here for quality
  (the one-session rule in the run guide is for clean benchmark comparison).
- Phase 2 suggestion: fresh `VOICE_NAME` (old checkpoints belong to a
  different dataset), EPOCHS 200, same Drive folder layout. If Colab
  disconnects mid-run, re-run with the same name — checkpoints resume.
