# B1 Colab run checklist (T4 GPU)

This is the step-by-step list for benchmark **B1** (DESIGN §3):
training time and quality on a real Colab T4, across
epochs (100 / 200 / 400) × data length (2 / 5 / 10 min).
It cannot run on the Mac — work through it top to bottom in Colab.

Background lives in [colab.md](colab.md). This page is the run list; it links
instead of repeating.

## 0. Before you open Colab

Work in **phases**. Each phase re-runs the same notebook with more
of your recordings, producing better weights each time.

- **Phase 1 (first run): end to end with ~2 minutes.**
  Proves the whole loop works: record → `dataset prepare` →
  Colab T4 → pack → import → `say`/`convert` → watermark detect.
  Expect a rough voice — 2 minutes is a smoke test, not a final clone.
- **Phase 2 (later): 5 minutes** of same-room, same-setup recordings.
  Better timbre, fewer artifacts.
- **Phase 3 (later): 10 minutes**, then the full B1 matrix
  (2/5/10 min × 100/200/400 epochs) logged in
  `docs/benchmarks/B1-training.md`. Trim the subsets from one session
  so only data length varies.

Never mix phases in one dataset: one pack, one session's recordings.

- [ ] Record your voice following [recording.md](recording.md).
  Aim for **10 minutes** of clean speech — you can cut shorter
  2-minute and 5-minute subsets from it later for the matrix.
  A ready-to-read 4-part script is [here](recording-script.md).
- [ ] On your Mac, make the dataset:
  `cyv dataset prepare <your-recordings-folder> -o dataset.zip`.
- [ ] Read the one-line consent statement it shows and type it only if it is
  true. It is saved in the pack as a record, not enforcement —
  see [USAGE_POLICY.md](../../USAGE_POLICY.md).
- [ ] Check you have a Google account with a few GB free on Drive
  (checkpoints + output live in `MyDrive/clone-your-voice/<name>/`).
- [ ] Reminder: only your own voice, or one you have explicit permission for.
  You are responsible for following Google Colab's terms
  (they ban creating deepfakes).

## 1. Colab setup

- [ ] Open `train_colab.ipynb` in Colab (File > Open notebook > GitHub).
- [ ] **Runtime > Change runtime type > T4 GPU.** CPU training would take days.
- [ ] Run the cells **top to bottom, in order**. Nothing else to install —
  the notebook builds its own Applio venv.
- [ ] **Never start Applio's Gradio web UI, and never open a tunnel**
  (ngrok and friends). Colab's free tier restricts UI-driven notebooks,
  and the repo's notebook test refuses any UI/tunnel code
  (`tests/test_notebook.py`). The notebook uses Applio's
  command-line path only — that is the supported path.
- [ ] If Colab disconnects: reconnect, mount Drive, and **run all cells again
  with the same voice name**. Checkpoints are mirrored to Drive while
  training runs, so it resumes instead of restarting.

## 2. Pin check (do this once per run)

- [ ] The notebook installs Applio at one exact commit
  (`APPLIO_COMMIT` in the install cell). Confirm it still equals
  `src/cyvd/engine_applio/pin.py`:
  both should read `c817f1f83f689ef72e0dfeb1c17ae990b9248506`.
  If they differ, stop — the pin must match
  (`CLAUDE.md` pin rule; `tests/test_notebook.py` checks this too).
- [ ] `notebooks/constraints.txt` is verified automatically: the install cell
  checks both helper files against embedded sha256 hashes and refuses to run
  on a mismatch. If it refuses, re-open the latest notebook from GitHub so
  notebook + helpers are the same version.

## 3. The experiment matrix

Notebook defaults are 40000 Hz, 200 epochs, batch size 8
(saved every 10 epochs). Keep 40000 Hz and batch 8 for every B1 run;
vary only epochs and data length.

| Run | Data | Epochs | Wall time | Notes |
|---|---|---|---|---|
| 1 | 2 min | 100 | | |
| 2 | 2 min | 200 | | |
| 3 | 2 min | 400 | | |
| 4 | 5 min | 100 | | |
| 5 | 5 min | 200 | | baseline (notebook default) |
| 6 | 5 min | 400 | | |
| 7 | 10 min | 100 | | |
| 8 | 10 min | 200 | | |
| 9 | 10 min | 400 | | |

Tips:

- Make the 2-minute and 5-minute datasets by trimming the same 10-minute
  recording (keep mic, room, and style identical — only length should vary).
- Use a **fresh voice name per run** (e.g. `b1-e100-d2`), so checkpoints on
  Drive never mix sample rates or epoch counts.
- For each run, note the wall-clock time from the train cell and,
  at the end, the last training-loss lines the self-test cell prints.
- Quality criteria: **listen** to every self-test output yourself.
  PASS/FAIL is only a heuristic (not silent, no NaN, sensible length).
  It should sound like you; robotic or noisy usually means too little
  or too noisy audio. Rank runs by ear, best to worst.

Log everything in `docs/benchmarks/B1-training.md` (create it).
Start the file with this hardware label
(CLAUDE.md requires it for every benchmark entry):

```markdown
# B1: T4 training time vs epochs × data

Environment: Colab <GPU shown by the check cell, e.g. Tesla T4>,
<n> GB GPU RAM, Applio @ c817f1f83f689ef72e0dfeb1c17ae990b9248506,
notebook + cyv <repo commit, e.g. b642ba5>.
```

then one row per run: data minutes, epochs, wall time, last loss lines,
self-test PASS/FAIL, your listening rank, and the `.cyvpack.zip` size.

## 4. Self-test gate and export

- [ ] Each run ends in the self-test cell. **FAIL = do not use that pack.**
  Either accept it as a B1 data point (short data / few epochs failing is
  itself a finding) or re-run with more/cleaner audio.
- [ ] If the audio sounds fine but the heuristic says FAIL, you may tick
  `EXPORT_EVEN_IF_SELFTEST_FAILS` and export — note in the log that you did.
- [ ] Download `<name>.cyvpack.zip` from
  `My Drive > clone-your-voice > <name> > output` in your browser.
- [ ] On your Mac: `cyv voices import ~/Downloads/<name>.cyvpack.zip`.
  Import only packs you trust (see [SECURITY.md](../../SECURITY.md)).
- [ ] Privacy cleanup, once the pack is safe on your Mac
  (DESIGN §6.4 — deleting in Drive only moves files to Trash):
  1. Run the notebook's delete-dataset cell (tick the box), or delete
     `dataset/` and `checkpoints/prepared.tar` by hand.
  2. Open [Drive Trash](https://drive.google.com/drive/trash) and
     **Empty trash** (or right-click > Delete forever).
  3. **Runtime > Disconnect and delete runtime** to wipe Colab's `/content`.

## 5. If something fails

| Message | What to do |
|---|---|
| **No GPU** (`No GPU found`) | Runtime > Change runtime type > T4 GPU > Save, run the cell again. |
| **Quota exhausted** (T4 greyed out / no GPU available) | Free GPU time is used up; it resets after a while (often within a day). Come back later, or use Colab Pro. |
| **Drive auth failed** | Run the Drive cell again and approve the popup. With several Google accounts, pick the one that owns the Drive; third-party cookies must be allowed for colab.research.google.com. |
| **Pin/hash mismatch** (commit differs, or helper sha256 refuses) | Stop. Do not run a mixed-version notebook. Re-open the latest `train_colab.ipynb` from GitHub and check the pin (§2). |
| **CUDA out of memory** | Lower `BATCH_SIZE` (try 4) in the settings cell and re-run. |
| **Train fails: `Key backend: 'module://matplotlib_inline...'`** | Fixed Oct 2026 in `cyv_colab.run_and_sync` (forces headless `Agg` in Applio subprocesses). Re-open the latest notebook from GitHub and re-run from the top. |
| **Preprocess/extract/train silently unfinished** | Applio's `core.py` exits 0 even on failure — the notebook checks for output files instead. Scroll up in that cell's output for the real error. |
| **Self-test FAIL** | Do not use the pack. Improve the recordings (see [recording.md](recording.md)) or change epochs, and train again. |
| **Checkpoint sample-rate clash** | You resumed with a different `SAMPLE_RATE` than the Drive checkpoints. Use the same value, or delete that voice's `checkpoints/` folder to start over. |

Paths cited: notebook and helpers in [`../../notebooks/`](../../notebooks/),
pin in `src/cyvd/engine_applio/pin.py`, checks in `tests/test_notebook.py`,
policy in [USAGE_POLICY.md](../../USAGE_POLICY.md).
