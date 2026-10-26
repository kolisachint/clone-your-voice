# M1 setup replication (uv path) + `uv lock` groundwork

## Question
Can B0's "clean-Mac install" be replicated with `uv` (no brew), and what
blocks a committed `uv.lock` for the daemon set (DESIGN §4.1)? Also: is the
notebook↔daemon Applio pin consistency (B1's Colab-only premise) covered?

## Method
Fresh build driven by the repo's own plan: `src/cyv/setup.py::plan()`
(steps 1–9: venv → `project[daemon]` → Applio fetch @ pin →
`requirements-engine.txt` → spaCy model), then model downloads (steps 10+,
all sha256-verified by `download_model`). `uv` 0.13.5 installed via the
direct script (`curl -LsSf https://astral.sh/uv/install.sh | sh`) —
**no brew, per policy**. `uv python install 3.12` provided CPython 3.12.15.
Scratch locations: venv + Applio + models all under `/tmp/cyv-heavy`
(`CYV_HOME=/tmp/cyv-heavy`); the real `~/.cyv` was left untouched.
`uv lock` was attempted in a throwaway copy (`/tmp/uvlock-probe`), not in
the repo.

## Environment
Apple M1, 8 GB, macOS 27.2. `xcode-select -p` → CLT present.
ffmpeg 9.0.2 at `/opt/homebrew/bin/ffmpeg` (pre-existing; `libopus` and
`libmp3lame` encoders confirmed). cyv 0.0.1 @ `3e5c009`.

## Results

Setup works end to end with uv. Installed versions (matches B0 except spacy):

| Package | This build | B0 |
|---|---|---|
| python | 3.12.15 | 3.12.16 |
| torch / torchaudio | 2.11.0 / 2.11.0 | same |
| numpy / transformers | 2.4.6 / 5.13.1 | same |
| kokoro / audioseal | 0.9.4 / 0.2.0 | same |
| spacy / preshed | **3.8.16** / 3.0.9 | 3.8.14 / 3.0.9 |
| faiss-cpu / soxr | 1.14.3 / 1.1.0 | same |
| en_core_web_sm | 3.8.0 | same |
| Applio | `c817f1f8` (= pin) | — (no checkout existed) |

spacy resolved 3.8.16 (newer than B0's 3.8.14) because nothing pins it;
`preshed==3.0.9` + `en_core_web_sm` 3.8.0 still install and load fine, and
Kokoro G2P works with the bundled espeak-ng dylib (no `CYV_ESPEAK_LIBRARY`
override needed on this Mac — same as B0).
Weights: 985 MB total under `models/` (RMVPE 181 MB, ContentVec 378 MB,
Kokoro 327 MB, AudioSeal ~93 MB, 54 voices), every file passing its
`models.toml` sha256 on first download.

`uv lock` groundwork:

- Plain `uv lock` on the repo **succeeds** (128 packages) — but it does not
  lock the daemon *as installed*. It resolves **torch 2.14.1**, while
  `cyv setup` actually installs **torch 2.11.0** (step 8 installs Applio's
  `requirements-engine.txt`, whose `torch==2.11.0` overwrites step 6).
  A lockfile that drifts from the real install on its largest dependency
  is worse than no lockfile.
- What lives outside any `uv.lock` today (blockers, in order of annoyance):
  1. **Applio itself** — git-cloned at `APPLIO_COMMIT`, and it has no
     `pyproject.toml`/`setup.py`, so it cannot become a
     `git+https://…@sha` PEP 508 dependency without restructuring how
     Applio ships. The pin (`src/cyvd/engine_applio/pin.py`) is the lock.
  2. **`requirements-engine.txt`** — Applio-upstream exact pins
     (`torch==2.11.0`, `transformers==5.13.1`, …) installed as a second
     `-r` step. Representing these in `pyproject` would fork from upstream
     and rot against every Applio update.
  3. **`daemon-constraints.txt`** (`preshed==3.0.9`, `torch>=2.6`) — pip
     constraints, not lockable; only honored because setup passes `-c`.
  4. **`daemon-spacy.txt`** — installed with `--require-hashes` from a
     GitHub release URL, outside project metadata.
- No restructure was attempted (per instructions). The honest options for
  M2: (a) keep `setup.py` as the installer and treat its logs as the record
  (status quo, reproducible only while PyPI + the Applio commit stay up);
  (b) vendor Applio's engine requirements into a `requirements-daemon.txt`
  with `uv pip compile` output checked in (a lock by another name, but one
  more file to rot); (c) make Applio a git dependency (blocked on Applio
  gaining build metadata). Recommendation: (a) for v1, revisit only if
  installs start breaking.

Notebook pin consistency (B1 stays Colab-T4, no training attempted):
`tests/test_notebook.py` passes — **36 passed, 4 skipped** (run via
`uv run --with pytest`; skips are pre-existing GPU/ Colab-only markers).
It asserts the notebook's `APPLIO_COMMIT` equals `pin.py`
(`c817f1f83f689ef72e0dfeb1c17ae990b9248506`) and that
`notebooks/constraints.txt` carries pins — so the Colab↔daemon consistency
rule is already enforced in CI. Nothing to do here.

## Caveats
- `/tmp/cyv-heavy` (venv, Applio checkout, 985 MB models) and
  `/tmp/uvlock-probe` (lock experiment + `uv.lock`) are scratch left behind
  for re-runs; safe to delete.
- Setup was run with `CYV_HOME` redirected; the product path `~/.cyv` was
  not exercised (directory creation, socket path length) — `cyv doctor`
  on a real `~/.cyv` is still untested on this Mac.
- `cyvd` daemon socket/serve path (`serve`/`say` through the unix socket)
  was not exercised — no voice pack, so `Pipeline.say` has nothing to load.
  The E2E smoke covered every stage except the voice-conditioned convert.
