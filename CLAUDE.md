# CLAUDE.md

Guidance for AI agents (and humans who like the same rules) working in this repo.

## Read first

- [docs/DESIGN.md](docs/DESIGN.md) is the source of truth. Follow it.
- Decisions are recorded as ADRs in `docs/adr/`. Do not contradict an accepted ADR. If you think one is wrong, propose a new ADR.
- Do not add features that are not in the design. If something is missing, ask or open an issue.

## Layout

- `src/cyv/`: the thin CLI. It is pure Python and **must not import torch** or other heavy packages. `cyv --help` has to stay instant.
- `src/cyvd/`: the daemon, which runs in its own venv. It holds the engine (`engine_applio/`), TTS (`tts_kokoro/`), watermark (`watermark/`) and server (`server/`) code.
- `tests/`: unit tests, pack-import tests, and a CPU smoke test.
- `notebooks/`: the Colab training notebook and `constraints.txt`.
- `docs/`: design, ADRs, benchmarks, guides, and `licenses.md`.

## Commands

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest
CYV_E2E=1 .venv/bin/python -m pytest   # slow tests, real models
```

Set up the environment as described in [CONTRIBUTING.md](CONTRIBUTING.md).

## Hard rules

- **Never commit weights or voices.** That covers `.pth`, `.index`, `.cyvpack.zip`, `.safetensors`, `.onnx`, and any voice data. Do not work around `.gitignore`.
- **Pin rule:** the Applio commit in `src/cyvd/engine_applio/pin.py` must match the Applio commit used in `notebooks/train_colab.ipynb`. If you change one, change the other in the same commit, and say so in the PR.
- **Dependencies:** a new dependency needs its license recorded in `docs/licenses.md`. Mark anything you have not verified as "verify".
- **Engine or TTS changes** need a benchmark entry in `docs/benchmarks/`.
- **Do not commit or push** unless the maintainer asks you to.

## Security rules

- Never call `torch.load` without `weights_only=True`. Never use `weights_only=False`. If a checkpoint needs extra types, add them with `torch.serialization.add_safe_globals`.
- Never use `zipfile.extractall`. Allowlist the exact member names and check each one.
- The daemon never takes a filesystem path for output. The CLI writes files and runs ffmpeg.
- Do not log or print voice data, consent text, or socket contents beyond what a user needs.

## Benchmarks

- Label the hardware for every number: chip, RAM, macOS version, and the cyv version.
- Report bad results as clearly as good ones. Do not present provisional targets as facts.
- Performance targets stay out of the README until the M1 spikes (B2, B3) report.

## Working style

- Keep changes small and focused. Do not touch files outside the task.
- Use plain, friendly English in docs. Explain terms for non-experts.
- Prefer editing an existing file over creating a new one.
