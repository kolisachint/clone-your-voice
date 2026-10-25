# Contributing

Thanks for helping. This project is small and aims to stay easy to read. Please read [docs/DESIGN.md](docs/DESIGN.md) and [USAGE_POLICY.md](USAGE_POLICY.md) before you start.

## Development setup

You need macOS on Apple Silicon for the full app. Tests also run on Linux in CI.

You also need **ffmpeg** (`brew install ffmpeg` on a Mac, `sudo apt-get install -y ffmpeg` on Ubuntu).

```bash
uv venv .venv --python 3.12
uv pip install -e '.[dev]'
```

The tests use **torch on CPU**. On Linux, install it from the CPU index:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On a Mac, the default torch wheel is fine.

## Lint and format

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

To fix formatting, run `.venv/bin/ruff format .`.

## Tests

```bash
.venv/bin/python -m pytest
```

Slow tests download models or run real inference. They are skipped unless you set `CYV_E2E=1`:

```bash
CYV_E2E=1 .venv/bin/python -m pytest
```

## Ground rules

- **The `cyv` package must not import torch.** `cyv` is the thin CLI. It has to start fast. Heavy code lives in `cyvd`, the daemon.
- **New dependency?** Record its license in [docs/licenses.md](docs/licenses.md) in the same pull request. Say "verify" if you have not checked it yet.
- **Engine or TTS change?** Add a benchmark entry in `docs/benchmarks/`. Label the hardware (chip, RAM, macOS version). Report what you measured, including bad results.
- **Never commit weights or voices.** That means `.pth`, `.index`, `.cyvpack.zip`, and similar files. `.gitignore` blocks most of them, but check your `git status` anyway.
- **Keep the design in sync.** If the code and `docs/DESIGN.md` disagree, change the design doc in the same pull request, or open an issue first. Decisions that change the design get an ADR in `docs/adr/`.

## Pull requests

Use the PR template. Before you ask for review, check:

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest` passes
- [ ] No torch import in `cyv`
- [ ] `docs/licenses.md` is updated if a dependency changed
- [ ] A benchmark entry is added if the engine or TTS changed
- [ ] No weights or voices are committed
- [ ] The usage policy is respected

## Labels

Maintainers use these labels, from the design doc:

- `good-first-issue`: a small, well-scoped starter task
- `preset`: output presets (`presets.toml`)
- `docs`: documentation and guides
- `benchmark`: measurements and results
- `notebook`: the Colab training notebook
- `security`: security reports and fixes (see [SECURITY.md](SECURITY.md))

## AI-assisted contributions

Contributions made with help from AI tools are welcome. They follow the same rules as any other contribution. [CLAUDE.md](CLAUDE.md) lists the conventions that AI agents follow in this repo. Please review what an agent wrote before you open a pull request, and make sure you understand the change.

## Conduct

Be kind and respectful. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
