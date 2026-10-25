## Summary

<!-- What does this change do, and why? -->

## Type

<!-- Pick one -->
- [ ] Bug fix
- [ ] Feature
- [ ] Preset
- [ ] Docs
- [ ] Benchmark
- [ ] Notebook
- [ ] Security
- [ ] Refactor or chore

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest` passes
- [ ] No torch import in `cyv`
- [ ] `docs/licenses.md` is updated, if a dependency changed
- [ ] A benchmark entry is in `docs/benchmarks/`, if the engine or TTS changed (label the hardware)
- [ ] No weights or voices are committed
- [ ] The usage policy is respected ([USAGE_POLICY.md](../USAGE_POLICY.md))
