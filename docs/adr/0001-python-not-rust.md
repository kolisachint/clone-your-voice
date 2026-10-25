# ADR-001: Python, not Rust

Status: Accepted

## Context

The app runs machine-learning models (voice conversion and text-to-speech) on a Mac. Two things make it slow: the GPU math inside the models, and the time it takes to start the program and load models. Neither is limited by CPU-heavy code written in our own language.

We also want contributors to be able to read and change the code easily.

## Decision

Write the project in Python. Do not write it in Rust or another systems language.

The thin CLI (`cyv`) stays small and light so it starts fast. Heavy work runs in a long-lived daemon (`cyvd`) that keeps models warm.

## Consequences

- The ML ecosystem (PyTorch, Applio, Kokoro, AudioSeal) is available directly.
- Contributors need only Python knowledge to work on most of the code.
- Startup cost is managed by keeping `cyv` free of torch and by the idle-exit daemon, not by rewriting in another language.
- Speed is limited by the GPU and model loading, not by Python itself. We measure this in the benchmarks.
