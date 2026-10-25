# Troubleshooting on a Mac

Start here: run the built-in check.

```bash
cyv doctor
```

It checks the things below and prints the exact fix for anything missing. Most problems are solved by following its advice.

## Apple Silicon only

clone-your-voice runs on Apple Silicon (M1 or newer) in v1. Intel Macs are not supported.

## Xcode Command Line Tools

Some Python packages are compiled on your Mac, so they need the developer tools.

```bash
xcode-select --install
```

If an install step fails with a compiler error, install these tools and try again.

## ffmpeg from Homebrew

The output presets (Telegram and WhatsApp voice notes, MP3, WAV) use ffmpeg.

```bash
brew install ffmpeg
```

Check that it works with `ffmpeg -version`.

## MPS fallback (Apple GPU)

The models can use the Apple GPU through PyTorch's MPS backend. Some operations are not available there, so PyTorch needs a fallback to the CPU for them.

If you see an error about an operation not being supported on MPS, set this before you run `cyv`:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

This is slower for the parts that run on the CPU, but it works.

## macOS 26 and the M1 attention bug

There is a known problem with one of PyTorch's attention functions (SDPA) on some M1 Macs running macOS 26. We are measuring how it affects speed and quality (benchmark B3). If you see strange errors or noise in the output on macOS 26, try the CPU path, and please report it with the benchmark template in `docs/benchmarks/`.

## Memory on an 8 GB Mac

The models take a lot of memory. On an 8 GB Mac:

- Close other large apps, especially browsers with many tabs, while you work.
- Use **one voice at a time**. The daemon keeps one voice warm.
- The daemon **exits when it has been idle** (15 minutes by default). This frees its memory. The next `cyv say` starts it again, which takes a few seconds longer.
- You can stop it yourself with `cyv stop`.

Measured memory numbers will be in `docs/benchmarks/` once benchmark B2 is done.

## Socket path too long

The daemon talks to the CLI through a unix socket file. macOS limits the length of that path (about 104 bytes). If your home folder path is long, the socket moves to a shorter folder:

```
$TMPDIR/cyv-<your-user-id>/
```

You do not need to do anything. If you see an error about a socket path, run `cyv doctor`, and include its output in any bug report.

## Still stuck?

Open a bug report with the template in `.github/ISSUE_TEMPLATE/`. Include your chip, RAM, macOS version, the cyv version, and the output of `cyv doctor`. Do not include voice recordings or voice packs.
