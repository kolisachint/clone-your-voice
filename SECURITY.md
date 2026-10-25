# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Use GitHub's **private vulnerability reporting**: go to the repository's **Security** tab and choose **Report a vulnerability**. This keeps the details private until a fix is ready.

## Supported versions

Only the `main` branch is supported. There are no released versions yet. Fixes land on `main`.

## Threat model (summary)

The full design is in [docs/DESIGN.md](docs/DESIGN.md), sections 6.1 and 6.2. In short:

**Voice packs are code-equivalent.** A `.pth` model file is a Python pickle. Loading an untrusted pickle can run code on your machine. A checksum inside a pack does not prove who made it. So:

- Import only packs from people you trust.
- Import never uses `extractall`. It allows exactly four files, rejects absolute paths, `..`, and symlinks, and caps the size.
- Model weights are loaded only with `torch.load(..., weights_only=True)`, in a separate process, and re-saved as plain tensors before the daemon uses them.

**The local daemon runs as you, for you only.** `cyvd` listens on a unix socket, not a network port. The socket directory is private (mode 0700), and the daemon checks the peer user ID. This stops other users on your Mac. It does **not** stop malware already running as your user. If your account is compromised, an attacker can also use your voice. The daemon exits when idle, and it never accepts file paths for output, so the CLI writes files.

**Out of scope:** this project does not prevent misuse of a voice you are allowed to use. See [USAGE_POLICY.md](USAGE_POLICY.md). The watermark is provenance, not security.
