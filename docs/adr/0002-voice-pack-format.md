# ADR-002: Voice pack v1 format

Status: Accepted

## Context

Training happens in the cloud (Colab), and the result has to be moved to a Mac and loaded there. The pack is a file made by someone else, so it is also an attack surface. The format must be simple to check, and it must fail safely when something is unknown.

## Decision

A voice pack is a zip file named `<name>.cyvpack.zip`. It contains exactly four files:

- `model.pth`: the model weights. These are sanitized on import.
- `model.index`: the retrieval index.
- `voice.json`: metadata. It includes `schema_version`, `name` (lowercase letters, digits, `_` or `-`, 1 to 32 characters), `engine` (`applio-rvc`), `engine_version` (a commit), `rvc_version`, `sample_rate`, `embedder`, `f0_method`, `vocoder`, `epochs`, `dataset_seconds`, `tts_base` (voice and language), `pitch_shift`, and sha256 checksums.
- `consent.json`: the consent statement, its version, the owner's name, and the creation time.

**Unknown values fail closed.** If `schema_version`, `embedder`, `vocoder` or `rvc_version` is unknown, import refuses with a clear message. A pack must never silently sound wrong.

## Decision details

- The checksums detect corruption. They do not prove who made the pack.
- Consent is recorded but not enforced on import, because a file can be forged.
- `tts_base` is chosen when the dataset is prepared.

## Consequences

- Import is strict. Some future packs will need a new schema version and a matching release.
- Checking four known names keeps the import code small and easy to test.
- Users must trust the source of a pack, because a pack can contain code-equivalent pickles.
