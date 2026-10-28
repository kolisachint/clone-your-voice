# ADR-006: Reference clips in the voice pack

Status: Proposed

## Context

The quality tier (ADR-005) uses a voice-cloning text-to-speech model, such as F5-TTS (ADR-007). These models copy the **delivery** of a short reference clip: its rhythm, intonation and tone. The B7 tests showed this clearly:

- F5 output follows the tone of the reference clip. A calm reference gave calm output on every line.
- Our one reference clip was calm and explanatory, so every line came out that way (`docs/benchmarks/B7-clone-tts-m1.md`, Notes).
- F5 also needs the exact words spoken in the clip. A wrong transcript made the pacing stiff in the experiment (`experiments/f5-say/TUNING.md`, section 5).

Today a voice pack (ADR-002) holds only the trained model, the index, `voice.json` and `consent.json`. It has no reference clip, so the quality tier would need the user's clip kept outside the pack, handled by hand, and never checked. That would be fragile and would bypass the import checks in DESIGN §6.1.

Reference clips are the user's voice data. A transcript of the user's speech is voice data too.

## Decision

Propose an optional `references` list in the voice pack, as **`voice.json` schema_version 2**.

### Pack contents (schema 2)

```
<name>.cyvpack.zip
├── model.pth          (as ADR-002)
├── model.index        (as ADR-002)
├── voice.json         schema_version = 2, plus the fields in ADR-002
│                      and an optional "references" list (below)
├── consent.json       (as ADR-002, with the consent text version bumped, see Consent)
└── ref/<id>.wav       0 to N reference clips, only when listed in voice.json
```

### The `references` entry

Each item in `references` has these fields:

| Field | Rule |
|---|---|
| `file` | `ref/<id>.wav`, where `<id>` matches `^[a-z0-9_-]{1,32}$` |
| `transcript` | The exact words spoken in the clip. Checked against nothing on import (a machine cannot verify it), but required to be non-empty text |
| `tone` | A short label from a fixed list, for example `calm`, `warm`, `excited`, `neutral`. The list is proposed in the implementation, not here. Unknown labels fail closed |
| `duration_s` | Between 8 and 12 seconds, inclusive. Checked against the audio on import |
| `sha256` | The sha256 of the WAV file. Checked on import |

Proposed audio format: mono WAV, 24 kHz, 16-bit PCM. This is the format used in B7. The format is part of the schema, so a different format needs a new schema version.

Proposed limit: at most 3 references per pack. The limit is a starting point and can change with a new schema version.

Example (illustrative values only):

```json
{
  "schema_version": 2,
  "name": "example_voice",
  "references": [
    {
      "file": "ref/calm_01.wav",
      "transcript": "<exact words, written at recording time>",
      "tone": "calm",
      "duration_s": 10.6,
      "sha256": "<64 hex characters>"
    }
  ]
}
```

### Import rules

These extend DESIGN §6.1 and ADR-002. They do not weaken them.

1. **Zip hygiene, unchanged in spirit.** Never use `extractall`. Allow only the member names listed in `voice.json`, plus `model.pth`, `model.index`, `voice.json` and `consent.json`. A member named `ref/<id>.wav` is allowed only when `<id>` appears in `references` and the file name matches `file` exactly. Reject absolute paths, `..`, symlinks and any other member. Each member is checked by name before it is written.
2. **Size caps.** Each reference is capped at 2 MB, which is more than 12 seconds of 24 kHz mono 16-bit audio (about 0.6 MB). The total pack stays under the 1 GB decompressed cap in §6.1.
3. **Checks before install.** The sha256 must match. The duration must be 8 to 12 seconds. The WAV header must match the proposed format. Any failure refuses the whole pack.
4. **Fail closed on unknown schema.** `schema_version` 1 and 2 are known. Any other value refuses import with a clear message, as ADR-002 already says.
5. **Schema 1 still imports.** A pack with `schema_version` 1 has no `references` key and no `ref/` files, and imports as before. A pack without references is never refused for lacking them.
6. **Sanitizing does not touch the clips.** The clips are audio, not pickles. They are never passed to `torch.load`. They are read by the audio reader that already runs in the sanitize subprocess, or by a separate check, and never executed.

### Privacy

- **Transcripts are voice data.** A `transcript` is a written copy of the user's words. It is never logged or printed, including in errors, `cyv doctor`, `cyv voices list` and debug output. Those commands may show "2 references" and nothing more.
- **Never commit.** Reference clips and their transcripts are never committed to git, including as test fixtures and notebook outputs. This matches CLAUDE.md, which already bans voice data in commits. Tests use a generated tone or a synthetic clip, never a real recording.
- **Colab outputs.** Notebook outputs can contain transcripts and audio. Clear all outputs before saving a notebook (see `docs/guides/sounding-like-you.md`). A test in `tests/test_notebook.py` should check that the notebook has no saved outputs.
- **Local storage.** Clips are installed under `~/.cyv/voices/<name>/` with the rest of the pack, not in the repository.

### Consent

- The clips are the user's voice, so the consent attestation covers them. Consent is still an attestation record, not enforcement (ADR-002, DESIGN §6.3). This ADR does not change that.
- Proposed: a new consent text version that says reference clips are included in the pack, with the same "your own voice, or one you have permission for" rule.
- `dataset prepare` asks for the new attestation when references are included.

### How clips get into a pack

Out of scope for this ADR and proposed as a follow-up: whether the clips are recorded and packed in `dataset prepare`, or added by the Colab notebook from a folder in the dataset zip. Either way, the rules above apply to the finished pack.

## Consequences

- **Packs without references still import.** Schema 1 stays valid. The quality tier is opt-in per voice.
- **Changes ADR-002.** ADR-002 says a pack contains **exactly four files**. A schema 2 pack with references has more. If this ADR is accepted, the ADR-002 rule must be read as "exactly four files for schema 1, and four plus the listed `ref/` files for schema 2". ADR-002 is not edited here.
- **Older cyv versions refuse schema 2 packs.** This is the fail-closed rule working as intended. A user needs a cyv release that knows schema 2 to import such a pack.
- **More attack surface.** Audio files add parsing code to import. The size caps, the exact name list and the sha256 check keep it small. A fuzz test (zip-slip and bad WAV headers, per DESIGN §7) must cover the new members.
- **Tone is set by the user.** The pack can store a calm clip, a warm clip, or both. Which one a line uses is chosen at synthesis time (a later CLI decision, not part of this ADR).
- **Limit:** a reference clip's transcript is not checked by the machine. A wrong transcript makes the speech stiff, as seen in the B7 tests. The docs must tell users to check it word by word.
