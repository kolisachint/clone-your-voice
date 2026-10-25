# B6: SynthID audio re-check (embedding access for third parties)

## Question
As of Oct 2026, does Google offer a SynthID audio embedding SDK or API that third parties can run offline? (DESIGN §3/§6.3 assume: no.)

## Method
Web search plus fetches of Google's pages (searched 2026-10-25). Secondary sources are marked as such; anything not confirmed on a Google-owned page is marked UNVERIFIED.

## Environment
Desk research only; no code run.

## Results

| Claim | Evidence | Status |
|---|---|---|
| SynthID audio watermarking is applied to audio from Google's own products (Lyria, NotebookLM podcast feature) | [Google DeepMind SynthID page](https://deepmind.google/models/synthid/) says it embeds a watermark into audio "generated or published through our AI music generation model Lyria or the podcast generation feature of Notebook LM" | Verified (Google) |
| The DeepMind page offers no SDK/API/open-source link for audio; partners are mentioned only for detection (OpenAI, NVIDIA, Kakao, "soon" Apple) | same page | Verified (absence on that page) |
| Only SynthID **Text** is open source | [Google blog, 2025-05-20](https://blog.google/innovation-and-ai/products/google-synthid-ai-content-detector/): open-source item is text watermarking; no detector SDK/API/offline mode mentioned | Verified for the 2025 post |
| SynthID Detector portal does audio detection (pinpoints watermarked segments) but only for content from Google/partners; requires uploading to Google's portal | same blog; portal says "not a general AI detector" per [secondary coverage](https://www.unite.ai/?p=481248) | Verified (2025 blog); not-general-detector quote UNVERIFIED (secondary) |
| Portal opened to everyone on 2026-10-07 | only [Unite.ai](https://www.unite.ai/?p=481248) and one blog ([braindetox.kr](https://braindetox.kr/en/posts/google_synthid_detector_ai_media_watermark_2026.html)); not found on a Google page | UNVERIFIED |
| Partner adoption (ElevenLabs partnered with Google to embed SynthID in generated audio) | search summary of ElevenLabs/[InfoQ](https://infoq.com/news/2026/05/google-synthid-content-detection); ElevenLabs' own [help page](https://elevenlabs.io/docs/help-center/legal/audio-detector/what-is-synth-id-and-why-is-eleven-labs-using-it) does not mention SynthID when fetched | UNVERIFIED; suggests case-by-case partnership, not public SDK |
| Google "AI Content Detection API" (Google Cloud, May 2026) exists for select partners (Shutterstock, Snap, Avid, Fox Sports, Canva); accepts JPEG/PNG/WebP images | [InfoQ](https://infoq.com/news/2026/05/google-synthid-content-detection), [Search Engine Journal](https://www.searchenginejournal.com/google-brings-ai-content-verification-to-search/575320/) via search summary | Secondary only; it is image detection, not audio embedding. UNVERIFIED details |
| Any public SynthID audio embedding SDK/API/offline model | none found in any source | Not found |

## Conclusion
No evidence of a public or offline SynthID audio embedding SDK/API for third parties as of 2026-10-25. Access appears to be by partnership; detection is via Google-hosted portal/limited API and requires uploading audio. **B6 result: still "no"**; DESIGN §6.3 wording stands. Keep "SynthID-style", never "SynthID". Keep the `Watermarker` interface so a backend can be added if this flips.

## Caveats
- Search tool is US-only and returned several low-authority secondary sources; I could not retrieve a Google page dated Oct 2026 about audio specifically. Absence of evidence, not proof; a private partner program could exist (and would not be offline anyway).
- The 2026-10-07 portal opening and the Cloud detection API come from secondary sources.
- Re-check before M3 (the SynthID backend item).
