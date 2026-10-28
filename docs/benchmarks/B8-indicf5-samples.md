# B8: IndicF5 first listen (Marathi, Hindi, English)

## Question

Does IndicF5 sound good enough in Marathi and Hindi to be our speech model for those languages? And can it also speak English, so that one model covers all three?

## Method

- **Model:** `ai4bharat/IndicF5`, Hugging Face revision `fb652a96e508091045890dae5bb99fe511facbcf`.
- **Where it ran:** AI4Bharat's official demo Space (`ai4bharat/IndicF5`, Gradio 5.20.1, Hugging Face ZeroGPU), called with `gradio_client` from a Linux cloud container. The GPU model is not reported by the Space. The model weights are gated, and the container had no Hugging Face token, so it could not run the model itself.
- **cyv version:** 0.0.1 (branch `shortlist-f5-indicf5-rvc`). No cyv code was involved.
- **Reference voices:** AI4Bharat's own example clips were not reachable from the container, so two female speakers came from Google FLEURS (CC-BY-4.0), dev split. The pick was the cleanest clip of 7 to 11 s by a rough noise measure, peak-normalized to 0.8:
  - Hindi: `hi_in` `14156493267883433925.wav`, 7.2 s.
  - Marathi: `mr_in` `3400252328446804299.wav`, 9.5 s.
- **Lines:**
  - Marathi: नमस्कार! आज वातावरण किती छान आहे, नाही का? संध्याकाळी आपण चहा प्यायला बाहेर जाऊया. तू किती वाजता मोकळी होशील?
  - Hindi: नमस्ते! आज मौसम कितना अच्छा है, है ना? चलो शाम को बाहर चाय पीने चलते हैं। तुम कितने बजे फ्री हो जाओगी?
  - English, with the Hindi reference voice: "Hello! The weather is really nice today. Shall we go out for a cup of tea this evening?"
- **Listener:** the maintainer, one listener, not blind.

## Results

| Line | Output | Wall time | Listened |
|---|---|---|---|
| Marathi | 12.4 s of audio | not recorded | **Sounds good** |
| English | 2.7 s of audio | 6 s, including the Space queue | **Bad** |
| Hindi | none | n/a | Not generated |

- **Hindi failed.** After the first request, the Space answered "You have exceeded your ZeroGPU runs limit" for anonymous use. Retries did not help. Hindi has not been heard yet.
- **English is bad, as expected.** English is not one of IndicF5's 11 languages. The 2.7 s output is far too short for an 18-word sentence, so words were probably dropped.

## Verdict

- **Marathi:** good enough to adopt IndicF5 for Marathi (ADR-008).
- **Hindi:** adopted on the strength of the Marathi result. It still needs its own listen, with a Hugging Face token.
- **English:** IndicF5 cannot be used. English stays with F5-TTS (B7).
- This is a first listen, not a measurement. There was one listener, one line per language, and voices that are not the user's. Speed on our own hardware has not been measured at all.

## Licenses

- IndicF5: MIT per the model card, gated on Hugging Face (verify; it is fine-tuned from F5-TTS, whose weights are CC-BY-NC-4.0).
- Google FLEURS: CC-BY-4.0 (verify). The clips were used only as test references and are not stored in this repository.

## Clip cutter test (voice lab, `notebooks/cyv_lab.py`)

Can the voice lab cut one long recording into clips, using line numbers spoken in English?

- **Setup:** a fake Hindi recording, 107 s long, made with espeak-ng: the 9 Hindi lines of batch 001, each after its number in English ("one" ... "nine"), 1 s pause after the number, 2 s after the line. Line 3 was started, cut off, and read again from its number (a retake).
- **Speech recognition:** faster-whisper 1.2.1, model `small`, int8, on a 4-core Linux cloud container (CPU only). The notebook uses `large-v3` on a GPU instead.

| Version | Lines cut correctly (start and end within 0.4 s) | Time |
|---|---|---|
| No prompt | 4 of 9. "five", "seven", "eight" and "nine" were heard as other words ("I'm", "And then", "Thanks for watching...") | 201 s |
| Number prompt, line guard, first 2 s of each region only | **9 of 9**, retake handled | 26 s |

- The fix is a prompt that lists the number words, plus two guards: only the next few numbers count, and only after 1 s of the current line has been spoken.
- **Limits:** synthetic speech, one take, not the user's voice. Clip ends came out about 0.3 s early, so the cutter now keeps 0.3 s after each line. The real check is batch 001.
