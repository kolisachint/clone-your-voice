# Recording guide

A good recording is the most important part of a good voice clone. Aim for **5 to 10 minutes** of clean speech. Use the sentences below, or read anything you like, as long as it is varied.

## Recording batches (ADR-008)

Most recording now happens in small **batches**. Each batch is a list of lines in [`batches/`](batches/). Every line has a file name, a language, a style and the text.

1. **Record one long file per section** of the reading sheet: English, Hindi, Marathi, and singing. Name each file as its section heading says, for example `batch-001_hi.m4a`. Any audio format is fine. WAV is best, and a phone's voice-memo format (`.m4a`) also works.
   - Before each line, **say its number in English** ("one", "two", ...), even in the Hindi and Marathi files. Then pause for about one second, read the line, and pause for about two seconds.
   - **Made a mistake?** Say the same number again and read the whole line again. The last take of each number is kept, and the spoken numbers are cut out before training.
   - You can still record one file per line instead, named after the line (`hi_conv_003.m4a`).
2. Read the line in the **style** its sheet entry says:
   - `read`: calm and clear, like reading aloud.
   - `conv`: casual, like talking to a friend.
   - `happy`, `sad`, `angry`: act it, as if you mean it. Do not shout into the mic. Step back a little for loud lines.
   - `sing`: sing it. Singing clips teach only the voice model, never the speech models.
3. If a word comes out wrong, record the whole line again. Do not join parts of takes together.
4. Upload the clips to Google Drive, into the folder named for the batch, for example `MyDrive/clone-your-voice/batches/001/`.

English clips (`en_…`) teach F5-TTS. Hindi and Marathi clips (`hi_…`, `mr_…`) teach IndicF5. Every clip teaches the voice model (Applio RVC), which is what makes the result sound like you.

The tips below apply to every clip. The rule "do not shout or sing" does not apply to batch lines whose style asks for it.

## Before you record

- **Find a quiet room.** Close windows and doors. Turn off fans, air conditioners, and the fridge if you can.
- **Avoid echo.** Hard rooms (bathrooms, kitchens with tiles, empty rooms) make an echo. Soft furniture, curtains, a bed, or a closet full of clothes help. A closet is a classic trick.
- **No music or background sound.** No TV, music, traffic, or other voices.
- **Use a close microphone.** Keep it about 15 to 20 cm (6 to 8 inches) from your mouth, slightly to the side, so breath noise is less loud. A headset or a phone's built-in mic held close works fine.
- **Keep the distance the same.** Changing distance makes the voice sound uneven.
- **Turn off effects.** No noise cancellation, echo, reverb, or auto-gain if your app has them.

## While you record

- **Speak naturally.** Read like you are talking to a friend. Do not whisper, shout, or sing.
- **Vary your sentences.** Mix questions, statements, short and long lines, and different emotions.
- **Keep a steady pace.** Pause between sentences.
- **Avoid clipping.** If the level meter goes into the red, move back a little or turn the gain down. A clipped (distorted) peak ruins the recording.
- **Breathe normally.** Do not hide your breath, but try not to gasp.
- **Stop if you get tired.** Your voice changes when you are tired. Take a break.

## File format

- Record **mono** if you can. Stereo is fine, because `cyv dataset prepare` converts it to mono.
- Save as **WAV**. Avoid lossy formats like MP3 if you can.
- Keep the file uncompressed until `cyv dataset prepare` has run.
- Do not edit the audio yourself. `cyv dataset prepare` trims silence, resamples, and normalizes loudness for you.

## Sentences to read

For a full ~10-minute script in four moods (chat, story, questions, calm facts), use the [recording script](recording-script.md) — each part stands alone, so you can record 2, 5, or 10 minutes across sittings.

Or read these aloud, one at a time. You do not need to read them in order, and you can read a few twice in different moods.

1. The kettle started whistling just as the rain began to fall.
2. Could you pass me the blue notebook from the top shelf, please?
3. I never expected the train to be so empty on a Sunday morning.
4. "Are you sure about that?" she asked, smiling over her coffee.
5. We planted tomatoes, basil, and a row of tiny green beans this spring.
6. The museum closes at six, so we should leave the hotel by four-thirty.
7. Honestly, I think the new bridge looks better than the old one ever did.
8. Turn left at the second light, then keep going until you see the red barn.
9. My grandmother kept her recipes in a battered tin box under the stairs.
10. Please remember to bring your umbrella, because the forecast says storms tonight.
11. The quick brown fox jumps over the lazy dog, and then it goes back to sleep.
12. She opened the letter slowly, read it twice, and then laughed out loud.
13. Three hundred and twelve people signed up for the workshop in only one afternoon.
14. "Good morning, everyone. Let's begin with the first item on the agenda."
15. Why do the leaves change color in autumn, and where do they go in winter?
16. He carried the boxes up three flights of stairs without once complaining.
17. The river runs quietly past the old mill, under a row of tall willow trees.
18. I still remember the smell of fresh bread from the bakery on our corner.
19. Wait, did you hear that? It sounded like someone knocking at the back door.
20. Thank you for your patience, and we will get back to you by Friday.

## Check your recording

- Listen to the whole recording with headphones. Listen for echo, hum, clicks, and clipping.
- Redo any part that sounds rough, or that you read in a different voice from the rest.
- Run `cyv dataset prepare` and follow what it reports. It rejects clipping, music, and sets that are too short.
