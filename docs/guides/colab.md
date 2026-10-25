# Training on Google Colab

This guide walks you through training your voice in the cloud. Training needs a GPU, which most Macs do not have, so we use a free Colab GPU.

The notebook is `notebooks/train_colab.ipynb` in this repository. It uses Applio's command-line path. **Do not use the Gradio web UI** of Applio. The notebook is built only for the command-line path.

> You are responsible for following Google Colab's terms. Some of them ban creating deepfakes, and the free tier limits some kinds of use. See [USAGE_POLICY.md](../../USAGE_POLICY.md).

## What you need

- A Google account.
- Your `dataset.zip`, made on your Mac with `cyv dataset prepare`. See [recording.md](recording.md) first.
- About 1 to 2 hours of time, and some Google Drive space for the checkpoints.

## Steps

### 1. Make your dataset

On your Mac, run:

```bash
cyv dataset prepare <your-recordings-folder> -o dataset.zip
```

Read the one-line consent statement it shows you. Type it only if it is true. The statement is saved in your voice pack.

### 2. Open the notebook in Colab

Open `notebooks/train_colab.ipynb` from the repository in Google Colab.

### 3. Choose a GPU

In Colab, open **Runtime**, then **Change runtime type**, and choose a **GPU** (a T4 is the one the project tests on). If Colab does not give you a GPU, it will say so in the notebook. Wait for quota to reset, or try again later.

### 4. Mount Google Drive

Run the Drive cell and allow access to your Google Drive. Checkpoints are saved to Drive, so that a disconnect does not lose your progress.

### 5. Upload your dataset

Upload `dataset.zip` to the location the notebook shows. Keep the file name the notebook expects.

### 6. Run the cells in order

Run the cells from top to bottom. They do four things:

1. Preprocess the audio.
2. Extract features.
3. Train the model.
4. Build the index.

Training can take a while. Keep the Colab tab open, or come back later.

### 7. Resume after a disconnect

If Colab disconnects, open the notebook again, mount Drive, and run the cells again. The notebook picks up from the last checkpoint saved on Drive.

### 8. Check the self-test

At the end, the notebook renders a fixed sentence and shows a loss summary. It prints **PASS** or **FAIL**.

- **PASS:** continue to the export step.
- **FAIL:** do not use this voice pack. Try again with more or cleaner recordings, or a different number of training epochs. See [recording.md](recording.md).

### 9. Export the voice pack

When the self-test passes, the notebook writes `<name>.cyvpack.zip` to your Google Drive.

### 10. Download the pack

Download the file from Google Drive in your browser. v1 does not connect to Drive automatically.

Then, on your Mac:

```bash
cyv voices import ~/Downloads/<name>.cyvpack.zip
```

Only import packs you trust. A voice pack can run code on your Mac if it is malicious. See [SECURITY.md](../../SECURITY.md).

### 11. Delete the files from Drive

Once you have downloaded the pack, delete the working files from Drive to save space and protect your voice data:

1. Delete `dataset.zip`, the checkpoints, and the exported pack from your Drive folder.
2. Open **Trash** in Google Drive and **Empty trash**.

Deleting a file in Drive only moves it to **Trash**. It is still there until you empty the Trash.

## Common problems

| Message | What to do |
|---|---|
| **No GPU** | Change the runtime to GPU and try again. Free GPU time is limited. |
| **Quota exhausted** | You have used your free GPU time for now. Try again after the reset, usually the next day. |
| **Drive auth failed** | Run the Drive cell again and allow access. Check that you are signed in to the right Google account. |
| **Self-test FAIL** | Do not use the pack. Improve the recordings and train again. |
