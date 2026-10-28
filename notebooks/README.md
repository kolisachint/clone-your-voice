# notebooks/

| File | What it is |
|---|---|
| `train_colab.ipynb` | The Colab training notebook. Run top to bottom on a free T4 GPU. Uses Applio's command-line path only (no web UI, no tunnels). |
| `cyv_colab.py` | Helper functions the notebook imports (dataset validation, `voice.json`, pack writing, Drive resume). Standard library only, so they are unit-tested in `tests/test_notebook.py`. |
| `constraints.txt` | Version pins for the packages most likely to break when Applio's requirements are installed. |

## Using it

1. On your Mac: `cyv dataset prepare <dir>` produces `dataset.zip`.
2. Open `train_colab.ipynb` in Google Colab (File > Open notebook > GitHub), choose **Runtime > Change runtime type > T4 GPU**, and run the cells in order.
3. Download `<name>.cyvpack.zip` from `MyDrive/clone-your-voice/<name>/output/` and run `cyv voices import <file>`.
4. Delete the dataset from Drive and **empty the Drive Trash** (step 9 of the notebook explains why).

If Colab disconnects during training, reconnect and run all cells again with the same voice name; it resumes from the checkpoints on your Drive.

## Notes for maintainers

* `APPLIO_COMMIT` in the install cell must equal `src/cyvd/timbre/applio/pin.py` (a test checks this). When you bump the pin, re-check every `core.py` flag used in the notebook and regenerate `constraints.txt` with
  `uv pip compile requirements-engine.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28 --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match`.
* The install cell downloads `cyv_colab.py` and `constraints.txt` from the `main` branch on GitHub (`CYV_RAW`). Point it at a release tag for stable releases. The cell checks both files against sha256 values embedded in the notebook and refuses to run on a mismatch, so after editing either file, update `HELPER_SHA256` in the install cell (`tests/test_notebook.py` fails until you do).
* Applio's `core.py` exits 0 even when preprocess/extract/train fail, so the notebook checks for the output files instead of trusting exit codes.
* Do not commit the notebook with outputs; it contains your Drive paths.
