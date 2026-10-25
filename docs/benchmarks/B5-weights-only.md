# B5: `torch.load(weights_only=True)` on real Applio / RVC v2 checkpoints

## Question
Does `torch.load(..., weights_only=True)` load real-world RVC v2 / Applio `.pth` voice models, so the DESIGN §6.1 sanitize plan (load in a subprocess with `weights_only=True`, re-save only tensors + plain config) works without an `add_safe_globals` allowlist?

## Method
Script: `scripts/bench/b5_weights_only.py`. For each model it (1) statically scans the pickle with `pickletools` (no execution) for imported globals, (2) runs `torch.load(weights_only=True)`, (3) lists top-level keys/types and recursively counts non-tensor/non-primitive objects, (4) does the sanitize round trip (re-save tensors + primitives, reload with `weights_only=True`, check bit-identical). Zip members are read in memory (no `extractall`). Also grepped Applio's `torch.load` call sites at the pinned commit.

## Environment
Linux x86 CPU box, Python 3.12, torch 2.14.1+cpu. Applio checkout `c817f1f83f689ef72e0dfeb1c17ae990b9248506`. Models downloaded 2026-10-25 from Hugging Face (6 models, 6 uploaders/repos).

## Results: checkpoints

| Model (HF repo) | Produced by | sr / f0 | `weights_only=True` | Pickle globals | Non-tensor/non-primitive | Sanitize round trip |
|---|---|---|---|---|---|---|
| Rootreck/Applio_3.2.0-RVC_v2-The_Witcher_3_Wild_Hunt (Cerys) | Applio 3.2.0 | "40000" str / 1 | OK | OrderedDict, HalfStorage, _rebuild_tensor_v2 | none | identical |
| Rootreck/Applio_3.2.9-RVC-Other (Marjorie Hughes) | Applio 3.2.9 (newer schema) | 40000 int / True | OK | same | none | identical |
| sorton/sorton_RVC_models (Caboose) | original RVC WebUI layout (`weights/`, `logs/`) | "40k" / 1 | OK | same | none | identical |
| thebuddyadrian/RVC_Models (LeahKazuno2) | RVC WebUI | "40k" / 1 | OK | same | none | identical |
| Mattfleu/Tamamo_RVC_v2_model | RVC WebUI | "40k" / 1 | OK | same | none | identical |
| rikineko1/RVCv2_Models (kaede_aga) | RVC WebUI | "40k" / 1 | OK | same | none | identical |
| SyntheticVoicess/StevenPacey_XTTS_RVC_models (stevenpaceyv2) | RVC WebUI-style | "40k" / 1 | OK | same | none | identical |

(7 files tested; "Produced by" is inferred from the key schema and repo naming, not stated by uploaders for the non-Applio ones.)

Top-level keys:
- Classic RVC WebUI: `weight` (dict, 457 fp16 tensors), `config` (list of 18 ints/str/nested lists), `info` (str), `sr` (str), `f0` (int), `version` ("v2").
- Applio 3.2.0: adds `epoch`, `step`, `creation_date`, `model_hash`.
- Applio 3.2.9: further adds `overtrain_info`, `dataset_length`, `model_name`, `author`, `embedder_model`, `speakers_id`, `vocoder`; **`sr` is an int and `f0` a bool** (types differ from classic).

## Results: how Applio (pin) loads checkpoints
All inference and training load paths pass `weights_only=True`: `rvc/infer/infer.py:466`, `rvc/realtime/pipeline.py:71`, `rvc/train/utils.py:48`, `rvc/train/train.py:430/528/552`, `rvc/train/process/{model_blender.py:21-22, change_info.py:8}`, `rvc/lib/predictors/{RMVPE.py:433, FCPE.py:731}`, `tabs/inference/inference.py:514`, `tabs/realtime/realtime.py:256`.
**Exceptions (fallback to unsafe):** `rvc/train/process/model_information.py:_load_checkpoint` (lines ~40-60) and `rvc/lib/tools/inspect_checkpoint.py` (~25-50) try `weights_only=True`, then on any exception fall back to plain `torch.load` / `weights_only=False` / a custom `_TolerantPickle`. These are info/inspect tools; cyv must not call them. Applio's own `weights_only=True` default on the inference path is consistent with our finding that its models need no extra globals.

## Conclusion
The §6.1 plan works as written. All 7 real checkpoints, from both Applio and original RVC WebUI, load with plain `weights_only=True` and **need no `add_safe_globals`**; the pickles reference only `OrderedDict`, `torch.HalfStorage` and `_rebuild_tensor_v2`. The sanitize re-save (tensors + primitives) round-trips bit-identically and is ~9 KB smaller. Refuse on any load failure; do not add a fallback.

Design implications:
- Sanitizer must be schema-tolerant: `sr` may be str ("40k", "40000") or int, `f0` int or bool; normalize in `voice.json`, not in the checkpoint.
- Weights are fp16 (HalfStorage); don't assume fp32.
- Sanitizer should whitelist value types (Tensor, int/float/str/bool/None, list/tuple/dict) as in the script, and reject anything else.
- Keep the CI grep for `torch.load` without `weights_only=True`; additionally ban importing Applio's `model_information`/`inspect_checkpoint`.

## Caveats
- Only models from 2023-2025 era tooling, all v2 / 40k; v1, 32k/48k and exotic forks (e.g. Mangio, RVC-Boss variants with custom metadata) untested. A checkpoint with non-primitive extras (e.g. numpy scalars/arrays in `info`) would fail by design; that is the desired refusal, but expect some user-visible rejections in the wild.
- `weights_only=True` blocks code execution via pickle globals but is not a defense against malformed tensors / resource exhaustion; keep the subprocess, rlimits, and size caps.
- Tested with torch 2.14 only; behaviour of `weights_only` is stable since torch 2.6 default flip, but pin in CI.
- Models are untrusted downloads; they were never loaded without `weights_only=True`.
