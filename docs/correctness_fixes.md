# Differences from the paper's code

The release follows the code of the paper's runs by default wherever that affects
results: prompts, message layouts, video settings, decoding, rewards, controller
updates, and data conversion. This page lists every place where the release
behaves differently, why, how to get the original behaviour back where that
matters, and what the change does to results. In the entries below, "paper's code"
means the code used for the paper's runs, and test files are under `tests/`.

## Training

### 1. Validation reward controller frozen by default

- **Paper's code:** in the ATPO runs, validation used its own adaptive controller,
  updated on every validation batch. Its coefficients drifted between evaluations, and the "best"
  checkpoint was chosen on that drifting score.
- **Release:** `validation: frozen` scores validation with the same reward settings
  and `update_controller: false` (α = β = c/2 at every evaluation).
- **Original behaviour:** `atpo train --validation historical`, or
  `validation: historical` in the config.
- **Effect:** validation scores and best-checkpoint selection differ. The training
  reward is unchanged.
- **Tests:** `test_training_configs.py`, `test_controller.py::test_frozen_validation_runtime_never_updates`.

### 2. Controller state saved with checkpoints (EasyR1 patch P1)

- **Paper's code:** the controller was not saved, so a resumed run restarted from
  balanced coefficients.
- **Release:** `reward_state.json` and `val_reward_state.json` are saved in every
  checkpoint and restored bit-exactly on resume ([controller.md](controller.md)).
- **Original behaviour:** `trainer.allow_missing_reward_state=true` resumes from a
  checkpoint without saved state, restarting the controller.
- **Effect:** none for uninterrupted runs. Whether any paper run was resumed is not
  recorded.
- **Tests:** `test_easyr1_patch.py` (CPU only).

### 3. Ground-truth rows of equal length accepted

- **Paper's code:** EasyR1's `collate_fn` stores labels with
  `np.array(..., dtype=object)`. When every label list in a dataloader batch has
  the same length, this produces a 2-D array whose rows are arrays, not lists. The
  paper's reward functions accepted only lists or tuples, so such rows scored
  accuracy 0 and counted as ground truth "no category" in the FP/FN statistics.
- **Release:** array rows are converted to lists.
- **Effect:** none of the validation batches (batch size 16, unshuffled) of the
  paper's validation files has equal-length rows, and for the training files the
  chance of a 64-prompt batch with equal lengths is below 1e-12. The change is not
  expected to alter results.
- **Tests:** `test_controller.py::test_equal_length_ground_truth_arrays_are_accepted`.

### 4. Explicit invalid-output policy

- **Paper's code:** the policy was implicit, and it differs between runs of the
  same mode ([controller.md](controller.md#controller)).
- **Release:** `invalid_prediction_policy` is required. The released configs state
  the policy of the paper's run they follow.
- **Effect:** none for the released configs.
- **Tests:** `test_controller.py`.

### 5. Launch failures are reported

- **Paper's code:**
  - The launch scripts reported success after training, whatever the outcome.
  - EasyR1's `get_abs_path` turns a missing optional path (for example a format
    template or video directory) into `None` with only a printed message.
- **Release:**
  - `atpo train` returns the trainer's exit status.
  - Preflight checks stop before launch if data files, templates, the
    initialization, or the first training videos are missing (`--skip-checks` to
    bypass).
  - Fingerprint differences from the paper's data files are reported as notes.
- **Tests:** `test_training_cli.py`.

### 6. Logging and paths

- **Paper's code:** paths were hard-coded, and logging went to W&B.
- **Release:**
  - Every path is a config variable.
  - The default logger is `file`; add W&B with `--logger wandb` and your own
    credentials.
- **Effect:** none on optimisation.

### 7. LoRA merge without config replacement

- **Paper's code:** a merge script replaced the merged model's `config.json` with
  the base model's after `llamafactory-cli export`.
- **Release:** `atpo merge-lora` keeps the exported `config.json`.

## Inference and evaluation

### 8. Every input gets a prediction record

- **Paper's code:** videos that failed to load, or whose prompt exceeded the
  context length, were skipped without a record.
- **Release:** `atpo predict` writes one line per input, with `status` ∈ `ok`,
  `partial`, `parse_error`, `video_error`, `generation_error`. Run metadata counts
  them.
- **Tests:** `test_inference.py`.

### 9. Unparseable outputs are not silently benign

- **Paper's code:** missing category keys were treated as false. A response with
  no parseable decision was scored as predicting no category.
- **Release:**
  - Predictions carry `labels: null` and `status: parse_error`, or
    `status: partial` with `missing_categories`.
  - `atpo evaluate` requires a choice of `--invalid-policy`:
    - `negative` is the paper's convention;
    - `exclude` drops them;
    - `error` stops.
  - Coverage is always reported.
- **Original behaviour:** `--invalid-policy negative --label-source historical`
  reproduces the paper's scoring.
- **Tests:** `test_evaluation.py`.

### 10. Simple-format parser requires the block prefix

- **Paper's code:** the "simple" parser (SFT outputs such as `GUARDRAIL: C1, C3`)
  read any response without a category ID as benign, including malformed ones.
- **Release:** the SFT presets require the `GUARDRAIL:` prefix and report other
  outputs as parse errors.
- **Original behaviour:** `--set parser.require_block_prefix=false`.

### 11. Inference settings travel with the checkpoint

- **Paper's code:** prompts, video settings, and parsers were command-line
  arguments of each evaluation run.
- **Release:**
  - Exported checkpoints carry `atpo_config.json`.
  - Presets reproduce the paper's evaluation settings, byte-exact prompts included.
  - Overrides are recorded.
- **Effect:** none when the matching preset is used.
- **Tests:** `test_inference.py`, `test_model_index.py`.

## Data preparation

### 12. Media filters report what they drop

- **Paper's code:** the duration and frame filters dropped missing or unreadable
  videos silently.
- **Release:**
  - `atpo filter-media` lists every dropped record with its reason.
  - It stops on unreadable videos unless `--on-unreadable drop` is given;
    `scripts/prepare_paper_data.sh` passes `drop`.
- **Effect:** none. On our copy of the SafeWatch videos the release filter selects
  exactly the paper's training videos.
- **Tests:** `test_sources.py`.

### 13. Benign benchmark labels are an explicit choice

- **Paper's code:** benign SafeWatch-Bench videos have no annotated category. The
  conversion gave them their folder's category, so the benign videos among the
  validation videos used for checkpoint selection are labelled harmful.
- **Release:**
  - `atpo import-data --safewatch-bench` requires `--benign-labels folder`
    (the paper's training-time rule) or `empty` (benign videos have no category).
  - The paper data recipe uses `folder`, so it reproduces the paper's validation file.
- **Effect:** none for the paper's setting. New experiments should consider `empty`.
- **Tests:** `test_sources.py`.

### 14. Repeated records and dataset variants are kept and reported

- **Paper's code:**
  - A few SafeWatch training videos appear twice upstream and were kept twice.
  - The C2-oversampled and base training files were used by different runs.
- **Release:** nothing is deduplicated or truncated. Validation reports the
  repeats, and each config records the fingerprint of the paper's training file it
  uses.
