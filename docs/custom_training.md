# Training ATPO on your own categories

The released checkpoints learned the SafeWatch or XD-Violence categories. Their
category definitions belong to that training. A new taxonomy needs its own
training data and its own validation; editing the prompt of a released model is
not enough. This guide follows `examples/custom_training/`, which you can run end
to end on synthetic clips to check your setup.

## 1. Define the taxonomy

```yaml
# examples/custom_training/taxonomy.yaml
name: workplace-safety
block_name: RESULT                 # the model answers 'RESULT: {"W1(No helmet)": true, ...}'
description: >-                    # first paragraph of the task prompt
  You are a video safety classifier for workplace footage. ...
categories:
  - {id: W1, name: No helmet, description: A person in a marked construction area without a hard hat.}
  - {id: W2, name: Fire or smoke, description: Visible flames or smoke that is not part of a controlled process.}
  - {id: W3, name: Fall, description: A person or object falling from height, or a collapsing structure.}
```

IDs are what your data and the outputs use. The output key of each category is
`key_template` (default `{id}({name})`). The taxonomy fingerprint covers the label
space: name, block name, key template, and category IDs, names, and aliases.
Descriptions are excluded, because they change only the prompt text. Saved
controller states and exported inference configs record the fingerprint, and
loading them with a different label space fails.

## 2. Prepare data

Write canonical records (`{"id", "video", "labels"}` per line; `[]` for benign),
then validate and build the files:

```bash
python examples/custom_training/make_synthetic_dataset.py --out atpo_example   # synthetic stand-in
atpo validate-data --taxonomy examples/custom_training/taxonomy.yaml \
  --split train=atpo_example/train.jsonl --split val=atpo_example/val.jsonl \
  --video-root atpo_example/videos --check-media decode
atpo prepare examples/custom_training/data_recipe.yaml --var DATA_DIR=atpo_example
```

The recipe renders the task prompt from the taxonomy (`prompt: {render: true}`):

- the category list;
- each definition;
- the output format with every key;
- the rule to answer false for absent categories.

`atpo show-config --training-run` prints the exact text after a training run is
set up. To use your own wording, pass `prompt: {path: my_prompt.txt}`. It must ask
for the same keys.

Keep validation and test videos separate from training videos;
`split_overlap: error` checks IDs and video paths.

## 3. Initialization

ATPO improves a policy that already answers in the expected format; the reward is
0 for unparseable outputs. Start from either:

- **SFT warm-up** (recommended):
  `atpo train --config examples/custom_training/sft.yaml --output-dir runs/sft --var DATASET_DIR=atpo_example/sft --var MEDIA_DIR=atpo_example/videos --nproc-per-node <gpus>`,
  then `atpo merge-lora`. The SFT targets are the JSON answer (`target_style: json`).
- **A base model** that follows the rendered prompt reasonably often. Check a few
  outputs with `atpo predict` first.

## 4. ATPO-G or ATPO-C

```yaml
# examples/custom_training/atpo_g.yaml (abridged)
extends: ../../configs/easyr1/base.yaml
reward:
  taxonomy: taxonomy.yaml          # relative to this file; inlined into the run config
  reward: atpo
  response_format: think_answer    # Qwen2.5-VL recipes; Qwen3-VL recipes used triple_newline
  parse_scope: answer_only
  controller:
    mode: global
    target_ratio: 1.0
    scale: 2.0                     # alpha + beta = 2: starts at the Jaccard index
    invalid_prediction_policy: as_empty
validation: frozen
```

```yaml
# examples/custom_training/atpo_c.yaml
extends: atpo_g.yaml
reward:
  controller:
    mode: category
    target_ratio: {W1: 1.0, W2: 0.5, W3: 1.0}   # every category must be listed
    scale: 2.0
    invalid_prediction_policy: exclude
```

```bash
atpo train --config examples/custom_training/atpo_g.yaml --output-dir runs/atpo_g \
  --var TRAIN_FILE=atpo_example/easyr1/train.parquet --var VAL_FILE=atpo_example/easyr1/val.parquet \
  --var VIDEO_ROOT=atpo_example/videos --var INIT_MODEL=runs/sft_merged \
  --set trainer.n_gpus_per_node=<gpus>
```

**Choosing targets.** The controller steers the ratio of false negatives to false
positives towards `target_ratio` (r\* = FN/FP):

- r\* > 1 tolerates more misses and favours precision;
- r\* < 1 favours recall;
- ATPO-C sets a target per category, for example recall-leaning for a
  safety-critical category.

Watch the logged `ctrl_ema_ratio_*` and `coef_beta_*` during training, and the
per-category precision and recall from `atpo evaluate` afterwards. See
[controller.md](controller.md) for the update rule and every setting.

**Invalid outputs.** `as_empty` treats an unparseable answer as "no category" in
the controller statistics (pushing towards recall). `exclude` ignores it. Pick one
deliberately; the paper's configs use both.

## 5. Export, predict, evaluate

```bash
atpo export --checkpoint runs/atpo_g/checkpoints/global_step_N --output-dir exports/atpo_g \
  --training-run runs/atpo_g             # inference prompt and layout exactly as in training
atpo predict --model exports/atpo_g --input atpo_example/test.jsonl \
  --video-root atpo_example/videos --output preds.jsonl
atpo evaluate --predictions preds.jsonl --ground-truth atpo_example/test.jsonl \
  --taxonomy examples/custom_training/taxonomy.yaml --invalid-policy exclude
```

## What has been verified

On 2026-10-06 the example ran through these steps:

- synthetic data generation, decode validation, and file preparation;
- `atpo train --dry-run` for the SFT, ATPO-G, and ATPO-C configs;
- GPU inference and evaluation with an **untrained** base model, using the
  configuration derived from the ATPO-G run (environment:
  `environments/verified-2026-10-06.txt`).

Actual SFT or RL training on a custom taxonomy has not been run. The synthetic
clips only check the plumbing; they cannot show that ATPO learns anything.
