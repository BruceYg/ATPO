# Training on the paper's datasets

This page walks through the pipeline of *Controllable Multi-label Video Safety
Detection via Adaptive Tversky Policy Optimization*
([arXiv:2610.02019](https://arxiv.org/abs/2610.02019)) on SafeWatch and
XD-Violence. The configs carry the settings of the paper's runs. Results also
depend on hardware, library versions, and sampling, so expect numbers close to, but
not identical with, the paper's.

## 1. Data

Obtain the datasets from their publishers:

- **SafeWatch-Bench-200K:** training videos and annotations.
- **SafeWatch-Bench:** validation and SafeWatch-Real evaluation videos.
- **XD-Violence:** videos and `train_list.txt` / `test_list.txt`.

Then build the training and validation files:

```bash
SAFEWATCH_200K_ANNOTATIONS=<SafeWatch-Bench-200K>/main_annotation SAFEWATCH_BENCH=<SafeWatch-Bench> \
XD_TRAIN_LIST=<XD-Violence>/train_list.txt XD_TEST_LIST=<XD-Violence>/test_list.txt OUT=paper_data \
  scripts/prepare_paper_data.sh
```

See [data_preparation.md](data_preparation.md#the-papers-datasets) for:

- the steps;
- the output files;
- the properties the files keep from the paper's runs.

## 2. SFT

```bash
atpo train --config configs/sft/safewatch_qwen25vl.yaml --output-dir runs/sft_sw \
  --var DATASET_DIR=paper_data/files/sft --var MEDIA_DIR=<SafeWatch videos> --nproc-per-node 4
atpo merge-lora --config configs/sft/safewatch_qwen25vl.yaml --adapter runs/sft_sw/lora --output-dir runs/sft_sw_merged
```

## 3. GRPO or ATPO

```bash
atpo train --config configs/atpo/safewatch_qwen25vl_atpo_g_o2.yaml --output-dir runs/g_o2 \
  --var TRAIN_FILE=paper_data/files/safewatch/train_oversample_c2.parquet \
  --var VAL_FILE=paper_data/files/safewatch/val.parquet \
  --var INIT_MODEL=runs/sft_sw_merged --var VIDEO_ROOT=<SafeWatch videos>
```

The configs, with their training files, are listed in
[training.md](training.md#configs).

## 4. Export, predict, evaluate

```bash
atpo export --checkpoint runs/g_o2/checkpoints/global_step_N --output-dir exports/g_o2 \
  --preset safewatch-qwen2.5vl-rl
atpo import-data --safewatch-bench <SafeWatch-Bench> --bench-source real --benign-labels empty \
  --taxonomy safewatch --video-prefix test/ --output safewatch_real.jsonl
atpo predict --model exports/g_o2 --input safewatch_real.jsonl --video-root <SafeWatch videos> \
  --backend vllm --output preds.jsonl
atpo evaluate --predictions preds.jsonl --ground-truth safewatch_real.jsonl \
  --taxonomy safewatch --invalid-policy negative --label-source historical
```

| Checkpoint | Preset |
| --- | --- |
| Qwen2.5-VL, SafeWatch | `safewatch-qwen2.5vl-rl` |
| Qwen3-VL, SafeWatch | `safewatch-qwen3vl-rl` |
| XD-Violence | `xdviolence-qwen2.5vl-rl` / `xdviolence-qwen3vl-rl` |
| SFT | the `*-sft` presets |

## Settings of the paper's runs

Release defaults differ from the paper's runs in a few places, each described in
[correctness_fixes.md](correctness_fixes.md). To match the paper:

| Paper behaviour | Use |
| --- | --- |
| validation controller adapting on validation batches (checkpoint selection) | `atpo train --validation historical` |
| unparseable outputs scored as "no category" | `atpo evaluate --invalid-policy negative` |
| parser with missing keys read as false | `atpo evaluate --label-source historical` |
| evaluation prompts, message layouts, and video settings | the bundled presets |
| benign SafeWatch validation videos labelled with their folder category | `--benign-labels folder` (the paper data recipe does this) |

The paper's RL runs used 4 GPUs (Qwen3-VL on XD-Violence: 8). Smaller batch or
resolution settings change the results.
