# Training

The paper's pipeline has five steps:

1. LoRA SFT with LLaMA-Factory;
2. merge the adapter;
3. GRPO or ATPO with the vendored EasyR1 (`third_party/EasyR1`);
4. export a checkpoint;
5. evaluate it.

`atpo train` launches steps 1 and 3 from a release config. It composes the
backend's own configuration file, checks paths, and records what it ran. The
backends run inside the run directory, so relative paths given with `--var` are
resolved against the directory you launch from when the configuration is composed.

> Status: training configurations, controller, and checkpoint resumption are tested
> on CPU. End-to-end GPU training runs with this release are pending.

## Environments

- **SFT:** needs LLaMA-Factory (`llamafactory-cli` on `PATH`); `environments/sft.txt`.
- **RL:** needs the vendored EasyR1's requirements; `environments/rl-paper.txt` pins the
  paper's PyTorch 2.8.0, vLLM 0.11.0, Transformers 4.57.3 (CUDA 12.8).

See [environments/README.md](../environments/README.md).

## Configs

```
configs/easyr1/base.yaml        EasyR1 settings shared by the paper's RL runs
configs/atpo/*.yaml             ATPO-G / ATPO-C settings of the paper
configs/grpo/*.yaml             GRPO baselines
configs/sft/*.yaml              LLaMA-Factory LoRA SFT + merge settings
examples/custom_training/       configs for your own taxonomy
```

A config has:

- `kind` (`easyr1` or `llamafactory`);
- `variables` for paths, supplied with `--var NAME=VALUE`;
- the backend tree (`easyr1:` or `llamafactory:`);
- for RL, the `reward` (see [controller.md](controller.md)) and `validation`;
- optionally, `paper_data`: the name and fingerprint of the paper's training and
  validation files ([data_preparation.md](data_preparation.md#the-papers-datasets)).

`extends:` merges a parent file.

| Config | Training file | Reward | GPUs |
| --- | --- | --- | --- |
| `grpo/safewatch_qwen25vl_grpo_str` | `safewatch/train` | static Tversky α=β=1, full response | 4 |
| `grpo/safewatch_qwen25vl_grpo_em`, `_jaccard` | `safewatch/train` | exact match / Jaccard | 4 |
| `atpo/safewatch_qwen25vl_atpo_g_c1` | `safewatch/train` | ATPO-G, c=1 | 4 |
| `atpo/safewatch_qwen25vl_atpo_g_o2` | `safewatch/train_oversample_c2` | ATPO-G, c=2, r\*=1 | 4 |
| `atpo/safewatch_qwen25vl_atpo_g_r5` | `safewatch/train` | ATPO-G, r\*=5 | 4 |
| `atpo/safewatch_qwen25vl_atpo_g_r0.2` | `safewatch/train` | ATPO-G, r\*=0.2, η=0.05 | 4 |
| `atpo/safewatch_qwen25vl_atpo_c_o2` | `safewatch/train_oversample_c2` | ATPO-C, r\*=1 | 4 |
| `atpo/safewatch_qwen25vl_atpo_c_o2_c3r0.2` | `safewatch/train_oversample_c2` | ATPO-C, C3 r\*=0.2 | 4 |
| `atpo/safewatch_qwen25vl_atpo_c_o2_c4r5_c6r0.2` | `safewatch/train_oversample_c2` | ATPO-C, C4 r\*=5, C6 r\*=0.2 | 4 |
| `atpo/safewatch_qwen3vl_atpo_g`, `_c_o2` | `safewatch/train`, `safewatch/train_oversample_c2` | ATPO-G / ATPO-C | 4 |
| `atpo/xdviolence_qwen25vl_atpo_g`, `_c` | `xdviolence/train` | ATPO-G / ATPO-C | 4 |
| `atpo/xdviolence_qwen3vl_atpo_g`, `_c` | `xdviolence/train` | ATPO-G / ATPO-C | 8 |

Every config validates on `safewatch/val` or `xdviolence/val`.

## SFT (LLaMA-Factory)

```bash
scripts/prepare_paper_data.sh                     # or: atpo prepare <your recipe>
atpo train --config configs/sft/safewatch_qwen25vl.yaml --output-dir runs/sft_sw \
  --var DATASET_DIR=paper_data/files/sft --var MEDIA_DIR=<SafeWatch videos> --nproc-per-node 4
atpo merge-lora --config configs/sft/safewatch_qwen25vl.yaml \
  --adapter runs/sft_sw/lora --output-dir runs/sft_sw_merged
```

`MEDIA_DIR` is the directory relative video paths are resolved against; it
defaults to `DATASET_DIR`.

`configs/sft/base.yaml` trains one epoch of LoRA (rank 64, alpha 128, all linear
layers) with learning rate 1e-5 (cosine, 10% warm-up), batch 1 per device with
gradient accumulation 8, cutoff length 8192, a frozen vision tower, and video at
1 fps.

## GRPO / ATPO (EasyR1)

```bash
atpo train --config configs/atpo/safewatch_qwen25vl_atpo_g_o2.yaml --output-dir runs/g_o2 \
  --var TRAIN_FILE=paper_data/files/safewatch/train_oversample_c2.parquet \
  --var VAL_FILE=paper_data/files/safewatch/val.parquet \
  --var INIT_MODEL=runs/sft_sw_merged --var VIDEO_ROOT=<SafeWatch videos>
```

EasyR1 joins the stored video paths (`train/...`, `test/...`) to `VIDEO_ROOT`.

| Option | Effect |
| --- | --- |
| `--dry-run` | write `easyr1_config.yaml` and `atpo_run.json`, print the command, do not train |
| `--set trainer.n_gpus_per_node=8` | override any EasyR1 setting (recorded) |
| `--validation historical` | the paper runs' validation reward (adapting controller); default `frozen` |
| `--resume` | continue from the last checkpoint in `--output-dir` (controller state included) |
| `--logger file --logger wandb` | loggers; W&B needs your own credentials in the environment |
| `--skip-checks` | skip preflight |

Preflight stops before launch if:

- the data files, format template, or a local initialization are missing;
- the reward settings are invalid;
- any of the first 20 training videos cannot be found.

It notes when the training or validation file differs from the paper's file
(`paper_data` fingerprints).

Run directory layout:

```
runs/g_o2/
  easyr1_config.yaml        full EasyR1 configuration actually run
  atpo_run.json             config path, variables, git commit, command
  checkpoints/global_step_N/{actor/, reward_state.json, val_reward_state.json}
```

`atpo inspect-state runs/g_o2/checkpoints/global_step_N` prints the controller
coefficients and statistics saved at that step.

## Export

```bash
# paper-style evaluation settings
atpo export --checkpoint runs/g_o2/checkpoints/global_step_N --output-dir exports/g_o2 \
  --preset safewatch-qwen2.5vl-rl
# or: the prompt and layout used in training
atpo export --checkpoint ... --output-dir ... --training-run runs/g_o2
# SFT model merged with merge-lora
atpo export --hf-model runs/sft_sw_merged --output-dir exports/sft_sw --preset safewatch-qwen2.5vl-sft
```

`atpo export` does the following:

- merges FSDP shards with the vendored `scripts/model_merger.py`, without its
  upload option;
- copies weights, configuration, and processor files;
- writes `atpo_config.json` (inference settings) and `atpo_export.json` (source
  checkpoint, training run record, controller state at the step, library versions);
- writes `checksums.sha256`.

The result loads with `atpo predict --model exports/g_o2`. Nothing is uploaded;
publishing is a separate, manual step.

## Hardware

The paper's RL runs used 4 GPUs (Qwen3-VL XD-Violence: 8), with rollout batch
64 × 8 responses, vLLM rollouts with tensor parallel size 1 at 60% GPU memory, and
FSDP. On smaller machines, reduce `data.max_pixels`,
`worker.rollout.gpu_memory_utilization`, or the batch sizes with `--set`; results
will then differ from the paper's setting.
