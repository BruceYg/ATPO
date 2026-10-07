# EasyR1 snapshot provenance

This directory vendors the RL backend used for the GRPO and ATPO experiments. It
is a source snapshot, not a Git submodule, so that the release does not depend on
an unpinned or inaccessible fork.

| Item | Value |
| --- | --- |
| Upstream project | EasyR1, https://github.com/hiyouga/EasyR1 (Apache-2.0, see `LICENSE`) |
| Upstream base commit | `4d13ebe55ad75230305e58a3e9cf48cd58042258` (2025-12-04, `v0.3.2-27-g4d13ebe`) |
| Snapshot | the research fork used for the paper's runs (2026-05-06), 99 commits on top of the base |
| Included paths | `verl/`, `scripts/model_merger.py`, `setup.py`, `pyproject.toml`, `requirements.txt`, `LICENSE`, `README.md` |

The base commit is the parent of the fork's first commit.

## What the fork adds to upstream (summary)

- Video RL pipeline: `video_fps`/`video_nframes`/`video_cache_dir` data settings,
  cached frame decoding (`verl/utils/dataset.py`), video metadata passed to vLLM
  (`verl/workers/rollout/vllm_rollout_spmd.py`) and to the log-prob processor
  (`verl/workers/fsdp_workers.py`), Qwen3-VL support.
- Separate validation reward configuration (`worker.val_reward`), NCCL environment
  passthrough and Ray object-store sizing (`verl/trainer/main.py`).
- Log-probs computed only for response tokens (`logits_indices`; commit `848dfef`).
- `scripts/model_merger.py` writes `model.safetensors` directly.

## Excluded from the snapshot

These fork files are not needed for SFT-initialized GRPO/ATPO training and are
excluded pending a licence review of the baseline code they adapt:

- `verl/trainer/assess_difficulty.py` (difficulty assessment tooling)
- `verl/trainer/guardreasoner_main.py`, `verl/trainer/guardreasoner_ray_trainer.py`
- `verl/trainer/vad_r1_main.py`, `verl/trainer/vad_r1_ray_trainer.py`

The VAD-R1 frame-masking branch inside `vllm_rollout_spmd.py` is part of the shared
rollout path and is kept verbatim; it is inactive unless samples carry
`ignored_intervals`.

The reward-function files of the fork (`examples/reward_function/`) are not vendored
here. The release implements the rewards in `atpo/rewards/` (`atpo/rewards/easyr1.py`
is the EasyR1 entry point); they were checked to give bit-identical rewards.

## Release patches

Changes made by the ATPO release on top of the snapshot are committed separately
and listed in [`PATCHES.md`](PATCHES.md). Each patch states whether it changes
training behaviour.
