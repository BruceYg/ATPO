# Third-party notices

This project is released under the MIT licence (`LICENSE`). The components below
keep their own licences.

## Included in this repository

| Component | Location | Licence | Notes |
| --- | --- | --- | --- |
| EasyR1 (snapshot of a research fork) | `third_party/EasyR1/` | Apache-2.0 (`third_party/EasyR1/LICENSE`) | EasyR1 is a fork of veRL (Apache-2.0, Copyright Bytedance Ltd. and/or its affiliates); file headers are kept. Provenance: `third_party/EasyR1/PROVENANCE.md`; release modifications: `third_party/EasyR1/PATCHES.md`. |

## Used, not included

| Component | Use | Licence (see upstream) |
| --- | --- | --- |
| LLaMA-Factory | SFT (`llamafactory-cli`) | Apache-2.0 |
| qwen-vl-utils | video frame sampling | Apache-2.0 |
| Hugging Face Transformers, huggingface_hub, Accelerate | inference, model loading | Apache-2.0 |
| vLLM | inference and RL rollouts | Apache-2.0 |
| PyTorch | everything | BSD-style |
| decord, PyAV | video decoding | Apache-2.0 / BSD |
| NumPy, pandas, PyArrow, PyYAML | data handling | BSD / Apache-2.0 / MIT |

Model checkpoints derived from Qwen2.5-VL-7B-Instruct or Qwen3-VL-8B-Instruct are
also subject to the licence of their base model; see the base models' model cards.

## Datasets

SafeWatch-Bench, SafeWatch-Bench-200K, and XD-Violence are not redistributed.
Obtain them from their publishers under their terms. `configs/data/manifests/`
lists the IDs of the videos selected from these datasets for the paper's training
and validation files; it contains no videos or annotations.
