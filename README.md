# ATPO: Adaptive Tversky Policy Optimization

Code for *Controllable Multi-label Video Safety Detection via Adaptive Tversky
Policy Optimization* ([arXiv:2610.02019](https://arxiv.org/abs/2610.02019)).

ATPO trains a video-language model (Qwen2.5-VL or Qwen3-VL) to decide which safety
categories a video violates. Its reward is a Tversky index whose weights on false
positives and false negatives a controller adapts during training, steering the
ratio of misses to false alarms towards a target. ATPO-G uses one global target;
ATPO-C sets a target per category.

> **Pre-release.**
>
> - No checkpoint has been published yet. The model names below need a local
>   checkpoint until then.
> - Inference has been tested on a GPU. Training has been tested on CPU
>   (configurations, controller, checkpointing); GPU training runs are pending.

## Install

```bash
git clone <repository> && cd ATPO
python -m venv .venv && . .venv/bin/activate
pip install -e ".[inference]"            # run models (Transformers backend)
pip install -e ".[inference,vllm]"       # plus the vLLM backend
pip install -e ".[data]"                 # build training files
```

Training needs LLaMA-Factory (SFT) or the vendored EasyR1 (RL); see
[environments/README.md](environments/README.md).

## 1. Run a model on your videos

```python
from atpo import VideoSafetyModel

model = VideoSafetyModel.from_pretrained("path/to/checkpoint")   # or a Hub ID with revision=...
prediction = model.predict("clip.mp4")
print(prediction.status, prediction.labels, prediction.label_names)
```

```bash
atpo predict --model path/to/checkpoint --video clip.mp4
atpo predict --model path/to/checkpoint --backend vllm --input videos.jsonl --output preds.jsonl
```

- **Output:** each prediction holds the category decisions, the label IDs and
  names, and the raw response. It also has an explicit `status`. A video that
  fails to load or an answer that cannot be parsed is reported as such, never as
  "safe".
- **Settings:** checkpoints exported by `atpo export` carry their prompt, video,
  and decoding settings.
- **Named checkpoints:** `atpo models` lists the planned released checkpoints.
  Point a name at a local copy with
  `ATPO_MODEL_<NAME>=/path` (for example `ATPO_MODEL_SAFEWATCH_QWEN2_5VL_ATPO_G`).

See [docs/inference.md](docs/inference.md).

## 2. Train ATPO on your own categories

```bash
# a synthetic stand-in dataset; replace with your own canonical records
python examples/custom_training/make_synthetic_dataset.py --out atpo_example
atpo validate-data --taxonomy examples/custom_training/taxonomy.yaml \
  --split train=atpo_example/train.jsonl --split val=atpo_example/val.jsonl \
  --video-root atpo_example/videos --check-media decode
atpo prepare examples/custom_training/data_recipe.yaml --var DATA_DIR=atpo_example
atpo train --config examples/custom_training/atpo_g.yaml --output-dir runs/atpo_g \
  --var TRAIN_FILE=atpo_example/easyr1/train.parquet --var VAL_FILE=atpo_example/easyr1/val.parquet \
  --var VIDEO_ROOT=atpo_example/videos --var INIT_MODEL=<SFT-initialized model> --dry-run
atpo export --checkpoint runs/atpo_g/checkpoints/global_step_N --output-dir exports/atpo_g --training-run runs/atpo_g
```

Categories, their definitions, and the answer format come from a taxonomy file.
The prompt and SFT targets are generated from it, and the controller works for
any number of categories. Remove `--dry-run` to train. See
[docs/custom_training.md](docs/custom_training.md) and
[docs/controller.md](docs/controller.md).

## 3. Train on the paper's datasets

```bash
scripts/prepare_paper_data.sh            # SafeWatch / XD-Violence training files from the datasets' annotations
atpo train --config configs/sft/safewatch_qwen25vl.yaml --output-dir runs/sft_sw --var ...
atpo train --config configs/atpo/safewatch_qwen25vl_atpo_g_o2.yaml --output-dir runs/g_o2 --var ...
```

The configs in `configs/` carry the settings of the paper's SFT, GRPO, ATPO-G, and
ATPO-C runs. See [docs/paper_setup.md](docs/paper_setup.md).

## Repository

| Path | Content |
| --- | --- |
| `atpo/` | the package: taxonomy and parsing, controllers and rewards, data, inference, evaluation, training launchers, export |
| `configs/` | SFT, GRPO, ATPO, and data-recipe configurations; split manifests of the paper's data |
| `examples/` | inference and custom-training examples |
| `models/` | model index and model-card template (no weights) |
| `prompts/` | where the bundled prompts live and where they come from |
| `scripts/` | data preparation for the paper's datasets |
| `third_party/EasyR1/` | vendored EasyR1 with documented patches |
| `docs/` | documentation ([index](docs/README.md)) |

## Citation

```bibtex
@article{yang2026atpo,
  title   = {Controllable Multi-label Video Safety Detection via Adaptive Tversky Policy Optimization},
  author  = {Yang, Guangyu and Mei, Jingbiao and Sun, Mingsheng and Chen, Jinghong and Bu, Yingtong and
             Qin, Pengda and Chen, Da and Byrne, Bill},
  journal = {arXiv preprint arXiv:2610.02019},
  year    = {2026}
}
```

## Licence

MIT ([LICENSE](LICENSE)). Third-party components keep their own licences; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The datasets are not
redistributed.
