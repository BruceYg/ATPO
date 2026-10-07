# Environments

The release has three independent environments, so inference does not pull in the
training stack.

## 1. Inference, data preparation, evaluation

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[inference,data]"     # Transformers backend
pip install -e ".[inference,vllm]"     # optional: vLLM backend (install a build matching your CUDA/torch)
pip install -e ".[dev]"                # tests
```

`verified-2026-10-06.txt` lists the versions with which the tests passed and the
GPU inference tests ran. Notes:

- torchvision ≥ 0.26 removed `read_video`. qwen-vl-utils then needs `decord`
  (installed by the extras on Linux x86-64) or `torchcodec`. Set
  `FORCE_QWENVL_VIDEO_READER=decord` to match the paper's runs.
- vLLM's FlashInfer sampler JIT-compiles on first use and needs `ninja` on `PATH`;
  otherwise set `VLLM_USE_FLASHINFER_SAMPLER=0`.

## 2. RL training (GRPO / ATPO)

Use the vendored EasyR1 (`third_party/EasyR1`) and its requirements:

```bash
pip install -r third_party/EasyR1/requirements.txt
pip install -e ".[data]"
```

`atpo train` adds `third_party/EasyR1` and the repository to `PYTHONPATH`;
EasyR1 itself need not be installed. Upstream EasyR1 also documents a Docker image
(`hiyouga/verl`, see `third_party/EasyR1/README.md`). EasyR1's requirements cap
Transformers at `<=4.57.0`.

## 3. SFT

LLaMA-Factory (Apache-2.0) is called through `llamafactory-cli`; install it
separately. The configs need a version that provides the `qwen2_vl` and
`qwen3_vl_nothink` templates and accepts the video arguments used in
`configs/sft/base.yaml` (`video_max_pixels`, `video_min_pixels`, `video_fps`,
`video_maxlen`, `media_dir`).

Every key of the composed SFT and merge configs is an argument of LLaMA-Factory
commit `591fc9ed` (2025-11-24). To check another version, point
`ATPO_LLAMAFACTORY_SRC` at its checkout and run `pytest tests/test_training_cli.py`.
