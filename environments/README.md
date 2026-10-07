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
- The paper's evaluation used vLLM 0.11.0. The RL environment below has exactly that
  version, so `atpo predict --backend vllm` can run there to match the paper's engine.

## 2. RL training (GRPO / ATPO)

The paper's runs used PyTorch 2.8.0, vLLM 0.11.0, Transformers 4.57.3, and CUDA 12.8,
with BF16 on four NVIDIA H800 80GB GPUs (paper, appendix C.2). `rl-paper.txt` pins
these versions together with the requirements of the vendored EasyR1 (Python 3.12, an
NVIDIA driver for CUDA 12.8):

```bash
python -m venv .venv-rl && . .venv-rl/bin/activate
pip install -r environments/rl-paper.txt
# flash-attn 2.8.3, prebuilt for torch 2.8 / CUDA 12 / Python 3.12
ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl"
pip install -e ".[data]"
```

`atpo train` adds `third_party/EasyR1` and the repository to `PYTHONPATH`; EasyR1
itself need not be installed. The vendored EasyR1's `requirements.txt` caps
Transformers at 4.57.0; the paper used 4.57.3, and `rl-paper.txt` follows the paper.
Upstream EasyR1 also documents Docker images (`third_party/EasyR1/README.md`).

## 3. SFT

LLaMA-Factory (Apache-2.0) is called through `llamafactory-cli`. `sft.txt` installs
LLaMA-Factory at commit `591fc9ed` (0.9.4.dev0) with PyTorch 2.8.0:

```bash
python -m venv .venv-sft && . .venv-sft/bin/activate
pip install -r environments/sft.txt
pip install -e ".[data]"
```

This LLaMA-Factory checks for Transformers ≤ 4.57.1 at start-up, so the SFT
environment uses 4.57.1, one patch release below the RL environment. Every key of
the composed SFT and merge configs is an argument at this commit, and the
`qwen2_vl` and `qwen3_vl_nothink` templates exist. To check another LLaMA-Factory
version, point `ATPO_LLAMAFACTORY_SRC` at its checkout and run
`pytest tests/test_training_cli.py`.

## Status

Both files resolve with `pip install --dry-run` (2026-10-07). No training run of the
release has been executed in these environments yet. `verified-2026-10-06.txt`
records the environment of the CPU tests and GPU inference tests.
