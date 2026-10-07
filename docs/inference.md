# Inference

`atpo predict` and `atpo.VideoSafetyModel` classify videos with any checkpoint of
this project. Inference needs only the `inference` extra; it does not import the
training stack.

```bash
pip install -e ".[inference]"          # Transformers backend
pip install -e ".[inference,vllm]"     # plus vLLM (install a build matching your CUDA)
```

## Quick start

```python
from atpo import VideoSafetyModel

model = VideoSafetyModel.from_pretrained("path/or/hub-id")   # see "Choosing a model"
prediction = model.predict("clip.mp4")
print(prediction.status, prediction.labels, prediction.unsafe)
```

```bash
atpo predict --model path/or/hub-id --video clip.mp4                 # prints one JSON line
atpo predict --model path/or/hub-id --backend vllm \
  --input videos.jsonl --video-root videos --output preds.jsonl
atpo predict ... --output preds.jsonl --resume                       # continue an interrupted run
```

`--input` is JSON Lines with `video` (and optionally `id`) per line. Canonical data
files with `labels` work as input; the labels are ignored.

## Choosing a model

`--model` / `from_pretrained(model)` accepts:

- a local directory (an `atpo export` output, or any merged checkpoint);
- a Hugging Face repository ID, optionally with `--revision` (branch, tag, or
  commit; pin a commit for reproducibility);
- a name from the model index (`atpo models`).

**No checkpoint has been published yet.** The index names the planned checkpoints
with empty Hub references. Point a name at your own copy with
`ATPO_MODEL_<NAME>=<path or repo id>` (plus `ATPO_MODEL_<NAME>_REVISION`), or with a
YAML file named by `ATPO_MODELS_FILE`. An unconfigured name fails with these
instructions. See `models/index.yaml`.

## Inference settings

Results depend on the exact prompt, video sampling, and parser. These travel as
an inference configuration:

| Field | Meaning |
| --- | --- |
| `taxonomy` | categories, their names and definitions; the output key template (`C1(Sexual Content)`) |
| `prompt`, `system_prompt` | exact text (SHA-256 recorded) |
| `response_format` | `think_answer`, `triple_newline`, or `plain` |
| `video` | `fps`, `min_pixels`, `max_pixels`, `image_patch_size`, and `position`: `after` (text, then video; the paper's evaluation layout), `before`, or `placeholder` (split at `<video>`; the RL training layout) |
| `parser` | `output_style` `json` or `simple`; `scope` `full_response` or `answer_only`; `require_block_prefix` |
| `generation` | greedy by default: `max_new_tokens` 1024, `temperature` 0, `top_k` -1, `repetition_penalty` 1 |
| `processor` | `checkpoint` (files shipped with the model) or a Hub ID/path with revision |

The configuration is resolved in this order:

1. an explicit config (`--config file.json`);
2. a preset (`--preset`);
3. the checkpoint's `atpo_config.json`;
4. the model index's preset for the named model.

When a checkpoint's own config is overridden, a warning is issued and the run
metadata says so. Individual values can be changed with
`--set video.max_pixels=50176` (an unknown key is an error). Inspect the result
before running:

```bash
atpo show-config --model path/or/hub-id            # full resolved config
atpo show-config --preset safewatch-qwen2.5vl-rl --prompt   # exact prompt text
atpo presets                                        # bundled presets
```

### Presets

The presets reproduce the paper's evaluation settings byte for byte (prompt files
are checked against SHA-256 in `atpo/resources/prompts/CHECKSUMS.sha256`).

| Preset | For |
| --- | --- |
| `safewatch-qwen2.5vl-rl` | SafeWatch GRPO/ATPO checkpoints, Qwen2.5-VL-7B |
| `safewatch-qwen3vl-rl` | SafeWatch GRPO/ATPO checkpoints, Qwen3-VL-8B (`v3_qwen3`) |
| `xdviolence-qwen2.5vl-rl`, `xdviolence-qwen3vl-rl` | XD-Violence GRPO/ATPO checkpoints |
| `safewatch-qwen2.5vl-sft`, `safewatch-qwen3vl-sft` | SafeWatch SFT initializations (prompt and output format of the SFT data) |
| `xdviolence-qwen2.5vl-sft`, `xdviolence-qwen3vl-sft` | XD-Violence SFT initializations |

### Evaluation layout and training layout

Evaluation in the paper used a different message layout and a higher resolution
than RL training:

| | RL training | Evaluation (presets) |
| --- | --- | --- |
| System message | none | "You are a helpful video content moderation assistant." (SafeWatch) or "You are a video violence detection assistant." (XD-Violence) |
| User turn | task text, video, format instruction | full prompt text, then the video |
| Pixels per frame | 3,136 to 50,176 | 12,544 to 200,704 |
| Sampling | 1 fps | 1 fps |

Both use `image_patch_size` 14 (the qwen-vl-utils default, also for Qwen3-VL,
whose native patch size is 16) and decord as the frame reader. To query a model
exactly as it was trained instead, derive the configuration from its training run:
`atpo show-config --training-run runs/x --output cfg.json`.

## Output

One JSON object per video (`atpo.prediction/v1`; values below are illustrative):

```json
{
  "id": "clip_0032", "video": "videos/clip_0032.mp4",
  "status": "ok",
  "labels": ["C3"], "label_names": ["Threats, Violence & Harm"],
  "decisions": {"C1": false, "C2": false, "C3": true, "C4": false, "C5": false, "C6": false},
  "unsafe": true, "missing_categories": [],
  "explanation": "…text inside <think>…",
  "raw_response": "…",
  "parse": {"scope": "full_response", "source": "answer", "format_ok": true, "answer_found": true,
            "error": null, "historical_labels": ["C3"]},
  "generation": {"prompt_tokens": 2481, "completion_tokens": 212, "finish_reason": "stop"},
  "error": null
}
```

| `status` | Meaning | `labels` / `unsafe` |
| --- | --- | --- |
| `ok` | every category decided | list / bool |
| `partial` | some categories missing (`missing_categories`) | flagged ones; `unsafe` true if any flagged, else null |
| `parse_error` | no decision could be read (`parse.error`) | null / null |
| `video_error` | the video could not be read | null / null |
| `generation_error` | the backend failed (for example, the prompt exceeds the context length) | null / null |

Nothing is silently counted as benign. `parse.historical_labels` holds what the
paper's parser would have reported (missing keys false), for comparisons with the
paper's results. `--output` also writes `<output>.meta.json`, which records:

- input file hash and counts per status;
- the resolved config and its fingerprint;
- the model and processor source, the resolved Hub revision, or local file hashes;
- backend settings, library versions, GPU, and video reader.

Runs are append-only: `--resume` skips IDs already in the output, and
`--overwrite` starts over.

## Backends

| | `transformers` (default) | `vllm` |
| --- | --- | --- |
| Use | single videos, small batches, CPU/GPU | throughput; the paper's evaluation engine |
| Settings | `--device-map`, `--attn-implementation` | `--max-model-len`, `--tensor-parallel-size`, `--gpu-memory-utilization` |

Both receive the same prepared request: the chat template applied by the
checkpoint's processor, and frames from `qwen_vl_utils.process_vision_info`.
Outputs are not guaranteed to be token-identical across backends. The vLLM
backend uses the paper's engine settings (`enforce_eager`, one video per
prompt, bfloat16), and it retries a failed batch request by request, so one bad
video only affects its own record.

## Environment notes

- torchvision ≥ 0.26 has no `read_video`. Install `decord` (included in the
  `inference` extra on Linux x86-64) or `torchcodec`. The paper's runs used
  decord (`FORCE_QWENVL_VIDEO_READER=decord`).
- vLLM's FlashInfer sampler compiles kernels at first use and needs `ninja` on
  `PATH`; set `VLLM_USE_FLASHINFER_SAMPLER=0` if it is missing.
- The presets' `max_model_len` is 98,304 tokens. Smaller GPUs need a lower
  `--max-model-len`, which makes long videos fail as `generation_error`, or a lower
  `video.max_pixels` (which changes results).
