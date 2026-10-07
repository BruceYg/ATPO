# Inference examples

Single video (Python API): `predict_one.py`.

Batch over a JSONL file with `{"id", "video"}` records (labels are ignored), resumable:

```bash
atpo predict --model <hub-id-or-dir> --backend vllm \
  --input data/test.jsonl --video-root data/videos --output preds.jsonl
# continue an interrupted run
atpo predict ... --output preds.jsonl --resume
```

Score the predictions (choose how invalid outputs count; see docs/evaluation.md):

```bash
atpo evaluate --predictions preds.jsonl --ground-truth data/test.jsonl \
  --taxonomy safewatch --invalid-policy exclude
```

Checkpoints without `atpo_config.json` need a preset (`atpo presets` lists them) or an
explicit `--config`. Smaller GPUs: lower the video resolution with
`--set video.max_pixels=50176`, or cap the context with `--max-model-len` (vLLM).
