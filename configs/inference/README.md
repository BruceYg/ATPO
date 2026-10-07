# Inference configurations

The bundled inference presets live in the package so that `pip install` is
enough for inference: [`atpo/resources/presets/`](../../atpo/resources/presets/).
`atpo presets` lists them; [docs/inference.md](../../docs/inference.md) explains
each field and the order in which settings are resolved.

To make your own, start from a preset or a checkpoint, then edit the JSON:

```bash
atpo show-config --preset xdviolence-qwen2.5vl-rl --set video.max_pixels=50176 --output my_inference.json
atpo predict --model <checkpoint> --config my_inference.json --video clip.mp4
atpo export --checkpoint <step dir> --output-dir <dir> --inference-config my_inference.json
```

Exported checkpoints carry their configuration in `atpo_config.json`, so a
config file is only needed to override it.
