# Prompts

The prompts and answer-format templates ship inside the package, so inference works
after `pip install` without this repository:
[`atpo/resources/prompts/`](../atpo/resources/prompts/). SHA-256 values are in
`CHECKSUMS.sha256` there, and `tests/test_prompts.py` checks them. The files are the
exact texts of the paper's runs.

| Bundled file | Used for |
| --- | --- |
| `safewatch/v3.txt` | SafeWatch RL training and validation files |
| `safewatch/v5.txt` | SafeWatch SFT training file; presets `safewatch-*-sft` |
| `safewatch/v15.txt` | an alternative SafeWatch SFT evaluation prompt |
| `safewatch/v3_qwen3.txt` | evaluation of Qwen3-VL SafeWatch RL checkpoints (preset `safewatch-qwen3vl-rl`) |
| `safewatch/inference_default_rl.txt` | evaluation of Qwen2.5-VL SafeWatch RL checkpoints (preset `safewatch-qwen2.5vl-rl`) |
| `xdviolence/v1.txt` | XD-Violence RL and SFT training files; presets `xdviolence-*-sft` |
| `xdviolence/v1reason.txt` | evaluation of Qwen2.5-VL XD-Violence RL checkpoints (preset `xdviolence-qwen2.5vl-rl`) |
| `xdviolence/v1_qwen3.txt` | evaluation of Qwen3-VL XD-Violence RL checkpoints (preset `xdviolence-qwen3vl-rl`) |
| `xdviolence/inference_default_rl.txt` | an alternative XD-Violence RL evaluation prompt |
| `formats/safewatch.jinja` | answer format appended during Qwen2.5-VL RL training, both datasets (`think_answer`) |
| `formats/qwen3_2.jinja` | answer format appended during Qwen3-VL RL training, both datasets (`triple_newline`) |
| `formats/safewatch_qwen3.jinja` | an alternative Qwen3-VL answer format (not used by the configs) |

Which prompt each inference preset uses is stated in `atpo/resources/presets/*.yaml`
and in [docs/inference.md](../docs/inference.md). Custom taxonomies render their
prompt from the taxonomy file (`prompt: {render: true}`).
