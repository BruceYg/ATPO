---
# Hugging Face metadata. Every {{...}} must be filled before upload. The code and weights are
# released under the MIT licence; the base model's licence terms also apply.
license: mit
base_model: "{{base_model}}"            # e.g. Qwen/Qwen2.5-VL-7B-Instruct
library_name: transformers
pipeline_tag: video-text-to-text
tags: [video-safety, content-moderation, multi-label-classification, atpo]
---

# {{model_name}}

{{one_sentence_summary}}: a {{base_model}} model trained with {{method}} (SFT
initialization followed by {{GRPO | ATPO-G | ATPO-C}}) to flag which of the
{{taxonomy_name}} categories a video contains.

Paper: *Controllable Multi-label Video Safety Detection via Adaptive Tversky Policy
Optimization* ([arXiv:2610.02019](https://arxiv.org/abs/2610.02019)).
Code: {{repository_url}} at commit {{code_commit}}.

## Usage

```python
from atpo import VideoSafetyModel

model = VideoSafetyModel.from_pretrained("{{hub_id}}", revision="{{revision}}")
prediction = model.predict("video.mp4")
print(prediction.status, prediction.labels, prediction.unsafe)
```

`atpo_config.json` in this repository holds the exact prompt, video sampling
({{fps}} fps, {{min_pixels}}-{{max_pixels}} pixels per frame), decoding settings, and output
parser used for the evaluation below. Changing them changes the results.

## Categories

{{category_table}}  <!-- id | name | definition, from atpo_config.json -->

## Training

| | |
|---|---|
| Training config | `{{training_config}}` |
| Training data | {{training_data}} (fingerprint `{{train_fingerprint}}`) |
| Controller | {{controller_summary}} <!-- mode, target ratio(s), scale c, invalid-output policy --> |
| Selected step | {{global_step}} ({{step_selection}}) |
| Controller state at this step | see `atpo_export.json` |

{{training_notes}}  <!-- e.g. frozen or adaptive validation reward -->

## Evaluation

{{benchmark}} ({{coverage_note}}). Invalid outputs counted as {{invalid_policy}}.

{{metrics_table}}  <!-- from `atpo evaluate --output`; state the exact command -->

Corresponding paper results: {{paper_reference}}.

## Intended use and limitations

Intended for research on video content moderation and as a component of human-reviewed
moderation workflows. It is not a substitute for human review and must not be the sole
basis of decisions that affect people.

- Trained on {{training_data}}; performance on other content, languages, and video
  styles is unknown.
- Errors in both directions occur: harmful videos are missed and benign videos are
  flagged. The controller's target ratio shifts this trade-off; it does not remove it.
- Outputs that cannot be parsed are reported with `status: parse_error` and no labels.
- {{additional_limitations}}

## Files

`checksums.sha256` lists the SHA-256 of every file; `atpo_export.json` records the
source checkpoint, training run, and library versions.

## Citation

{{bibtex}}
