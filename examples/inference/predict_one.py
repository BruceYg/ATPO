#!/usr/bin/env python3
"""Classify one video with a released or locally trained checkpoint.

    python examples/inference/predict_one.py --model <hub-id-or-dir> --video clip.mp4
    python examples/inference/predict_one.py --model ./merged_sft --preset safewatch-qwen2.5vl-sft --video clip.mp4

Checkpoints exported with `atpo export` carry atpo_config.json and need no preset.
"""

import argparse
import json

from atpo import VideoSafetyModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--preset")
    parser.add_argument("--video", required=True)
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    args = parser.parse_args()

    model = VideoSafetyModel.from_pretrained(args.model, revision=args.revision, preset=args.preset,
                                             backend=args.backend)
    prediction = model.predict(args.video)
    # status is ok | partial | parse_error | video_error | generation_error;
    # labels/unsafe are null unless the output could be parsed.
    print(json.dumps(prediction.to_dict(include_raw=False), indent=2))


if __name__ == "__main__":
    main()
