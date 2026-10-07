#!/usr/bin/env python3
"""Generate a tiny synthetic video dataset for checking the pipeline end to end.

Each clip is 4 s of 160x120 video at 8 fps. Categories are drawn as abstract cues
(W1: a yellow square, W2: a flickering orange band, W3: a block falling from the
top); benign clips show only the background. The data is meant for smoke tests of
data preparation, training launch, export, and inference, not for learning anything
useful.

    python examples/custom_training/make_synthetic_dataset.py --out atpo_example
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def render(path: Path, labels: list[str], rng: random.Random) -> None:
    import av

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=8)
    stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
    shade = rng.randint(40, 90)
    for t in range(32):
        frame = np.full((120, 160, 3), shade, dtype=np.uint8)
        if "W1" in labels:
            frame[20:50, 20 + t:50 + t] = (230, 200, 30)
        if "W2" in labels and t % 2 == 0:
            frame[90:120, :] = (255, 120, 0)
        if "W3" in labels:
            y = min(3 * t, 100)
            frame[y:y + 20, 110:140] = (60, 60, 230)
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--val", type=int, default=8)
    parser.add_argument("--test", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    (args.out / "videos").mkdir(parents=True, exist_ok=True)
    label_sets = [[], ["W1"], ["W2"], ["W3"], ["W1", "W2"], ["W2", "W3"]]
    counter = 0
    for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for _ in range(n):
                labels = rng.choice(label_sets)
                name = f"clip_{counter:04d}.mp4"
                render(args.out / "videos" / name, labels, rng)
                f.write(json.dumps({"id": f"clip_{counter:04d}", "video": name, "labels": labels}) + "\n")
                counter += 1
    print(f"wrote {counter} clips and train/val/test.jsonl under {args.out}")


if __name__ == "__main__":
    main()
