"""Shared test helpers: synthetic model responses and reward batches."""

from __future__ import annotations

import random
from pathlib import Path

from atpo.taxonomy import Taxonomy

REPO_ROOT = Path(__file__).resolve().parents[1]


def _json_block(taxonomy: Taxonomy, decisions: dict[str, bool], *, short: bool, upper: bool) -> str:
    lines = []
    for category in taxonomy.categories:
        if category.id not in decisions:
            continue
        key = category.id if short else taxonomy.output_key(category)
        value = str(decisions[category.id])
        value = value if upper else value.lower()
        lines.append(f'"{key}": {value}')
    return f"{taxonomy.block_name}: {{\n" + ",\n".join(lines) + "\n}"


def make_response(rng: random.Random, taxonomy: Taxonomy, response_format: str, gt: list[str]) -> str:
    """Generate a well-formed, malformed, or adversarial response."""
    ids = taxonomy.ids
    # Predictions correlated with the ground truth so all of TP/FP/FN occur.
    decisions = {cid: (cid in gt) != (rng.random() < 0.25) for cid in ids}
    kind = rng.choice(
        [
            "good", "good", "good", "good_upper", "short_keys", "asterisk", "partial",
            "think_leak", "no_answer", "empty_answer", "trailing", "garbage", "empty",
        ]
    )
    upper = kind == "good_upper"
    if kind == "partial":
        keep = rng.sample(ids, k=rng.randint(1, len(ids) - 1))
        decisions = {cid: decisions[cid] for cid in keep}
    if kind == "asterisk":
        body = "\n".join(
            f"* {taxonomy.output_key(cid)}: {str(v).lower()}" for cid, v in decisions.items()
        )
    else:
        body = _json_block(taxonomy, decisions, short=kind == "short_keys", upper=upper)
    reasoning = "The video shows a person walking."
    if kind == "think_leak":
        flipped = {cid: not v for cid, v in decisions.items()}
        reasoning += " Draft: " + _json_block(taxonomy, flipped, short=False, upper=False)
    if kind == "garbage":
        return "I cannot determine the content of this video."
    if kind == "empty":
        return ""
    if response_format == "think_answer":
        if kind == "no_answer":
            return f"<think>{reasoning}</think>\n{body}"
        if kind == "empty_answer":
            return f"<think>{reasoning}</think>\n<answer>No decision.</answer>"
        text = f"<think>{reasoning}</think>\n<answer>{body}</answer>"
        if kind == "trailing":
            text += "\nThanks!"
        return text
    # triple_newline (Qwen3 recipes)
    if kind == "no_answer":
        return f"{reasoning}\n{body}"
    if kind == "empty_answer":
        return f"{reasoning}\n\n\nNo decision."
    return f"{reasoning}\n\n\n{body}"


def make_batch(
    rng: random.Random, taxonomy: Taxonomy, response_format: str, size: int
) -> list[dict]:
    """A batch of reward inputs with integer (1-based) ground-truth labels, as in the EasyR1 files."""
    batch = []
    for _ in range(size):
        r = rng.random()
        if r < 0.3:
            gt: list[str] = []
        elif r < 0.85:
            gt = [rng.choice(taxonomy.ids)]
        else:
            gt = rng.sample(taxonomy.ids, k=rng.randint(2, 3))
        response = make_response(rng, taxonomy, response_format, gt)
        gt_int = [taxonomy.index(cid) + 1 for cid in gt]
        rng.shuffle(gt_int)
        batch.append({"response": response, "response_length": len(response), "ground_truth": gt_int})
    return batch
