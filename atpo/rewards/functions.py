"""Per-sample accuracy rewards over predicted and ground-truth label sets.

Each function reproduces the arithmetic of the corresponding historical EasyR1
reward file, including edge cases:

* an unparseable prediction (``None``) scores 0;
* an empty prediction for an empty ground truth (a correctly identified benign
  sample) scores 1;
* a zero denominator scores 1.

Inputs are sets of category IDs (or any hashable labels) except for
:func:`category_tversky`, which takes 0/1 vectors in taxonomy order.
"""

from __future__ import annotations

from typing import AbstractSet, Hashable, Optional, Sequence

import numpy as np

LabelSet = AbstractSet[Hashable]


def exact_match(pred: Optional[LabelSet], gt: LabelSet) -> float:
    """1 if the predicted set equals the ground truth, else 0."""
    if pred is None:
        return 0.0
    return 1.0 if set(pred) == set(gt) else 0.0


def jaccard(pred: Optional[LabelSet], gt: LabelSet) -> float:
    """|P ∩ G| / |P ∪ G|."""
    if pred is None:
        return 0.0
    pred, gt = set(pred), set(gt)
    if not pred and not gt:
        return 1.0
    union = pred | gt
    if not union:
        return 1.0
    return len(pred & gt) / len(union)


def tversky(pred: Optional[LabelSet], gt: LabelSet, alpha: float, beta: float) -> float:
    """Tversky index TP / (TP + alpha·FP + beta·FN) with scalar weights.

    Used by static-Tversky GRPO and ATPO-G. ``alpha = beta = 1`` gives the Jaccard
    index; ``alpha = beta = 0.5`` gives the Dice coefficient.
    """
    if pred is None:
        return 0.0
    pred, gt = set(pred), set(gt)
    if not pred and not gt:
        return 1.0
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    denominator = tp + alpha * fp + beta * fn
    if denominator == 0:
        return 1.0
    return tp / denominator


def category_tversky(
    pred_vec: Optional[Sequence[int]],
    gt_vec: Sequence[int],
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    """Micro category-aware Tversky reward (ATPO-C, ``*_tversky_adaptive_category.py``).

    ``sum_c TP_c / (sum_c TP_c + sum_c alpha_c·FP_c + sum_c beta_c·FN_c)`` with 0/1
    indicators per category.
    """
    if pred_vec is None:
        return 0.0
    pred = np.asarray(pred_vec)
    gt = np.asarray(gt_vec)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    tp_vec = (gt == 1) & (pred == 1)
    fp_vec = (gt == 0) & (pred == 1)
    fn_vec = (gt == 1) & (pred == 0)
    sum_tp = float(np.sum(tp_vec))
    sum_weighted_fp = float(np.sum(alpha * fp_vec))
    sum_weighted_fn = float(np.sum(beta * fn_vec))
    denominator = sum_tp + sum_weighted_fp + sum_weighted_fn
    if denominator == 0:
        return 1.0
    return sum_tp / denominator
