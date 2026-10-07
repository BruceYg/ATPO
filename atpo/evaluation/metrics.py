"""Multi-label classification metrics.

The definitions reproduce the evaluation script used for the paper,
which produced the paper's SafeWatch numbers, including scikit-learn's
``zero_division=0`` conventions. Only NumPy is required.

* ``jaccard_similarity``: mean over samples of |P ∩ G| / |P ∪ G|; a sample with
  empty prediction and empty ground truth (a correctly identified benign video)
  scores 1, a sample where exactly one side is empty scores 0.
* ``micro_*``: pooled TP/FP/FN over all samples and categories (benign samples
  contribute only false positives). This is the "P/R" reported in the paper.
* ``macro_*``: unweighted mean over *all* categories; categories that are never
  predicted (or have no positives) contribute 0.
* ``sample_*``: per-sample precision/recall/F1 averaged over samples; a benign
  sample predicted benign contributes 0 (scikit-learn ``average="samples"``).
* ``binary``: unsafe = at least one category. ``false_refusal_rate`` is the share
  of benign samples flagged; ``violation_leakage_rate`` the share of unsafe
  samples missed.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _f1(tp: float, fp: float, fn: float) -> float:
    # scikit-learn's F-beta (beta=1) from counts, zero_division=0.
    den = 2 * tp + fp + fn
    return _div(2 * tp, den)


def jaccard(gt: frozenset | set, pred: frozenset | set) -> float:
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    return len(gt & pred) / len(gt | pred)


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, category_ids: Sequence[str]) -> dict[str, Any]:
    """Compute the full metric set for binary indicator matrices of shape (N, K)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError(f"expected matching (N, K) matrices, got {y_true.shape} and {y_pred.shape}")
    n, k = y_true.shape
    if k != len(category_ids):
        raise ValueError("category_ids do not match the matrix width")
    if n == 0:
        raise ValueError("no samples to evaluate")

    results: dict[str, Any] = {"num_samples": int(n)}
    jaccards = []
    for row_t, row_p in zip(y_true, y_pred):
        g = {i for i in range(k) if row_t[i]}
        p = {i for i in range(k) if row_p[i]}
        jaccards.append(jaccard(g, p))
    results["jaccard_similarity"] = float(np.mean(jaccards))
    results["exact_match_ratio"] = float(np.mean(np.all(y_true == y_pred, axis=1)))
    results["hamming_accuracy"] = float(1 - np.mean(y_true != y_pred))
    results["micro_accuracy"] = float(np.mean(y_true.ravel() == y_pred.ravel()))

    per_category: dict[str, dict[str, Any]] = {}
    precisions, recalls, f1s, accuracies, specificities, fprs, balanced = [], [], [], [], [], [], []
    supports = []
    for i, cid in enumerate(category_ids):
        t, p = y_true[:, i], y_pred[:, i]
        tp = int(np.sum((t == 1) & (p == 1)))
        fp = int(np.sum((t == 0) & (p == 1)))
        fn = int(np.sum((t == 1) & (p == 0)))
        tn = int(np.sum((t == 0) & (p == 0)))
        precision = _div(tp, tp + fp)
        recall = _div(tp, tp + fn)
        f1 = _f1(tp, fp, fn)
        accuracy = float(np.mean(t == p))
        specificity = _div(tn, tn + fp)
        fpr = _div(fp, fp + tn)
        bal = (recall + specificity) / 2
        per_category[cid] = {
            "n_positive_gt": int(np.sum(t)),
            "n_positive_pred": int(np.sum(p)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "false_positive_rate": fpr,
            "balanced_accuracy": bal,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)
        specificities.append(specificity)
        fprs.append(fpr)
        balanced.append(bal)
        supports.append(tp + fn)
    results["per_category"] = per_category
    results["macro_accuracy"] = float(np.mean(accuracies))
    results["macro_precision"] = float(np.mean(precisions))
    results["macro_recall"] = float(np.mean(recalls))
    results["macro_f1"] = float(np.mean(f1s))
    results["macro_specificity"] = float(np.mean(specificities))
    results["macro_false_positive_rate"] = float(np.mean(fprs))
    results["macro_balanced_accuracy"] = float(np.mean(balanced))

    tp_all = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp_all = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn_all = int(np.sum((y_true == 1) & (y_pred == 0)))
    results["micro_precision"] = _div(tp_all, tp_all + fp_all)
    results["micro_recall"] = _div(tp_all, tp_all + fn_all)
    results["micro_f1"] = _f1(tp_all, fp_all, fn_all)

    sp, sr, sf = [], [], []
    for row_t, row_p in zip(y_true, y_pred):
        tp = int(np.sum((row_t == 1) & (row_p == 1)))
        fp = int(np.sum((row_t == 0) & (row_p == 1)))
        fn = int(np.sum((row_t == 1) & (row_p == 0)))
        sp.append(_div(tp, tp + fp))
        sr.append(_div(tp, tp + fn))
        sf.append(_f1(tp, fp, fn))
    results["sample_precision"] = float(np.mean(sp))
    results["sample_recall"] = float(np.mean(sr))
    results["sample_f1"] = float(np.mean(sf))

    total_support = float(np.sum(supports))
    if total_support > 0:
        w = np.asarray(supports, dtype=np.float64) / total_support
        results["weighted_precision"] = float(np.sum(w * np.asarray(precisions)))
        results["weighted_recall"] = float(np.sum(w * np.asarray(recalls)))
        results["weighted_f1"] = float(np.sum(w * np.asarray(f1s)))
    else:
        results["weighted_precision"] = results["weighted_recall"] = results["weighted_f1"] = 0.0

    binary_gt = (np.sum(y_true, axis=1) > 0).astype(int)
    binary_pred = (np.sum(y_pred, axis=1) > 0).astype(int)
    btp = int(np.sum((binary_gt == 1) & (binary_pred == 1)))
    bfp = int(np.sum((binary_gt == 0) & (binary_pred == 1)))
    bfn = int(np.sum((binary_gt == 1) & (binary_pred == 0)))
    n_benign = int(np.sum(binary_gt == 0))
    n_unsafe = int(np.sum(binary_gt == 1))
    results["binary"] = {
        "accuracy": float(np.mean(binary_gt == binary_pred)),
        "precision": _div(btp, btp + bfp),
        "recall": _div(btp, btp + bfn),
        "f1_score": _f1(btp, bfp, bfn),
        "false_refusal_rate": _div(bfp, n_benign),
        "violation_leakage_rate": _div(bfn, n_unsafe),
        "n_benign": n_benign,
        "n_unsafe": n_unsafe,
        "n_false_refusal": bfp,
        "n_missed_violation": bfn,
    }
    return results
