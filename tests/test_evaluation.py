"""Evaluation metrics and coverage handling."""

from __future__ import annotations

import numpy as np
import pytest

from atpo.evaluation import (
    EvalSample,
    EvaluationError,
    evaluate_samples,
    multilabel_metrics,
)
from atpo.taxonomy import load_taxonomy

SAFEWATCH = load_taxonomy("safewatch")


def _compare(ours: dict, theirs: dict, keys) -> None:
    for key in keys:
        assert ours[key] == pytest.approx(theirs[key], abs=1e-12), key


SCALAR_KEYS = [
    "jaccard_similarity", "exact_match_ratio", "hamming_accuracy", "micro_accuracy",
    "macro_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_specificity",
    "macro_false_positive_rate", "macro_balanced_accuracy", "micro_precision", "micro_recall",
    "micro_f1", "sample_precision", "sample_recall", "sample_f1", "weighted_precision",
    "weighted_recall", "weighted_f1",
]


def _random_matrices(seed: int, n: int = 300, k: int = 6):
    rng = np.random.default_rng(seed)
    y_true = (rng.random((n, k)) < 0.2).astype(int)
    y_true[rng.random(n) < 0.3] = 0  # benign rows
    y_pred = np.where(rng.random((n, k)) < 0.15, 1 - y_true, y_true)
    y_pred[:, 1] = 0  # a never-predicted category exercises zero-division handling
    return y_true, y_pred


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_match_sklearn(seed):
    sk = pytest.importorskip("sklearn.metrics")
    y_true, y_pred = _random_matrices(seed)
    m = multilabel_metrics(y_true, y_pred, SAFEWATCH.ids)
    for avg in ("micro", "macro", "samples", "weighted"):
        prefix = "sample" if avg == "samples" else avg
        assert m[f"{prefix}_precision"] == pytest.approx(sk.precision_score(y_true, y_pred, average=avg, zero_division=0.0))
        assert m[f"{prefix}_recall"] == pytest.approx(sk.recall_score(y_true, y_pred, average=avg, zero_division=0.0))
        assert m[f"{prefix}_f1"] == pytest.approx(sk.f1_score(y_true, y_pred, average=avg, zero_division=0.0))
    assert m["jaccard_similarity"] == pytest.approx(
        sk.jaccard_score(y_true, y_pred, average="samples", zero_division=1.0)
    )


def test_coverage_policies():
    samples = [
        EvalSample("a", frozenset({"C1"}), frozenset({"C1"})),
        EvalSample("b", frozenset(), None, status="invalid"),
        EvalSample("c", frozenset({"C2"}), None, status="missing"),
    ]
    with pytest.raises(EvaluationError):
        evaluate_samples(samples, SAFEWATCH, invalid_policy="negative")  # missing_policy defaults to error
    with pytest.raises(EvaluationError):
        evaluate_samples(samples, SAFEWATCH, invalid_policy="error", missing_policy="exclude")
    neg = evaluate_samples(samples, SAFEWATCH, invalid_policy="negative", missing_policy="negative")
    assert neg["coverage"]["scored_samples"] == 3
    # The invalid benign sample counts as correct when scored as negative.
    assert neg["metrics"]["jaccard_similarity"] == pytest.approx(2 / 3)
    exc = evaluate_samples(samples, SAFEWATCH, invalid_policy="exclude", missing_policy="exclude")
    assert exc["coverage"]["scored_samples"] == 1
    assert exc["coverage"]["invalid_ids"] == ["b"] and exc["coverage"]["missing_ids"] == ["c"]
