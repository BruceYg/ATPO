"""Evaluation: metrics and coverage handling."""

from .evaluate import (
    EvalSample,
    EvaluationError,
    evaluate_samples,
    ground_truth_from_records,
    load_jsonl,
    samples_from_predictions,
    summary_table,
)
from .metrics import jaccard, multilabel_metrics

__all__ = [
    "EvalSample",
    "EvaluationError",
    "evaluate_samples",
    "ground_truth_from_records",
    "jaccard",
    "load_jsonl",
    "multilabel_metrics",
    "samples_from_predictions",
    "summary_table",
]
