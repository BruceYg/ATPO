"""Evaluation of predictions against ground truth with explicit coverage handling.

Predictions come from ``atpo predict`` JSONL files; ground truth comes from
canonical data records (``{"id", "video", "labels"}``).

Two decisions are always explicit and reported:

``invalid_policy``
    What to do with predictions whose output could not be parsed or whose video
    failed to process. ``negative`` scores them as predicting no category; this is
    how the paper's numbers were computed. ``exclude`` drops
    them from all metrics. ``error`` refuses to evaluate if any exist.
``missing_policy``
    What to do with ground-truth samples that have no prediction record at all.
    ``error`` (default) refuses, ``negative`` scores them as predicting no
    category, ``exclude`` drops them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

import numpy as np

from ..taxonomy import Taxonomy
from .metrics import multilabel_metrics

InvalidPolicy = Literal["negative", "exclude", "error"]
MissingPolicy = Literal["negative", "exclude", "error"]


@dataclass(frozen=True)
class EvalSample:
    """One sample to score. ``pred`` is ``None`` when the prediction is invalid."""

    id: str
    gt: frozenset[str]
    pred: Optional[frozenset[str]]
    status: str = "ok"
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False)


class EvaluationError(ValueError):
    pass


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def ground_truth_from_records(records: Iterable[Mapping[str, Any]], taxonomy: Taxonomy) -> dict[str, frozenset[str]]:
    """Map sample ID to ground-truth category IDs from canonical data records."""
    gt: dict[str, frozenset[str]] = {}
    for record in records:
        sid = str(record["id"])
        if sid in gt:
            raise EvaluationError(f"duplicate ground-truth id {sid!r}")
        gt[sid] = frozenset(taxonomy.resolve_labels(record.get("labels") or []))
    return gt


def samples_from_predictions(
    predictions: Iterable[Mapping[str, Any]],
    ground_truth: Mapping[str, frozenset[str]],
    taxonomy: Taxonomy,
    *,
    label_source: Literal["labels", "historical"] = "labels",
) -> tuple[list[EvalSample], dict[str, Any]]:
    """Join ``atpo predict`` records with ground truth by sample ID.

    ``label_source="labels"`` scores the release interpretation: only ``ok`` and
    ``partial`` predictions carry labels, everything else is invalid. ``"historical"``
    scores ``parse.historical_labels`` (what the research scripts recorded: unparsable or
    missing decisions count as not flagged) for every parsed response; video and
    generation errors stay invalid (the research scripts dropped those samples, which
    corresponds to ``invalid_policy="exclude"``).
    """
    if label_source not in ("labels", "historical"):
        raise EvaluationError(f"label_source must be 'labels' or 'historical', got {label_source!r}")
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicates = []
    for record in predictions:
        sid = str(record.get("id"))
        if sid in by_id:
            duplicates.append(sid)
        by_id[sid] = record
    if duplicates:
        raise EvaluationError(f"duplicate prediction ids, e.g. {duplicates[:5]}")
    samples = []
    for sid, gt in ground_truth.items():
        record = by_id.get(sid)
        if record is None:
            samples.append(EvalSample(id=sid, gt=gt, pred=None, status="missing"))
            continue
        status = str(record.get("status", "ok"))
        labels = record.get("labels")
        if label_source == "historical" and status in ("ok", "partial", "parse_error"):
            historical = (record.get("parse") or {}).get("historical_labels")
            if historical is None:
                raise EvaluationError(f"prediction {sid!r} has no parse.historical_labels")
            samples.append(EvalSample(id=sid, gt=gt, pred=frozenset(taxonomy.resolve_labels(historical)), status=status))
        elif status in ("ok", "partial") and labels is not None:
            pred = frozenset(taxonomy.resolve_labels(labels))
            samples.append(EvalSample(id=sid, gt=gt, pred=pred, status=status))
        else:
            samples.append(EvalSample(id=sid, gt=gt, pred=None, status=status if status != "ok" else "invalid"))
    extra = sorted(set(by_id) - set(ground_truth))
    return samples, {"extra_prediction_ids": extra}


def evaluate_samples(
    samples: Sequence[EvalSample],
    taxonomy: Taxonomy,
    *,
    invalid_policy: InvalidPolicy,
    missing_policy: MissingPolicy = "error",
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Score samples and return ``{"metrics": ..., "coverage": ...}``."""
    if invalid_policy not in ("negative", "exclude", "error"):
        raise EvaluationError(f"invalid_policy must be negative, exclude, or error; got {invalid_policy!r}")
    if missing_policy not in ("negative", "exclude", "error"):
        raise EvaluationError(f"missing_policy must be negative, exclude, or error; got {missing_policy!r}")
    missing = [s.id for s in samples if s.status == "missing"]
    invalid = [s for s in samples if s.pred is None and s.status != "missing"]
    unparsed_scored = [s.id for s in samples if s.pred is not None and s.status == "parse_error"]
    partial = [s.id for s in samples if s.status == "partial"]
    if missing and missing_policy == "error":
        raise EvaluationError(
            f"{len(missing)} ground-truth samples have no prediction (e.g. {missing[:5]}); "
            "choose missing_policy=negative or exclude explicitly"
        )
    if invalid and invalid_policy == "error":
        raise EvaluationError(
            f"{len(invalid)} predictions are invalid (e.g. {[s.id for s in invalid[:5]]}); "
            "choose invalid_policy=negative (historical convention) or exclude explicitly"
        )
    ids = taxonomy.ids
    rows_t, rows_p, scored_ids = [], [], []
    for s in samples:
        pred = s.pred
        if pred is None:
            policy = missing_policy if s.status == "missing" else invalid_policy
            if policy == "exclude":
                continue
            pred = frozenset()
        rows_t.append(taxonomy.to_vector(s.gt))
        rows_p.append(taxonomy.to_vector(pred))
        scored_ids.append(s.id)
    if not rows_t:
        raise EvaluationError("no samples left to evaluate")
    metrics = multilabel_metrics(np.array(rows_t), np.array(rows_p), ids)
    status_counts: dict[str, int] = {}
    for s in samples:
        status_counts[s.status] = status_counts.get(s.status, 0) + 1
    coverage = {
        "ground_truth_samples": len(samples),
        "scored_samples": len(scored_ids),
        "status_counts": status_counts,
        "missing": len(missing),
        "invalid": len(invalid),
        "partial": len(partial),
        "invalid_policy": invalid_policy,
        "missing_policy": missing_policy,
        "missing_ids": missing,
        "invalid_ids": [s.id for s in invalid],
        "partial_ids": partial,
        "parse_errors_scored_with_historical_labels": len(unparsed_scored),
    }
    if extra:
        coverage.update(extra)
    return {"taxonomy": taxonomy.name, "metrics": metrics, "coverage": coverage}


def summary_table(result: Mapping[str, Any], taxonomy: Taxonomy) -> str:
    """Human-readable summary of the main metrics (percentages)."""
    m = result["metrics"]
    c = result["coverage"]
    lines = [
        f"Samples scored: {c['scored_samples']} / {c['ground_truth_samples']} "
        f"(invalid {c['invalid']} -> {c['invalid_policy']}, missing {c['missing']} -> {c['missing_policy']}, "
        f"partial {c['partial']})",
        f"Jaccard {100 * m['jaccard_similarity']:.2f} | exact match {100 * m['exact_match_ratio']:.2f} | "
        f"micro P/R/F1 {100 * m['micro_precision']:.2f}/{100 * m['micro_recall']:.2f}/{100 * m['micro_f1']:.2f} | "
        f"macro F1 {100 * m['macro_f1']:.2f}",
        f"Binary acc/P/R/F1 {100 * m['binary']['accuracy']:.2f}/{100 * m['binary']['precision']:.2f}/"
        f"{100 * m['binary']['recall']:.2f}/{100 * m['binary']['f1_score']:.2f} | "
        f"false refusal {100 * m['binary']['false_refusal_rate']:.2f} | "
        f"violation leakage {100 * m['binary']['violation_leakage_rate']:.2f}",
        "Per category (P/R/F1, support):",
    ]
    for cid in taxonomy.ids:
        pc = m["per_category"][cid]
        lines.append(
            f"  {cid} {taxonomy.category(cid).name}: {100 * pc['precision']:.2f}/{100 * pc['recall']:.2f}/"
            f"{100 * pc['f1_score']:.2f} ({pc['n_positive_gt']})"
        )
    return "\n".join(lines)
