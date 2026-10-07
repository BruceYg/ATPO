"""Structured prediction records (one JSON object per video).

``status`` is one of

``ok``                every category has an explicit decision;
``partial``           some categories were found, others are missing (``missing_categories``);
``parse_error``       the response could not be parsed;
``video_error``       the video could not be read or preprocessed;
``generation_error``  the model failed to generate (for example the prompt exceeded the
                      maximum model length).

A failed prediction is never reported as safe: ``labels``, ``decisions`` and ``unsafe``
are ``null`` unless the output was parsed. ``parse.historical_labels`` gives the labels
the historical scripts would have recorded (missing or unparsable decisions counted as
"not flagged"), so historical metrics can be reproduced from release outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..parsing import ParsedResponse
from ..taxonomy import Taxonomy

PREDICTION_FORMAT = "atpo.prediction/v1"
STATUSES = ("ok", "partial", "parse_error", "video_error", "generation_error")


@dataclass
class Prediction:
    id: str
    video: str
    status: str
    labels: Optional[list[str]] = None
    label_names: Optional[list[str]] = None
    decisions: Optional[dict[str, Optional[bool]]] = None
    unsafe: Optional[bool] = None
    missing_categories: list[str] = field(default_factory=list)
    explanation: Optional[str] = None
    raw_response: Optional[str] = None
    parse: Optional[dict[str, Any]] = None
    generation: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"invalid status {self.status!r}")

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self, *, include_explanation: bool = True, include_raw: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "video": self.video,
            "status": self.status,
            "labels": self.labels,
            "label_names": self.label_names,
            "decisions": self.decisions,
            "unsafe": self.unsafe,
            "missing_categories": self.missing_categories,
        }
        if include_explanation:
            out["explanation"] = self.explanation
        if include_raw:
            out["raw_response"] = self.raw_response
        out["parse"] = self.parse
        out["generation"] = self.generation
        out["error"] = self.error
        return out


def prediction_from_parse(
    *,
    sample_id: str,
    video: str,
    raw_response: str,
    parsed: ParsedResponse,
    taxonomy: Taxonomy,
    scope: str,
    generation: Optional[dict[str, Any]] = None,
) -> Prediction:
    parse_info = {
        "scope": scope,
        "source": parsed.source,
        "format_ok": parsed.format_ok,
        "answer_found": parsed.answer_found,
        "error": parsed.error,
        "historical_labels": list(parsed.historical_labels),
    }
    if parsed.status == "invalid":
        return Prediction(
            id=sample_id,
            video=video,
            status="parse_error",
            explanation=parsed.explanation,
            raw_response=raw_response,
            parse=parse_info,
            generation=generation,
            error=parsed.error,
        )
    labels = list(parsed.labels or [])
    missing = [cid for cid, value in parsed.decisions.items() if value is None]
    if parsed.status == "ok":
        unsafe: Optional[bool] = bool(labels)
    else:  # partial: unsafe if anything was flagged, otherwise unknown
        unsafe = True if labels else None
    return Prediction(
        id=sample_id,
        video=video,
        status="ok" if parsed.status == "ok" else "partial",
        labels=labels,
        label_names=[taxonomy.category(cid).name for cid in labels],
        decisions=dict(parsed.decisions),
        unsafe=unsafe,
        missing_categories=missing,
        explanation=parsed.explanation,
        raw_response=raw_response,
        parse=parse_info,
        generation=generation,
    )


def error_prediction(sample_id: str, video: str, status: str, error: str,
                     raw_response: Optional[str] = None) -> Prediction:
    if status not in ("video_error", "generation_error"):
        raise ValueError("error predictions must have status video_error or generation_error")
    return Prediction(id=sample_id, video=video, status=status, raw_response=raw_response, error=error)
