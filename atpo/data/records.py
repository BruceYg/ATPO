"""Canonical dataset records and validation.

A dataset is a JSON Lines file with one record per video::

    {"id": "clip-0001", "video": "videos/clip-0001.mp4", "labels": ["C2", "C3"]}

``labels`` lists the categories present (an empty list means benign). Labels may be
given as category IDs or as aliases defined by the taxonomy (for the built-in
taxonomies, the 1-based integers used by the research code). Any other fields are
kept in ``Record.extra`` and can be passed through to the training files.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

from ..taxonomy import Taxonomy

RepeatPolicy = Literal["allow", "warn", "error"]


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Record:
    id: str
    video: str
    labels: tuple[str, ...]
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "video": self.video, "labels": list(self.labels), **dict(self.extra)}


@dataclass
class ValidationReport:
    num_records: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    label_counts: dict[str, int] = field(default_factory=dict)
    cardinality_counts: dict[int, int] = field(default_factory=dict)
    repeated_ids: dict[str, int] = field(default_factory=dict)
    repeated_videos: dict[str, int] = field(default_factory=dict)
    missing_media: list[str] = field(default_factory=list)
    unreadable_media: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "num_records": self.num_records,
            "errors": self.errors,
            "warnings": self.warnings,
            "label_counts": self.label_counts,
            "num_benign": self.cardinality_counts.get(0, 0),
            "label_cardinality": {str(k): v for k, v in sorted(self.cardinality_counts.items())},
            "repeated_ids": self.repeated_ids,
            "repeated_videos": self.repeated_videos,
            "missing_media": self.missing_media,
            "unreadable_media": self.unreadable_media,
        }

    def raise_if_errors(self) -> None:
        if self.errors:
            shown = "\n  ".join(self.errors[:20])
            more = f"\n  ... and {len(self.errors) - 20} more" if len(self.errors) > 20 else ""
            raise DataValidationError(f"{len(self.errors)} validation error(s):\n  {shown}{more}")


def parse_record(data: Mapping[str, Any], taxonomy: Taxonomy, *, where: str = "") -> Record:
    prefix = f"{where}: " if where else ""
    if not isinstance(data, Mapping):
        raise DataValidationError(f"{prefix}record must be a JSON object")
    missing = [k for k in ("id", "video", "labels") if k not in data]
    if missing:
        raise DataValidationError(f"{prefix}missing field(s) {missing}")
    sid, video, labels = data["id"], data["video"], data["labels"]
    if not isinstance(sid, (str, int)) or isinstance(sid, bool) or str(sid) == "":
        raise DataValidationError(f"{prefix}'id' must be a non-empty string")
    if not isinstance(video, str) or not video:
        raise DataValidationError(f"{prefix}'video' must be a non-empty string")
    if not isinstance(labels, list):
        raise DataValidationError(f"{prefix}'labels' must be a list (use [] for benign)")
    try:
        resolved = taxonomy.resolve_labels(labels)
    except (KeyError, ValueError, TypeError) as exc:
        raise DataValidationError(f"{prefix}{exc}") from exc
    if len(resolved) != len(labels):
        raise DataValidationError(f"{prefix}duplicate labels {labels}")
    extra = {k: v for k, v in data.items() if k not in ("id", "video", "labels")}
    return Record(id=str(sid), video=video, labels=tuple(resolved), extra=extra)


def load_records(path: str | Path, taxonomy: Taxonomy) -> list[Record]:
    """Read and parse a canonical JSONL file (errors name the offending line)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_no}"
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"{where}: invalid JSON ({exc})") from exc
            records.append(parse_record(data, taxonomy, where=where))
    return records


def write_records(records: Iterable[Record], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def resolve_video(video: str, video_root: Optional[str | Path]) -> str:
    if "://" in video or os.path.isabs(video) or video_root is None:
        return video
    return str(Path(video_root) / video)


def validate_records(
    records: Sequence[Record],
    taxonomy: Taxonomy,
    *,
    video_root: Optional[str | Path] = None,
    repeated_ids: RepeatPolicy = "error",
    repeated_videos: RepeatPolicy = "warn",
    check_media: Literal["none", "exists", "decode"] = "none",
    require_all_categories: bool = False,
) -> ValidationReport:
    """Validate parsed records.

    Repeated IDs are errors by default. Repeated videos (the same file under several
    IDs, as in the historical oversampled training sets) are reported as warnings by
    default; choose ``"error"`` to forbid them or ``"allow"`` to accept silently.
    ``check_media="exists"`` checks that local files exist; ``"decode"`` also opens each
    video and reads its first frame.
    """
    report = ValidationReport(num_records=len(records))
    id_counts = Counter(r.id for r in records)
    video_counts = Counter(resolve_video(r.video, video_root) for r in records)
    report.repeated_ids = {k: v for k, v in id_counts.items() if v > 1}
    report.repeated_videos = {k: v for k, v in video_counts.items() if v > 1}
    for policy, found, what in ((repeated_ids, report.repeated_ids, "ids"),
                                (repeated_videos, report.repeated_videos, "videos")):
        if found and policy != "allow":
            examples = list(found)[:5]
            message = f"{len(found)} repeated {what} ({sum(found.values()) - len(found)} extra rows), e.g. {examples}"
            (report.errors if policy == "error" else report.warnings).append(message)

    label_counts = Counter(label for r in records for label in r.labels)
    report.label_counts = {cid: label_counts.get(cid, 0) for cid in taxonomy.ids}
    report.cardinality_counts = dict(Counter(len(r.labels) for r in records))
    absent = [cid for cid, n in report.label_counts.items() if n == 0]
    if absent:
        message = f"categories with no positive example: {absent}"
        (report.errors if require_all_categories else report.warnings).append(message)

    if check_media != "none":
        for video in video_counts:
            if "://" in video:
                continue
            if not os.path.isfile(video):
                report.missing_media.append(video)
            elif check_media == "decode" and not _can_decode(video):
                report.unreadable_media.append(video)
        if report.missing_media:
            report.errors.append(f"{len(report.missing_media)} video file(s) not found, e.g. {report.missing_media[:3]}")
        if report.unreadable_media:
            report.errors.append(
                f"{len(report.unreadable_media)} video file(s) could not be decoded, e.g. {report.unreadable_media[:3]}"
            )
    return report


def _can_decode(path: str) -> bool:
    try:
        import av  # PyAV is part of the inference extra
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("check_media='decode' requires PyAV (pip install av)") from exc
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            for _ in container.decode(stream):
                return True
    except Exception:
        return False
    return False


def check_split_overlap(
    splits: Mapping[str, Sequence[Record]], *, video_root: Optional[str | Path] = None
) -> list[str]:
    """Report IDs or videos shared between splits (each overlap is one message)."""
    problems = []
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ids = {r.id for r in splits[a]} & {r.id for r in splits[b]}
            videos = ({resolve_video(r.video, video_root) for r in splits[a]}
                      & {resolve_video(r.video, video_root) for r in splits[b]})
            if ids:
                problems.append(f"{len(ids)} id(s) in both {a} and {b}, e.g. {sorted(ids)[:3]}")
            if videos:
                problems.append(f"{len(videos)} video(s) in both {a} and {b}, e.g. {sorted(videos)[:3]}")
    return problems
