"""Convert existing training files into canonical records.

These adapters let existing EasyR1 or LLaMA-Factory files with this layout be re-expressed as canonical ``{"id", "video", "labels"}``
records. They do not repair anything: repeated IDs and videos are kept, and
unknown labels raise errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..taxonomy import Taxonomy
from .records import DataValidationError, Record


def _strip_prefix(path: str, prefix: Optional[str]) -> str:
    if not prefix:
        return path
    normalized = prefix.rstrip("/") + "/"
    if not path.startswith(normalized):
        raise DataValidationError(f"video path {path!r} does not start with {normalized!r}")
    return path[len(normalized):]


def records_from_easyr1_parquet(
    path: str | Path, taxonomy: Taxonomy, *, path_prefix: Optional[str] = None,
    keep_prompt_hash: bool = True,
) -> list[Record]:
    import hashlib

    import pyarrow.parquet as pq

    records = []
    for row in pq.read_table(path).to_pylist():
        extra: dict[str, Any] = {}
        if keep_prompt_hash and row.get("prompt") is not None:
            extra["source_prompt_sha256"] = hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
        records.append(Record(
            id=str(row["id"]),
            video=_strip_prefix(row["video_path"], path_prefix),
            labels=tuple(taxonomy.resolve_labels(list(row["response"]))),
            extra=extra,
        ))
    return records


def records_from_sharegpt(
    path: str | Path, taxonomy: Taxonomy, *, path_prefix: Optional[str] = None,
    extra_fields: tuple[str, ...] = (),
) -> list[Record]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            videos = row.get("videos") or []
            if len(videos) != 1:
                raise DataValidationError(f"{path}:{line_no}: expected exactly one video, got {len(videos)}")
            records.append(Record(
                id=str(row["id"]),
                video=_strip_prefix(videos[0], path_prefix),
                labels=tuple(taxonomy.resolve_labels(list(row.get("labels") or []))),
                extra={name: row[name] for name in extra_fields if name in row},
            ))
    return records
