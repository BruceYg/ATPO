"""Training-file builders: LLaMA-Factory ShareGPT JSONL and EasyR1 parquet.

Both builders take canonical :class:`~atpo.data.records.Record` objects and a task
prompt. The layouts reproduce the training files of the paper's runs exactly:

* ShareGPT rows: ``{"id", "messages": [user, assistant], "videos": [path], "labels", ...}``
  where the user turn is ``prompt.strip() + " <video>"``.
* EasyR1 rows: ``{"id", "video_path", "prompt", "response"}`` where ``prompt`` is
  ``prompt.strip() + " <video>"`` (the answer-format instruction is added at load time
  by EasyR1's ``data.format_prompt`` template) and ``response`` holds the labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

from ..taxonomy import Taxonomy
from .prompts import TargetStyle, sft_target, with_video_token
from .records import Record

LabelFormat = Literal["index", "id"]


def _video_path(record: Record, path_prefix: Optional[str]) -> str:
    if not path_prefix:
        return record.video
    return path_prefix.rstrip("/") + "/" + record.video.lstrip("/")


def _labels(record: Record, taxonomy: Taxonomy, label_format: LabelFormat) -> list[Any]:
    if label_format == "index":
        return [taxonomy.index(cid) + 1 for cid in record.labels]
    if label_format == "id":
        return list(record.labels)
    raise ValueError("label_format must be 'index' or 'id'")


def build_sharegpt_rows(
    records: Sequence[Record],
    taxonomy: Taxonomy,
    *,
    prompt: str,
    target_style: TargetStyle,
    block_name: Optional[str] = None,
    label_format: LabelFormat = "id",
    path_prefix: Optional[str] = None,
    extra_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    user = with_video_token(prompt)
    rows = []
    for record in records:
        row: dict[str, Any] = {
            "id": record.id,
            "messages": [  # key order as in the paper's files
                {"content": user, "role": "user"},
                {"content": sft_target(record.labels, taxonomy, style=target_style, block_name=block_name),
                 "role": "assistant"},
            ],
            "videos": [_video_path(record, path_prefix)],
            "labels": _labels(record, taxonomy, label_format),
        }
        for name in extra_fields:
            if name not in record.extra:
                raise KeyError(f"record {record.id!r} has no field {name!r}")
            row[name] = record.extra[name]
        rows.append(row)
    return rows


def build_easyr1_rows(
    records: Sequence[Record],
    taxonomy: Taxonomy,
    *,
    prompt: str,
    label_format: LabelFormat = "id",
    path_prefix: Optional[str] = None,
) -> list[dict[str, Any]]:
    task = with_video_token(prompt)
    return [
        {
            "id": record.id,
            "video_path": _video_path(record, path_prefix),
            "prompt": task,
            "response": _labels(record, taxonomy, label_format),
        }
        for record in records
    ]


def oversample_rows(
    rows: Sequence[dict[str, Any]],
    class_index: int,
    *,
    labels_key: str = "response",
) -> list[dict[str, Any]]:
    """Append one copy of every row whose labels contain ``class_index`` (1-based).

    Reproduces the paper's oversampling (the C2-oversampled SafeWatch training file is
    this operation with class 2 applied to the base file): copies are appended after
    all original rows in their original order, with IDs ``<id>__aug_cls<k>_<n>`` where
    ``n`` is the smallest counter not already used.
    """
    existing = {str(r["id"]) for r in rows}
    copies = []
    for row in rows:
        labels = row[labels_key]
        indices = {int(v) for v in labels if not isinstance(v, str) or v.isdigit()}
        if class_index not in indices:
            continue
        counter = 1
        while f"{row['id']}__aug_cls{class_index}_{counter}" in existing:
            counter += 1
        new_id = f"{row['id']}__aug_cls{class_index}_{counter}"
        existing.add(new_id)
        copies.append({**row, "id": new_id})
    return list(rows) + copies


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_easyr1_parquet(rows: Sequence[dict[str, Any]], path: str | Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_labels = any(isinstance(v, str) for r in rows for v in r["response"])
    schema = pa.schema([
        ("id", pa.string()),
        ("video_path", pa.string()),
        ("prompt", pa.string()),
        ("response", pa.list_(pa.string() if string_labels else pa.int64())),
    ])
    table = pa.Table.from_pylist([{k: r[k] for k in ("id", "video_path", "prompt", "response")} for r in rows],
                                 schema=schema)
    pq.write_table(table, path)
    return table.num_rows


def read_easyr1_parquet(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def llamafactory_dataset_info(file_name: str) -> dict[str, Any]:
    """``dataset_info.json`` entry for a ShareGPT video dataset (as used historically)."""
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "videos": "videos"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    }
