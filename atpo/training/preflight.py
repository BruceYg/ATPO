"""Checks run by ``atpo train`` before launching a backend.

EasyR1 replaces missing optional paths (``data.format_prompt``, ``data.image_dir``,
``data.video_cache_dir``) with ``None`` after printing a message, which would train
without the answer-format instruction or resolve videos against the wrong
directory. These checks fail before any GPU work instead. They also compare the
training/validation files with the fingerprints of the paper's files (``paper_data``
in the config, when present) and report differences.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from ..configuration import ConfigError
from .easyr1 import TrainingConfig


class PreflightError(ConfigError):
    pass


def rows_fingerprint(path: str | Path) -> tuple[str, int]:
    """SHA-256 over the rows of an EasyR1 parquet (independent of parquet encoding)."""
    import pyarrow.parquet as pq

    digest = hashlib.sha256()
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        payload = {k: row[k] for k in ("id", "video_path", "prompt", "response")}
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
    return digest.hexdigest(), len(rows)


def _looks_local(reference: str) -> bool:
    return reference.startswith(("/", "./", "../", "~")) or os.path.exists(reference)


def preflight_easyr1(config: TrainingConfig, tree: Mapping[str, Any], *, output_dir: str | Path,
                     sample_videos: int = 20) -> list[str]:
    from ..rewards.runtime import build_settings

    problems: list[str] = []
    notes: list[str] = []
    data = tree["data"]
    for key in ("train_files", "val_files"):
        value = str(data[key])
        if "@" not in value and not os.path.exists(value):
            problems.append(f"data.{key} not found: {value}")
    for key in ("format_prompt", "image_dir", "override_chat_template"):
        value = data.get(key)
        if value is not None and not os.path.exists(str(value)):
            problems.append(f"data.{key} not found: {value} (EasyR1 would silently ignore it)")
    model = str(tree["worker"]["actor"]["model"]["model_path"])
    if _looks_local(model) and not os.path.isdir(os.path.expanduser(model)):
        problems.append(f"worker.actor.model.model_path not found: {model}")
    try:
        build_settings(dict(tree["worker"]["reward"]["reward_function_kwargs"]))
    except Exception as exc:
        problems.append(f"invalid reward settings: {exc}")
    if problems:
        raise PreflightError("preflight failed:\n  " + "\n  ".join(problems))

    for key, fp_key in (("train_files", "train_fingerprint"), ("val_files", "val_fingerprint")):
        expected = config.paper_data.get(fp_key)
        path = str(data[key])
        if expected and os.path.exists(path) and path.endswith(".parquet"):
            actual, _ = rows_fingerprint(path)
            if actual != expected:
                notes.append(f"data.{key} differs from the paper's file "
                             f"({config.paper_data.get(key.replace('_files', '_file'))}, built by "
                             f"scripts/prepare_paper_data.sh)")
    missing = _sample_missing_videos(str(data["train_files"]), data.get("image_dir"), Path(output_dir), sample_videos)
    if missing:
        raise PreflightError(f"{len(missing)} of the first {sample_videos} training videos were not found, "
                             f"e.g. {missing[:3]}; set VIDEO_ROOT (data.image_dir) or fix the paths")
    for note in notes:
        print(f"[preflight] note: {note}")
    return notes


def _sample_missing_videos(train_file: str, image_dir: Optional[str], run_dir: Path, limit: int) -> list[str]:
    if not train_file.endswith(".parquet") or not os.path.exists(train_file):
        return []
    import pyarrow.parquet as pq

    table = pq.read_table(train_file, columns=["video_path"]).slice(0, limit)
    missing = []
    for video in table.column("video_path").to_pylist():
        if "://" in video:
            continue
        path = os.path.join(image_dir, video) if image_dir else video
        if not os.path.isabs(path):
            path = str(run_dir / path)  # EasyR1 runs with the run directory as working directory
        if not os.path.exists(path):
            missing.append(path)
    return missing


def preflight_llamafactory(config: TrainingConfig, tree: Mapping[str, Any]) -> None:
    problems = []
    dataset_dir = tree.get("dataset_dir")
    if dataset_dir:
        info_path = Path(dataset_dir) / "dataset_info.json"
        if not info_path.exists():
            problems.append(f"{info_path} not found")
        else:
            registered = json.loads(info_path.read_text(encoding="utf-8"))
            for name in str(tree.get("dataset", "")).split(","):
                name = name.strip()
                if name and name not in registered:
                    problems.append(f"dataset {name!r} is not registered in {info_path}")
                elif name and not (Path(dataset_dir) / registered[name]["file_name"]).exists():
                    problems.append(f"dataset file for {name!r} not found: {registered[name]['file_name']}")
    model = str(tree.get("model_name_or_path", ""))
    if _looks_local(model) and not os.path.isdir(os.path.expanduser(model)):
        problems.append(f"model_name_or_path not found: {model}")
    if problems:
        raise PreflightError("preflight failed:\n  " + "\n  ".join(problems))
