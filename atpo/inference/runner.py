"""Batch prediction to JSONL with a metadata sidecar.

Each prediction is appended to the output file as soon as it is available, so an
interrupted run can continue with ``resume=True`` (already written IDs are skipped).
``<output>.meta.json`` records the model reference and resolved revision, the full
inference configuration, backend settings, library versions, hardware, input hash,
and per-status counts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import os
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from .model import VideoSafetyModel, iter_jsonl
from .prediction import PREDICTION_FORMAT

RUN_FORMAT = "atpo.prediction_run/v1"


def library_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in ("atpo", "torch", "transformers", "vllm", "qwen_vl_utils", "torchvision", "av", "decord",
                 "torchcodec", "huggingface_hub", "numpy"):
        try:
            if name == "qwen_vl_utils":
                from importlib.metadata import version

                versions[name] = version("qwen-vl-utils")
                continue
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform()}
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return info


def video_reader_backend() -> Optional[str]:
    try:
        from qwen_vl_utils.vision_process import get_video_reader_backend

        return get_video_reader_backend()
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(
    *,
    videos: Optional[Sequence[str]] = None,
    input_path: Optional[str] = None,
    video_root: Optional[str] = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return ``(ids, video_paths, input_info)`` from explicit videos or a JSONL file.

    JSONL records need ``video`` (absolute, or relative to ``video_root`` / the JSONL's
    directory) and may carry ``id`` (defaults to the video string). Other fields such
    as ``labels`` are ignored here.
    """
    if (videos is None) == (input_path is None):
        raise ValueError("give exactly one of videos or input_path")
    if videos is not None:
        ids = [str(v) for v in videos]
        return ids, [str(v) for v in videos], {"type": "videos", "count": len(videos)}
    path = Path(input_path)  # type: ignore[arg-type]
    root = Path(video_root) if video_root else path.parent
    ids, paths = [], []
    for record in iter_jsonl(path):
        if "video" not in record:
            raise ValueError(f"{path}: record without 'video': {record}")
        video = str(record["video"])
        if "://" not in video and not os.path.isabs(video):
            video = str(root / video)
        ids.append(str(record.get("id", record["video"])))
        paths.append(video)
    duplicates = [k for k, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ValueError(f"{path}: duplicate ids, e.g. {duplicates[:5]}")
    return ids, paths, {"type": "jsonl", "path": str(path), "sha256": _sha256_file(path), "count": len(ids),
                        "video_root": str(root)}


def run_predictions(
    model: VideoSafetyModel,
    ids: Sequence[str],
    videos: Sequence[str],
    output_path: str | Path,
    *,
    input_info: Optional[dict[str, Any]] = None,
    batch_size: int = 8,
    resume: bool = False,
    overwrite: bool = False,
    include_raw: bool = True,
    include_explanation: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_path)
    meta_path = output_path.with_name(output_path.name + ".meta.json")
    done: set[str] = set()
    if output_path.exists():
        if resume:
            for record in iter_jsonl(output_path):
                done.add(str(record["id"]))
        elif not overwrite:
            raise FileExistsError(f"{output_path} exists; use resume=True to continue or overwrite=True")
        else:
            output_path.unlink()
    unknown = done - set(ids)
    if unknown:
        raise ValueError(f"{output_path} contains ids that are not in the input, e.g. {sorted(unknown)[:5]}")
    todo = [(i, v) for i, v in zip(ids, videos) if i not in done]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    meta: dict[str, Any] = {
        "format": RUN_FORMAT,
        "prediction_format": PREDICTION_FORMAT,
        "started": started,
        "finished": None,
        "status": "running",
        "input": input_info or {},
        "output": str(output_path),
        "resumed_from": len(done) if resume else 0,
        "batch_size": batch_size,
        "versions": library_versions(),
        "hardware": hardware_info(),
        "video_reader_backend": video_reader_backend(),
        **model.describe(),
    }
    _write_json(meta_path, meta)

    with open(output_path, "a", encoding="utf-8") as out:
        def write(prediction):
            out.write(json.dumps(prediction.to_dict(include_explanation=include_explanation, include_raw=include_raw),
                                 ensure_ascii=False) + "\n")
            out.flush()

        model.predict_batch([v for _, v in todo], [i for i, _ in todo], batch_size=batch_size, on_prediction=write)

    counts: Counter[str] = Counter()
    total = 0
    for record in iter_jsonl(output_path):
        counts[record["status"]] += 1
        total += 1
    meta.update(
        finished=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        status="complete" if total == len(ids) else "incomplete",
        counts=dict(counts),
        num_records=total,
        video_reader_backend=video_reader_backend(),
    )
    _write_json(meta_path, meta)
    return meta


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
