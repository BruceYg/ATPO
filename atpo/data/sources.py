"""Canonical records from the datasets' own annotation files, plus selection and media filtering.

The adapters reproduce the conversion scripts used for the paper: the same records,
order, IDs, and label mapping. With :func:`filter_by_media` and :func:`select_records`
they rebuild the paper's training and validation files (``scripts/prepare_paper_data.sh``).

``safewatch_annotation_records``  SafeWatch-Bench-200K ``main_annotation/<folder>/{full,full_gt}.json``
``safewatch_benchmark_records``   SafeWatch-Bench ``{genai,real}/C*/<subcategory>_benchmark.json``
``xdviolence_list_records``       XD-Violence list files (``<name>_label_<codes>`` per line)

``video`` in each record is the path inside the dataset, prefixed by ``video_prefix``
so that it can be resolved against one video root per dataset.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Iterable, Literal, Optional, Sequence

from ..taxonomy import Taxonomy
from .records import DataValidationError, Record

BenignLabels = Literal["folder", "empty"]
SelectMethod = Literal["random_sample", "sorted_index_sample"]


# ------------------------------------------------------------------ SafeWatch-200K
def safewatch_annotation_records(
    annotation_dir: str | Path,
    taxonomy: Taxonomy,
    *,
    video_type: Literal["full", "clip", "all"] = "full",
    video_prefix: str = "",
) -> list[Record]:
    """Training records from SafeWatch-Bench-200K ``main_annotation``.

    As ``filter_training_data.py``: folders in sorted order; within a folder, the
    order of ``full.json``; labels and subcategories from the folder's
    ``full_gt.json`` keyed by video path (a later duplicate replaces an earlier
    one); the leading ``dataset/`` is removed from the path, which is also the ID.
    Repeated videos are kept.
    """
    records = []
    for folder in sorted(p for p in Path(annotation_dir).iterdir() if p.is_dir()):
        data_file, gt_file = folder / "full.json", folder / "full_gt.json"
        if not data_file.exists():
            continue
        ground_truth: dict[str, dict[str, Any]] = {}
        if gt_file.exists():
            for item in json.loads(gt_file.read_text(encoding="utf-8")):
                ground_truth[item.get("video_path", "")] = item
        for datapoint in json.loads(data_file.read_text(encoding="utf-8")):
            video = datapoint.get("video", "")
            if video_type != "all" and not video.startswith(f"dataset/{video_type}/"):
                continue
            gt = ground_truth.get(video, {})
            path = video[len("dataset/"):] if video.startswith("dataset/") else video
            records.append(Record(
                id=path,
                video=video_prefix + path,
                labels=tuple(taxonomy.resolve_labels(gt.get("labels", []))),
                extra={"subcategories": list(gt.get("subcategories", []))},
            ))
    return records


# --------------------------------------------------------------- SafeWatch-Bench
def _folder_category(path: str) -> list[str]:
    for part in path.split("/"):
        if part.startswith("C") and len(part) == 2 and part[1].isdigit():
            return [part]
    return []


def _path_subcategory(path: str) -> list[str]:
    for part in path.split("/"):
        if part.endswith("_benchmark"):
            return [part.replace("_benchmark", "")]
    return []


def safewatch_benchmark_records(
    benchmark_dir: str | Path,
    taxonomy: Taxonomy,
    *,
    benign_labels: BenignLabels,
    sources: Sequence[str] = ("real", "genai"),
    video_prefix: str = "",
    skip_unknown_video: bool = True,
) -> list[Record]:
    """Evaluation records from SafeWatch-Bench (real and generated videos).

    As the paper's conversion script: all JSON files under the
    sources, sorted by path (``genai`` before ``real``); the ``videos`` directory is
    removed from each path; labels come from the annotation, or from the ``C<k>``
    folder when the annotation has none.

    ``benign_labels`` must be chosen explicitly. The benign videos of the benchmark
    have no annotated category, so the paper's rule gives them their folder's
    category (``"folder"``; used for the validation file of the paper's RL runs).
    ``"empty"`` labels a video whose subcategory is exactly ``["benign"]`` as benign
    (the labels the paper's evaluation used).
    """
    if benign_labels not in ("folder", "empty"):
        raise ValueError("benign_labels must be 'folder' (historical) or 'empty'")
    root = Path(benchmark_dir)
    files: list[Path] = []
    for source in sources:
        if (root / source).exists():
            files.extend((root / source).rglob("*.json"))
    records = []
    for file in sorted(files):
        data = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            values = list(data.values())
            items = values[0] if len(data) == 1 and isinstance(values[0], list) else [data]
        else:
            items = data
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_path = item.get("video_path") or item.get("videoPath", item.get("video", ""))
            if not raw_path:
                continue
            parts = raw_path.split("/")
            if "videos" in parts:
                parts.remove("videos")
            path = "/".join(parts)
            if skip_unknown_video and path.endswith(".unknown_video"):
                continue
            annotated = item.get("labels", [])
            labels = list(annotated) if isinstance(annotated, list) and annotated else _folder_category(path)
            annotated_sub = item.get("subcategories", [])
            subcategories = list(annotated_sub) if isinstance(annotated_sub, list) and annotated_sub \
                else _path_subcategory(path)
            if benign_labels == "empty" and subcategories == ["benign"]:
                labels = []
            records.append(Record(
                id=path,
                video=video_prefix + path,
                labels=tuple(taxonomy.resolve_labels(labels)),
                extra={"subcategories": subcategories, "harmfulness": 0 if "benign" in path.lower() else 1},
            ))
    return records


# -------------------------------------------------------------------- XD-Violence
XD_LABEL_ORDER = ("B1", "B2", "B3", "B4", "B5", "B6")


def parse_xdviolence_label_codes(label_part: str) -> list[str]:
    """``"B1-G-0"`` -> ``["B1", "B3"]``: G is Explosion (B3); A (normal) and 0 are dropped."""
    codes = [code.strip() for code in label_part.split("-") if code.strip()]
    codes = ["B3" if code == "G" else code for code in codes]
    return sorted((code for code in codes if code in XD_LABEL_ORDER), key=XD_LABEL_ORDER.index)


def xdviolence_list_records(
    list_file: str | Path,
    taxonomy: Taxonomy,
    *,
    video_path: Literal["id", "basename"],
    video_prefix: str = "",
) -> list[Record]:
    """Records from an XD-Violence list file, one ``<name>_label_<codes>`` per line.

    The ID is the line plus ``.mp4``. The paper's training file stores videos under
    the full ID (``video_path="id"``), the validation file under the file name only
    (``"basename"``).
    """
    records = []
    for line_no, raw in enumerate(Path(list_file).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if "_label_" not in line:
            raise DataValidationError(f"{list_file}:{line_no}: missing '_label_' in {line!r}")
        _, label_part = line.split("_label_", 1)
        video_id = line if line.endswith(".mp4") else f"{line}.mp4"
        video = video_id if video_path == "id" else Path(video_id).name
        records.append(Record(id=video_id, video=video_prefix + video,
                              labels=tuple(taxonomy.resolve_labels(parse_xdviolence_label_codes(label_part)))))
    return records


# ----------------------------------------------------------------------- selection
def select_records(records: Sequence[Record], n: int, *, seed: int, method: SelectMethod) -> list[Record]:
    """Random subset, with the two sampling procedures of the historical scripts.

    ``random_sample``        ``random.seed(seed); random.sample(records, n)``; result in
                             sampling order (SafeWatch validation file). No sampling if
                             ``n >= len(records)``.
    ``sorted_index_sample``  ``sorted(Random(seed).sample(range(len), n))``; result in
                             input order (XD-Violence validation file).
    """
    if method == "random_sample":
        return list(records) if n >= len(records) else Random(seed).sample(list(records), n)
    if method == "sorted_index_sample":
        indices = sorted(Random(seed).sample(range(len(records)), min(n, len(records))))
        return [records[i] for i in indices]
    raise ValueError(f"unknown selection method {method!r}")


def read_id_manifest(path: str | Path) -> list[str]:
    """IDs, one per line, in order (repeats allowed); blank lines and '#' comments are ignored."""
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            ids.append(line)
    return ids


def write_id_manifest(records: Iterable[Record], path: str | Path, *, header: Sequence[str] = ()) -> None:
    lines = [f"# {h}" for h in header] + [r.id for r in records]
    if any("\n" in line for line in lines):
        raise DataValidationError("IDs must not contain newlines")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_by_manifest(records: Sequence[Record], ids: Sequence[str]) -> list[Record]:
    """Keep the records whose IDs are listed, in record order; the result must reproduce
    the manifest exactly (same IDs, same order, same repeats)."""
    wanted = set(ids)
    kept = [r for r in records if r.id in wanted]
    if [r.id for r in kept] != list(ids):
        kept_ids = [r.id for r in kept]
        missing = wanted.difference(kept_ids)
        detail = f"{len(missing)} listed IDs are absent" if missing else "order or repeats differ"
        raise DataValidationError(f"records do not match the manifest ({detail}; {len(kept)} vs {len(ids)} rows)")
    return kept


# -------------------------------------------------------------------- media filter
@dataclass
class MediaProbe:
    frames: Optional[int] = None
    fps: Optional[float] = None
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        if self.error is None and self.frames and self.fps:
            return self.frames / self.fps
        return None


@dataclass
class MediaFilterReport:
    kept: int = 0
    dropped: list[dict[str, Any]] = field(default_factory=list)
    criteria: dict[str, Any] = field(default_factory=dict)
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for item in self.dropped:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        return {"criteria": self.criteria, "backend": self.backend, "kept": self.kept,
                "dropped": len(self.dropped), "dropped_by_reason": reasons, "dropped_records": self.dropped}


def _probe_decord(path: str) -> MediaProbe:
    from decord import VideoReader

    try:
        reader = VideoReader(path)
        frames, fps = len(reader), float(reader.get_avg_fps())
    except Exception as exc:  # decord raises its own error types
        return MediaProbe(error=f"{type(exc).__name__}: {exc}")
    if not (fps > 0 and frames > 0):
        return MediaProbe(frames=frames, fps=fps, error=f"invalid fps={fps} or frame_count={frames}")
    return MediaProbe(frames=frames, fps=fps)


def _probe_opencv(path: str) -> MediaProbe:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return MediaProbe(error="OpenCV cannot open file")
    frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not (fps > 0 and frames > 0):
        return MediaProbe(frames=frames, fps=fps, error=f"invalid fps={fps} or frame_count={frames}")
    return MediaProbe(frames=frames, fps=fps)


def _probe_backend(name: str):
    if name == "decord":
        import decord  # noqa: F401  (fail early if missing)

        return _probe_decord
    if name == "opencv":
        import cv2  # noqa: F401

        return _probe_opencv
    raise ValueError("backend must be 'decord' (historical) or 'opencv'")


def _cache_key(path: str) -> Optional[str]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return f"{path}|{stat.st_size}|{int(stat.st_mtime)}"


def filter_by_media(
    records: Sequence[Record],
    video_root: str | Path,
    *,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
    min_frames: Optional[int] = None,
    drop_suffixes: Sequence[str] = (".unknown_video",),
    on_unreadable: Literal["drop", "error"] = "error",
    backend: str = "decord",
    workers: int = 8,
    cache_file: Optional[str | Path] = None,
) -> tuple[list[Record], MediaFilterReport]:
    """Keep records whose video satisfies the duration and frame-count bounds.

    Matches the paper's duration and frame filters: duration is ``frames / average fps`` from decord;
    a video is kept if ``min_duration <= duration <= max_duration`` (both inclusive)
    and ``frames >= min_frames``; files ending in ``drop_suffixes`` are removed; all
    records of one video are kept or dropped together, in input order. The
    historical scripts dropped missing or unreadable videos silently;
    ``on_unreadable="drop"`` reproduces that and lists them in the report, ``"error"``
    stops instead. ``cache_file`` stores probe results (keyed by path, size, and
    modification time) for later runs.
    """
    probe = _probe_backend(backend)
    root = Path(video_root)
    cache: dict[str, dict[str, Any]] = {}
    if cache_file and Path(cache_file).exists():
        cache = json.loads(Path(cache_file).read_text(encoding="utf-8"))
    results: dict[str, MediaProbe] = {}
    pending = []
    for record in records:
        if any(record.video.endswith(s) for s in drop_suffixes):
            continue
        path = str(root / record.video)
        if path in results:
            continue
        key = _cache_key(path)
        if key is None:
            results[path] = MediaProbe(error="file not found")
        elif key in cache:
            results[path] = MediaProbe(**cache[key])
        else:
            results[path] = MediaProbe()
            pending.append((path, key))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (path, key), result in zip(pending, pool.map(lambda item: probe(item[0]), pending)):
            results[path] = result
            cache[key] = {"frames": result.frames, "fps": result.fps, "error": result.error}
    if cache_file and pending:
        Path(cache_file).write_text(json.dumps(cache), encoding="utf-8")

    report = MediaFilterReport(backend=backend, criteria={
        "max_duration": max_duration, "min_duration": min_duration, "min_frames": min_frames,
        "drop_suffixes": list(drop_suffixes), "on_unreadable": on_unreadable})
    kept = []
    for record in records:
        info: dict[str, Any] = {"id": record.id, "video": record.video}
        if any(record.video.endswith(s) for s in drop_suffixes):
            report.dropped.append({**info, "reason": "suffix"})
            continue
        result = results[str(root / record.video)]
        info.update(frames=result.frames, fps=result.fps)
        if result.error is not None:
            if on_unreadable == "error":
                raise DataValidationError(f"cannot read {root / record.video}: {result.error}")
            report.dropped.append({**info, "reason": "unreadable", "error": result.error})
            continue
        duration = result.duration
        info["duration"] = duration
        if max_duration is not None and duration > max_duration:
            report.dropped.append({**info, "reason": "too_long"})
        elif min_duration is not None and duration < min_duration:
            report.dropped.append({**info, "reason": "too_short"})
        elif min_frames is not None and result.frames < min_frames:
            report.dropped.append({**info, "reason": "too_few_frames"})
        else:
            kept.append(record)
    report.kept = len(kept)
    return kept, report
