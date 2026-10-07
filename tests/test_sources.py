"""Dataset adapters, selection, and media filtering on synthetic data."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from atpo.cli import main as cli_main
from atpo.data import load_records
from atpo.data.records import DataValidationError
from atpo.data.sources import (filter_by_media, parse_xdviolence_label_codes, read_id_manifest,
                               safewatch_annotation_records, safewatch_benchmark_records, select_by_manifest,
                               select_records, xdviolence_list_records)
from atpo.taxonomy import load_taxonomy

from helpers import REPO_ROOT

SAFEWATCH = load_taxonomy("safewatch")
XD = load_taxonomy("xdviolence")
MANIFESTS = REPO_ROOT / "configs" / "data" / "manifests"


def _dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ----------------------------------------------------------------- adapters
def test_safewatch_annotation_adapter(tmp_path):
    root = tmp_path / "main_annotation"
    _dump(root / "b_folder" / "full.json", [
        {"id": 1, "video": "dataset/full/b_folder/target/x.mp4"},
        {"id": 2, "video": "dataset/clip/b_folder/target/x_0.mp4"},   # clips are not used
        {"id": 3, "video": "dataset/full/b_folder/target/x.mp4"},     # listed twice upstream: kept twice
    ])
    _dump(root / "b_folder" / "full_gt.json", [
        {"video_path": "dataset/full/b_folder/target/x.mp4", "labels": [2], "subcategories": ["a"]},
        {"video_path": "dataset/full/b_folder/target/x.mp4", "labels": [3, 1], "subcategories": ["b"]},  # wins
    ])
    _dump(root / "a_folder" / "full.json", [{"id": 1, "video": "dataset/full/a_folder/target/y.mp4"}])
    (root / "c_folder").mkdir()  # no full.json: skipped
    records = safewatch_annotation_records(root, SAFEWATCH, video_prefix="train/")
    assert [r.id for r in records] == ["full/a_folder/target/y.mp4", "full/b_folder/target/x.mp4",
                                       "full/b_folder/target/x.mp4"]
    assert records[0].labels == () and records[0].extra == {"subcategories": []}
    assert records[1].labels == ("C1", "C3") and records[1].extra == {"subcategories": ["b"]}
    assert records[1].video == "train/full/b_folder/target/x.mp4"


def test_safewatch_benchmark_adapter(tmp_path):
    root = tmp_path / "bench"
    _dump(root / "real" / "C2" / "benign_benchmark.json", [
        {"video_path": "real/videos/C2/benign_benchmark/1.mp4", "labels": [], "subcategories": []},
        {"video_path": "real/videos/C2/benign_benchmark/2.unknown_video", "labels": [], "subcategories": []},
    ])
    _dump(root / "real" / "C2" / "bully_benchmark.json", [
        {"video_path": "real/videos/C2/bully_benchmark/1.mp4", "labels": [2, 3], "subcategories": ["bully"]},
        {"video_path": "real/videos/C2/bully_benchmark/2.mp4", "labels": [], "subcategories": []},
    ])
    _dump(root / "genai" / "C1" / "benign_benchmark.json", [
        {"video_path": "genai/videos/C1/benign_benchmark/1.mp4", "labels": [], "subcategories": []},
    ])
    with pytest.raises(ValueError, match="benign_labels"):
        safewatch_benchmark_records(root, SAFEWATCH, benign_labels=None)
    historical = safewatch_benchmark_records(root, SAFEWATCH, benign_labels="folder")
    assert [(r.id, r.labels) for r in historical] == [
        ("genai/C1/benign_benchmark/1.mp4", ("C1",)),      # genai sorts before real
        ("real/C2/benign_benchmark/1.mp4", ("C2",)),       # historical: folder category for benign videos
        ("real/C2/bully_benchmark/1.mp4", ("C2", "C3")),
        ("real/C2/bully_benchmark/2.mp4", ("C2",)),        # no annotation: folder category
    ]
    assert historical[1].extra == {"subcategories": ["benign"], "harmfulness": 0}
    corrected = safewatch_benchmark_records(root, SAFEWATCH, benign_labels="empty")
    assert [r.labels for r in corrected] == [(), (), ("C2", "C3"), ("C2",)]


def test_xdviolence_list_adapter(tmp_path):
    assert parse_xdviolence_label_codes("B4-G-0") == ["B3", "B4"]
    assert parse_xdviolence_label_codes("A") == [] and parse_xdviolence_label_codes("B6-B1-0") == ["B1", "B6"]
    lists = tmp_path / "list.txt"
    lists.write_text("1-1004/Movie__#00-00-01_00-00-09_label_B2-G-0\n\ntest_videos/Other_label_A\n")
    by_id = xdviolence_list_records(lists, XD, video_path="id", video_prefix="train/")
    assert [(r.id, r.video, r.labels) for r in by_id] == [
        ("1-1004/Movie__#00-00-01_00-00-09_label_B2-G-0.mp4", "train/1-1004/Movie__#00-00-01_00-00-09_label_B2-G-0.mp4",
         ("B2", "B3")),
        ("test_videos/Other_label_A.mp4", "train/test_videos/Other_label_A.mp4", ()),
    ]
    by_name = xdviolence_list_records(lists, XD, video_path="basename")
    assert by_name[1].video == "Other_label_A.mp4" and by_name[1].id == "test_videos/Other_label_A.mp4"
    lists.write_text("no-marker-here\n")
    with pytest.raises(DataValidationError, match="_label_"):
        xdviolence_list_records(lists, XD, video_path="id")


# ---------------------------------------------------------------- selection
def test_selection_procedures():
    from atpo.data import Record

    records = [Record(f"v{i}", f"v{i}.mp4", ()) for i in range(50)]
    random.seed(123)
    expected = random.sample(list(records), 10)  # SafeWatch validation sampling of the paper
    assert select_records(records, 10, seed=123, method="random_sample") == expected
    expected_idx = sorted(random.Random(42).sample(range(50), 10))  # XD-Violence validation sampling
    assert select_records(records, 10, seed=42, method="sorted_index_sample") == [records[i] for i in expected_idx]
    assert select_records(records, 80, seed=1, method="random_sample") == records
    repeated = records[:3] + [records[1]]
    assert select_by_manifest(repeated, ["v1", "v1"]) == [records[1], records[1]]
    with pytest.raises(DataValidationError, match="absent"):
        select_by_manifest(records, ["v1", "missing"])
    with pytest.raises(DataValidationError, match="order"):
        select_by_manifest(records, ["v2", "v1"])


# ------------------------------------------------------------- media filter
def _make_video(path: Path, frames: int, rate: int = 8) -> None:
    av = pytest.importorskip("av")
    import numpy as np

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=rate)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    for t in range(frames):
        image = np.full((48, 64, 3), t * 7 % 255, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_media_filter_semantics(tmp_path, monkeypatch):
    pytest.importorskip("decord")
    from atpo.data import Record
    from atpo.data import sources

    for name, frames in (("two_s.mp4", 16), ("three_s.mp4", 24), ("one_frame.mp4", 1)):
        _make_video(tmp_path / name, frames)
    records = [Record("a", "two_s.mp4", ()), Record("b", "three_s.mp4", ("C1",)), Record("c", "one_frame.mp4", ()),
               Record("d", "missing.mp4", ()), Record("e", "x.unknown_video", ()), Record("a", "two_s.mp4", ())]
    cache = tmp_path / "probe.json"
    kept, report = filter_by_media(records, tmp_path, max_duration=2.0, min_frames=2, on_unreadable="drop",
                                   cache_file=cache)
    assert [r.id for r in kept] == ["a", "a"]  # 2.0 s is kept (inclusive bound); repeats stay together
    reasons = {d["id"]: d["reason"] for d in report.dropped}
    assert reasons == {"b": "too_long", "c": "too_few_frames", "d": "unreadable", "e": "suffix"}
    assert report.to_dict()["dropped_by_reason"] == {"too_long": 1, "too_few_frames": 1, "unreadable": 1, "suffix": 1}
    with pytest.raises(DataValidationError, match="missing.mp4"):
        filter_by_media(records, tmp_path, max_duration=2.0, on_unreadable="error")

    def no_decoding(path):
        raise AssertionError(f"{path} should come from the cache")

    monkeypatch.setattr(sources, "_probe_decord", no_decoding)
    cached, cached_report = filter_by_media(records, tmp_path, max_duration=2.0, min_frames=2, on_unreadable="drop",
                                            cache_file=cache)
    assert cached == kept and cached_report.to_dict() == report.to_dict()


def test_cli_source_chain(tmp_path):
    lists = tmp_path / "list.txt"
    lists.write_text("".join(f"dir/clip{i}_label_{'B1-0-0' if i % 2 else 'A'}\n" for i in range(10)))
    out = tmp_path / "all.jsonl"
    assert cli_main(["import-data", "--xdviolence-list", str(lists), "--taxonomy", "xdviolence",
                     "--video-path", "basename", "--output", str(out)]) == 0
    manifest = json.loads((tmp_path / "all.jsonl.manifest.json").read_text())
    assert manifest["records"] == 10 and manifest["benign"] == 5 and manifest["source"]["kind"] == "xdviolence-list"
    with pytest.raises(SystemExit):
        cli_main(["import-data", "--xdviolence-list", str(lists), "--taxonomy", "xdviolence", "--output", str(out)])
    assert cli_main(["select", "--input", str(out), "--taxonomy", "xdviolence", "--n", "4", "--seed", "42",
                     "--method", "sorted_index_sample", "--output", str(tmp_path / "sub.jsonl"),
                     "--write-manifest", str(tmp_path / "ids.txt")]) == 0
    ids = read_id_manifest(tmp_path / "ids.txt")
    assert len(ids) == 4 and cli_main(["select", "--input", str(out), "--taxonomy", "xdviolence", "--ids-file",
                                       str(tmp_path / "ids.txt"), "--output", str(tmp_path / "again.jsonl")]) == 0
    assert [r.id for r in load_records(tmp_path / "again.jsonl", XD)] == ids


def test_manifests_are_well_formed():
    manifests = {p.name: read_id_manifest(p) for p in sorted(MANIFESTS.glob("*.txt"))}
    assert set(manifests) == {"safewatch_train.txt", "safewatch_val.txt", "xdviolence_train.txt", "xdviolence_val.txt"}
    for name, ids in manifests.items():
        assert ids and all(i.endswith(".mp4") and i == i.strip() for i in ids), name
        assert (MANIFESTS / name).read_text(encoding="utf-8").startswith("# ATPO split manifest"), name
    for dataset in ("safewatch", "xdviolence"):
        val = manifests[f"{dataset}_val.txt"]
        assert len(set(val)) == len(val)
        assert not set(val) & set(manifests[f"{dataset}_train.txt"])
    assert len(set(manifests["xdviolence_train.txt"])) == len(manifests["xdviolence_train.txt"])
