"""Data preparation: validation, builders, recipes, and oversampling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atpo.data import (DataValidationError, Record, build_easyr1_rows, check_split_overlap, load_records,
                       oversample_rows, render_task_prompt, run_recipe, sft_target, validate_records)
from atpo.data.prompts import apply_format_template
from atpo.parsing import parse_response
from atpo.taxonomy import Taxonomy, load_taxonomy

SAFEWATCH = load_taxonomy("safewatch")
XD = load_taxonomy("xdviolence")

CUSTOM = Taxonomy.from_dict({
    "name": "kitchen-safety",
    "block_name": "RESULT",
    "description": "You are a video safety classifier for kitchen footage.",
    "categories": [
        {"id": "K1", "name": "Knife misuse", "description": "Unsafe handling of knives."},
        {"id": "K2", "name": "Fire hazard", "description": "Open flames left unattended or grease fires."},
        {"id": "K3", "name": "Spill", "description": "Liquid on the floor that could cause slipping."},
    ],
})


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_validation_reports_problems(tmp_path):
    good = _write_jsonl(tmp_path / "a.jsonl", [
        {"id": "a", "video": "a.mp4", "labels": ["K1"]},
        {"id": "b", "video": "a.mp4", "labels": []},
        {"id": "a", "video": "c.mp4", "labels": ["K2", "K3"]},
    ])
    records = load_records(good, CUSTOM)
    report = validate_records(records, CUSTOM)
    assert not report.ok and "repeated ids" in report.errors[0]
    assert any("repeated videos" in w for w in report.warnings)
    report = validate_records(records, CUSTOM, repeated_ids="warn", check_media="exists", video_root=tmp_path)
    assert report.missing_media and not report.ok
    bad = _write_jsonl(tmp_path / "b.jsonl", [{"id": "x", "video": "x.mp4", "labels": ["K9"]}])
    with pytest.raises(DataValidationError, match=r"b\.jsonl:1"):
        load_records(bad, CUSTOM)
    dup = _write_jsonl(tmp_path / "c.jsonl", [{"id": "x", "video": "x.mp4", "labels": ["K1", "K1"]}])
    with pytest.raises(DataValidationError, match="duplicate labels"):
        load_records(dup, CUSTOM)
    overlap = check_split_overlap({"train": records, "val": [Record("b", "z.mp4", ())]})
    assert overlap and "id(s) in both train and val" in overlap[0]


def test_rendered_prompt_round_trips_through_parser():
    prompt = render_task_prompt(CUSTOM)
    assert '"K1(Knife misuse)": boolean' in prompt and 'beginning with "RESULT:"' in prompt
    target = sft_target(["K2"], CUSTOM, style="json")
    parsed = parse_response(target, CUSTOM, response_format="plain", output_style="json", scope="full_response")
    assert parsed.status == "ok" and parsed.labels == ["K2"]
    simple = sft_target([], CUSTOM, style="simple")
    assert simple == "RESULT: "
    parsed = parse_response(simple, CUSTOM, response_format="plain", output_style="simple", scope="full_response")
    assert parsed.status == "ok" and parsed.labels == []
    rl_prompt = apply_format_template(prompt + " <video>", "think_answer")
    assert rl_prompt.endswith("<answer> </answer> tags.") and rl_prompt.count("<video>") == 1


def test_oversampling_appends_copies_in_order():
    records = [Record(f"v{i}", f"v{i}.mp4", tuple(SAFEWATCH.resolve_labels(labels)))
               for i, labels in enumerate([[2], [], [1, 2], [3], [2, 6], [2]])]
    records.append(Record("v0__aug_cls2_1", "v0.mp4", ("C2",)))  # forces the counter to advance
    rows = build_easyr1_rows(records, SAFEWATCH, prompt="task", label_format="index")
    out = oversample_rows(rows, 2)
    assert out[:len(rows)] == rows
    copies = out[len(rows):]
    assert [r["video_path"] for r in copies] == ["v0.mp4", "v2.mp4", "v4.mp4", "v5.mp4", "v0.mp4"]
    assert copies[0]["id"] == "v0__aug_cls2_2"           # _1 is already taken
    assert all(r["response"] == orig["response"] for r, orig in zip(copies, [rows[0], rows[2], rows[4], rows[5], rows[6]]))
    assert len({r["id"] for r in out}) == len(out)


def test_recipe_end_to_end(tmp_path):
    (tmp_path / "taxonomy.yaml").write_text(json.dumps(CUSTOM.to_dict()))
    _write_jsonl(tmp_path / "train.jsonl", [
        {"id": "t1", "video": "t1.mp4", "labels": ["K1"]}, {"id": "t2", "video": "t2.mp4", "labels": []},
        {"id": "t3", "video": "t3.mp4", "labels": ["K2", "K3"]},
    ])
    _write_jsonl(tmp_path / "val.jsonl", [{"id": "v1", "video": "v1.mp4", "labels": ["K3"]}])
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "taxonomy: taxonomy.yaml\n"
        "splits:\n  train: ${D}/train.jsonl\n  val: ${D}/val.jsonl\n"
        "outputs:\n"
        "  - {kind: easyr1, split: train, path: '${D}/out/train.parquet', prompt: {render: true}}\n"
        "  - {kind: sharegpt, split: train, path: '${D}/out/sft.jsonl', prompt: {render: true},\n"
        "     target_style: json, dataset_name: kitchen_sft}\n"
    )
    summary = run_recipe(recipe, variables={"D": str(tmp_path)})
    assert [o["rows"] for o in summary["outputs"]] == [3, 3]
    import pyarrow.parquet as pq

    rows = pq.read_table(tmp_path / "out" / "train.parquet").to_pylist()
    assert rows[2]["response"] == ["K2", "K3"] and rows[0]["prompt"].endswith(" <video>")
    info = json.loads((tmp_path / "out" / "dataset_info.json").read_text())
    assert info["kitchen_sft"]["file_name"] == "sft.jsonl"
    manifest = json.loads((tmp_path / "out" / "train.parquet.manifest.json").read_text())
    assert manifest["label_counts"] == {"K1": 1, "K2": 1, "K3": 1} and len(manifest["sha256"]) == 64
    with pytest.raises(FileExistsError):
        run_recipe(recipe, variables={"D": str(tmp_path)})
    with pytest.raises(Exception, match="undefined variable"):
        run_recipe(recipe)
