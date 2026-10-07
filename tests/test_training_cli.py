"""`atpo train` dry runs, preflight checks, `atpo export`, and `atpo inspect-state` (no GPU).

The LLaMA-Factory compatibility test reads a LLaMA-Factory checkout without importing it:

    ATPO_LLAMAFACTORY_SRC=/path/to/LLaMA-Factory
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import re
from pathlib import Path

import pytest
import yaml

from atpo.cli import main as cli_main
from atpo.data import Record, build_easyr1_rows, render_task_prompt, write_easyr1_parquet
from atpo.data.prompts import apply_format_template
from atpo.export import export_easyr1_checkpoint, export_hf_model, verify_checksums
from atpo.inference import InferenceConfig, load_preset
from atpo.inference.model import resolve_inference_config
from atpo.rewards.runtime import RewardRuntime, build_settings
from atpo.taxonomy import Taxonomy
from atpo.training.preflight import PreflightError

from helpers import REPO_ROOT

CUSTOM = Taxonomy.from_dict({
    "name": "kitchen-safety", "block_name": "RESULT",
    "categories": [{"id": "K1", "name": "Knife misuse"}, {"id": "K2", "name": "Fire hazard"}],
})


@pytest.fixture
def workspace(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    records = []
    for i, labels in enumerate([["K1"], [], ["K1", "K2"]]):
        (videos / f"v{i}.mp4").write_bytes(b"\x00")
        records.append(Record(f"v{i}", f"v{i}.mp4", tuple(labels)))
    prompt = render_task_prompt(CUSTOM)
    rows = build_easyr1_rows(records, CUSTOM, prompt=prompt)
    write_easyr1_parquet(rows, tmp_path / "train.parquet")
    write_easyr1_parquet(rows[:1], tmp_path / "val.parquet")
    (tmp_path / "taxonomy.json").write_text(json.dumps(CUSTOM.to_dict()))
    (tmp_path / "init").mkdir()
    config = tmp_path / "atpo.yaml"
    config.write_text(yaml.safe_dump({
        "extends": str(REPO_ROOT / "configs" / "easyr1" / "base.yaml"),
        "name": "kitchen_atpo_g",
        "reward": {"taxonomy": str(tmp_path / "taxonomy.json"), "reward": "atpo", "response_format": "think_answer",
                   "parse_scope": "answer_only", "controller": {"mode": "global", "scale": 2.0,
                                                                "invalid_prediction_policy": "as_empty"}},
        "easyr1": {"trainer": {"n_gpus_per_node": 1}},
    }))
    return tmp_path, config, prompt


def _train_args(ws, config, out, *extra):
    return ["train", "--config", str(config), "--output-dir", str(out), "--var", f"TRAIN_FILE={ws / 'train.parquet'}",
            "--var", f"VAL_FILE={ws / 'val.parquet'}", "--var", f"INIT_MODEL={ws / 'init'}", *extra]


def test_train_dry_run_writes_config(workspace):
    ws, config, _ = workspace
    out = ws / "run"
    assert cli_main(_train_args(ws, config, out, "--var", f"VIDEO_ROOT={ws / 'videos'}", "--dry-run",
                                "--set", "trainer.total_epochs=1")) == 0
    composed = yaml.safe_load((out / "easyr1_config.yaml").read_text())
    assert composed["data"]["image_dir"] == str(ws / "videos")
    assert composed["trainer"]["total_epochs"] == 1 and composed["trainer"]["n_gpus_per_node"] == 1
    assert composed["trainer"]["save_checkpoint_path"] == str(out.resolve() / "checkpoints")
    assert composed["worker"]["val_reward"]["reward_function_kwargs"]["update_controller"] is False
    run = json.loads((out / "atpo_run.json").read_text())
    assert run["dry_run"] is True and "verl.trainer.main" in run["command"]


def test_relative_paths_resolve_against_launch_directory(workspace, monkeypatch):
    """The backend runs in the run directory, so relative launch paths are made absolute."""
    ws, config, _ = workspace
    monkeypatch.chdir(ws)
    assert cli_main(["train", "--config", str(config), "--output-dir", "run", "--var", "TRAIN_FILE=train.parquet",
                     "--var", "VAL_FILE=val.parquet", "--var", "INIT_MODEL=init", "--var", "VIDEO_ROOT=videos",
                     "--dry-run"]) == 0
    composed = yaml.safe_load((ws / "run" / "easyr1_config.yaml").read_text())
    assert composed["data"]["train_files"] == str(ws / "train.parquet")
    assert composed["data"]["image_dir"] == str(ws / "videos")
    assert composed["worker"]["actor"]["model"]["model_path"] == str(ws / "init")


def test_preflight_rejects_missing_paths(workspace):
    ws, config, _ = workspace
    with pytest.raises(PreflightError, match="training videos were not found"):
        cli_main(_train_args(ws, config, ws / "run1", "--dry-run"))  # VIDEO_ROOT not set
    with pytest.raises(PreflightError, match="image_dir not found"):
        cli_main(_train_args(ws, config, ws / "run2", "--var", "VIDEO_ROOT=/nonexistent", "--dry-run"))
    bad = ["train", "--config", str(config), "--output-dir", str(ws / "run3"), "--var", "TRAIN_FILE=/no.parquet",
           "--var", f"VAL_FILE={ws / 'val.parquet'}", "--var", "INIT_MODEL=/no/model", "--dry-run"]
    with pytest.raises(PreflightError, match="train_files not found"):
        cli_main(bad)


def test_sft_without_llamafactory_explains_how_to_install(tmp_path, monkeypatch):
    from atpo.configuration import ConfigError

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "dataset_info.json").write_text(json.dumps({"safewatch_sft": {"file_name": "sft.jsonl"}}))
    (tmp_path / "data" / "sft.jsonl").write_text("")
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    with pytest.raises(ConfigError, match="environments/sft.txt"):
        cli_main(["train", "--config", str(REPO_ROOT / "configs" / "sft" / "safewatch_qwen25vl.yaml"),
                  "--output-dir", str(tmp_path / "run"), "--var", f"DATASET_DIR={tmp_path / 'data'}"])


def test_console_reports_user_errors_without_traceback(tmp_path, monkeypatch, capsys):
    from atpo.cli import console_main

    monkeypatch.delenv("ATPO_DEBUG", raising=False)
    missing = ["train", "--config", str(REPO_ROOT / "configs" / "sft" / "safewatch_qwen25vl.yaml"),
               "--output-dir", str(tmp_path / "run"), "--var", f"DATASET_DIR={tmp_path / 'nowhere'}"]
    assert console_main(missing) == 2
    assert capsys.readouterr().err.startswith("atpo: error: ")
    monkeypatch.setenv("ATPO_DEBUG", "1")
    with pytest.raises(PreflightError):
        console_main(missing)


def test_sft_dry_run(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sft.jsonl").write_text("{}\n")
    (tmp_path / "data" / "dataset_info.json").write_text(json.dumps({"safewatch_sft": {"file_name": "sft.jsonl"}}))
    out = tmp_path / "sft"
    assert cli_main(["train", "--config", str(REPO_ROOT / "configs/sft/safewatch_qwen25vl.yaml"), "--output-dir", str(out),
                     "--var", f"DATASET_DIR={tmp_path / 'data'}", "--nproc-per-node", "4", "--dry-run"]) == 0
    composed = yaml.safe_load((out / "llamafactory_train.yaml").read_text())
    assert composed["model_name_or_path"] == "Qwen/Qwen2.5-VL-7B-Instruct" and composed["template"] == "qwen2_vl"
    assert composed["output_dir"] == str(out.resolve() / "lora") and "media_dir" not in composed
    with pytest.raises(PreflightError, match="not registered"):
        cli_main(["train", "--config", str(REPO_ROOT / "configs/sft/xdviolence_qwen25vl.yaml"), "--output-dir",
                  str(out), "--var", f"DATASET_DIR={tmp_path / 'data'}", "--dry-run"])



@pytest.mark.skipif(not os.environ.get("ATPO_LLAMAFACTORY_SRC"), reason="set ATPO_LLAMAFACTORY_SRC to a checkout")
def test_sft_configs_use_known_llamafactory_arguments(tmp_path):
    """Every key of the composed SFT and merge configs is an argument of the given LLaMA-Factory checkout."""
    transformers = pytest.importorskip("transformers")
    src = Path(os.environ["ATPO_LLAMAFACTORY_SRC"]) / "src" / "llamafactory"
    known = {field.name for field in dataclasses.fields(transformers.Seq2SeqTrainingArguments)}
    for path in (src / "hparams").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                known |= {item.target.id for item in node.body
                          if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
    templates = set(re.findall(r'register_template\(\s*name="([^"]+)"',
                               (src / "data" / "template.py").read_text(encoding="utf-8")))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "dataset_info.json").write_text(json.dumps(
        {name: {"file_name": "sft.jsonl"} for name in ("safewatch_sft", "xdviolence_sft_v1_sv1")}))
    for config in sorted((REPO_ROOT / "configs" / "sft").glob("*_qwen*.yaml")):
        out = tmp_path / config.stem
        common = ["--config", str(config), "--var", f"DATASET_DIR={tmp_path / 'data'}"]
        assert cli_main(["train", *common, "--output-dir", str(out), "--dry-run", "--skip-checks"]) == 0
        assert cli_main(["merge-lora", *common, "--adapter", str(out / "lora"),
                         "--output-dir", str(tmp_path / f"{config.stem}_merged"), "--dry-run"]) == 0
        for composed in (out / "llamafactory_train.yaml",
                         tmp_path / f"{config.stem}_merged_merge_run" / "llamafactory_export.yaml"):
            values = yaml.safe_load(composed.read_text())
            assert sorted(set(values) - known) == [], composed
            assert values["template"] in templates

def _fake_checkpoint(root: Path, reward_kwargs: dict) -> Path:
    step = root / "checkpoints" / "global_step_15"
    hf = step / "actor" / "huggingface"
    hf.mkdir(parents=True)
    for name, content in (("config.json", "{}"), ("model.safetensors", "weights"), ("tokenizer_config.json", "{}"),
                          ("preprocessor_config.json", "{}"), ("chat_template.json", "{}")):
        (hf / name).write_text(content)
    runtime = RewardRuntime(build_settings(dict(reward_kwargs)))
    (step / "reward_state.json").write_text(json.dumps(runtime.state_dict()))
    return step


def test_export_derives_training_prompt(workspace, capsys):
    ws, config, prompt = workspace
    out = ws / "run"
    cli_main(_train_args(ws, config, out, "--var", f"VIDEO_ROOT={ws / 'videos'}", "--dry-run"))
    composed = yaml.safe_load((out / "easyr1_config.yaml").read_text())
    step = _fake_checkpoint(out, composed["worker"]["reward"]["reward_function_kwargs"])
    export_dir = ws / "export"
    export_easyr1_checkpoint(step, export_dir, training_run=out)
    config_out = InferenceConfig.from_file(export_dir / "atpo_config.json")
    assert config_out.prompt == apply_format_template(prompt + " <video>", "think_answer")
    assert config_out.video.position == "placeholder" and config_out.system_prompt is None
    assert (config_out.video.min_pixels, config_out.video.max_pixels, config_out.video.fps) == (3136, 50176, 1.0)
    assert config_out.parser.scope == "answer_only" and config_out.processor == "checkpoint"
    assert config_out.taxonomy.fingerprint() == CUSTOM.fingerprint()
    export_info = json.loads((export_dir / "atpo_export.json").read_text())
    assert export_info["source"]["global_step"] == 15 and export_info["controller_at_export"]["mode"] == "global"
    assert verify_checksums(export_dir) == []
    (export_dir / "model.safetensors").write_text("tampered")
    assert verify_checksums(export_dir) == ["model.safetensors"]
    resolved, info = resolve_inference_config(str(export_dir))
    assert info["source"] == "checkpoint" and resolved.fingerprint() == config_out.fingerprint()
    assert cli_main(["inspect-state", str(step)]) == 0
    assert "alpha" in capsys.readouterr().out


def test_export_hf_model_with_preset(tmp_path):
    model = tmp_path / "merged"
    model.mkdir()
    for name in ("config.json", "model.safetensors", "tokenizer_config.json", "preprocessor_config.json"):
        (model / name).write_text("{}")
    export_hf_model(model, tmp_path / "out", preset="safewatch-qwen2.5vl-sft")
    config = InferenceConfig.from_file(tmp_path / "out" / "atpo_config.json")
    assert config.prompt == load_preset("safewatch-qwen2.5vl-sft").prompt and config.processor == "checkpoint"
    assert config.provenance["processor_before_export"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    with pytest.raises(Exception, match="not empty"):
        export_hf_model(model, tmp_path / "out", preset="safewatch-qwen2.5vl-sft")
