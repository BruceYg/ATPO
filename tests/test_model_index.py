"""Named checkpoint references: bundled index, environment and file overrides (no GPU)."""

from __future__ import annotations

import json

import pytest

from atpo.cli import main as cli_main
from atpo.inference import load_preset
from atpo.inference.model import resolve_inference_config, resolve_model
from atpo.inference.model_index import (ModelIndexError, bundled_index, env_var_name, list_models, lookup_model,
                                        resolve_named_model)

from helpers import REPO_ROOT


def test_bundled_index_is_consistent():
    assert (REPO_ROOT / "models" / "index.yaml").read_text() == \
        (REPO_ROOT / "atpo" / "resources" / "models.yaml").read_text()
    index = bundled_index()
    assert index
    for name, entry in index.items():
        assert entry["hub_id"] is None and entry["revision"] is None  # nothing is published yet
        load_preset(entry["inference_preset"])
        if entry["training_config"]:
            assert (REPO_ROOT / entry["training_config"]).is_file(), name


def test_unpublished_name_explains_how_to_configure(monkeypatch):
    monkeypatch.delenv("ATPO_MODELS_FILE", raising=False)
    name = "safewatch-qwen2.5vl-atpo-g"
    monkeypatch.delenv(env_var_name(name), raising=False)
    assert env_var_name(name) == "ATPO_MODEL_SAFEWATCH_QWEN2_5VL_ATPO_G"
    assert lookup_model(name).location is None and lookup_model("no-such-model") is None
    with pytest.raises(ModelIndexError, match="ATPO_MODEL_SAFEWATCH_QWEN2_5VL_ATPO_G"):
        resolve_named_model(name)
    with pytest.raises(ModelIndexError, match="has not been published"):
        cli_main(["show-config", "--model", name])


def test_environment_override_and_fallback_preset(monkeypatch, tmp_path, capsys):
    name = "xdviolence-qwen3vl-sft"
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    monkeypatch.setenv(env_var_name(name), str(model_dir))
    path, info, fallback = resolve_model(name)
    assert path == str(model_dir.resolve()) and fallback == "xdviolence-qwen3vl-sft"
    assert info["model_index"] == {"name": name, "source": f"environment:{env_var_name(name)}"}
    config, config_info = resolve_inference_config(path, fallback_preset=fallback)
    assert config_info["source"] == "model_index_preset:xdviolence-qwen3vl-sft"
    assert config.base_model == "Qwen/Qwen3-VL-8B-Instruct"
    # A checkpoint with its own atpo_config.json takes precedence over the index preset.
    load_preset("xdviolence-qwen2.5vl-sft").save(model_dir / "atpo_config.json")
    config, config_info = resolve_inference_config(path, fallback_preset=fallback)
    assert config_info["source"] == "checkpoint" and config.base_model == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert cli_main(["show-config", "--model", name, "--prompt"]) == 0
    assert "DETECTION" in capsys.readouterr().out


def test_user_file_override(monkeypatch, tmp_path, capsys):
    (tmp_path / "local_ckpt").mkdir()
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  safewatch-qwen2.5vl-atpo-c:\n    path: local_ckpt\n"
        "  my-custom-model:\n    path: local_ckpt\n    inference_preset: safewatch-qwen2.5vl-rl\n"
        "    description: my model\n"
    )
    monkeypatch.setenv("ATPO_MODELS_FILE", str(tmp_path / "models.yaml"))
    entry = lookup_model("safewatch-qwen2.5vl-atpo-c")
    assert entry.location == str((tmp_path / "local_ckpt").resolve()) and entry.status == "configured"
    assert entry.inference_preset == "safewatch-qwen2.5vl-rl"  # kept from the bundled entry
    assert "my-custom-model" in [e.name for e in list_models()]
    assert cli_main(["models", "--json"]) == 0
    listed = {e["name"]: e for e in json.loads(capsys.readouterr().out)}
    assert listed["my-custom-model"]["location"] == str((tmp_path / "local_ckpt").resolve())
    (tmp_path / "bad.yaml").write_text("models:\n  x:\n    hub: typo\n")
    monkeypatch.setenv("ATPO_MODELS_FILE", str(tmp_path / "bad.yaml"))
    with pytest.raises(ModelIndexError, match="unknown keys"):
        lookup_model("x")
