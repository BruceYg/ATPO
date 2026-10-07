"""Release training configs compose into valid EasyR1 configurations."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from atpo.configuration import ConfigError
from atpo.rewards import RewardRuntime
from atpo.rewards.runtime import build_settings
from atpo.training import compose_easyr1_config, load_training_config

from helpers import REPO_ROOT

CONFIGS = sorted((REPO_ROOT / "configs" / "atpo").glob("*.yaml")) + sorted((REPO_ROOT / "configs" / "grpo").glob("*.yaml"))
VARS = {"TRAIN_FILE": "train.parquet", "VAL_FILE": "val.parquet", "INIT_MODEL": "init"}


def _recipe_outputs() -> set[str]:
    outputs = set()
    for recipe in (REPO_ROOT / "configs" / "data").glob("*_paper.yaml"):
        for output in yaml.safe_load(recipe.read_text())["outputs"]:
            outputs.add(output["path"].removeprefix("${OUT}/"))
    return outputs


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_composes(path, tmp_path):
    config = load_training_config(path)
    composed = compose_easyr1_config(config, variables=VARS, output_dir=tmp_path)
    assert Path(composed["data"]["format_prompt"]).is_file()
    RewardRuntime(build_settings(dict(config.reward)))
    # Frozen validation reward by default; historical mode leaves validation to the train reward settings.
    assert composed["worker"]["val_reward"]["reward_function_kwargs"]["update_controller"] is False
    historical = compose_easyr1_config(config, variables=VARS, output_dir=tmp_path, validation="historical")
    assert "val_reward" not in historical["worker"]
    # The paper's data files are named after outputs of the data recipes.
    outputs = _recipe_outputs()
    assert config.paper_data["train_file"] in outputs and config.paper_data["val_file"] in outputs
    assert len(config.paper_data["train_fingerprint"]) == len(config.paper_data["val_fingerprint"]) == 64


def test_undefined_variable_is_an_error(tmp_path):
    config = load_training_config(CONFIGS[0])
    with pytest.raises(ConfigError, match="INIT_MODEL"):
        compose_easyr1_config(config, variables={"TRAIN_FILE": "a", "VAL_FILE": "b"}, output_dir=tmp_path)
    with pytest.raises(ConfigError, match="unknown variable"):
        compose_easyr1_config(config, variables={**VARS, "TYPO": "x"}, output_dir=tmp_path)


def test_composed_config_is_accepted_by_easyr1(tmp_path):
    """The composed YAML merges into EasyR1's structured PPOConfig without errors."""
    pytest.importorskip("omegaconf")
    import sys

    sys.path.insert(0, str(REPO_ROOT / "third_party" / "EasyR1"))
    try:
        from omegaconf import OmegaConf
        from verl.trainer.config import PPOConfig
    except Exception as exc:  # pragma: no cover - depends on the training environment
        pytest.skip(f"EasyR1 config classes not importable: {exc}")
    finally:
        sys.path.pop(0)
    for path in CONFIGS:
        composed = compose_easyr1_config(load_training_config(path), variables=VARS, output_dir=tmp_path)
        merged = OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.create(composed))
        ppo = OmegaConf.to_object(merged)
        assert ppo.worker.val_reward.reward_function_kwargs["update_controller"] is False
