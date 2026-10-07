"""Compose and launch EasyR1 (GRPO / ATPO) training runs from release configs.

A training config (``configs/atpo/*.yaml``, ``configs/grpo/*.yaml``) contains:

``easyr1``      EasyR1 configuration tree (usually via ``extends: ../easyr1/base.yaml``);
``reward``      settings for the release reward (``atpo/rewards/easyr1.py``);
``validation``  ``frozen`` (default) or ``historical`` (see below);
``variables``   paths and names supplied at launch time (``--var NAME=VALUE``);
``paper_data``  optional fingerprints of the paper's training and validation files
                (``scripts/prepare_paper_data.sh``); preflight notes when the given files differ.

Validation reward. The paper's runs did not set ``worker.val_reward``, so
validation used its own copy of the adaptive reward whose controller kept
updating on validation batches; the "best" checkpoint was selected with those
drifting coefficients. ``validation: frozen`` (the release default) scores
validation with the same reward settings but ``update_controller: false``, so the
coefficients stay at their initial values and validation scores are comparable
across steps. ``validation: historical`` restores the behaviour of the paper's runs.

The composed EasyR1 config is written to ``<output_dir>/easyr1_config.yaml`` and
run with ``python -m verl.trainer.main config=...`` from the vendored EasyR1
(``third_party/EasyR1``). The exit status of the trainer is returned unchanged.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from ..configuration import (ConfigError, absolute_local_path, expand_variables, load_yaml_with_extends,
                             resolve_variables)
from ..data.prompts import format_template_path
from ..taxonomy import BUILTIN_TAXONOMIES, load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[2]
EASYR1_DIR = REPO_ROOT / "third_party" / "EasyR1"
REWARD_FILE = REPO_ROOT / "atpo" / "rewards" / "easyr1.py"
# Local path settings made absolute at compose time (EasyR1 runs in the run directory);
# True: resolve only if the path exists (otherwise it may be a Hugging Face ID).
LOCAL_PATHS = (("data.train_files", True), ("data.val_files", True), ("data.image_dir", False),
               ("data.video_cache_dir", False), ("worker.actor.model.model_path", True),
               ("worker.actor.model.tokenizer_path", True))
TRAINING_FORMAT = "atpo.training/v1"


@dataclass
class TrainingConfig:
    name: str
    kind: str
    path: Path
    description: str = ""
    easyr1: dict[str, Any] = field(default_factory=dict)
    reward: dict[str, Any] = field(default_factory=dict)
    validation: str = "frozen"
    variables: dict[str, Any] = field(default_factory=dict)
    paper_data: dict[str, Any] = field(default_factory=dict)
    llamafactory: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)


def load_training_config(path: str | Path) -> TrainingConfig:
    path = Path(path)
    data = load_yaml_with_extends(path)
    fmt = data.pop("format", TRAINING_FORMAT)
    if fmt != TRAINING_FORMAT:
        raise ConfigError(f"{path}: unsupported format {fmt!r}")
    kind = data.pop("kind", None)
    if kind not in ("easyr1", "llamafactory"):
        raise ConfigError(f"{path}: kind must be 'easyr1' or 'llamafactory'")
    known = {"name", "description", "easyr1", "reward", "validation", "variables", "paper_data", "llamafactory",
             "export"}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    config = TrainingConfig(name=data.get("name") or path.stem, kind=kind, path=path,
                            description=" ".join(str(data.get("description", "")).split()),
                            easyr1=data.get("easyr1", {}) or {}, reward=data.get("reward", {}) or {},
                            validation=data.get("validation", "frozen"), variables=data.get("variables", {}) or {},
                            paper_data=data.get("paper_data", {}) or {}, llamafactory=data.get("llamafactory", {}) or {},
                            export=data.get("export", {}) or {})
    if kind == "easyr1":
        if not config.reward:
            raise ConfigError(f"{path}: 'reward' is required for EasyR1 training")
        if config.validation not in ("frozen", "historical"):
            raise ConfigError(f"{path}: validation must be 'frozen' or 'historical'")
    return config


def _set(tree: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _get(tree: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = tree
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def compose_easyr1_config(
    config: TrainingConfig,
    *,
    variables: Mapping[str, Any],
    output_dir: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
    validation: Optional[str] = None,
    resume: bool = False,
    loggers: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Return the full EasyR1 config tree for this run (no files are written)."""
    if config.kind != "easyr1":
        raise ConfigError(f"{config.name} is a {config.kind} config")
    values = resolve_variables(config.variables, variables)
    values.setdefault("OUTPUT_DIR", str(Path(output_dir).resolve()))
    tree = expand_variables(copy.deepcopy(config.easyr1), values)
    reward = expand_variables(copy.deepcopy(config.reward), values)
    taxonomy = reward.get("taxonomy")
    if isinstance(taxonomy, str) and taxonomy not in BUILTIN_TAXONOMIES:
        # Inline custom taxonomies so the composed config does not depend on the working
        # directory of the Ray workers (relative paths resolve against the config file).
        path = Path(taxonomy).expanduser()
        if not path.is_absolute():
            path = config.path.parent / path
        reward["taxonomy"] = load_taxonomy(path).to_dict()

    response_format = reward.get("response_format", "think_answer")
    if _get(tree, "data.format_prompt") in (None, "auto"):
        _set(tree, "data.format_prompt", format_template_path(response_format) if response_format != "plain" else None)
    _set(tree, "worker.reward.reward_function", f"{REWARD_FILE}:compute_score")
    _set(tree, "worker.reward.reward_function_kwargs", reward)
    mode = validation or config.validation
    if mode == "frozen":
        _set(tree, "worker.val_reward", {
            "reward_function": f"{REWARD_FILE}:compute_score",
            "reward_function_kwargs": {**reward, "update_controller": False},
        })
    elif mode != "historical":
        raise ConfigError("validation must be 'frozen' or 'historical'")
    if _get(tree, "trainer.save_checkpoint_path") in (None, "auto"):
        _set(tree, "trainer.save_checkpoint_path", str(Path(output_dir).resolve() / "checkpoints"))
    if _get(tree, "trainer.experiment_name") in (None, "auto"):
        _set(tree, "trainer.experiment_name", config.name)
    if loggers is not None:
        _set(tree, "trainer.logger", list(loggers))
    if resume:
        _set(tree, "trainer.find_last_checkpoint", True)
    for dotted, value in (overrides or {}).items():
        _set(tree, dotted, value)
    for dotted, must_exist in LOCAL_PATHS:
        if _get(tree, dotted) is not None:
            _set(tree, dotted, absolute_local_path(_get(tree, dotted), must_exist=must_exist))
    return tree


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True)
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return None


def launch_easyr1(
    config: TrainingConfig,
    tree: Mapping[str, Any],
    *,
    output_dir: str | Path,
    variables: Mapping[str, Any],
    dry_run: bool = False,
    extra_env: Optional[Mapping[str, str]] = None,
) -> int:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "easyr1_config.yaml"
    config_path.write_text(yaml.safe_dump(dict(tree), sort_keys=False), encoding="utf-8")
    command = [sys.executable, "-m", "verl.trainer.main", f"config={config_path}"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(EASYR1_DIR), str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update(extra_env or {})
    record = {
        "format": "atpo.training_run/v1",
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": str(config.path),
        "name": config.name,
        "variables": {k: v for k, v in variables.items()},
        "paper_data": config.paper_data,
        "atpo_commit": _git_commit(),
        "easyr1_config": str(config_path),
        "command": " ".join(shlex.quote(c) for c in command),
        "cwd": str(output_dir),
        "dry_run": dry_run,
    }
    (output_dir / "atpo_run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if dry_run:
        print(f"[dry-run] wrote {config_path}\n[dry-run] would run: {record['command']}")
        return 0
    process = subprocess.run(command, cwd=output_dir, env=env, check=False)
    return process.returncode
