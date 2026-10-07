"""Compose and launch LLaMA-Factory SFT runs and LoRA merges from release configs.

An SFT config (``configs/sft/*.yaml``, ``kind: llamafactory``) holds the complete
LLaMA-Factory training arguments under ``llamafactory`` and the merge arguments
under ``export``. LLaMA-Factory is an external dependency (see
``environments/README.md`` for the pinned revision); it is called through
``llamafactory-cli``.
"""

from __future__ import annotations

import copy
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..configuration import ConfigError, absolute_local_path, expand_variables, resolve_variables
from .easyr1 import TrainingConfig, _git_commit


def compose_llamafactory_config(
    config: TrainingConfig,
    *,
    variables: Mapping[str, Any],
    output_dir: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
    loggers: Optional[list[str]] = None,
) -> dict[str, Any]:
    if config.kind != "llamafactory":
        raise ConfigError(f"{config.name} is a {config.kind} config")
    values = resolve_variables(config.variables, variables)
    values.setdefault("OUTPUT_DIR", str(Path(output_dir).resolve()))
    tree = expand_variables(copy.deepcopy(config.llamafactory), values)
    if tree.get("output_dir") in (None, "auto"):
        tree["output_dir"] = str(Path(output_dir).resolve() / "lora")
    if tree.get("media_dir") is None:
        tree.pop("media_dir", None)  # LLaMA-Factory then resolves media against dataset_dir
    if loggers is not None:
        tree["report_to"] = loggers[0] if len(loggers) == 1 else list(loggers)
    tree.update(overrides or {})
    # LLaMA-Factory runs in the run directory, so local paths given at launch are made absolute.
    for key, must_exist in (("dataset_dir", False), ("media_dir", False), ("model_name_or_path", True)):
        if tree.get(key) is not None:
            tree[key] = absolute_local_path(tree[key], must_exist=must_exist)
    return tree


def compose_merge_config(
    config: TrainingConfig,
    *,
    variables: Mapping[str, Any],
    adapter: str,
    export_dir: str,
) -> dict[str, Any]:
    if not config.export:
        raise ConfigError(f"{config.name} has no 'export' section")
    values = resolve_variables(config.variables, variables)
    values.setdefault("OUTPUT_DIR", str(Path(export_dir).resolve().parent))
    tree = expand_variables(copy.deepcopy(config.export), values)
    tree["adapter_name_or_path"] = absolute_local_path(adapter, must_exist=True)
    tree["export_dir"] = absolute_local_path(export_dir, must_exist=False)
    if tree.get("model_name_or_path") is not None:
        tree["model_name_or_path"] = absolute_local_path(tree["model_name_or_path"], must_exist=True)
    return tree


def launch_llamafactory(
    command: str,
    tree: Mapping[str, Any],
    *,
    output_dir: str | Path,
    config: TrainingConfig,
    variables: Mapping[str, Any],
    nproc_per_node: Optional[int] = None,
    dry_run: bool = False,
) -> int:
    """Run ``llamafactory-cli <command> <yaml>`` (command is ``train`` or ``export``)."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"llamafactory_{command}.yaml"
    yaml_path.write_text(yaml.safe_dump(dict(tree), sort_keys=False), encoding="utf-8")
    cli = ["llamafactory-cli", command, str(yaml_path)]
    env = dict(os.environ)
    if command == "train" and nproc_per_node and nproc_per_node > 1:
        env.update(FORCE_TORCHRUN="1", NNODES=env.get("NNODES", "1"), NPROC_PER_NODE=str(nproc_per_node))
    record = {
        "format": "atpo.training_run/v1",
        "config": str(config.path),
        "name": config.name,
        "variables": dict(variables),
        "paper_data": config.paper_data,
        "atpo_commit": _git_commit(),
        "llamafactory_yaml": str(yaml_path),
        "command": " ".join(shlex.quote(c) for c in cli),
        "nproc_per_node": nproc_per_node,
        "dry_run": dry_run,
    }
    (output_dir / f"atpo_{command}_run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if dry_run:
        print(f"[dry-run] wrote {yaml_path}\n[dry-run] would run: {record['command']}")
        return 0
    if shutil.which("llamafactory-cli") is None:
        raise ConfigError("llamafactory-cli not found on PATH; install LLaMA-Factory with "
                          "`pip install -r environments/sft.txt` (see environments/README.md). "
                          f"The composed config is at {yaml_path}")
    return subprocess.run(cli, cwd=output_dir, env=env, check=False).returncode
