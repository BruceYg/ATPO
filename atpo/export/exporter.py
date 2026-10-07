"""Export trained checkpoints as self-contained inference directories.

``export_easyr1_checkpoint`` merges an EasyR1 FSDP checkpoint
(``global_step_N/actor``) into Hugging Face weights with the vendored
``scripts/model_merger.py`` (never with its ``--hf_upload_path`` option, which
creates a public repository), copies the result with the processor and tokenizer
files, and writes:

``atpo_config.json``   inference configuration (taxonomy, exact prompt, video and
                       generation settings, parser), see :mod:`atpo.inference.config`;
``atpo_export.json``   provenance: source checkpoint, training configuration, the ATPO
                       controller state at the exported step, tool versions;
``checksums.sha256``   SHA-256 of every exported file.

``export_hf_model`` does the same for a directory that already holds Hugging Face
weights (for example an SFT model merged with ``atpo merge-lora``).

The inference configuration is, in order of precedence: an explicit config file, a
named preset (for the paper's checkpoints, whose evaluation settings differ from
training), or one derived from the training run (prompt and layout exactly as in
training).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..data.prompts import apply_format_template
from ..inference.config import CONFIG_FILENAME, GenerationConfig, InferenceConfig, ParserConfig, VideoConfig
from ..inference.presets import load_preset
from ..taxonomy import load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGER = REPO_ROOT / "third_party" / "EasyR1" / "scripts" / "model_merger.py"
SKIP_FILES = {"checksums.sha256"}


class ExportError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(directory: Path) -> Path:
    lines = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.name not in SKIP_FILES):
        lines.append(f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}")
    out = directory / "checksums.sha256"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def verify_checksums(directory: str | Path) -> list[str]:
    """Return the files whose hashes do not match ``checksums.sha256`` (empty if all match)."""
    directory = Path(directory)
    bad = []
    for line in (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = directory / name
        if not path.exists() or sha256_file(path) != digest:
            bad.append(name)
    return bad


def inference_config_from_training(easyr1_config: Mapping[str, Any], *, max_new_tokens: int = 1024,
                                   base_model: Optional[str] = None) -> InferenceConfig:
    """Inference settings that reproduce the training-time prompt and preprocessing."""
    import pyarrow.parquet as pq

    data = easyr1_config["data"]
    reward = easyr1_config["worker"]["reward"]["reward_function_kwargs"]
    prompts = set(pq.read_table(data["train_files"], columns=["prompt"]).column("prompt").to_pylist())
    if len(prompts) != 1:
        raise ExportError(f"training file has {len(prompts)} distinct prompts; pass an inference config explicitly")
    response_format = reward.get("response_format", "think_answer")
    prompt = apply_format_template(prompts.pop(), response_format)
    position = "placeholder" if "<video>" in prompt else "after"
    taxonomy = load_taxonomy(reward["taxonomy"])
    config = InferenceConfig(
        taxonomy=taxonomy,
        prompt=prompt,
        system_prompt=None,  # EasyR1 training sends no system message
        response_format=response_format,
        video=VideoConfig(fps=float(data.get("video_fps", 2.0)), min_pixels=int(data["min_pixels"]),
                          max_pixels=int(data["max_pixels"]), image_patch_size=14, position=position),
        parser=ParserConfig(output_style="json", scope=reward.get("parse_scope", "full_response")),
        generation=GenerationConfig(max_new_tokens=max_new_tokens),
        base_model=base_model,
        processor="checkpoint",
        description="Derived from the training run: same prompt, message layout, and video settings as training.",
        provenance={"derived_from": "training", "train_file": data["train_files"]},
    )
    return config.validate()


def _resolve_inference_config(*, config_path: Optional[str], preset: Optional[str],
                              easyr1_config: Optional[Mapping[str, Any]], base_model: Optional[str],
                              overrides: Optional[Mapping[str, Any]]) -> InferenceConfig:
    if config_path:
        config = InferenceConfig.from_file(config_path)
    elif preset:
        config = load_preset(preset)
    elif easyr1_config is not None:
        config = inference_config_from_training(easyr1_config, base_model=base_model)
    else:
        raise ExportError("no inference configuration: pass --preset, --inference-config, or --training-run")
    config = config.with_overrides(overrides)
    data = config.to_dict()
    data["processor"] = {"source": "checkpoint", "revision": None, "use_fast": config.processor_use_fast}
    data["provenance"] = {**config.provenance, "processor_before_export": config.processor}
    data.pop("taxonomy_fingerprint", None)
    data["prompt"] = config.prompt
    return InferenceConfig.from_dict(data)


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.iterdir()):
        if path.is_file():
            shutil.copy2(path, dst / path.name)


def _library_versions() -> dict[str, Optional[str]]:
    from ..inference.runner import library_versions

    return library_versions()


def _controller_summary(step_dir: Path) -> Optional[dict[str, Any]]:
    path = step_dir / "reward_state.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    ctrl = state.get("controller")
    summary: dict[str, Any] = {"reward": state.get("reward"), "batches_scored": state.get("batches_scored")}
    if ctrl:
        summary.update(mode=ctrl["mode"], category_ids=ctrl["category_ids"], alpha=ctrl["readable"]["alpha"],
                       beta=ctrl["readable"]["beta"], num_updates=ctrl["num_updates"])
    return summary


def export_easyr1_checkpoint(
    step_dir: str | Path,
    output_dir: str | Path,
    *,
    training_run: Optional[str | Path] = None,
    preset: Optional[str] = None,
    inference_config: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    remerge: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    step_dir = Path(step_dir).resolve()
    actor = step_dir / "actor"
    if not step_dir.name.startswith("global_step_") or not actor.is_dir():
        raise ExportError(f"{step_dir} is not an EasyR1 global_step_* directory with an actor/ folder")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ExportError(f"{output_dir} exists and is not empty")
    easyr1_config = None
    run_record: dict[str, Any] = {}
    if training_run:
        run_path = Path(training_run)
        if run_path.is_dir():
            run_path = run_path / "easyr1_config.yaml"
        easyr1_config = yaml.safe_load(run_path.read_text(encoding="utf-8"))
        record_path = run_path.parent / "atpo_run.json"
        run_record = json.loads(record_path.read_text()) if record_path.exists() else {}
    base_model = None
    if easyr1_config is not None:
        base_model = str(easyr1_config["worker"]["actor"]["model"]["model_path"])
    config = _resolve_inference_config(config_path=inference_config, preset=preset, easyr1_config=easyr1_config,
                                       base_model=base_model, overrides=overrides)

    hf_dir = actor / "huggingface"
    weights = list(hf_dir.glob("*.safetensors")) + list(hf_dir.glob("pytorch_model*.bin"))
    command = [sys.executable, str(MERGER), "--local_dir", str(actor)]
    if dry_run:
        return {"would_run": command if (remerge or not weights) else None, "output_dir": str(output_dir),
                "inference_config": config.to_dict()}
    if remerge or not weights:
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise ExportError(f"model merger failed with exit code {result.returncode}")
    if not (hf_dir / "config.json").exists():
        raise ExportError(f"{hf_dir} has no config.json after merging")
    _copy_tree(hf_dir, output_dir)
    config.save(output_dir / CONFIG_FILENAME)
    provenance = {
        "format": "atpo.export/v1",
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {"kind": "easyr1", "checkpoint": str(step_dir), "global_step": int(step_dir.name.split("_")[-1])},
        "training_run": run_record or None,
        "controller_at_export": _controller_summary(step_dir),
        "inference_config_fingerprint": config.fingerprint(),
        "versions": _library_versions(),
    }
    (output_dir / "atpo_export.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    write_checksums(output_dir)
    return provenance


def export_hf_model(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    preset: Optional[str] = None,
    inference_config: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    processor_source: Optional[str] = None,
) -> dict[str, Any]:
    """Package an existing Hugging Face model directory with an inference config."""
    model_dir = Path(model_dir).resolve()
    if not (model_dir / "config.json").exists():
        raise ExportError(f"{model_dir} has no config.json")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ExportError(f"{output_dir} exists and is not empty")
    config = _resolve_inference_config(config_path=inference_config, preset=preset, easyr1_config=None,
                                       base_model=None, overrides=overrides)
    _copy_tree(model_dir, output_dir)
    if processor_source:
        _copy_processor_files(processor_source, output_dir)
    missing = [n for n in ("preprocessor_config.json", "tokenizer_config.json") if not (output_dir / n).exists()]
    if missing:
        raise ExportError(f"exported model lacks processor files {missing}; pass processor_source")
    config.save(output_dir / CONFIG_FILENAME)
    provenance = {
        "format": "atpo.export/v1",
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {"kind": "huggingface_dir", "path": str(model_dir), "processor_source": processor_source},
        "inference_config_fingerprint": config.fingerprint(),
        "versions": _library_versions(),
    }
    (output_dir / "atpo_export.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    write_checksums(output_dir)
    return provenance


def _copy_processor_files(source: str, output_dir: Path) -> None:
    from ..inference.model import PROCESSOR_PATTERNS, resolve_model_reference

    path, _ = resolve_model_reference(source, allow_patterns=PROCESSOR_PATTERNS)
    import fnmatch

    for file in Path(path).iterdir():
        if file.is_file() and any(fnmatch.fnmatch(file.name, p) for p in PROCESSOR_PATTERNS):
            if file.name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
                continue
            target = output_dir / file.name
            if not target.exists():
                shutil.copy2(file, target)
