"""Data recipes: build training files from canonical records with one YAML file.

Example (``configs/data/custom_example.yaml``)::

    taxonomy: examples/custom_training/taxonomy.yaml
    splits:
      train: ${DATA_DIR}/train.jsonl
      val: ${DATA_DIR}/val.jsonl
    video_root: ${DATA_DIR}/videos
    validate: {repeated_ids: error, repeated_videos: warn, check_media: exists}
    outputs:
      - kind: easyr1            # EasyR1 parquet (RL)
        split: train
        path: ${OUT_DIR}/easyr1/train.parquet
        prompt: {render: true}  # or {file: safewatch/v3.txt, strip: true} or {path: my_prompt.txt}
      - kind: sharegpt          # LLaMA-Factory JSONL (SFT)
        split: train
        path: ${OUT_DIR}/sft/train.jsonl
        prompt: {render: true}
        target_style: json
        dataset_name: custom_sft   # also writes dataset_info.json in the same directory

``${VAR}`` references are expanded from ``variables`` (command line ``--var``), the
defaults declared in an optional ``variables:`` section
(``NAME: {default: ..., description: ...}``), and the environment; an undefined
variable is an error. Every run writes
``<output>.manifest.json`` next to each output with row counts, label counts, the
prompt hash, and the SHA-256 of inputs and outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..taxonomy import Taxonomy, load_taxonomy
from .builders import (build_easyr1_rows, build_sharegpt_rows, llamafactory_dataset_info, oversample_rows,
                       write_easyr1_parquet, write_jsonl)
from .prompts import read_resource_prompt, render_task_prompt
from .records import check_split_overlap, load_records, validate_records

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class RecipeError(ValueError):
    pass


def expand(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        def sub(match):
            name = match.group(1)
            if name in variables:
                return str(variables[name])
            if name in os.environ:
                return os.environ[name]
            raise RecipeError(f"undefined variable ${{{name}}}; pass --var {name}=...")
        return _VAR.sub(sub, value)
    if isinstance(value, list):
        return [expand(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, variables) for k, v in value.items()}
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_prompt(spec: Mapping[str, Any], taxonomy: Taxonomy, base_dir: Path) -> tuple[str, dict[str, Any]]:
    """Return ``(prompt_text, provenance)`` for a recipe prompt spec."""
    if spec.get("render"):
        text = render_task_prompt(taxonomy, intro=spec.get("intro"), subject=spec.get("subject", "categories"))
        source = {"render": True, "intro": spec.get("intro")}
    elif "file" in spec:
        text = read_resource_prompt(spec["file"], strip=bool(spec.get("strip", False)))
        source = {"file": spec["file"], "strip": bool(spec.get("strip", False))}
    elif "path" in spec:
        path = Path(spec["path"])
        path = path if path.is_absolute() else base_dir / path
        text = path.read_bytes().decode("utf-8")
        if spec.get("strip", False):
            text = text.strip()
        source = {"path": str(path), "strip": bool(spec.get("strip", False))}
    elif "text" in spec:
        text = spec["text"]
        source = {"text": True}
    else:
        raise RecipeError("prompt needs one of render, file, path, or text")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if spec.get("sha256") and spec["sha256"] != digest:
        raise RecipeError(f"prompt sha256 {digest} does not match recipe value {spec['sha256']}")
    return text, {**source, "sha256": digest}


def run_recipe(
    recipe_path: str | Path,
    *,
    variables: Optional[Mapping[str, str]] = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    recipe_path = Path(recipe_path)
    raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    declared = raw.pop("variables", None) or {}
    values = {name: str(spec["default"]) for name, spec in declared.items()
              if isinstance(spec, Mapping) and spec.get("default") is not None}
    values.update(variables or {})
    recipe = expand(raw, values)
    base_dir = recipe_path.parent
    taxonomy_spec = recipe["taxonomy"]
    if isinstance(taxonomy_spec, str) and taxonomy_spec not in ("safewatch", "xdviolence"):
        candidate = Path(taxonomy_spec)
        taxonomy_spec = str(candidate if candidate.is_absolute() or candidate.exists() else base_dir / candidate)
    taxonomy = load_taxonomy(taxonomy_spec)

    splits = {name: load_records(path, taxonomy) for name, path in recipe["splits"].items()}
    validation = recipe.get("validate", {})
    video_root = recipe.get("video_root") or None
    reports = {}
    for name, records in splits.items():
        report = validate_records(
            records, taxonomy, video_root=video_root,
            repeated_ids=validation.get("repeated_ids", "error"),
            repeated_videos=validation.get("repeated_videos", "warn"),
            check_media=validation.get("check_media", "none"),
            require_all_categories=bool(validation.get("require_all_categories", False)),
        )
        reports[name] = report.to_dict()
        report.raise_if_errors()
    overlap = check_split_overlap(splits, video_root=video_root)
    if overlap and validation.get("split_overlap", "error") == "error":
        raise RecipeError("split overlap: " + "; ".join(overlap))

    summary: dict[str, Any] = {"recipe": str(recipe_path), "taxonomy": taxonomy.name,
                               "taxonomy_fingerprint": taxonomy.fingerprint(), "validation": reports,
                               "split_overlap": overlap, "outputs": []}
    for output in recipe.get("outputs", []):
        kind, split, path = output["kind"], output["split"], Path(output["path"])
        if split not in splits:
            raise RecipeError(f"output refers to unknown split {split!r}")
        prompt, prompt_info = resolve_prompt(output["prompt"], taxonomy, base_dir)
        label_format = output.get("label_format", "id")
        prefix = output.get("path_prefix")
        if kind == "easyr1":
            rows = build_easyr1_rows(splits[split], taxonomy, prompt=prompt, label_format=label_format,
                                     path_prefix=prefix)
            for class_index in output.get("oversample", []) or []:
                if label_format != "index":
                    raise RecipeError("oversample requires label_format: index (historical behaviour)")
                rows = oversample_rows(rows, int(class_index))
        elif kind == "sharegpt":
            rows = build_sharegpt_rows(
                splits[split], taxonomy, prompt=prompt, target_style=output["target_style"],
                block_name=output.get("block_name"), label_format=label_format, path_prefix=prefix,
                extra_fields=tuple(output.get("extra_fields", []) or []),
            )
        else:
            raise RecipeError(f"unknown output kind {kind!r}")
        label_key = "response" if kind == "easyr1" else "labels"
        info: dict[str, Any] = {
            "kind": kind, "split": split, "path": str(path), "rows": len(rows),
            "unique_ids": len({r["id"] for r in rows}), "prompt": prompt_info,
            "label_counts": dict(Counter(str(v) for r in rows for v in r[label_key])),
            "oversample": output.get("oversample") or [],
        }
        if not dry_run:
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists; pass overwrite=True")
            if kind == "easyr1":
                write_easyr1_parquet(rows, path)
            else:
                write_jsonl(rows, path)
                if output.get("dataset_name"):
                    info_path = path.parent / "dataset_info.json"
                    existing = json.loads(info_path.read_text()) if info_path.exists() else {}
                    existing[output["dataset_name"]] = llamafactory_dataset_info(path.name)
                    info_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
                    info["dataset_info"] = str(info_path)
            info["sha256"] = _sha256(path)
            info["inputs"] = {name: {"path": str(p), "sha256": _sha256(p)} for name, p in recipe["splits"].items()
                              if name == split}
            Path(str(path) + ".manifest.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        summary["outputs"].append(info)
    return summary
