"""Named inference presets for checkpoints that have no ``atpo_config.json``.

Each preset reproduces a historical evaluation setting (prompt bytes, system prompt,
video sampling, parser). Prompt files are read from ``atpo/resources/prompts`` and
checked against the SHA-256 recorded in the preset.
"""

from __future__ import annotations

import hashlib
from importlib import resources
from typing import Any

import yaml

from .config import InferenceConfig

_PRESET_PACKAGE = "atpo.resources.presets"
_PROMPT_PACKAGE = "atpo.resources.prompts"


def list_presets() -> dict[str, str]:
    """Return ``{name: description}`` for all bundled presets."""
    out = {}
    for entry in sorted(resources.files(_PRESET_PACKAGE).iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".yaml"):
            data = yaml.safe_load(entry.read_text(encoding="utf-8"))
            out[data["name"]] = " ".join(str(data.get("description", "")).split())
    return out


def read_prompt_file(relpath: str) -> str:
    """Read a bundled prompt file exactly (no newline translation)."""
    node = resources.files(_PROMPT_PACKAGE)
    for part in relpath.split("/"):
        node = node / part
    return node.read_bytes().decode("utf-8")


def preset_dict(name: str) -> dict[str, Any]:
    path = resources.files(_PRESET_PACKAGE) / f"{name}.yaml"
    if not path.is_file():
        raise KeyError(f"unknown preset {name!r}; available: {', '.join(list_presets())}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("name") != name:
        raise ValueError(f"preset file {name}.yaml declares name {data.get('name')!r}")
    return data


def load_preset(name: str) -> InferenceConfig:
    data = preset_dict(name)
    prompt_spec = data.pop("prompt")
    text = read_prompt_file(prompt_spec["file"])
    if prompt_spec.get("strip", False):
        text = text.strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if prompt_spec.get("sha256") and digest != prompt_spec["sha256"]:
        raise ValueError(f"preset {name}: prompt {prompt_spec['file']} has sha256 {digest}, expected {prompt_spec['sha256']}")
    data.pop("name")
    data["preset"] = name
    data["prompt"] = text
    provenance = dict(data.get("provenance") or {})
    provenance["prompt_file"] = prompt_spec["file"]
    provenance["prompt_stripped"] = bool(prompt_spec.get("strip", False))
    data["provenance"] = provenance
    return InferenceConfig.from_dict(data)
