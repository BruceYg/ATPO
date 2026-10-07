"""Named checkpoint references (``atpo models``).

The bundled index (``atpo/resources/models.yaml``, mirrored at ``models/index.yaml``)
names the checkpoints described in the paper. Until they are published every entry
has ``hub_id: null``. A name can be pointed at a local directory or a (private)
repository without editing the package:

``ATPO_MODEL_<NAME>``           path or repository ID (NAME upper-cased, any character
                                other than a letter or digit replaced by ``_``);
``ATPO_MODEL_<NAME>_REVISION``  revision for a repository ID;
``ATPO_MODELS_FILE``            YAML file with the same layout as the index; its entries
                                (``path`` or ``hub_id``, ``revision``, ...) are merged
                                over the bundled ones.

Resolution order: environment variable, user file, bundled index. An entry's
``inference_preset`` is used only when the checkpoint has no ``atpo_config.json``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Optional

import yaml

INDEX_FORMAT = "atpo.model_index/v1"
MODELS_FILE_ENV = "ATPO_MODELS_FILE"


class ModelIndexError(ValueError):
    pass


@dataclass(frozen=True)
class ModelEntry:
    name: str
    location: Optional[str]
    revision: Optional[str]
    inference_preset: Optional[str]
    base_model: Optional[str]
    description: str
    status: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def env_var_name(name: str) -> str:
    return "ATPO_MODEL_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def _read_index(text: str, origin: str) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(text) or {}
    if data.get("format", INDEX_FORMAT) != INDEX_FORMAT:
        raise ModelIndexError(f"{origin}: unsupported format {data.get('format')!r}")
    models = data.get("models") or {}
    if not isinstance(models, dict):
        raise ModelIndexError(f"{origin}: 'models' must be a mapping")
    for name, entry in models.items():
        if not isinstance(entry, dict):
            raise ModelIndexError(f"{origin}: entry {name!r} must be a mapping")
        unknown = set(entry) - {"description", "hub_id", "path", "revision", "base_model", "inference_preset",
                                "training_config", "status"}
        if unknown:
            raise ModelIndexError(f"{origin}: entry {name!r} has unknown keys {sorted(unknown)}")
    return models


def bundled_index() -> dict[str, dict[str, Any]]:
    text = resources.files("atpo.resources").joinpath("models.yaml").read_text(encoding="utf-8")
    return _read_index(text, "bundled model index")


def _user_index() -> tuple[dict[str, dict[str, Any]], Optional[str]]:
    path = os.environ.get(MODELS_FILE_ENV)
    if not path:
        return {}, None
    file = Path(path).expanduser()
    if not file.is_file():
        raise ModelIndexError(f"{MODELS_FILE_ENV}={path} does not exist")
    return _read_index(file.read_text(encoding="utf-8"), str(file)), str(file)


def lookup_model(name: str) -> Optional[ModelEntry]:
    """Return the entry for ``name``, or None if no index or variable defines it."""
    bundled = bundled_index()
    user, user_file = _user_index()
    env_location = os.environ.get(env_var_name(name))
    if name not in bundled and name not in user and env_location is None:
        return None
    entry = {**bundled.get(name, {})}
    source = "bundled" if name in bundled else None
    if name in user:
        override = user[name]
        if "path" in override or "hub_id" in override:
            entry.pop("path", None)
            entry.pop("hub_id", None)
            entry.pop("revision", None)
        entry.update(override)
        source = f"file:{user_file}"
    location = entry.get("path") or entry.get("hub_id")
    revision = entry.get("revision")
    if env_location is not None:
        location, revision, source = env_location, os.environ.get(env_var_name(name) + "_REVISION"), \
            f"environment:{env_var_name(name)}"
    if location is not None and entry.get("path") and source.startswith("file:") and not Path(location).is_absolute():
        location = str((Path(user_file).parent / location).resolve())  # relative to the user file
    return ModelEntry(
        name=name,
        location=location,
        revision=revision,
        inference_preset=entry.get("inference_preset"),
        base_model=entry.get("base_model"),
        description=str(entry.get("description", "")),
        status="configured" if location else str(entry.get("status", "not published")),
        source=source or "environment",
    )


def list_models() -> list[ModelEntry]:
    names = list(bundled_index())
    user, _ = _user_index()
    names += [n for n in user if n not in names]
    return [entry for entry in (lookup_model(n) for n in names) if entry is not None]


def resolve_named_model(name: str) -> Optional[ModelEntry]:
    """Like :func:`lookup_model`, but raise if the name exists without a location."""
    entry = lookup_model(name)
    if entry is not None and entry.location is None:
        raise ModelIndexError(
            f"model {name!r} has not been published ({entry.status}). Point it at a local directory or a "
            f"repository with {env_var_name(name)}=<path or repo id> (and {env_var_name(name)}_REVISION), "
            f"or list it in a file named by {MODELS_FILE_ENV}; or pass the path or repository ID directly."
        )
    return entry
