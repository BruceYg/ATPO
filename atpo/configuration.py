"""Shared helpers for YAML configuration files: ``extends`` chains, deep merge, and
``${VAR}`` substitution.

A value that is exactly ``"${VAR}"`` is replaced by the variable's value with its
type preserved (so a variable may be ``null`` or a number); ``${VAR}`` inside a
longer string is interpolated as text. Variables come from, in order: explicit
values (``--var``), the ``variables`` block of the file (``default``), and the
environment. An unresolved variable is an error.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_WHOLE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigError(ValueError):
    pass


_MISSING = object()


def _lookup(name: str, variables: Mapping[str, Any], use_env: bool) -> Any:
    if name in variables:
        return variables[name]
    if use_env and name in os.environ:
        return os.environ[name]
    raise ConfigError(f"undefined variable ${{{name}}}; pass --var {name}=...")


def expand_variables(value: Any, variables: Mapping[str, Any], *, use_env: bool = True) -> Any:
    if isinstance(value, str):
        whole = _WHOLE.match(value)
        if whole:
            return _lookup(whole.group(1), variables, use_env)
        return _VAR.sub(lambda m: str(_lookup(m.group(1), variables, use_env)), value)
    if isinstance(value, list):
        return [expand_variables(v, variables, use_env=use_env) for v in value]
    if isinstance(value, dict):
        return {k: expand_variables(v, variables, use_env=use_env) for k, v in value.items()}
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalars in ``override`` replace ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml_with_extends(path: str | Path, _seen: Optional[set[Path]] = None) -> dict[str, Any]:
    """Load YAML; ``extends: other.yaml`` (relative to the file) is merged underneath."""
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ConfigError(f"circular 'extends' at {path}")
    seen.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    base = load_yaml_with_extends(path.parent / parent, seen)
    # Variable declarations accumulate across the chain.
    variables = deep_merge(base.get("variables", {}) or {}, data.get("variables", {}) or {})
    merged = deep_merge(base, data)
    if variables:
        merged["variables"] = variables
    return merged


def resolve_variables(declared: Mapping[str, Any], given: Mapping[str, Any]) -> dict[str, Any]:
    """Combine ``--var`` values with declared defaults (environment handled at expansion)."""
    values: dict[str, Any] = {}
    for name, spec in (declared or {}).items():
        if isinstance(spec, Mapping) and "default" in spec:
            values[name] = spec["default"]
    unknown = set(given) - set(declared or {})
    if unknown and declared:
        raise ConfigError(f"unknown variable(s) {sorted(unknown)}; declared: {sorted(declared)}")
    values.update(given)
    return values


def absolute_local_path(value: Any, *, must_exist: bool) -> Any:
    """Resolve a relative local path against the current working directory.

    The training backends run with the run directory as their working directory, so
    relative paths given at launch are made absolute when the config is composed.
    URLs and other non-path values are returned unchanged; with ``must_exist`` only
    paths that exist are resolved (so Hugging Face IDs pass through).
    """
    if not isinstance(value, str) or not value or "://" in value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    if must_exist and not path.exists():
        return value
    return str(path.resolve())


def parse_scalar(text: str) -> Any:
    """Interpret a command-line value as YAML (numbers, booleans, null, lists)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text
