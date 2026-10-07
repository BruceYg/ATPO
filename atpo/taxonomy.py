"""Category taxonomies shared by data preparation, rewards, controllers, and inference.

A taxonomy fixes the label space of a task: stable category IDs (used in data
records, configuration mappings, and controller state), display names, prompt
descriptions, and the JSON keys a model is asked to emit (``"C1(Sexual Content)"``).

Category order is significant: controller state vectors, reward vectors, and
historical integer labels follow it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_CATEGORY_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]*$")
_BLOCK_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

BUILTIN_TAXONOMIES = ("safewatch", "xdviolence")


class TaxonomyError(ValueError):
    """Invalid taxonomy definition."""


class UnknownLabelError(TaxonomyError):
    """A label does not resolve to any category of the taxonomy."""


@dataclass(frozen=True)
class Category:
    """One category of a taxonomy.

    ``aliases`` are alternative spellings accepted in annotations, for example the
    historical 1-based integer labels of the benchmark files (``"1"`` for ``C1``).
    """

    id: str
    name: str
    description: str = ""
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Taxonomy:
    """An ordered set of categories plus the output-format conventions of a task.

    ``block_name`` is the label that introduces the JSON decision block in model
    outputs (``GUARDRAIL`` for SafeWatch, ``DETECTION`` for XD-Violence).
    ``key_template`` builds the JSON key for each category.
    """

    name: str
    categories: tuple[Category, ...]
    block_name: str = "RESULT"
    key_template: str = "{id}({name})"
    description: str = ""
    _lookup: dict[str, str] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise TaxonomyError("taxonomy name must be non-empty")
        if not self.categories:
            raise TaxonomyError(f"taxonomy {self.name!r} has no categories")
        if not _BLOCK_NAME.match(self.block_name):
            raise TaxonomyError(f"invalid block_name {self.block_name!r}")
        try:
            self.key_template.format(id="X", name="Y")
        except (KeyError, IndexError) as exc:
            raise TaxonomyError(
                f"key_template {self.key_template!r} may only use {{id}} and {{name}}"
            ) from exc

        lookup: dict[str, str] = {}
        keys: set[str] = set()
        for category in self.categories:
            if not _CATEGORY_ID.match(category.id):
                raise TaxonomyError(
                    f"category id {category.id!r} must start with a letter and contain only "
                    "letters, digits, '_', '.', or '-'"
                )
            if not category.name:
                raise TaxonomyError(f"category {category.id!r} has an empty name")
            for spelling in (category.id, *category.aliases):
                spelling = str(spelling)
                if spelling in lookup and lookup[spelling] != category.id:
                    raise TaxonomyError(
                        f"label spelling {spelling!r} is ambiguous between "
                        f"{lookup[spelling]!r} and {category.id!r}"
                    )
                lookup[spelling] = category.id
            key = self.key_template.format(id=category.id, name=category.name)
            if key in keys:
                raise TaxonomyError(f"duplicate output key {key!r}")
            keys.add(key)
        ids = [c.id for c in self.categories]
        if len(set(ids)) != len(ids):
            raise TaxonomyError(f"duplicate category ids in taxonomy {self.name!r}")
        object.__setattr__(self, "_lookup", lookup)

    # ------------------------------------------------------------------ access
    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.categories]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.categories]

    def __len__(self) -> int:
        return len(self.categories)

    def index(self, category_id: str) -> int:
        for i, category in enumerate(self.categories):
            if category.id == category_id:
                return i
        raise UnknownLabelError(f"unknown category id {category_id!r} for taxonomy {self.name!r}")

    def category(self, category_id: str) -> Category:
        return self.categories[self.index(category_id)]

    def output_key(self, category: Category | str) -> str:
        if isinstance(category, str):
            category = self.category(category)
        return self.key_template.format(id=category.id, name=category.name)

    def resolve_label(self, label: Any) -> str:
        """Map an annotation label (ID, alias, or integer alias) to a category ID."""
        if hasattr(label, "item") and not isinstance(label, (str, bytes)):
            label = label.item()  # numpy scalar -> Python scalar
        if isinstance(label, bool):  # bool is an int subclass; never a valid label
            raise UnknownLabelError(f"boolean {label!r} is not a category label")
        if isinstance(label, float):
            if not label.is_integer():
                raise UnknownLabelError(f"non-integer numeric label {label!r}")
            label = int(label)
        if not isinstance(label, (str, int)):
            raise UnknownLabelError(f"unsupported label type {type(label).__name__}: {label!r}")
        key = str(label).strip()
        if key not in self._lookup:
            raise UnknownLabelError(
                f"label {label!r} is not a category ID or alias of taxonomy {self.name!r}; "
                f"valid IDs: {', '.join(self.ids)}"
            )
        return self._lookup[key]

    def resolve_labels(self, labels: Iterable[Any]) -> list[str]:
        """Resolve labels and return category IDs in taxonomy order (deduplicated)."""
        resolved = {self.resolve_label(label) for label in labels}
        return [cid for cid in self.ids if cid in resolved]

    def names_for(self, category_ids: Iterable[str]) -> list[str]:
        return [self.category(cid).name for cid in category_ids]

    def to_vector(self, category_ids: Iterable[str]) -> list[int]:
        chosen = set(category_ids)
        unknown = chosen.difference(self.ids)
        if unknown:
            raise UnknownLabelError(f"unknown category ids {sorted(unknown)}")
        return [1 if cid in chosen else 0 for cid in self.ids]

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "block_name": self.block_name,
            "key_template": self.key_template,
            "categories": [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "aliases": list(c.aliases),
                }
                for c in self.categories
            ],
        }

    def fingerprint(self) -> str:
        """SHA-256 over the label space: IDs, names, aliases, block name, key template.

        Category descriptions are excluded because they only affect prompt text,
        not label identity or output parsing.
        """
        payload = {
            "name": self.name,
            "block_name": self.block_name,
            "key_template": self.key_template,
            "categories": [[c.id, c.name, sorted(c.aliases)] for c in self.categories],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Taxonomy":
        if not isinstance(data, Mapping):
            raise TaxonomyError("taxonomy definition must be a mapping")
        unknown = set(data).difference(
            {"name", "description", "block_name", "key_template", "categories"}
        )
        if unknown:
            raise TaxonomyError(f"unknown taxonomy fields: {sorted(unknown)}")
        raw_categories = data.get("categories")
        if not isinstance(raw_categories, Sequence) or isinstance(raw_categories, (str, bytes)):
            raise TaxonomyError("taxonomy 'categories' must be a list")
        categories = []
        for raw in raw_categories:
            if not isinstance(raw, Mapping):
                raise TaxonomyError(f"category entry must be a mapping, got {raw!r}")
            extra = set(raw).difference({"id", "name", "description", "aliases"})
            if extra:
                raise TaxonomyError(f"unknown category fields {sorted(extra)} in {raw!r}")
            if "id" not in raw or "name" not in raw:
                raise TaxonomyError(f"category entry needs 'id' and 'name': {raw!r}")
            aliases = raw.get("aliases") or ()
            categories.append(
                Category(
                    id=str(raw["id"]),
                    name=str(raw["name"]),
                    description=str(raw.get("description") or "").strip(),
                    aliases=tuple(str(a) for a in aliases),
                )
            )
        return cls(
            name=str(data.get("name", "")),
            categories=tuple(categories),
            block_name=str(data.get("block_name", "RESULT")),
            key_template=str(data.get("key_template", "{id}({name})")),
            description=str(data.get("description") or "").strip(),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Taxonomy":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
        # Allow a taxonomy nested under a "taxonomy:" key in larger config files.
        if isinstance(data, Mapping) and "taxonomy" in data and "categories" not in data:
            data = data["taxonomy"]
        return cls.from_dict(data)


def load_taxonomy(spec: "str | Path | Mapping[str, Any] | Taxonomy") -> Taxonomy:
    """Load a taxonomy from a built-in name, a YAML/JSON path, or a mapping."""
    if isinstance(spec, Taxonomy):
        return spec
    if isinstance(spec, Mapping):
        return Taxonomy.from_dict(spec)
    spec_str = str(spec)
    if spec_str in BUILTIN_TAXONOMIES:
        resource = resources.files("atpo.resources.taxonomies").joinpath(f"{spec_str}.yaml")
        import yaml

        return Taxonomy.from_dict(yaml.safe_load(resource.read_text(encoding="utf-8")))
    path = Path(spec_str).expanduser()
    if not path.exists():
        raise TaxonomyError(
            f"taxonomy {spec_str!r} is neither a built-in ({', '.join(BUILTIN_TAXONOMIES)}) "
            "nor an existing file"
        )
    return Taxonomy.from_file(path)
