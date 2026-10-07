"""Inference configuration stored with a checkpoint as ``atpo_config.json``.

The configuration fixes everything needed to turn a video into a structured
prediction: taxonomy, exact prompt text, system prompt, video sampling, response
format, parser settings, and generation settings. A checkpoint exported by
``atpo export`` carries this file; checkpoints without it need an explicit preset
(see :mod:`atpo.inference.presets`).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from ..parsing import OUTPUT_STYLES, PARSE_SCOPES, RESPONSE_FORMATS
from ..taxonomy import Taxonomy, load_taxonomy

CONFIG_FORMAT = "atpo.inference_config/v1"
CONFIG_FILENAME = "atpo_config.json"


@dataclass
class VideoConfig:
    fps: float = 1.0
    min_pixels: int = 12544
    max_pixels: int = 200704
    # qwen-vl-utils resizes frames to multiples of 2 * image_patch_size. The historical
    # scripts used the library default (14) for both Qwen2.5-VL and Qwen3-VL.
    image_patch_size: int = 14
    # "after"/"before": the video follows/precedes the whole prompt text (historical
    # evaluation scripts). "placeholder": the prompt contains "<video>" where the video goes
    # (EasyR1 training layout).
    position: str = "after"

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("video.fps must be positive")
        if not 0 < self.min_pixels <= self.max_pixels:
            raise ValueError("video pixels must satisfy 0 < min_pixels <= max_pixels")
        if self.position not in ("after", "before", "placeholder"):
            raise ValueError("video.position must be 'after', 'before', or 'placeholder'")
        if self.image_patch_size <= 0:
            raise ValueError("video.image_patch_size must be positive")


@dataclass
class ParserConfig:
    output_style: str = "json"
    scope: str = "full_response"
    require_block_prefix: bool = True

    def validate(self) -> None:
        if self.output_style not in OUTPUT_STYLES:
            raise ValueError(f"parser.output_style must be one of {OUTPUT_STYLES}")
        if self.scope not in PARSE_SCOPES:
            raise ValueError(f"parser.scope must be one of {PARSE_SCOPES}")


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    seed: Optional[int] = None

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("generation.max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("generation.temperature must be >= 0")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass
class InferenceConfig:
    taxonomy: Taxonomy
    prompt: str
    system_prompt: Optional[str]
    response_format: str
    video: VideoConfig = field(default_factory=VideoConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    base_model: Optional[str] = None
    # Where the processor (chat template, image/video processor) comes from:
    # "checkpoint" uses the files saved with the model; anything else is a model ID or path.
    processor: str = "checkpoint"
    processor_revision: Optional[str] = None
    processor_use_fast: bool = False
    max_model_len: Optional[int] = None
    preset: Optional[str] = None
    description: Optional[str] = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "InferenceConfig":
        if self.response_format not in RESPONSE_FORMATS:
            raise ValueError(f"response_format must be one of {RESPONSE_FORMATS}")
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        self.video.validate()
        if self.video.position == "placeholder" and self.prompt.count("<video>") != 1:
            raise ValueError("video.position 'placeholder' needs exactly one '<video>' in the prompt")
        if self.video.position != "placeholder" and "<video>" in self.prompt:
            raise ValueError("prompt contains '<video>'; set video.position to 'placeholder'")
        self.parser.validate()
        self.generation.validate()
        return self

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": CONFIG_FORMAT,
            "preset": self.preset,
            "description": self.description,
            "taxonomy": self.taxonomy.to_dict(),
            "taxonomy_fingerprint": self.taxonomy.fingerprint(),
            "prompt": {"text": self.prompt, "sha256": self.prompt_sha256},
            "system_prompt": self.system_prompt,
            "response_format": self.response_format,
            "video": asdict(self.video),
            "parser": asdict(self.parser),
            "generation": asdict(self.generation),
            "base_model": self.base_model,
            "processor": {
                "source": self.processor,
                "revision": self.processor_revision,
                "use_fast": self.processor_use_fast,
            },
            "max_model_len": self.max_model_len,
            "provenance": self.provenance,
        }

    def fingerprint(self) -> str:
        data = self.to_dict()
        data.pop("description", None)
        data.pop("provenance", None)
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InferenceConfig":
        data = copy.deepcopy(dict(data))
        fmt = data.pop("format", CONFIG_FORMAT)
        if fmt != CONFIG_FORMAT:
            raise ValueError(f"unsupported inference config format {fmt!r}")
        taxonomy_spec = data.pop("taxonomy")
        taxonomy = load_taxonomy(taxonomy_spec) if isinstance(taxonomy_spec, str) else Taxonomy.from_dict(taxonomy_spec)
        expected_tax = data.pop("taxonomy_fingerprint", None)
        if expected_tax is not None and expected_tax != taxonomy.fingerprint():
            raise ValueError("taxonomy_fingerprint does not match the taxonomy in the config")
        prompt = data.pop("prompt")
        if isinstance(prompt, Mapping):
            text = prompt["text"]
            expected = prompt.get("sha256")
            if expected is not None and hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
                raise ValueError("prompt text does not match its recorded sha256")
        else:
            text = prompt
        processor = data.pop("processor", "checkpoint")
        if isinstance(processor, Mapping):
            proc_source = processor.get("source", "checkpoint")
            proc_rev = processor.get("revision")
            proc_fast = bool(processor.get("use_fast", False))
        else:
            proc_source, proc_rev, proc_fast = processor, None, False
        known = {"system_prompt", "response_format", "video", "parser", "generation", "base_model",
                 "max_model_len", "preset", "description", "provenance"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown inference config keys: {sorted(unknown)}")
        sections = {}
        for key, section_cls in (("video", VideoConfig), ("parser", ParserConfig), ("generation", GenerationConfig)):
            try:
                sections[key] = section_cls(**(data.get(key) or {}))
            except TypeError as exc:
                raise ValueError(f"invalid {key} settings: {exc}") from exc
        config = cls(
            taxonomy=taxonomy,
            prompt=text,
            system_prompt=data.get("system_prompt"),
            response_format=data["response_format"],
            video=sections["video"],
            parser=sections["parser"],
            generation=sections["generation"],
            base_model=data.get("base_model"),
            processor=proc_source,
            processor_revision=proc_rev,
            processor_use_fast=proc_fast,
            max_model_len=data.get("max_model_len"),
            preset=data.get("preset"),
            description=data.get("description"),
            provenance=dict(data.get("provenance") or {}),
        )
        return config.validate()

    @classmethod
    def from_file(cls, path: str | Path) -> "InferenceConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            path = path / CONFIG_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def with_overrides(self, overrides: Optional[Mapping[str, Any]]) -> "InferenceConfig":
        """Return a copy with dotted-key overrides applied, e.g. ``{"video.max_pixels": 50176}``."""
        if not overrides:
            return self
        data = self.to_dict()
        data.pop("prompt")
        data["prompt"] = self.prompt
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            target = data
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    raise ValueError(f"unknown override key {dotted!r}")
                target = target[part]
            if parts[-1] not in target:
                raise ValueError(f"unknown override key {dotted!r}")
            target[parts[-1]] = value
        data.pop("taxonomy_fingerprint", None)
        return InferenceConfig.from_dict(data)
