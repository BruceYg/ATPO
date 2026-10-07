"""High-level inference API.

    from atpo import VideoSafetyModel

    model = VideoSafetyModel.from_pretrained("path/or/hub-id", backend="vllm")
    prediction = model.predict("example.mp4")
    print(prediction.status, prediction.labels)

The model reference may be a local directory or a Hugging Face Hub repository ID
(optionally with ``revision``). The inference configuration comes from, in order of
precedence, an explicit ``config``, an explicit ``preset``, or the checkpoint's
``atpo_config.json``; a checkpoint without one requires an explicit choice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from ..parsing import parse_response
from .backends import Backend, PreparedRequest
from .config import CONFIG_FILENAME, InferenceConfig
from .prediction import Prediction, error_prediction, prediction_from_parse
from .presets import load_preset

logger = logging.getLogger(__name__)

PROCESSOR_PATTERNS = ["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"]


class ModelResolutionError(ValueError):
    pass


class ConfigResolutionError(ValueError):
    pass


def _is_local(reference: str) -> bool:
    return Path(reference).expanduser().exists()


def _local_fingerprint(path: Path) -> dict[str, Any]:
    """Hash small descriptive files of a local checkpoint (weights are not hashed here)."""
    hashes = {}
    for name in ("config.json", CONFIG_FILENAME, "model.safetensors.index.json", "checksums.sha256",
                 "preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json",
                 "chat_template.jinja", "tokenizer_config.json"):
        file = path / name
        if file.is_file():
            hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    return hashes


def resolve_model_reference(
    reference: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    allow_patterns: Optional[list[str]] = None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(local_path, provenance)`` for a local directory or a Hub repository."""
    if _is_local(reference):
        if revision is not None:
            raise ModelResolutionError(f"revision={revision!r} was given but {reference!r} is a local path")
        path = Path(reference).expanduser().resolve()
        if not path.is_dir():
            raise ModelResolutionError(f"{reference!r} is not a directory")
        return str(path), {"source": "local", "path": str(path), "file_sha256": _local_fingerprint(path)}
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency of the inference extra
        raise ModelResolutionError("huggingface_hub is required to load Hub models") from exc
    try:
        local = snapshot_download(
            repo_id=reference,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        raise ModelResolutionError(
            f"{reference!r} is neither an existing local directory nor a loadable Hub repository "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    return local, {
        "source": "huggingface",
        "repo_id": reference,
        "requested_revision": revision,
        "resolved_revision": Path(local).name,
    }


def resolve_model(
    model: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    allow_patterns: Optional[list[str]] = None,
) -> tuple[str, dict[str, Any], Optional[str]]:
    """Resolve a model name from the index, a local directory, or a repository ID.

    Returns ``(local_path, provenance, fallback_preset)``.
    """
    from .model_index import resolve_named_model

    entry = None if _is_local(model) else resolve_named_model(model)
    if entry is None:
        path, info = resolve_model_reference(model, revision=revision, cache_dir=cache_dir,
                                             local_files_only=local_files_only, allow_patterns=allow_patterns)
        return path, info, None
    path, info = resolve_model_reference(
        entry.location, revision=revision if revision is not None else entry.revision, cache_dir=cache_dir,
        local_files_only=local_files_only, allow_patterns=allow_patterns,
    )
    info = {**info, "model_index": {"name": entry.name, "source": entry.source}}
    return path, info, entry.inference_preset


def resolve_inference_config(
    model_path: str,
    *,
    preset: Optional[str] = None,
    config: Any = None,
    overrides: Optional[Mapping[str, Any]] = None,
    fallback_preset: Optional[str] = None,
) -> tuple[InferenceConfig, dict[str, Any]]:
    """Inference settings for a checkpoint: explicit config > preset > the checkpoint's
    atpo_config.json > ``fallback_preset`` (from the model index)."""
    checkpoint_file = Path(model_path) / CONFIG_FILENAME
    checkpoint_config = InferenceConfig.from_file(checkpoint_file) if checkpoint_file.is_file() else None
    if config is not None and preset is not None:
        raise ConfigResolutionError("pass either config or preset, not both")
    if config is not None:
        if isinstance(config, InferenceConfig):
            resolved = config
        elif isinstance(config, Mapping):
            resolved = InferenceConfig.from_dict(config)
        else:
            resolved = InferenceConfig.from_file(config)
        source = "explicit_config"
    elif preset is not None:
        resolved = load_preset(preset)
        source = f"preset:{preset}"
    elif checkpoint_config is not None:
        resolved = checkpoint_config
        source = "checkpoint"
    elif fallback_preset is not None:
        resolved = load_preset(fallback_preset)
        source = f"model_index_preset:{fallback_preset}"
    else:
        raise ConfigResolutionError(
            f"{model_path} has no {CONFIG_FILENAME}; pass preset=... (see `atpo presets`) or config=..."
        )
    info: dict[str, Any] = {"source": source}
    if checkpoint_config is not None and source != "checkpoint":
        same = checkpoint_config.fingerprint() == resolved.fingerprint()
        info["checkpoint_config_overridden"] = not same
        if not same:
            warnings.warn(
                f"the checkpoint's {CONFIG_FILENAME} differs from the requested {source}; using {source}",
                stacklevel=3,
            )
    if overrides:
        resolved = resolved.with_overrides(overrides)
        info["overrides"] = dict(overrides)
    return resolved, info


def build_messages(config: InferenceConfig, video: str) -> list[dict[str, Any]]:
    """Chat messages for one video, laid out as the configured historical script did."""
    video_item = {
        "type": "video",
        "video": video,
        "max_pixels": config.video.max_pixels,
        "fps": config.video.fps,
        "min_pixels": config.video.min_pixels,
    }
    if config.video.position == "after":
        content = [{"type": "text", "text": config.prompt}, video_item]
    elif config.video.position == "before":
        content = [video_item, {"type": "text", "text": config.prompt}]
    else:  # placeholder, as EasyR1's dataset splits the prompt at "<video>"
        content = []
        for index, part in enumerate(config.prompt.split("<video>")):
            if index:
                content.append(video_item)
            if part:
                content.append({"type": "text", "text": part})
    messages: list[dict[str, Any]] = []
    if config.system_prompt is not None:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


def _default_vision_fn(image_patch_size: int) -> Callable[[list[dict[str, Any]]], tuple[Any, Any, Any]]:
    from qwen_vl_utils import process_vision_info

    def vision(messages):
        return process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True, image_patch_size=image_patch_size
        )

    return vision


class VideoSafetyModel:
    def __init__(
        self,
        backend: Backend,
        processor: Any,
        config: InferenceConfig,
        *,
        vision_fn: Optional[Callable[[list[dict[str, Any]]], tuple[Any, Any, Any]]] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> None:
        self.backend = backend
        self.processor = processor
        self.config = config.validate()
        self._vision_fn = vision_fn
        self.provenance = dict(provenance or {})

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        revision: Optional[str] = None,
        preset: Optional[str] = None,
        config: Any = None,
        overrides: Optional[Mapping[str, Any]] = None,
        backend: str = "transformers",
        processor: Optional[str] = None,
        processor_revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        dtype: str = "bfloat16",
        **backend_kwargs: Any,
    ) -> "VideoSafetyModel":
        """Load a checkpoint for inference.

        ``backend_kwargs`` go to the backend: for ``"transformers"`` e.g. ``device_map``,
        ``attn_implementation``; for ``"vllm"`` e.g. ``max_model_len``,
        ``tensor_parallel_size``, ``gpu_memory_utilization``.
        """
        from transformers import AutoProcessor

        model_path, model_info, fallback_preset = resolve_model(
            model, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only
        )
        inference_config, config_info = resolve_inference_config(
            model_path, preset=preset, config=config, overrides=overrides, fallback_preset=fallback_preset
        )

        proc_ref = processor or inference_config.processor
        proc_rev = processor_revision if processor is not None else inference_config.processor_revision
        if proc_ref == "checkpoint":
            proc_path, proc_info = model_path, {"source": "checkpoint"}
        else:
            proc_path, proc_info = resolve_model_reference(
                proc_ref, revision=proc_rev, cache_dir=cache_dir, local_files_only=local_files_only,
                allow_patterns=PROCESSOR_PATTERNS,
            )
        loaded_processor = AutoProcessor.from_pretrained(
            proc_path, use_fast=inference_config.processor_use_fast, trust_remote_code=trust_remote_code
        )

        if backend == "transformers":
            from .backends import TransformersBackend

            engine: Backend = TransformersBackend(
                model_path, loaded_processor, dtype=dtype, trust_remote_code=trust_remote_code, **backend_kwargs
            )
        elif backend == "vllm":
            from .backends import VLLMBackend

            backend_kwargs.setdefault("max_model_len", inference_config.max_model_len)
            engine = VLLMBackend(model_path, dtype=dtype, trust_remote_code=trust_remote_code, **backend_kwargs)
        else:
            raise ValueError(f"unknown backend {backend!r}; use 'transformers' or 'vllm'")

        provenance = {
            "model": {"reference": model, **model_info},
            "config": config_info,
            "processor": {"reference": proc_ref, **proc_info},
        }
        return cls(engine, loaded_processor, inference_config, provenance=provenance)

    # ------------------------------------------------------------------ prediction
    @property
    def taxonomy(self):
        return self.config.taxonomy

    def _vision(self, messages):
        if self._vision_fn is None:
            self._vision_fn = _default_vision_fn(self.config.video.image_patch_size)
        return self._vision_fn(messages)

    def prepare(self, video: str) -> PreparedRequest:
        """Build the generation request for one video (raises on unreadable video)."""
        if "://" not in video and not os.path.isfile(video):
            raise FileNotFoundError(f"video not found: {video}")
        messages = build_messages(self.config, video)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs, video_kwargs = self._vision(messages)
        if not video_inputs:
            raise ValueError("no video frames were decoded")
        return PreparedRequest(prompt=prompt, videos=list(video_inputs), video_kwargs=dict(video_kwargs or {}))

    def parse(self, sample_id: str, video: str, text: str, info: Optional[dict[str, Any]] = None) -> Prediction:
        parsed = parse_response(
            text,
            self.taxonomy,
            response_format=self.config.response_format,  # type: ignore[arg-type]
            output_style=self.config.parser.output_style,  # type: ignore[arg-type]
            scope=self.config.parser.scope,  # type: ignore[arg-type]
            require_block_prefix=self.config.parser.require_block_prefix,
        )
        return prediction_from_parse(
            sample_id=sample_id, video=video, raw_response=text, parsed=parsed, taxonomy=self.taxonomy,
            scope=self.config.parser.scope, generation=info,
        )

    def predict(self, video: str, *, id: Optional[str] = None) -> Prediction:
        return self.predict_batch([video], ids=[id] if id is not None else None)[0]

    def predict_batch(
        self,
        videos: Sequence[str],
        ids: Optional[Sequence[str]] = None,
        *,
        batch_size: int = 8,
        on_prediction: Optional[Callable[[Prediction], None]] = None,
    ) -> list[Prediction]:
        if ids is None:
            ids = [str(v) for v in videos]
        if len(ids) != len(videos):
            raise ValueError("ids and videos must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique")
        results: list[Prediction] = []
        for start in range(0, len(videos), max(1, batch_size)):
            chunk = list(zip(ids[start:start + batch_size], videos[start:start + batch_size]))
            ordered: list[Optional[Prediction]] = [None] * len(chunk)
            requests, slots = [], []
            for slot, (sid, video) in enumerate(chunk):
                try:
                    requests.append(self.prepare(str(video)))
                    slots.append(slot)
                except Exception as exc:
                    ordered[slot] = error_prediction(str(sid), str(video), "video_error", f"{type(exc).__name__}: {exc}")
            if requests:
                outputs = self.backend.generate(requests, self.config.generation)
                for slot, output in zip(slots, outputs):
                    sid, video = chunk[slot]
                    if output.error is not None or output.text is None:
                        ordered[slot] = error_prediction(str(sid), str(video), "generation_error",
                                                         output.error or "no output")
                    else:
                        ordered[slot] = self.parse(str(sid), str(video), output.text, output.info)
            for prediction in ordered:
                assert prediction is not None
                results.append(prediction)
                if on_prediction is not None:
                    on_prediction(prediction)
        return results

    def describe(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "config_fingerprint": self.config.fingerprint(),
            "backend": self.backend.describe(),
            **self.provenance,
        }


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
