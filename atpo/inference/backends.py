"""Generation backends.

Both backends receive the same prepared request: the chat-formatted prompt string
plus the video tensors and metadata produced by ``qwen_vl_utils.process_vision_info``
(called with ``return_video_kwargs=True, return_video_metadata=True`` exactly as the
historical vLLM scripts did). The vLLM backend reproduces the historical engine
setup; the Transformers backend runs the same request through
``AutoModelForImageTextToText.generate`` and is not guaranteed to be token-identical
to vLLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .config import GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class PreparedRequest:
    prompt: str
    videos: list[Any]  # list of (tensor, metadata) tuples from qwen-vl-utils
    video_kwargs: dict[str, Any]


@dataclass
class GenerationOutput:
    text: Optional[str]
    info: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class Backend:
    name = "base"

    def generate(self, requests: Sequence[PreparedRequest], generation: GenerationConfig) -> list[GenerationOutput]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


def _torch_dtype(dtype: str):
    import torch

    if dtype == "auto":
        return "auto"
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"unknown dtype {dtype!r}") from exc


class TransformersBackend(Backend):
    name = "transformers"

    def __init__(
        self,
        model_path: str,
        processor: Any,
        *,
        dtype: str = "bfloat16",
        device_map: Any = "auto",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> None:
        from transformers import AutoModelForImageTextToText

        kwargs: dict[str, Any] = {"dtype": _torch_dtype(dtype), "device_map": device_map}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        kwargs.update(model_kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, **kwargs
        ).eval()
        self.processor = processor
        self._settings = {"dtype": dtype, "device_map": str(device_map), "attn_implementation": attn_implementation}
        eos = self.model.generation_config.eos_token_id
        self._eos = set(eos if isinstance(eos, (list, tuple)) else [eos] if eos is not None else [])

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, **self._settings}

    def _generate_one(self, request: PreparedRequest, generation: GenerationConfig) -> GenerationOutput:
        import torch

        videos = [v[0] if isinstance(v, tuple) else v for v in request.videos] or None
        metadata = [v[1] for v in request.videos if isinstance(v, tuple)] or None
        processor_kwargs = dict(request.video_kwargs)
        if metadata is not None:
            processor_kwargs["video_metadata"] = metadata
        inputs = self.processor(text=[request.prompt], videos=videos, return_tensors="pt", **processor_kwargs)
        inputs = inputs.to(self.model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": generation.max_new_tokens,
            "repetition_penalty": generation.repetition_penalty,
        }
        if generation.greedy:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            if generation.seed is not None:
                torch.manual_seed(generation.seed)
            gen_kwargs.update(
                do_sample=True,
                temperature=generation.temperature,
                top_p=generation.top_p,
                top_k=generation.top_k if generation.top_k > 0 else None,
            )
        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = int(inputs["input_ids"].shape[1])
        new_tokens = output[0, prompt_len:].tolist()
        stopped = any(token in self._eos for token in new_tokens)
        text = self.processor.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        info = {
            "prompt_tokens": prompt_len,
            "completion_tokens": len(new_tokens),
            "finish_reason": "stop" if stopped or len(new_tokens) < generation.max_new_tokens else "length",
        }
        return GenerationOutput(text=text, info=info)

    def generate(self, requests: Sequence[PreparedRequest], generation: GenerationConfig) -> list[GenerationOutput]:
        outputs = []
        for request in requests:
            try:
                outputs.append(self._generate_one(request, generation))
            except Exception as exc:  # reported per sample, never dropped
                logger.exception("generation failed")
                outputs.append(GenerationOutput(text=None, error=f"{type(exc).__name__}: {exc}"))
        return outputs


class VLLMBackend(Backend):
    """vLLM engine configured like the paper's evaluation scripts."""

    name = "vllm"

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = True,
        trust_remote_code: bool = False,
        seed: Optional[int] = None,
        **engine_kwargs: Any,
    ) -> None:
        from vllm import LLM

        args: dict[str, Any] = {
            "model": model_path,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "limit_mm_per_prompt": {"video": 1},
            "trust_remote_code": trust_remote_code,
            "disable_log_stats": True,
        }
        if max_model_len is not None:
            args["max_model_len"] = max_model_len
        if seed is not None:
            args["seed"] = seed
        args.update(engine_kwargs)
        self._settings = {k: v for k, v in args.items() if k != "model"}
        self.llm = LLM(**args)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, **self._settings}

    def _sampling_params(self, generation: GenerationConfig):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=generation.temperature,
            top_p=generation.top_p,
            top_k=generation.top_k,
            repetition_penalty=generation.repetition_penalty,
            max_tokens=generation.max_new_tokens,
            skip_special_tokens=True,
            seed=generation.seed,
        )

    @staticmethod
    def _to_input(request: PreparedRequest) -> dict[str, Any]:
        return {
            "prompt": request.prompt,
            "multi_modal_data": {"video": request.videos},
            "mm_processor_kwargs": request.video_kwargs,
        }

    @staticmethod
    def _convert(result: Any) -> GenerationOutput:
        completion = result.outputs[0]
        info = {
            "prompt_tokens": len(result.prompt_token_ids or []),
            "completion_tokens": len(completion.token_ids),
            "finish_reason": completion.finish_reason,
        }
        return GenerationOutput(text=completion.text, info=info)

    def generate(self, requests: Sequence[PreparedRequest], generation: GenerationConfig) -> list[GenerationOutput]:
        params = self._sampling_params(generation)
        inputs = [self._to_input(r) for r in requests]
        try:
            results = self.llm.generate(inputs, params, use_tqdm=False)
            return [self._convert(r) for r in results]
        except Exception as exc:
            if len(inputs) == 1:
                return [GenerationOutput(text=None, error=f"{type(exc).__name__}: {exc}")]
            # Like the historical script, retry one by one so a single failing request
            # (for example a prompt longer than max_model_len) does not fail the batch;
            # unlike it, failures are reported instead of dropped.
            logger.warning("batch generation failed (%s); retrying requests individually", exc)
            outputs = []
            for single in inputs:
                try:
                    outputs.append(self._convert(self.llm.generate([single], params, use_tqdm=False)[0]))
                except Exception as single_exc:
                    outputs.append(GenerationOutput(text=None, error=f"{type(single_exc).__name__}: {single_exc}"))
            return outputs
