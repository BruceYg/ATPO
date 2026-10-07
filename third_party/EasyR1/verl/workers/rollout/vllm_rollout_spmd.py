# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from PIL import Image
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video_cached
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _mask_video_frames(frames, ignored_interval: Optional[list[float]]):
    """Zero out frames whose normalized position falls in ignored_interval.

    Used by VAD-R1 AVA-GRPO verification pass to ablate a temporal interval without
    changing frame count (so video_grid_thw and prompt token positions stay valid).

    Handles three frame container types returned by qwen_vl_utils.fetch_video and friends:
    - torch.Tensor of shape (T, C, H, W)  → returns a cloned tensor with masked time-slices zeroed
    - numpy.ndarray of shape (T, ...)     → returns a copy with masked time-slices zeroed
    - list of PIL.Image / tensor / ndarray frames → returns a new list with masked entries replaced
      by an all-zeros frame of identical type/shape

    Never mutates the input (cached frames must stay clean).
    """
    if ignored_interval is None:
        return frames
    try:
        start = float(ignored_interval[0])
        end = float(ignored_interval[1])
    except (TypeError, ValueError, IndexError):
        return frames
    if not (0.0 <= start <= end <= 1.0):
        return frames

    # Tensor representation (T, C, H, W) — most common path from fetch_video.
    if isinstance(frames, torch.Tensor):
        n = frames.shape[0]
        if n == 0:
            return frames
        denom = max(n - 1, 1)
        positions = torch.arange(n, dtype=torch.float32) / denom if n > 1 else torch.zeros(n)
        mask = (positions >= start) & (positions <= end)
        if not bool(mask.any()):
            return frames
        masked = frames.clone()
        masked[mask] = 0
        return masked

    # Numpy array representation (T, ...).
    if isinstance(frames, np.ndarray):
        n = frames.shape[0]
        if n == 0:
            return frames
        denom = max(n - 1, 1)
        positions = np.arange(n, dtype=np.float32) / denom if n > 1 else np.zeros(n)
        mask = (positions >= start) & (positions <= end)
        if not bool(mask.any()):
            return frames
        masked = frames.copy()
        masked[mask] = 0
        return masked

    # List/tuple of per-frame objects (PIL Image, tensor, or ndarray).
    if isinstance(frames, (list, tuple)):
        n = len(frames)
        if n == 0:
            return frames
        denom = max(n - 1, 1)
        result = []
        for i, frame in enumerate(frames):
            pos = i / denom if n > 1 else 0.0
            if start <= pos <= end:
                if isinstance(frame, Image.Image):
                    result.append(Image.new(frame.mode or "RGB", frame.size, 0))
                elif isinstance(frame, torch.Tensor):
                    result.append(torch.zeros_like(frame))
                elif isinstance(frame, np.ndarray):
                    result.append(np.zeros_like(frame))
                else:
                    result.append(frame)
            else:
                result.append(frame)
        return result

    # Unknown container type — leave unchanged rather than crash the whole rollout.
    return frames


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    min_pixels: int,
    max_pixels: int,
    video_fps: Optional[float] = None,
    video_nframes: Optional[int] = None,
    video_cache_dir: Optional[str] = None,
    ignored_interval: Optional[list[float]] = None,
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """Process multi-modal data for vLLM.

    Returns:
        A tuple of (multi_modal_data, mm_processor_kwargs):
        - multi_modal_data: Dict with "image" or "video" key containing processed media.
          For videos, each entry is a tuple of (frames, metadata) where metadata contains fps info.
        - mm_processor_kwargs: Dict with additional processor kwargs like second_per_grid_ts for videos.
    """
    images, videos = [], []
    mm_processor_kwargs = {}
    video_cache_path = Path(video_cache_dir) if video_cache_dir else None
    print(f"[VLLM_ROLLOUT] Processing multi-modal data, video_cache_dir={video_cache_dir}")

    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        second_per_grid_ts = []
        for video in multi_modal_data["videos"]:
            frames, actual_fps = process_video_cached(
                video,
                min_pixels,
                max_pixels,
                video_fps=video_fps,
                video_nframes=video_nframes,
                cache_dir=video_cache_path,
                return_fps=True,
            )
            frames = _mask_video_frames(frames, ignored_interval)
            # Pass video as (frames, metadata) tuple for vLLM
            video_metadata = {"fps": actual_fps, "total_num_frames": len(frames), "frames_indices": list(range(len(frames)))}
            videos.append((frames, video_metadata))
            # Compute second_per_grid_ts (same as in dataset.py)
            second_per_grid_ts.append(2.0 / actual_fps)

        mm_processor_kwargs["second_per_grid_ts"] = second_per_grid_ts
        print(f"[VLLM_ROLLOUT] Video metadata: second_per_grid_ts={second_per_grid_ts}")

    if len(images) != 0:
        return {"image": images}, {}

    if len(videos) != 0:
        return {"video": videos}, mm_processor_kwargs

    return None, {}


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        # VAD-R1 AVA-GRPO: per-sample [start, end] in normalized [0,1] frame index to mask out.
        # Carried in non_tensor_batch (not meta_info) so it shards correctly under DP dispatch.
        # When set, frames in that range are replaced with black frames before feeding to vLLM.
        # Absent for normal rollouts.
        batch_ignored_intervals = non_tensor_batch.pop("ignored_intervals", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")
        if batch_ignored_intervals is not None and len(batch_ignored_intervals) != batch_size:
            raise RuntimeError(
                f"ignored_intervals length {len(batch_ignored_intervals)} != batch_size {batch_size}"
            )

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for idx, (raw_prompt_ids, multi_modal_data) in enumerate(zip(batch_raw_prompt_ids, batch_multi_modal_data)):
                ignored_interval = batch_ignored_intervals[idx] if batch_ignored_intervals is not None else None
                mm_data, mm_kwargs = _process_multi_modal_data(
                    multi_modal_data,
                    prompts.meta_info["min_pixels"],
                    prompts.meta_info["max_pixels"],
                    prompts.meta_info.get("video_fps"),
                    prompts.meta_info.get("video_nframes"),
                    prompts.meta_info.get("video_cache_dir"),
                    ignored_interval=ignored_interval,
                )
                vllm_input = {
                    "prompt_token_ids": list(raw_prompt_ids),
                    "multi_modal_data": mm_data,
                }
                if mm_kwargs:
                    vllm_input["mm_processor_kwargs"] = mm_kwargs
                vllm_inputs.append(vllm_input)
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)
                batch_raw_prompt_ids = _repeat_interleave(batch_raw_prompt_ids, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        # Preserve raw_prompt_ids in the output so subsequent rollout calls (e.g. VAD-R1
        # AVA-GRPO verification pass) can reuse it without reconstructing from input_ids.
        non_tensor_batch = {"raw_prompt_ids": batch_raw_prompt_ids}
        if batch_multi_modal_data is not None:
            non_tensor_batch["multi_modal_data"] = batch_multi_modal_data

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
