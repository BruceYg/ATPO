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

import hashlib
import math
import os
import pickle
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from filelock import FileLock
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: Optional[float] = None,
    video_nframes: Optional[int] = None,
    return_fps: bool = False,
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels}
    # Use nframes if specified, otherwise use fps (default to 2.0 if neither specified)
    if video_nframes is not None:
        vision_info["nframes"] = video_nframes
    else:
        vision_info["fps"] = video_fps if video_fps is not None else 2.0
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


def process_video_cached(
    video: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: Optional[float] = None,
    video_nframes: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    return_fps: bool = False,
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    """
    Process video with persistent disk caching.

    Cache is automatically invalidated if processing parameters change.
    The cache key is computed from the video path and processing parameters.

    Note: The cache always stores both frames and fps. The return_fps parameter
    only controls what is returned to the caller, not what is cached. This ensures
    cache consistency regardless of how callers request the data.

    Args:
        video: Path to the video file.
        min_pixels: Minimum number of pixels for resizing.
        max_pixels: Maximum number of pixels for resizing.
        video_fps: Frame rate for video processing. Ignored if video_nframes is set.
        video_nframes: Total number of frames to extract (evenly distributed). Takes precedence over video_fps.
        cache_dir: Directory to store cached processed videos. If None, no caching is performed.
        return_fps: Whether to return the actual FPS used.

    Returns:
        List of processed video frames (PIL Images), or tuple of (frames, fps_list) if return_fps=True.
    """
    if cache_dir is None:
        # Fallback to non-cached version - WARNING: this can cause token/feature mismatch!
        print(f"[VIDEO_CACHE] WARNING: cache_dir is None, processing video without caching: {video}")
        return process_video(video, min_pixels, max_pixels, video_fps, video_nframes, return_fps)

    # Create cache key from video path and processing parameters
    # Note: return_fps is NOT included in the key because the actual frames are the same
    # Include both fps and nframes in key to distinguish different sampling strategies
    cache_key_str = f"{video}|{min_pixels}|{max_pixels}|fps={video_fps}|nframes={video_nframes}"
    cache_hash = hashlib.sha256(cache_key_str.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{cache_hash}.pkl"
    lock_file = cache_dir / f"{cache_hash}.lock"

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use file lock for the ENTIRE check-process-write sequence to avoid race conditions
    # This ensures only one process extracts frames for a given video
    with FileLock(str(lock_file), timeout=600):
        frames = None
        actual_fps = None

        # Try to load from cache (inside lock to handle race conditions)
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                # Verify cache integrity by checking the video path
                if cached_data.get("video_path") == video:
                    frames = cached_data["frames"]
                    actual_fps = cached_data["fps"]
                    print(f"[VIDEO_CACHE] Cache HIT: {video} -> {len(frames)} frames")
            except (pickle.PickleError, KeyError, EOFError, Exception) as e:
                # Cache corrupted, will regenerate
                print(f"[VIDEO_CACHE] Cache corrupted for {video}, regenerating: {e}")
                try:
                    cache_file.unlink(missing_ok=True)
                except Exception:
                    pass

        # Process video if not in cache (cache miss or corrupted)
        if frames is None:
            print(f"[VIDEO_CACHE] Cache MISS, processing: {video}")
            # Always process with return_fps=True to get both frames and fps for caching
            frames, actual_fps = process_video(video, min_pixels, max_pixels, video_fps, video_nframes, return_fps=True)
            print(f"[VIDEO_CACHE] Processed video: {video} -> {len(frames)} frames")

            # Save to cache
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(
                        {
                            "video_path": video,
                            "frames": frames,
                            "fps": actual_fps,
                            "params": {
                                "min_pixels": min_pixels,
                                "max_pixels": max_pixels,
                                "video_fps": video_fps,
                                "video_nframes": video_nframes,
                            },
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                print(f"[VIDEO_CACHE] Saved to cache: {cache_file}")
            except Exception as e:
                print(f"[VIDEO_CACHE] WARNING: Failed to cache video {video}: {e}")

    # Return based on caller's request
    if return_fps:
        return frames, actual_fps
    else:
        return frames


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: Optional[float] = 2.0,
        video_nframes: Optional[int] = None,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        video_cache_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.video_nframes = video_nframes
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.video_cache_dir = Path(video_cache_dir) if video_cache_dir else None

        if self.video_cache_dir:
            self.video_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Video caching enabled: {self.video_cache_dir}")

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            # Normalize videos to always be a list
            if isinstance(videos, str):
                videos = [videos]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(
                    process_video_cached(
                        video,
                        self.min_pixels,
                        self.max_pixels,
                        video_fps=self.video_fps,
                        video_nframes=self.video_nframes,
                        cache_dir=self.video_cache_dir,
                    )
                )

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            # Normalize videos to always be a list
            if isinstance(videos, str):
                videos = [videos]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, actual_video_fps = process_video_cached(
                    video,
                    self.min_pixels,
                    self.max_pixels,
                    video_fps=self.video_fps,
                    video_nframes=self.video_nframes,
                    cache_dir=self.video_cache_dir,
                    return_fps=True,
                )
                processed_videos.append(processed_video)
                video_fps_list.append(actual_video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example
