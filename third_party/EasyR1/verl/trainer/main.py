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

import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_config = config.worker.val_reward or config.worker.reward
        ValRemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=val_reward_config.num_cpus)
        val_reward_fn = ValRemoteRewardManager.remote(val_reward_config, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        # Collect NCCL environment variables from parent process
        import os
        nccl_env_vars = {}
        for key in [
            "NCCL_DEBUG", "NCCL_IB_DISABLE", "NCCL_IB_TIMEOUT", "NCCL_IB_RETRY_CNT",
            "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_CACHE_DIR",
            "NCCL_ASYNC_ERROR_HANDLING", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE"
        ]:
            if key in os.environ:
                nccl_env_vars[key] = os.environ[key]
        
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
                **nccl_env_vars,  # Add NCCL variables from parent process
            }
        }
        # Configure object store memory and directory for multi-GPU setups
        # Ray can use a custom directory on a large mounted drive instead of /dev/shm
        import shutil
        from pathlib import Path
        
        object_store_memory = None
        temp_dir = None
        
        # Check for custom object store directory (preferred for large drives)
        if "RAY_OBJECT_STORE_DIR" in os.environ:
            temp_dir = os.environ["RAY_OBJECT_STORE_DIR"]
            # Ensure directory exists
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            # Validate path length: Unix sockets have 107-byte limit
            # Ray adds: /session_TIMESTAMP_PID/sockets/plasma_store (~53 chars)
            max_base_path_length = 107 - 53  # Reserve space for Ray's subdirectories
            if len(temp_dir) > max_base_path_length:
                raise ValueError(
                    f"Ray object store directory path too long: {len(temp_dir)} chars. "
                    f"Maximum is {max_base_path_length} chars (Unix socket limit is 107 bytes). "
                    f"Current path: {temp_dir}"
                )
            print(f"Using custom Ray object store directory: {temp_dir} (path length: {len(temp_dir)} chars)")
        
        # Configure object store memory
        if "RAY_OBJECT_STORE_MEMORY" in os.environ:
            # Allow explicit override via environment variable
            object_store_memory = int(os.environ["RAY_OBJECT_STORE_MEMORY"])
            print(f"Using explicit Ray object store memory: {object_store_memory / (1024**3):.2f} GB")
        elif temp_dir:
            # If using custom directory, use available space on that drive
            try:
                drive_stats = shutil.disk_usage(temp_dir)
                # Use 30% of available space on the custom drive, but cap at 500GB
                object_store_memory = min(int(drive_stats.free * 0.3), 500 * 1024 * 1024 * 1024)
                print(f"Auto-configured Ray object store memory from {temp_dir}: {object_store_memory / (1024**3):.2f} GB")
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not check disk usage for {temp_dir}: {e}")
                # Fall back to a reasonable default (100GB)
                object_store_memory = 100 * 1024 * 1024 * 1024
                print(f"Using default Ray object store memory: {object_store_memory / (1024**3):.2f} GB")
        else:
            # Try to use 50% of available /dev/shm if it exists, otherwise use default
            try:
                shm_stats = shutil.disk_usage("/dev/shm")
                # Use 50% of available /dev/shm, but cap at 200GB
                object_store_memory = min(shm_stats.free // 2, 200 * 1024 * 1024 * 1024)
                print(f"Auto-configured Ray object store memory from /dev/shm: {object_store_memory / (1024**3):.2f} GB")
            except (OSError, PermissionError):
                # Fall back to Ray's default
                pass
        
        init_kwargs = {"runtime_env": runtime_env}
        if object_store_memory is not None:
            init_kwargs["object_store_memory"] = object_store_memory
        if temp_dir is not None:
            init_kwargs["_temp_dir"] = temp_dir
        ray.init(**init_kwargs)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
