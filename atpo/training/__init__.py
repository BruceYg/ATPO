"""Training launchers: EasyR1 (GRPO/ATPO) and LLaMA-Factory (SFT)."""

from .easyr1 import TrainingConfig, compose_easyr1_config, launch_easyr1, load_training_config
from .llamafactory import compose_llamafactory_config, compose_merge_config, launch_llamafactory

__all__ = [
    "TrainingConfig",
    "compose_easyr1_config",
    "compose_llamafactory_config",
    "compose_merge_config",
    "launch_easyr1",
    "launch_llamafactory",
    "load_training_config",
]
