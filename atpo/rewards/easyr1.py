"""EasyR1 batch reward entry point for GRPO baselines and ATPO.

Point EasyR1 at this file::

    worker.reward.reward_function=<path to this file>:compute_score
    worker.reward.reward_function_kwargs={taxonomy: safewatch, reward: atpo, mode: global, ...}

``atpo train`` fills in the path automatically.

EasyR1 loads the file once per reward worker (a Ray actor); training and
validation use separate workers, so each has its own runtime and controller.
The bundled EasyR1 snapshot (``third_party/EasyR1``) additionally calls

* :func:`configure` when the worker starts, so configuration errors surface
  before any rollout is generated;
* :func:`get_state` / :func:`set_state` when saving and restoring checkpoints,
  so the controller resumes exactly where it stopped.

With an unmodified EasyR1 the runtime is created on the first ``compute_score``
call and controller state is not checkpointed.
"""

from __future__ import annotations

from typing import Any, Optional

from atpo.rewards.runtime import RewardRuntime

REWARD_NAME = "atpo"
REWARD_TYPE = "batch"

_RUNTIME: Optional[RewardRuntime] = None
_KWARGS: Optional[dict[str, Any]] = None
_PENDING_STATE: Optional[dict[str, Any]] = None
_PENDING_ALLOW_CHANGE = False


def _normalize(kwargs: dict[str, Any]) -> dict[str, Any]:
    # OmegaConf containers arrive as plain dicts/lists after OmegaConf.to_object.
    return {k: v for k, v in kwargs.items()}


def configure(**kwargs: Any) -> None:
    """Create the runtime from reward kwargs (idempotent for identical kwargs)."""
    global _RUNTIME, _KWARGS, _PENDING_STATE
    kwargs = _normalize(kwargs)
    if _RUNTIME is not None:
        if kwargs != _KWARGS:
            raise RuntimeError(
                "reward kwargs changed after the ATPO reward runtime was created; "
                f"configured with {_KWARGS}, called with {kwargs}"
            )
        return
    _RUNTIME = RewardRuntime.from_kwargs(kwargs)
    _KWARGS = kwargs
    if _PENDING_STATE is not None:
        _RUNTIME.load_state_dict(_PENDING_STATE, allow_config_change=_PENDING_ALLOW_CHANGE)
        _PENDING_STATE = None


def compute_score(reward_inputs: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, float]]:
    configure(**kwargs)
    assert _RUNTIME is not None
    return _RUNTIME.score_batch(reward_inputs)


def is_stateful() -> bool:
    """Whether the configured reward carries controller state that must be checkpointed."""
    if _RUNTIME is None:
        raise RuntimeError("reward runtime is not configured yet")
    return _RUNTIME.stateful


def get_state() -> Optional[dict[str, Any]]:
    if _RUNTIME is None:
        raise RuntimeError("reward runtime is not configured yet")
    return _RUNTIME.state_dict()


def set_state(state: dict[str, Any], allow_config_change: bool = False) -> None:
    """Restore state; applied at configuration time if the runtime does not exist yet."""
    global _PENDING_STATE, _PENDING_ALLOW_CHANGE
    if _RUNTIME is None:
        _PENDING_STATE = state
        _PENDING_ALLOW_CHANGE = allow_config_change
        return
    _RUNTIME.load_state_dict(state, allow_config_change=allow_config_change)


def describe() -> dict[str, Any]:
    if _RUNTIME is None:
        raise RuntimeError("reward runtime is not configured yet")
    return _RUNTIME.settings.describe()
