"""CPU integration test of the vendored EasyR1 reward-state checkpointing patch.

Uses real Ray actors running EasyR1's ``AutoRewardManager`` with the release reward
module, and calls the patched ``RayPPOTrainer`` checkpoint methods. Model workers
are replaced by recording fakes, so no GPU or model weights are needed. The
trainer module imports the FSDP worker stack (GPU training dependencies), which is
stubbed because the checkpoint logic under test does not use it.
"""

from __future__ import annotations

import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")
for _name in ("tensordict", "codetiming", "torchdata", "omegaconf"):
    pytest.importorskip(_name)

from atpo.taxonomy import load_taxonomy  # noqa: E402

from easyr1_fakes import FakeDataLoader, FakeWorkerGroup, TableTokenizer  # noqa: E402
from helpers import make_response  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EASYR1 = REPO_ROOT / "third_party" / "EasyR1"
REWARD_FILE = REPO_ROOT / "atpo" / "rewards" / "easyr1.py"

ATPO_C = {
    "taxonomy": "safewatch",
    "reward": "atpo",
    "mode": "category",
    "invalid_prediction_policy": "exclude",
    "c": 2.0,
    "answer_only": True,
    "target_ratio": [1, 1, 1, 5, 1, 0.2],
}


@pytest.fixture(scope="module")
def verl():
    sys.path.insert(0, str(EASYR1))
    if "verl.workers.fsdp_workers" not in sys.modules:
        stub = types.ModuleType("verl.workers.fsdp_workers")
        stub.FSDPWorker = type("FSDPWorker", (), {})
        sys.modules["verl.workers.fsdp_workers"] = stub
    from verl.protocol import DataProto
    from verl.trainer import ray_trainer
    from verl.trainer.config import PPOConfig
    from verl.workers.reward import AutoRewardManager, RewardConfig

    return types.SimpleNamespace(
        DataProto=DataProto,
        ray_trainer=ray_trainer,
        PPOConfig=PPOConfig,
        AutoRewardManager=AutoRewardManager,
        RewardConfig=RewardConfig,
    )


@pytest.fixture(scope="module")
def cluster():
    paths = [str(EASYR1), str(REPO_ROOT), str(REPO_ROOT / "tests")]
    env_path = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": env_path, "TOKENIZERS_PARALLELISM": "false"}},
    )
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def table():
    """Responses and ground truth for 12 batches of 24 samples."""
    taxonomy = load_taxonomy("safewatch")
    rng = random.Random(7)
    responses, gts = [], []
    for _ in range(12 * 24):
        r = rng.random()
        gt = [] if r < 0.3 else ([rng.choice(taxonomy.ids)] if r < 0.85 else rng.sample(taxonomy.ids, 2))
        responses.append(make_response(rng, taxonomy, "think_answer", gt))
        gts.append([taxonomy.index(c) + 1 for c in gt])
    return responses, gts


def _batch(verl, table, k: int, size: int = 24):
    _, gts = table
    ids = list(range(k * size, (k + 1) * size))
    ground_truth = np.empty(len(ids), dtype=object)
    for j, i in enumerate(ids):
        ground_truth[j] = gts[i]
    return verl.DataProto.from_single_dict(
        {
            "responses": torch.tensor(ids, dtype=torch.long).unsqueeze(-1),
            "response_mask": torch.ones(len(ids), 1, dtype=torch.long),
            "ground_truth": ground_truth,
        }
    )


def _manager(verl, table, kwargs):
    config = verl.RewardConfig(reward_function=f"{REWARD_FILE}:compute_score", reward_function_kwargs=dict(kwargs))
    config.post_init()
    return ray.remote(verl.AutoRewardManager).remote(config, TableTokenizer(table[0]))


def _trainer(verl, tmp_path, train, val, **trainer_overrides):
    config = verl.PPOConfig()
    config.trainer.save_checkpoint_path = str(tmp_path)
    config.trainer.find_last_checkpoint = False
    for key, value in trainer_overrides.items():
        setattr(config.trainer, key, value)
    trainer = verl.ray_trainer.RayPPOTrainer.__new__(verl.ray_trainer.RayPPOTrainer)
    trainer.config = config
    trainer.reward_fn = train
    trainer.val_reward_fn = val
    trainer.actor_rollout_ref_wg = FakeWorkerGroup()
    trainer.train_dataloader = FakeDataLoader()
    trainer.use_critic = False
    trainer.global_step = 0
    trainer.val_reward_score = 0.0
    trainer.best_val_reward_score = -1.0
    trainer.best_global_step = None
    return trainer


def _score(manager, batch):
    tensor, metrics = ray.get(manager.compute_reward.remote(batch))
    return tensor.sum(-1).tolist(), metrics


def test_reward_state_is_saved_and_restored_exactly(verl, cluster, table, tmp_path):
    val_kwargs = {**ATPO_C, "update_controller": False}
    train, val = _manager(verl, table, ATPO_C), _manager(verl, table, val_kwargs)
    trainer = _trainer(verl, tmp_path, train, val)
    for k in range(3):
        _score(train, _batch(verl, table, k))
        _score(val, _batch(verl, table, 10))
    trainer.global_step = 3
    trainer._save_checkpoint()

    folder = tmp_path / "global_step_3"
    state = json.loads((folder / "reward_state.json").read_text())
    val_state = json.loads((folder / "val_reward_state.json").read_text())
    assert state["controller"]["num_updates"] == 3 and state["batches_scored"] == 3
    assert val_state["controller"]["num_updates"] == 0 and val_state["batches_scored"] == 3
    assert (tmp_path / "checkpoint_tracker.json").exists()

    expected = [_score(train, _batch(verl, table, k)) for k in range(3, 6)]

    # Fresh reward workers, as after a job restart.
    train2, val2 = _manager(verl, table, ATPO_C), _manager(verl, table, val_kwargs)
    resumed = _trainer(verl, tmp_path, train2, val2, load_checkpoint_path=str(folder))
    resumed._load_checkpoint()
    assert resumed.global_step == 3
    restored = ray.get(train2.get_reward_state.remote())
    assert restored["controller"]["exact"] == state["controller"]["exact"]
    got = [_score(train2, _batch(verl, table, k)) for k in range(3, 6)]
    for (exp_scores, exp_metrics), (got_scores, got_metrics) in zip(expected, got):
        assert exp_scores == got_scores
        assert exp_metrics["coef_beta_C4"] == got_metrics["coef_beta_C4"]

    # Without the restore the controller restarts from balanced coefficients.
    fresh = _manager(verl, table, ATPO_C)
    assert _score(fresh, _batch(verl, table, 3))[0] != expected[0][0]


def test_missing_reward_state_fails_unless_allowed(verl, cluster, table, tmp_path):
    folder = tmp_path / "global_step_2"
    folder.mkdir()
    train = _manager(verl, table, ATPO_C)
    strict = _trainer(verl, tmp_path, train, None, load_checkpoint_path=str(folder))
    with pytest.raises(RuntimeError, match="allow_missing_reward_state"):
        strict._load_checkpoint()
    lenient = _trainer(
        verl, tmp_path, train, None, load_checkpoint_path=str(folder), allow_missing_reward_state=True
    )
    lenient._load_checkpoint()
    assert ray.get(train.get_reward_state.remote())["controller"]["num_updates"] == 0


def test_static_rewards_need_no_state(verl, cluster, table, tmp_path):
    folder = tmp_path / "global_step_1"
    folder.mkdir()
    train = _manager(verl, table, {"taxonomy": "safewatch", "reward": "jaccard"})
    assert ray.get(train.is_stateful.remote()) is False
    trainer = _trainer(verl, tmp_path, train, None, load_checkpoint_path=str(folder))
    trainer._load_checkpoint()  # no state file needed
    trainer.config.algorithm.online_filtering = True
    trainer._check_reward_call_pattern()  # allowed for stateless rewards


def test_stateful_reward_rejects_multiple_calls_per_step(verl, cluster, table, tmp_path):
    train = _manager(verl, table, ATPO_C)
    trainer = _trainer(verl, tmp_path, train, None)
    trainer.config.algorithm.online_filtering = True
    with pytest.raises(NotImplementedError):
        trainer._check_reward_call_pattern()
    trainer.config.algorithm.online_filtering = False
    trainer.config.algorithm.adv_estimator = "remax"
    with pytest.raises(NotImplementedError):
        trainer._check_reward_call_pattern()


def test_invalid_reward_config_fails_at_worker_start(verl, cluster, table):
    bad = _manager(verl, table, {"taxonomy": "safewatch", "reward": "atpo", "mode": "global"})
    with pytest.raises(Exception, match="invalid_prediction_policy"):
        ray.get(bad.is_stateful.remote())


def test_val_reward_override_from_cli(verl):
    from omegaconf import OmegaConf

    default = OmegaConf.structured(verl.PPOConfig())
    cli = OmegaConf.from_dotlist(
        [
            f"worker.val_reward.reward_function={REWARD_FILE}:compute_score",
            "worker.val_reward.reward_function_kwargs.taxonomy=safewatch",
            "worker.val_reward.reward_function_kwargs.update_controller=false",
            "worker.reward.reward_function_kwargs.target_ratio=[1,1,1,5,1,0.2]",
        ]
    )
    config = OmegaConf.to_object(OmegaConf.merge(default, cli))
    config.worker.val_reward.post_init()
    assert config.worker.val_reward.reward_function_name == "compute_score"
    assert config.worker.val_reward.reward_function_kwargs == {"taxonomy": "safewatch", "update_controller": False}
    assert config.worker.reward.reward_function_kwargs["target_ratio"] == [1, 1, 1, 5, 1, 0.2]
