"""Controller semantics: direction, scaling, validation, and exact checkpoint resumption."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest

from atpo.controllers import (
    AdaptiveTverskyController,
    ControllerConfig,
    ControllerConfigError,
    ControllerStateError,
)
from atpo.rewards import RewardConfigError, RewardRuntime
from atpo.taxonomy import load_taxonomy

from helpers import make_batch

IDS = ["a", "b", "c"]


def _controller(**overrides):
    cfg = {"mode": "category", "invalid_prediction_policy": "exclude", **overrides}
    return AdaptiveTverskyController(ControllerConfig.from_dict(cfg), IDS)


def test_initial_coefficients_are_balanced_and_scaled():
    ctrl = _controller(scale=2.0)
    assert np.allclose(ctrl.alpha, 1.0) and np.allclose(ctrl.beta, 1.0)
    ctrl = _controller()
    assert np.allclose(ctrl.alpha, 0.5) and np.allclose(ctrl.beta, 0.5)
    assert np.allclose(ctrl.alpha + ctrl.beta, 1.0)


def test_direction_more_false_negatives_raises_beta():
    ctrl = _controller(target_ratio=1.0, eta=0.5)
    ctrl.update([0.1, 0.1, 0.1], [0.4, 0.1, 0.01])  # FN/FP = 4, 1, 0.1
    beta = ctrl.beta
    assert beta[0] > 0.5  # FN-heavy: penalise misses more
    assert math.isclose(beta[1], 0.5, abs_tol=1e-6)  # on target: unchanged
    assert beta[2] < 0.5  # FP-heavy: penalise false alarms more
    assert np.allclose(ctrl.alpha + ctrl.beta, 1.0)


def test_target_ratio_shifts_equilibrium():
    # The same observed ratio of 1 is "too many false positives" for a target of 5.
    ctrl = _controller(target_ratio={"a": 5.0, "b": 1.0, "c": 0.2}, eta=0.5)
    ctrl.update([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    beta = ctrl.beta
    assert beta[0] < 0.5 and math.isclose(beta[1], 0.5, abs_tol=1e-6) and beta[2] > 0.5


def test_ema_weights_incoming_batch_by_rho():
    ctrl = _controller(rho=0.25)
    ctrl.update([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])  # first update initialises the EMA
    assert np.allclose(ctrl.fp_ema, 1.0)
    ctrl.update([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert np.allclose(ctrl.fp_ema, 0.75)  # (1 - rho) * history + rho * batch


def test_beta_clip_and_leakage():
    ctrl = _controller(beta_clip=0.2, eta=100.0)
    ctrl.update([0.0, 1.0, 0.1], [1.0, 0.0, 0.1])
    assert np.all(ctrl.beta <= 0.8 + 1e-12) and np.all(ctrl.beta >= 0.2 - 1e-12)
    leaky = _controller(logit_leakage=1.0, eta=0.0)
    leaky.update([0.0, 1.0, 0.1], [1.0, 0.0, 0.1])
    assert np.allclose(leaky.logit_u, 0.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"target_ratio": 0.0},
        {"target_ratio": -1.0},
        {"target_ratio": [1.0, 2.0]},  # wrong length
        {"target_ratio": {"a": 1.0, "b": 1.0}},  # missing category
        {"target_ratio": {"a": 1.0, "b": 1.0, "c": 1.0, "zzz": 1.0}},  # unknown category
        {"rho": 0.0},
        {"rho": 1.5},
        {"eta": -0.1},
        {"scale": 0.0},
        {"beta_clip": 0.5},
        {"logit_leakage": 2.0},
        {"init_fp_ema": -1.0},
        {"target_ratio": True},
        {"target_ratio": "1.0"},
        {"epsilon": 0.0},
    ],
)
def test_invalid_configurations_are_rejected(bad):
    with pytest.raises(ControllerConfigError):
        _controller(**bad)


def test_global_mode_requires_scalars_and_policy():
    with pytest.raises(ControllerConfigError):
        AdaptiveTverskyController(
            ControllerConfig(mode="global", target_ratio=[1, 1, 1], invalid_prediction_policy="as_empty"), IDS
        )
    with pytest.raises(ControllerConfigError):
        AdaptiveTverskyController(ControllerConfig(mode="global"), IDS)  # policy missing


def test_state_roundtrip_is_bit_exact_through_json():
    rng = np.random.default_rng(0)
    a = _controller(eta=0.3, target_ratio=[1, 5, 0.2], init_fp_ema=0.1)
    for _ in range(7):
        a.update(rng.random(3), rng.random(3))
    blob = json.dumps(a.state_dict())
    b = _controller(eta=0.3, target_ratio=[1, 5, 0.2], init_fp_ema=0.1)
    b.load_state_dict(json.loads(blob))
    for _ in range(9):
        fp, fn = rng.random(3), rng.random(3)
        a.update(fp, fn)
        b.update(fp, fn)
        assert np.array_equal(a.alpha, b.alpha) and np.array_equal(a.beta, b.beta)
        assert np.array_equal(a.fp_ema, b.fp_ema) and np.array_equal(a.logit_u, b.logit_u)
    assert a.num_updates == b.num_updates == 16


def test_state_restore_rejects_mismatches():
    a = _controller(target_ratio=2.0)
    a.update([0.1, 0.2, 0.3], [0.3, 0.2, 0.1])
    state = a.state_dict()
    with pytest.raises(ControllerStateError):
        _controller(target_ratio=3.0).load_state_dict(state)  # hyperparameters changed
    other = AdaptiveTverskyController(
        ControllerConfig(mode="category", target_ratio=2.0, invalid_prediction_policy="exclude"), ["a", "b", "x"]
    )
    with pytest.raises(ControllerStateError):
        other.load_state_dict(state)  # different categories
    changed = _controller(target_ratio=3.0)
    changed.load_state_dict(state, allow_config_change=True)  # explicit opt-in keeps statistics
    assert np.array_equal(changed.fp_ema, a.fp_ema) and changed.num_updates == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward": "atpo", "mode": "global", "invalid_prediction_policy": "as_empty", "c": 2.0, "answer_only": True},
        {
            "reward": "atpo", "mode": "category", "invalid_prediction_policy": "exclude", "c": 2.0,
            "answer_only": True, "target_ratio": [1, 1, 1, 5, 1, 0.2],
        },
    ],
)
def test_runtime_resume_matches_uninterrupted_run(kwargs):
    taxonomy = load_taxonomy("safewatch")
    rng = random.Random(1)
    batches = [make_batch(rng, taxonomy, "think_answer", 32) for _ in range(12)]
    full = RewardRuntime.from_kwargs({"taxonomy": "safewatch", **kwargs})
    expected = [full.score_batch(b) for b in batches]

    first = RewardRuntime.from_kwargs({"taxonomy": "safewatch", **kwargs})
    for b in batches[:5]:
        first.score_batch(b)
    saved = json.loads(json.dumps(first.state_dict()))
    resumed = RewardRuntime.from_kwargs({"taxonomy": "safewatch", **kwargs})
    resumed.load_state_dict(saved)
    for b, exp in zip(batches[5:], expected[5:]):
        got = resumed.score_batch(b)
        assert [g["overall"] for g in got] == [e["overall"] for e in exp]
    assert resumed.controller.num_updates == full.controller.num_updates

    # Without restoring, the controller restarts from balanced coefficients
    # (historical behaviour) and later rewards differ.
    fresh = RewardRuntime.from_kwargs({"taxonomy": "safewatch", **kwargs})
    got = fresh.score_batch(batches[5])
    assert [g["overall"] for g in got] != [e["overall"] for e in expected[5]]


def test_frozen_validation_runtime_never_updates():
    taxonomy = load_taxonomy("safewatch")
    rng = random.Random(2)
    val = RewardRuntime.from_kwargs(
        {
            "taxonomy": "safewatch", "reward": "atpo", "mode": "global", "invalid_prediction_policy": "as_empty",
            "c": 2.0, "answer_only": True, "update_controller": False,
        }
    )
    for _ in range(5):
        scores = val.score_batch(make_batch(rng, taxonomy, "think_answer", 16))
        assert scores[0]["ctrl_updated"] == 0.0
    assert val.controller.num_updates == 0
    # With c = 2 the frozen coefficients are alpha = beta = 1: the reward is the Jaccard index.
    assert np.allclose(val.controller.alpha, 1.0) and np.allclose(val.controller.beta, 1.0)


def test_reward_state_rejects_other_taxonomy():
    a = RewardRuntime.from_kwargs(
        {"taxonomy": "safewatch", "reward": "atpo", "mode": "global", "invalid_prediction_policy": "as_empty"}
    )
    b = RewardRuntime.from_kwargs(
        {"taxonomy": "xdviolence", "reward": "atpo", "mode": "global", "invalid_prediction_policy": "as_empty"}
    )
    with pytest.raises(ControllerStateError):
        b.load_state_dict(a.state_dict())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward": "atpo"},  # no mode
        {"reward": "atpo", "mode": "global"},  # no invalid policy
        {"reward": "tversky", "target_ratio": 2.0},  # controller key on a static reward
        {"reward": "jaccard", "alpha": 1.0},
        {"reward": "atpo", "mode": "global", "invalid_prediction_policy": "as_empty", "answer_only": True,
         "parse_scope": "answer_only"},
        {"reward": "atpo", "mode": "global", "invalid_prediction_policy": "as_empty", "unknown_key": 1},
        {"reward": "atpo", "mode": "global", "invalid_prediction_policy": "as_empty", "response_format": "plain"},
        {"reward": "nope"},
    ],
)
def test_reward_configuration_errors(kwargs):
    with pytest.raises(RewardConfigError):
        RewardRuntime.from_kwargs({"taxonomy": "safewatch", **kwargs})


def test_ground_truth_errors_are_loud():
    runtime = RewardRuntime.from_kwargs({"taxonomy": "safewatch", "reward": "jaccard"})
    with pytest.raises(ValueError):
        runtime.score_batch([{"response": "", "ground_truth": [7]}])
    with pytest.raises(ValueError):
        runtime.score_batch([{"response": "", "ground_truth": "[1]"}])


def test_equal_length_ground_truth_arrays_are_accepted():
    """EasyR1's collate turns equal-length label lists into a 2-D object array.

    The historical reward files rejected the resulting ndarray rows (accuracy 0).
    The release accepts them; see docs/correctness_fixes.md.
    """
    runtime = RewardRuntime.from_kwargs({"taxonomy": "safewatch", "reward": "jaccard"})
    gt = np.array([[2], [3]], dtype=object)
    response = '<think>x</think><answer>"C2(Harassment & Bullying)": true</answer>'
    scores = runtime.score_batch([{"response": response, "ground_truth": gt[0]}])
    assert scores[0]["accuracy"] == 1.0
