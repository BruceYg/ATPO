"""Batch reward runtime: parsing, accuracy rewards, ATPO controller, metrics, and state.

The runtime is independent of EasyR1. :mod:`atpo.rewards.easyr1` wraps one runtime
instance per reward worker process.

Timing contract (one call of :meth:`RewardRuntime.score_batch` = one batch):

1. read the controller coefficients (derived from earlier batches only);
2. score every sample of the batch with those coefficients;
3. compute FP/FN statistics over the batch and update the controller once
   (only if ``update_controller`` is true and at least one sample is counted).

In EasyR1 the training reward worker receives the full rollout batch of a step
(``rollout_batch_size × rollout.n`` responses) in a single call, so statistics are
aggregated over the global batch without cross-rank reduction, and the controller
is updated once per optimizer step. Configurations that call the training reward
more than once per step (``algorithm.online_filtering`` or ``adv_estimator=remax``)
are rejected by the patched trainer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..controllers import (
    AdaptiveTverskyController,
    ControllerConfig,
    ControllerStateError,
    ResolvedControllerConfig,
    resolve_controller_config,
)
from ..parsing import PARSE_SCOPES, parse_response
from ..taxonomy import Taxonomy, UnknownLabelError, load_taxonomy
from .functions import category_tversky, exact_match, jaccard, tversky

REWARD_KINDS = ("exact_match", "jaccard", "tversky", "atpo")
TRAINING_RESPONSE_FORMATS = ("think_answer", "triple_newline")
STATE_FORMAT = "atpo.reward_state/v1"

_CONTROLLER_KEYS = {
    "mode",
    "target_ratio",
    "rho",
    "eta",
    "epsilon",
    "scale",
    "c",
    "logit_leakage",
    "beta_clip",
    "init_logit_u",
    "init_fp_ema",
    "init_fn_ema",
    "invalid_prediction_policy",
}
_GENERAL_KEYS = {
    "taxonomy",
    "reward",
    "response_format",
    "parse_scope",
    "answer_only",
    "format_weight",
    "update_controller",
    "log_classification_metrics",
    "alpha",
    "beta",
    "controller",
}


class RewardConfigError(ValueError):
    """Invalid reward configuration."""


class GroundTruthError(ValueError):
    """A ground-truth record cannot be mapped onto the taxonomy."""


@dataclass(frozen=True)
class RewardSettings:
    taxonomy: Taxonomy
    kind: str
    response_format: str
    parse_scope: str
    format_weight: float
    alpha: Optional[float]
    beta: Optional[float]
    controller: Optional[ResolvedControllerConfig]
    update_controller: bool
    log_classification_metrics: bool

    def describe(self) -> dict[str, Any]:
        return {
            "taxonomy": self.taxonomy.name,
            "taxonomy_fingerprint": self.taxonomy.fingerprint(),
            "reward": self.kind,
            "response_format": self.response_format,
            "parse_scope": self.parse_scope,
            "format_weight": self.format_weight,
            "alpha": self.alpha,
            "beta": self.beta,
            "controller": None if self.controller is None else self.controller.to_dict(),
            "update_controller": self.update_controller,
            "log_classification_metrics": self.log_classification_metrics,
        }

    def fingerprint(self) -> str:
        blob = json.dumps(self.describe(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _as_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise RewardConfigError(f"{name} must be a boolean, got {value!r}")


def build_settings(kwargs: Mapping[str, Any]) -> RewardSettings:
    """Validate reward keyword arguments.

    Controller settings may be given flat (``target_ratio=5.0``; the historical
    launcher style) or nested under ``controller``. The historical keywords ``c``
    and ``answer_only`` are accepted.
    """
    kwargs = dict(kwargs)
    unknown = set(kwargs) - _GENERAL_KEYS - _CONTROLLER_KEYS
    if unknown:
        raise RewardConfigError(f"unknown reward settings: {sorted(unknown)}")
    if "taxonomy" not in kwargs:
        raise RewardConfigError("'taxonomy' is required (a built-in name or a taxonomy file)")
    taxonomy = load_taxonomy(kwargs.pop("taxonomy"))
    kind = kwargs.pop("reward", None)
    if kind not in REWARD_KINDS:
        raise RewardConfigError(f"'reward' must be one of {REWARD_KINDS}, got {kind!r}")

    response_format = kwargs.pop("response_format", "think_answer")
    if response_format not in TRAINING_RESPONSE_FORMATS:
        raise RewardConfigError(
            f"response_format must be one of {TRAINING_RESPONSE_FORMATS}, got {response_format!r}"
        )
    if "parse_scope" in kwargs and "answer_only" in kwargs:
        raise RewardConfigError("give either 'parse_scope' or the historical 'answer_only', not both")
    if "answer_only" in kwargs:
        parse_scope = "answer_only" if _as_bool("answer_only", kwargs.pop("answer_only")) else "full_response"
    else:
        # Historical default of every reward file: parse the full response.
        parse_scope = kwargs.pop("parse_scope", "full_response")
    if parse_scope not in PARSE_SCOPES:
        raise RewardConfigError(f"parse_scope must be one of {PARSE_SCOPES}, got {parse_scope!r}")

    format_weight = float(kwargs.pop("format_weight", 0.1))
    if not 0.0 <= format_weight <= 1.0:
        raise RewardConfigError(f"format_weight must be in [0, 1], got {format_weight}")
    update_controller = _as_bool("update_controller", kwargs.pop("update_controller", True))
    log_metrics = _as_bool("log_classification_metrics", kwargs.pop("log_classification_metrics", False))

    alpha = kwargs.pop("alpha", None)
    beta = kwargs.pop("beta", None)
    if kind == "tversky":
        alpha = 1.0 if alpha is None else float(alpha)
        beta = 1.0 if beta is None else float(beta)
        if alpha < 0 or beta < 0:
            raise RewardConfigError("static Tversky alpha and beta must be >= 0")
    elif alpha is not None or beta is not None:
        raise RewardConfigError("'alpha'/'beta' are only valid for reward=tversky")

    nested = kwargs.pop("controller", None)
    flat = {k: kwargs.pop(k) for k in list(kwargs) if k in _CONTROLLER_KEYS}
    controller: Optional[ResolvedControllerConfig] = None
    if kind == "atpo":
        if nested is not None and flat:
            raise RewardConfigError("give controller settings either nested or flat, not both")
        raw = dict(nested) if nested is not None else flat
        if "mode" not in raw:
            raise RewardConfigError("reward=atpo requires controller mode 'global' or 'category'")
        try:
            controller = resolve_controller_config(ControllerConfig.from_dict(raw), taxonomy.ids)
        except ValueError as exc:
            raise RewardConfigError(f"invalid controller settings: {exc}") from exc
    elif nested is not None or flat:
        raise RewardConfigError(
            f"controller settings {sorted(flat) or ['controller']} are only valid for reward=atpo"
        )
    if kwargs:  # pragma: no cover - guarded by the unknown-key check
        raise RewardConfigError(f"unused reward settings: {sorted(kwargs)}")
    return RewardSettings(
        taxonomy=taxonomy,
        kind=kind,
        response_format=response_format,
        parse_scope=parse_scope,
        format_weight=format_weight,
        alpha=alpha,
        beta=beta,
        controller=controller,
        update_controller=update_controller,
        log_classification_metrics=log_metrics,
    )


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, category_ids: Sequence[str]
) -> dict[str, float]:
    """Batch precision/recall/F1 (per category, macro, micro); zero-division gives 0.

    Matches the batch metrics logged by the reward functions of the paper's runs.
    """
    metrics: dict[str, float] = {}
    if y_true.shape[0] == 0:
        return metrics
    precisions, recalls, f1s = [], [], []
    total_tp = total_fp = total_fn = 0
    for i, cid in enumerate(category_ids):
        t, p = y_true[:, i], y_pred[:, i]
        tp = int(np.sum((t == 1) & (p == 1)))
        fp = int(np.sum((t == 0) & (p == 1)))
        fn = int(np.sum((t == 1) & (p == 0)))
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        metrics[f"{cid}_precision"] = float(precision)
        metrics[f"{cid}_recall"] = float(recall)
        metrics[f"{cid}_f1"] = float(f1)
        metrics[f"{cid}_support"] = float(tp + fn)
    metrics["macro_precision"] = float(np.mean(precisions))
    metrics["macro_recall"] = float(np.mean(recalls))
    metrics["macro_f1"] = float(np.mean(f1s))
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    metrics["micro_precision"] = float(micro_p)
    metrics["micro_recall"] = float(micro_r)
    metrics["micro_f1"] = float(micro_f1)
    return metrics


class RewardRuntime:
    """Scores batches and owns the ATPO controller of one reward worker."""

    def __init__(self, settings: RewardSettings) -> None:
        self.settings = settings
        self.taxonomy = settings.taxonomy
        self.controller: Optional[AdaptiveTverskyController] = (
            AdaptiveTverskyController(settings.controller) if settings.controller is not None else None
        )
        self.batches_scored = 0

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]) -> "RewardRuntime":
        return cls(build_settings(kwargs))

    @property
    def stateful(self) -> bool:
        return self.controller is not None

    # ------------------------------------------------------------- utilities
    def _ground_truth(self, value: Any) -> list[str]:
        if value is None:
            raise GroundTruthError("ground truth is missing")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise GroundTruthError(
                f"ground truth must be a list of category labels, got {type(value).__name__}: {value!r}"
            )
        try:
            return self.taxonomy.resolve_labels(value)
        except UnknownLabelError as exc:
            raise GroundTruthError(str(exc)) from exc

    def _coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.controller is not None
        return self.controller.alpha, self.controller.beta

    # ---------------------------------------------------------------- scoring
    def score_batch(self, reward_inputs: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
        s = self.settings
        tax = self.taxonomy
        ids = tax.ids
        k = len(ids)
        alpha_used = beta_used = None
        if self.controller is not None:
            alpha_used, beta_used = self._coefficients()

        scores: list[dict[str, float]] = []
        pred_rows: list[Optional[np.ndarray]] = []
        gt_rows: list[np.ndarray] = []
        for item in reward_inputs:
            response = item.get("response", "") or ""
            gt_ids = self._ground_truth(item.get("ground_truth"))
            parsed = parse_response(
                response,
                tax,
                response_format=s.response_format,  # type: ignore[arg-type]
                output_style="json",
                scope=s.parse_scope,  # type: ignore[arg-type]
            )
            pred_ids = None if parsed.labels is None else set(parsed.labels)
            gt_set = set(gt_ids)
            if s.kind == "exact_match":
                acc = exact_match(pred_ids, gt_set)
            elif s.kind == "jaccard":
                acc = jaccard(pred_ids, gt_set)
            elif s.kind == "tversky":
                acc = tversky(pred_ids, gt_set, s.alpha, s.beta)  # type: ignore[arg-type]
            elif self.controller is not None and self.controller.mode == "global":
                acc = tversky(pred_ids, gt_set, float(alpha_used[0]), float(beta_used[0]))
            else:
                pred_vec = None if pred_ids is None else np.array(tax.to_vector(pred_ids))
                acc = category_tversky(pred_vec, np.array(tax.to_vector(gt_set)), alpha_used, beta_used)
            fmt = 1.0 if parsed.format_ok else 0.0
            overall = (1.0 - s.format_weight) * acc + s.format_weight * fmt
            scores.append({"overall": float(overall), "format": float(fmt), "accuracy": float(acc)})
            pred_rows.append(None if pred_ids is None else np.array(tax.to_vector(pred_ids)))
            gt_rows.append(np.array(tax.to_vector(gt_set)))

        n = len(scores)
        if n == 0:
            return scores
        invalid = sum(1 for row in pred_rows if row is None)
        policy = "exclude"
        if self.controller is not None:
            policy = self.controller.config.invalid_prediction_policy
        elif s.kind in ("tversky", "exact_match", "jaccard"):
            # Static rewards have no controller; statistics are diagnostic only and
            # follow the static Tversky files, which counted invalid outputs as empty.
            policy = "as_empty"
        y_pred, y_true = [], []
        for pred, gt in zip(pred_rows, gt_rows):
            if pred is None:
                if policy == "exclude":
                    continue
                pred = np.zeros(k, dtype=np.int64)
            y_pred.append(pred)
            y_true.append(gt)

        metrics: dict[str, float] = {
            "invalid_rate": invalid / n,
            "counted_samples": float(len(y_pred)),
        }
        update_record = None
        if y_pred:
            yp = np.stack(y_pred, axis=0)
            yt = np.stack(y_true, axis=0)
            fp_matrix = (yp == 1) & (yt == 0)
            fn_matrix = (yp == 0) & (yt == 1)
            tp_matrix = (yp == 1) & (yt == 1)
            fp_per_sample = np.sum(fp_matrix, axis=1)
            fn_per_sample = np.sum(fn_matrix, axis=1)
            batch_fp = float(np.mean(fp_per_sample))
            batch_fn = float(np.mean(fn_per_sample))
            fp_rate = np.mean(fp_matrix, axis=0)
            fn_rate = np.mean(fn_matrix, axis=0)
            eps = self.controller.config.epsilon if self.controller is not None else 1e-8
            metrics.update(
                {
                    "batch_tp": float(np.mean(np.sum(tp_matrix, axis=1))),
                    "batch_fp": batch_fp,
                    "batch_fn": batch_fn,
                    "batch_fn_fp_ratio": (batch_fn + eps) / (batch_fp + eps),
                }
            )
            for i, cid in enumerate(ids):
                metrics[f"batch_fp_{cid}"] = float(fp_rate[i])
                metrics[f"batch_fn_{cid}"] = float(fn_rate[i])
            if s.log_classification_metrics:
                metrics.update(classification_metrics(yt, yp, ids))
            if self.controller is not None and s.update_controller:
                if self.controller.mode == "global":
                    update_record = self.controller.update([batch_fp], [batch_fn])
                else:
                    update_record = self.controller.update(fp_rate, fn_rate)

        if self.controller is not None:
            cfg = self.controller.config
            metrics["ctrl_updated"] = 1.0 if update_record is not None else 0.0
            metrics["ctrl_updates"] = float(self.controller.num_updates)
            labels = [""] if cfg.mode == "global" else [f"_{cid}" for cid in ids]
            fp_ema, fn_ema = self.controller.fp_ema, self.controller.fn_ema
            ratio, logit = self.controller.ema_ratio(), self.controller.logit_u
            for j, suffix in enumerate(labels):
                metrics[f"coef_alpha{suffix}"] = float(alpha_used[j])
                metrics[f"coef_beta{suffix}"] = float(beta_used[j])
                metrics[f"ctrl_target_ratio{suffix}"] = float(cfg.target_ratio[j])
                metrics[f"ctrl_fp_ema{suffix}"] = float(fp_ema[j])
                metrics[f"ctrl_fn_ema{suffix}"] = float(fn_ema[j])
                metrics[f"ctrl_ema_ratio{suffix}"] = float(ratio[j])
                metrics[f"ctrl_logit_u{suffix}"] = float(logit[j])

        for score in scores:
            score.update(metrics)
        self.batches_scored += 1
        return scores

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "format": STATE_FORMAT,
            "reward": self.settings.kind,
            "taxonomy": self.taxonomy.name,
            "taxonomy_fingerprint": self.taxonomy.fingerprint(),
            "settings": self.settings.describe(),
            "settings_fingerprint": self.settings.fingerprint(),
            "batches_scored": self.batches_scored,
            "controller": None if self.controller is None else self.controller.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any], *, allow_config_change: bool = False) -> None:
        if state.get("format") != STATE_FORMAT:
            raise ControllerStateError(f"unsupported reward state format {state.get('format')!r}")
        if state.get("reward") != self.settings.kind:
            raise ControllerStateError(
                f"saved reward {state.get('reward')!r} != configured {self.settings.kind!r}"
            )
        if state.get("taxonomy_fingerprint") != self.taxonomy.fingerprint():
            raise ControllerStateError(
                f"saved taxonomy {state.get('taxonomy')!r} does not match the configured taxonomy "
                f"{self.taxonomy.name!r} (label space fingerprint differs)"
            )
        saved_controller = state.get("controller")
        if self.controller is None:
            if saved_controller is not None:
                raise ControllerStateError("saved state has a controller but this reward has none")
        else:
            if saved_controller is None:
                raise ControllerStateError("saved state has no controller state")
            self.controller.load_state_dict(saved_controller, allow_config_change=allow_config_change)
        self.batches_scored = int(state.get("batches_scored", 0))
