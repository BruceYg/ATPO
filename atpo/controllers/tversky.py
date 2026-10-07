"""Adaptive Tversky coefficient controllers (ATPO-G and ATPO-C).

The controller tracks exponential moving averages (EMAs) of false positives (FP)
and false negatives (FN) and moves a logit ``u`` so that the observed FN/FP ratio
approaches a target ratio ``r*``. The Tversky coefficients used by the reward are

    beta  = c * clip(sigmoid(u), beta_clip, 1 - beta_clip)    (FN weight)
    alpha = c * (1 - clip(sigmoid(u), beta_clip, 1 - beta_clip))  (FP weight)

One call to :meth:`AdaptiveTverskyController.update` performs, in order
(as in the reward functions of the paper's runs)::

    first update:  EMA <- batch statistic        (or the smoothed value if an EMA initializer was given)
    later updates: EMA <- (1 - rho) * EMA + rho * batch statistic
    r   = (FN_EMA + eps) / (FP_EMA + eps)
    u   <- (1 - leakage) * u + eta * (log(r + eps) - log(r* + eps))
    alpha, beta <- coefficients from u (see above)

``rho`` therefore weights the *incoming batch*; ``1 - rho`` weights history.
If FN dominates (r > r*), ``u`` and ``beta`` increase, penalizing misses more.

Modes:

``global`` (ATPO-G)
    One coefficient pair shared by all categories. Statistics are scalars:
    the per-sample number of FP (or FN) summed over categories, averaged over samples.
``category`` (ATPO-C)
    One coefficient pair per category. Statistics are per-category FP/FN rates
    averaged over samples; targets may differ per category.

The coefficients returned by :attr:`alpha`/:attr:`beta` are the ones to use for the
*next* batch to be scored. The batch statistics passed to ``update`` must come
from that same batch, so a batch is always scored with coefficients derived only
from earlier batches.

Initial coefficients follow the paper's runs: with no initializer, ``u = 0``
and ``alpha = beta = c / 2``; no clipping is applied before the first update.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np

ControllerMode = Literal["global", "category"]
InvalidPolicy = Literal["as_empty", "exclude"]
ScalarOrPerCategory = Union[float, int, Sequence[float], Mapping[str, float]]

STATE_FORMAT = "atpo.controller_state/v1"

class ControllerConfigError(ValueError):
    """Invalid controller configuration."""


class ControllerStateError(ValueError):
    """Controller state cannot be restored into this controller."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid, identical in arithmetic to the historical implementation."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    pos = x >= 0
    neg = ~pos
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    result[neg] = exp_x / (1.0 + exp_x)
    return result


@dataclass
class ControllerConfig:
    """User-facing controller configuration.

    ``target_ratio``, ``init_logit_u``, ``init_fp_ema`` and ``init_fn_ema`` accept a
    scalar, a list in taxonomy order, or a mapping keyed by category ID that covers
    every category. In ``global`` mode they must be scalars.

    ``scale`` is the historical ``c`` keyword: ``alpha + beta = scale``. With
    ``scale=2`` and balanced coefficients the Tversky index equals the Jaccard index.

    ``invalid_prediction_policy`` (required) controls how unparseable model outputs
    enter the FP/FN statistics: ``as_empty`` counts them as predicting no category
    (so they add false negatives), ``exclude`` leaves them out. The paper's runs
    differ: ATPO-G and the Qwen3-VL ATPO-C runs used ``as_empty``, the Qwen2.5-VL
    ATPO-C runs ``exclude``. The accuracy reward of an
    unparseable output is 0 in both cases.
    """

    mode: ControllerMode = "global"
    target_ratio: ScalarOrPerCategory = 1.0
    rho: float = 0.05
    eta: float = 1e-2
    epsilon: float = 1e-8
    scale: float = 1.0
    logit_leakage: float = 0.0
    beta_clip: float = 0.0
    init_logit_u: Optional[ScalarOrPerCategory] = None
    init_fp_ema: Optional[ScalarOrPerCategory] = None
    init_fn_ema: Optional[ScalarOrPerCategory] = None
    invalid_prediction_policy: Optional[InvalidPolicy] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ControllerConfig":
        data = dict(data)
        if "c" in data:  # historical keyword
            if "scale" in data and data["scale"] != data["c"]:
                raise ControllerConfigError("both 'c' and 'scale' given with different values")
            data["scale"] = data.pop("c")
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ControllerConfigError(f"unknown controller settings: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, tuple):
                out[key] = list(value)
        return out


@dataclass(frozen=True)
class ResolvedControllerConfig:
    """Validated configuration with per-category settings expanded to vectors."""

    mode: ControllerMode
    category_ids: tuple[str, ...]
    target_ratio: tuple[float, ...]
    rho: float
    eta: float
    epsilon: float
    scale: float
    logit_leakage: float
    beta_clip: float
    init_logit_u: Optional[tuple[float, ...]]
    init_fp_ema: Optional[tuple[float, ...]]
    init_fn_ema: Optional[tuple[float, ...]]
    invalid_prediction_policy: InvalidPolicy

    @property
    def dim(self) -> int:
        return 1 if self.mode == "global" else len(self.category_ids)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, tuple):
                out[key] = list(value)
        return out

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _expand(
    name: str,
    value: Any,
    mode: str,
    category_ids: Sequence[str],
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> tuple[float, ...]:
    dim = 1 if mode == "global" else len(category_ids)
    if isinstance(value, (bool, np.bool_)):
        raise ControllerConfigError(f"{name} must be numeric, got {value!r}")
    if isinstance(value, Mapping):
        if mode == "global":
            raise ControllerConfigError(f"{name} must be a scalar in global mode")
        keys = {str(k) for k in value}
        missing = [cid for cid in category_ids if cid not in keys]
        extra = sorted(keys - set(category_ids))
        if missing or extra:
            raise ControllerConfigError(
                f"{name} mapping must cover exactly the taxonomy categories; "
                f"missing={missing}, unknown={extra}"
            )
        vector = [value[cid] for cid in category_ids]
    elif isinstance(value, (str, bytes)):
        raise ControllerConfigError(f"{name} must be numeric, got {value!r}")
    elif isinstance(value, Sequence) or isinstance(value, np.ndarray):
        vector = list(np.asarray(value).reshape(-1)) if isinstance(value, np.ndarray) else list(value)
        if np.asarray(value).ndim > 1:
            raise ControllerConfigError(f"{name} must be a flat sequence")
        if mode == "global":
            raise ControllerConfigError(f"{name} must be a scalar in global mode")
        if len(vector) != dim:
            raise ControllerConfigError(
                f"{name} has {len(vector)} values but the taxonomy has {dim} categories"
            )
    else:
        vector = [value] * dim
    out = []
    for item in vector:
        if isinstance(item, (bool, np.bool_)) or isinstance(item, (str, bytes)):
            raise ControllerConfigError(f"{name} values must be numeric, got {item!r}")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ControllerConfigError(f"{name} values must be numeric, got {item!r}") from exc
        if not math.isfinite(number):
            raise ControllerConfigError(f"{name} values must be finite, got {item!r}")
        if positive and number <= 0.0:
            raise ControllerConfigError(f"{name} values must be > 0, got {number}")
        if non_negative and number < 0.0:
            raise ControllerConfigError(f"{name} values must be >= 0, got {number}")
        out.append(number)
    return tuple(out)


def resolve_controller_config(
    config: ControllerConfig | Mapping[str, Any], category_ids: Sequence[str]
) -> ResolvedControllerConfig:
    """Validate a controller configuration against a taxonomy's category IDs."""
    if isinstance(config, Mapping):
        config = ControllerConfig.from_dict(config)
    category_ids = tuple(str(c) for c in category_ids)
    if not category_ids:
        raise ControllerConfigError("controller needs at least one category")
    if len(set(category_ids)) != len(category_ids):
        raise ControllerConfigError("duplicate category ids")
    mode = config.mode
    if mode not in ("global", "category"):
        raise ControllerConfigError(f"mode must be 'global' or 'category', got {mode!r}")

    def _scalar(name: str, value: Any, lo: float, hi: float, lo_open: bool, hi_open: bool) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ControllerConfigError(f"{name} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ControllerConfigError(f"{name} must be numeric, got {value!r}") from exc
        if not math.isfinite(number):
            raise ControllerConfigError(f"{name} must be finite")
        if (number < lo or (lo_open and number == lo)) or (number > hi or (hi_open and number == hi)):
            lo_b = "(" if lo_open else "["
            hi_b = ")" if hi_open else "]"
            raise ControllerConfigError(f"{name}={number} must be in {lo_b}{lo}, {hi}{hi_b}")
        return number

    rho = _scalar("rho", config.rho, 0.0, 1.0, True, False)
    eta = _scalar("eta", config.eta, 0.0, math.inf, False, True)
    epsilon = _scalar("epsilon", config.epsilon, 0.0, math.inf, True, True)
    scale = _scalar("scale", config.scale, 0.0, math.inf, True, True)
    leakage = _scalar("logit_leakage", config.logit_leakage, 0.0, 1.0, False, False)
    beta_clip = _scalar("beta_clip", config.beta_clip, 0.0, 0.5, False, True)

    policy = config.invalid_prediction_policy
    if policy is None:
        raise ControllerConfigError(
            "invalid_prediction_policy is required ('as_empty' or 'exclude'); the paper's "
            "recipes differ, see docs/controller.md"
        )
    if policy not in ("as_empty", "exclude"):
        raise ControllerConfigError(
            f"invalid_prediction_policy must be 'as_empty' or 'exclude', got {policy!r}"
        )
    return ResolvedControllerConfig(
        mode=mode,
        category_ids=category_ids,
        target_ratio=_expand("target_ratio", config.target_ratio, mode, category_ids, positive=True),
        rho=rho,
        eta=eta,
        epsilon=epsilon,
        scale=scale,
        logit_leakage=leakage,
        beta_clip=beta_clip,
        init_logit_u=None
        if config.init_logit_u is None
        else _expand("init_logit_u", config.init_logit_u, mode, category_ids),
        init_fp_ema=None
        if config.init_fp_ema is None
        else _expand("init_fp_ema", config.init_fp_ema, mode, category_ids, non_negative=True),
        init_fn_ema=None
        if config.init_fn_ema is None
        else _expand("init_fn_ema", config.init_fn_ema, mode, category_ids, non_negative=True),
        invalid_prediction_policy=policy,  # type: ignore[arg-type]
    )


@dataclass
class ControllerUpdate:
    """Record of one controller update (for logging and tests)."""

    update_index: int
    batch_fp: list[float]
    batch_fn: list[float]
    fp_ema: list[float]
    fn_ema: list[float]
    ratio: list[float]
    logit_u: list[float]
    alpha: list[float]
    beta: list[float]


@dataclass
class _State:
    fp_ema: np.ndarray
    fn_ema: np.ndarray
    logit_u: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    initialized: bool = False
    num_updates: int = 0
    init_fp_ema_applied: bool = False
    init_fn_ema_applied: bool = False


class AdaptiveTverskyController:
    """ATPO controller. See the module docstring for the update rule."""

    def __init__(
        self,
        config: ControllerConfig | Mapping[str, Any] | ResolvedControllerConfig,
        category_ids: Optional[Sequence[str]] = None,
    ) -> None:
        if isinstance(config, ResolvedControllerConfig):
            if category_ids is not None and tuple(category_ids) != config.category_ids:
                raise ControllerConfigError("category_ids do not match the resolved configuration")
            self.config = config
        else:
            if category_ids is None:
                raise ControllerConfigError("category_ids are required")
            self.config = resolve_controller_config(config, category_ids)
        self._reset()

    # ---------------------------------------------------------------- set-up
    def _reset(self) -> None:
        cfg = self.config
        dim = cfg.dim
        state = _State(
            fp_ema=np.zeros(dim, dtype=np.float64),
            fn_ema=np.zeros(dim, dtype=np.float64),
            logit_u=np.zeros(dim, dtype=np.float64),
            alpha=np.full(dim, 0.5, dtype=np.float64),
            beta=np.full(dim, 0.5, dtype=np.float64),
        )
        self._state = state
        target = np.asarray(cfg.target_ratio, dtype=np.float64)
        if cfg.init_logit_u is not None:
            state.logit_u = np.asarray(cfg.init_logit_u, dtype=np.float64).copy()
        if cfg.init_fp_ema is not None:
            state.fp_ema = np.asarray(cfg.init_fp_ema, dtype=np.float64).copy()
            state.init_fp_ema_applied = True
        if cfg.init_fn_ema is not None:
            state.fn_ema = np.asarray(cfg.init_fn_ema, dtype=np.float64).copy()
            state.init_fn_ema_applied = True
        if cfg.init_logit_u is None and (state.init_fp_ema_applied or state.init_fn_ema_applied):
            # Historical ATPO-C behaviour: start from the log-ratio gap of the initial EMAs.
            init_ratio = (state.fn_ema + cfg.epsilon) / (state.fp_ema + cfg.epsilon)
            state.logit_u = np.log(init_ratio + cfg.epsilon) - np.log(target + cfg.epsilon)
        base = _sigmoid(state.logit_u)  # not clipped before the first update (historical)
        state.beta = cfg.scale * base
        state.alpha = cfg.scale * (1.0 - base)

    # -------------------------------------------------------------- accessors
    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def category_ids(self) -> tuple[str, ...]:
        return self.config.category_ids

    @property
    def num_updates(self) -> int:
        return self._state.num_updates

    @property
    def initialized(self) -> bool:
        return self._state.initialized

    @property
    def alpha(self) -> np.ndarray:
        """Current FP weights (length 1 in global mode, one per category otherwise)."""
        return self._state.alpha.copy()

    @property
    def beta(self) -> np.ndarray:
        """Current FN weights (length 1 in global mode, one per category otherwise)."""
        return self._state.beta.copy()

    @property
    def fp_ema(self) -> np.ndarray:
        return self._state.fp_ema.copy()

    @property
    def fn_ema(self) -> np.ndarray:
        return self._state.fn_ema.copy()

    @property
    def logit_u(self) -> np.ndarray:
        return self._state.logit_u.copy()

    def ema_ratio(self) -> np.ndarray:
        eps = self.config.epsilon
        return (self._state.fn_ema + eps) / (self._state.fp_ema + eps)

    # ----------------------------------------------------------------- update
    def update(self, batch_fp: Sequence[float] | float, batch_fn: Sequence[float] | float) -> ControllerUpdate:
        """Apply one update from batch FP/FN statistics of the batch just scored."""
        cfg = self.config
        state = self._state
        fp = np.asarray(batch_fp, dtype=np.float64).reshape(-1)
        fn = np.asarray(batch_fn, dtype=np.float64).reshape(-1)
        if fp.shape != (cfg.dim,) or fn.shape != (cfg.dim,):
            raise ValueError(
                f"expected batch statistics of shape ({cfg.dim},), got {fp.shape} and {fn.shape}"
            )
        if not (np.all(np.isfinite(fp)) and np.all(np.isfinite(fn))):
            raise ValueError("batch statistics must be finite")
        if np.any(fp < 0) or np.any(fn < 0):
            raise ValueError("batch statistics must be non-negative")
        rho = cfg.rho
        if not state.initialized:
            if state.init_fp_ema_applied:
                state.fp_ema = (1 - rho) * state.fp_ema + rho * fp
            else:
                state.fp_ema = fp.copy()
            if state.init_fn_ema_applied:
                state.fn_ema = (1 - rho) * state.fn_ema + rho * fn
            else:
                state.fn_ema = fn.copy()
            state.initialized = True
        else:
            state.fp_ema = (1 - rho) * state.fp_ema + rho * fp
            state.fn_ema = (1 - rho) * state.fn_ema + rho * fn

        eps = cfg.epsilon
        target = np.asarray(cfg.target_ratio, dtype=np.float64)
        ratio = (state.fn_ema + eps) / (state.fp_ema + eps)
        log_diff = np.log(ratio + eps) - np.log(target + eps)
        state.logit_u = (1 - cfg.logit_leakage) * state.logit_u + cfg.eta * log_diff
        base = np.clip(_sigmoid(state.logit_u), cfg.beta_clip, 1.0 - cfg.beta_clip)
        state.beta = cfg.scale * base
        state.alpha = cfg.scale * (1.0 - base)
        state.num_updates += 1
        return ControllerUpdate(
            update_index=state.num_updates,
            batch_fp=fp.tolist(),
            batch_fn=fn.tolist(),
            fp_ema=state.fp_ema.tolist(),
            fn_ema=state.fn_ema.tolist(),
            ratio=ratio.tolist(),
            logit_u=state.logit_u.tolist(),
            alpha=state.alpha.tolist(),
            beta=state.beta.tolist(),
        )

    # ------------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, Any]:
        """JSON-serializable state sufficient to resume training exactly.

        Floats are stored with ``float.hex`` so restoration is bit-exact; decimal
        copies are included for readability.
        """
        s = self._state

        def _hex(arr: np.ndarray) -> list[str]:
            return [float(v).hex() for v in arr]

        return {
            "format": STATE_FORMAT,
            "mode": self.config.mode,
            "category_ids": list(self.config.category_ids),
            "config": self.config.to_dict(),
            "config_fingerprint": self.config.fingerprint(),
            "initialized": s.initialized,
            "num_updates": s.num_updates,
            "init_fp_ema_applied": s.init_fp_ema_applied,
            "init_fn_ema_applied": s.init_fn_ema_applied,
            "exact": {
                "fp_ema": _hex(s.fp_ema),
                "fn_ema": _hex(s.fn_ema),
                "logit_u": _hex(s.logit_u),
                "alpha": _hex(s.alpha),
                "beta": _hex(s.beta),
            },
            "readable": {
                "fp_ema": s.fp_ema.tolist(),
                "fn_ema": s.fn_ema.tolist(),
                "logit_u": s.logit_u.tolist(),
                "alpha": s.alpha.tolist(),
                "beta": s.beta.tolist(),
                "ema_ratio": self.ema_ratio().tolist(),
            },
        }

    def load_state_dict(self, state: Mapping[str, Any], *, allow_config_change: bool = False) -> None:
        """Restore state saved by :meth:`state_dict`.

        The mode and category IDs must match. A different hyperparameter
        configuration is rejected unless ``allow_config_change`` is set, in which case
        the saved statistics and logits are kept and the new settings apply to
        subsequent updates.
        """
        if state.get("format") != STATE_FORMAT:
            raise ControllerStateError(f"unsupported controller state format {state.get('format')!r}")
        if state.get("mode") != self.config.mode:
            raise ControllerStateError(
                f"saved controller mode {state.get('mode')!r} != configured {self.config.mode!r}"
            )
        if tuple(state.get("category_ids", ())) != self.config.category_ids:
            raise ControllerStateError(
                "saved controller categories "
                f"{state.get('category_ids')} != configured {list(self.config.category_ids)}"
            )
        if state.get("config_fingerprint") != self.config.fingerprint() and not allow_config_change:
            raise ControllerStateError(
                "controller hyperparameters differ from the saved state; "
                "set allow_config_change to resume with new settings"
            )
        exact = state["exact"]
        dim = self.config.dim

        def _arr(key: str) -> np.ndarray:
            values = np.array([float.fromhex(v) for v in exact[key]], dtype=np.float64)
            if values.shape != (dim,):
                raise ControllerStateError(f"saved {key} has shape {values.shape}, expected ({dim},)")
            return values

        restored = _State(
            fp_ema=_arr("fp_ema"),
            fn_ema=_arr("fn_ema"),
            logit_u=_arr("logit_u"),
            alpha=_arr("alpha"),
            beta=_arr("beta"),
            initialized=bool(state["initialized"]),
            num_updates=int(state["num_updates"]),
            init_fp_ema_applied=bool(state.get("init_fp_ema_applied", False)),
            init_fn_ema_applied=bool(state.get("init_fn_ema_applied", False)),
        )
        if allow_config_change and state.get("config_fingerprint") != self.config.fingerprint():
            # Coefficients follow the new scale/clip immediately; statistics are retained.
            base = np.clip(_sigmoid(restored.logit_u), self.config.beta_clip, 1.0 - self.config.beta_clip)
            if not restored.initialized:
                base = _sigmoid(restored.logit_u)
            restored.beta = self.config.scale * base
            restored.alpha = self.config.scale * (1.0 - base)
        self._state = restored
