# ATPO controller: semantics, timing, and checkpoints

This page describes what the release computes, when, and how controller state
survives checkpoint/resume. The code is `atpo/controllers/tversky.py`
(controller), `atpo/rewards/runtime.py` (batch scoring), `atpo/rewards/functions.py`
(per-sample rewards), and `atpo/rewards/easyr1.py` (EasyR1 entry point). All
statements below are covered by the tests named in each section.

## Per-response reward

For a response with predicted category set P and ground-truth set G:

| Reward (`reward:`) | Accuracy term | Used by |
| --- | --- | --- |
| `exact_match` | 1 if P = G else 0 | GRPO-EM baseline |
| `jaccard` | \|P ∩ G\| / \|P ∪ G\| | GRPO-Jaccard baseline |
| `tversky` | TP / (TP + α·FP + β·FN), fixed α, β | GRPO-STR (α = β = 1) |
| `atpo`, mode `global` | Tversky index with the controller's α, β | ATPO-G |
| `atpo`, mode `category` | Σ_c TP_c / (Σ_c TP_c + Σ_c α_c·FP_c + Σ_c β_c·FN_c) | ATPO-C |

Edge cases, as in the paper's reward functions: an unparseable response scores
0; an empty prediction for an empty ground truth (a correctly identified benign
video) scores 1; a zero denominator scores 1.

The training reward is `overall = (1 − w)·accuracy + w·format` with
`format_weight` w = 0.1. `format` is 1 when the response has the trained layout
(`response_format: think_answer` — `<think>…</think><answer>…</answer>` — for the
Qwen2.5-VL recipes, `triple_newline` for the Qwen3-VL recipes). `parse_scope`
selects where labels are read: `answer_only` (the paper's ATPO runs) or
`full_response` (the paper's GRPO-STR run).

The release rewards and controllers were checked against the reward functions of the
paper's runs on random batches: `overall`, `format`, `accuracy`, and the controller
trajectory are bit-identical.

## Controller

State per controlled unit (one unit for ATPO-G, one per category for ATPO-C):
FP and FN exponential moving averages, a logit `u`, and the coefficients α, β.

**Batch statistics.** For the responses counted in a batch (see invalid outputs
below), with 0/1 indicator vectors over the taxonomy:

- ATPO-G: `batch_FP` = mean over responses of the number of false-positive
  categories (summed over categories); same for `batch_FN`.
- ATPO-C: per category, the fraction of responses with a false positive
  (respectively false negative) for that category.

**Update** (once per scored batch):

```
first update:  FP_EMA = batch_FP;  FN_EMA = batch_FN
later:         FP_EMA = (1 − ρ)·FP_EMA + ρ·batch_FP      (same for FN)
r  = (FN_EMA + ε) / (FP_EMA + ε)
u  = (1 − leakage)·u + η·(log(r + ε) − log(r* + ε))
β  = c · clip(σ(u), b, 1 − b)        α = c − β
```

`ρ` weights the incoming batch. When misses dominate (r > r\*), `u` and β grow, so
false negatives cost more and the policy is pushed to flag more categories; when
false alarms dominate, α grows. The target ratio r\* = FN/FP sets the operating
point: r\* > 1 favours precision (more misses tolerated), r\* < 1 favours recall.

| Setting | Default | Notes |
| --- | --- | --- |
| `mode` | — (required) | `global` (ATPO-G) or `category` (ATPO-C) |
| `target_ratio` | 1.0 | ATPO-C: scalar, list in taxonomy order, or `{category_id: value}` |
| `scale` (`c`) | 1.0 | α + β = c. Most of the paper's runs used c = 2; release configs always state it. |
| `rho` | 0.05 | EMA weight of the incoming batch |
| `eta` | 0.01 | logit step size (the paper's r\*=0.2 ATPO-G run used 0.05) |
| `epsilon` | 1e-8 | |
| `logit_leakage` | 0 | pulls `u` towards 0 each update |
| `beta_clip` | 0 | b above; 0 means no clipping |
| `init_logit_u`, `init_fp_ema`, `init_fn_ema` | none | optional initial state |
| `invalid_prediction_policy` | — (required) | see below |

**Initial coefficients.** With no initializer, u = 0 and α = β = c/2: Jaccard for
c = 2, Dice for c = 1. No clipping is applied before the first update.

**Invalid outputs.** An unparseable response always gets accuracy 0. Whether it
enters the FP/FN statistics is `invalid_prediction_policy`: `as_empty` counts it as
predicting no category (adding false negatives), `exclude` leaves it out. The
paper's runs differ, so the release has no default and each config states the
policy of the run it follows:

| Paper runs | Policy |
| --- | --- |
| ATPO-G (SafeWatch, XD-Violence; both backbones) | `as_empty` |
| ATPO-C, Qwen2.5-VL | `exclude` |
| ATPO-C, Qwen3-VL | `as_empty` |

Tests: `tests/test_controller.py` (direction, equilibrium, EMA weighting, clipping,
leakage, configuration errors, required policy).

## Timing

One call of the reward function scores one batch, in this order:

1. read α, β (derived from earlier batches only);
2. score every response with them;
3. compute the batch statistics and update the controller once.

EasyR1 calls the training reward once per training step with the whole rollout
batch (64 prompts × 8 rollouts = 512 responses in the paper's configs), in one
reward worker. The controller therefore updates once per step on global-batch
statistics, without cross-rank reduction, exactly as in the paper's runs. The
patched trainer rejects stateful rewards combined with `algorithm.online_filtering`
or `algorithm.adv_estimator=remax`, which call the training reward more than once
per step; the paper's runs did not use them.

## Validation reward

EasyR1 scores validation with a separate reward worker (`worker.val_reward`,
falling back to the training settings when unset). The paper's ATPO runs did
not set it, so validation had **its own adaptive controller that kept updating on
validation batches** (one update per validation batch),
starting from balanced coefficients and drifting from pass to pass. Validation
responses were sampled (temperature 0.6, top-p 0.95), and the "best" checkpoint
was selected on this score.

The release default (`validation: frozen`) scores validation with the same reward
settings and `update_controller: false`: the validation controller never moves, so
validation uses α = β = c/2 at every evaluation and scores are comparable across
steps. `atpo train --validation historical` restores the behaviour of the paper's runs.
Sampling settings are unchanged. Tests: `tests/test_training_configs.py`,
`tests/test_controller.py::test_frozen_validation_runtime_never_updates`.

## Checkpoints and resume

In the paper's runs the controller lived only in the reward worker's memory and
was not saved; a resumed run restarted it from balanced coefficients. The release
patches the vendored EasyR1 (patch P1, `third_party/EasyR1/PATCHES.md`):

- every checkpoint `global_step_N/` gets `reward_state.json` (training reward) and
  `val_reward_state.json` (validation reward), written atomically before
  `checkpoint_tracker.json` is updated;
- floats are stored as `float.hex` next to readable decimals, so restoration is
  bit-exact; the files also record the reward settings, their fingerprint, the
  taxonomy fingerprint, and the number of batches scored;
- resuming restores both states. A stateful reward without saved state fails
  unless `trainer.allow_missing_reward_state=true` (the old restart-from-scratch
  behaviour, with a warning); a changed controller configuration fails unless
  `trainer.allow_reward_config_change=true` (statistics and logits are kept, the
  new settings apply from the next update); a different taxonomy always fails.

Resume a run with the same output directory:

```bash
atpo train --config configs/atpo/safewatch_qwen25vl_atpo_g_o2.yaml --output-dir runs/g_o2 \
  --var TRAIN_FILE=... --var VAL_FILE=... --var INIT_MODEL=... --resume
atpo inspect-state runs/g_o2/checkpoints/global_step_40      # alpha, beta, EMAs, logits
```

`--resume` sets `trainer.find_last_checkpoint=true`. EasyR1 restores the
best-checkpoint tracker only in that mode; an explicit
`--set trainer.load_checkpoint_path=...` resumes weights, optimizer, dataloader,
and reward states but not the tracker.

Verification: `tests/test_controller.py::test_runtime_resume_matches_uninterrupted_run`
(runtime level) and `tests/test_easyr1_patch.py` (real Ray reward workers driven by
the patched trainer methods, model workers replaced by fakes): a run resumed from a
saved step reproduces the uninterrupted run's rewards exactly. This is CPU-level
verification; the patch has not yet been exercised in a GPU training run.

## Logged quantities

Every response's reward dictionary carries batch-level diagnostics, which EasyR1
averages into its metrics: `coef_alpha[_<id>]`, `coef_beta[_<id>]` (the
coefficients used for this batch), `ctrl_fp_ema`, `ctrl_fn_ema`, `ctrl_ema_ratio`,
`ctrl_logit_u`, `ctrl_target_ratio`, `ctrl_updates`, `batch_fp`, `batch_fn`,
`batch_fp_<id>`, `batch_fn_<id>`, `invalid_rate`, `counted_samples`, and, with
`log_classification_metrics: true`, batch precision/recall/F1 per category, macro,
and micro.
