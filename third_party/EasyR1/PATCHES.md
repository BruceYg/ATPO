# Release patches on top of the EasyR1 snapshot

The unmodified snapshot is the first commit of this repository ("Vendor EasyR1
snapshot"). Every later change under `third_party/EasyR1/` is listed here; run
`git diff <first commit> -- third_party/EasyR1` to see the diffs.

## P1 Reward-controller state checkpointing (changes resume behaviour only)

Files: `verl/workers/reward/function.py`, `verl/trainer/ray_trainer.py`,
`verl/trainer/config.py`.

Problem: ATPO controller state (FP/FN EMAs, logits, coefficients) lived only in
the reward worker's memory. Checkpoints saved the actor, optimizer, scheduler,
RNG, and dataloader state but not the controller, so a resumed run restarted the
controller from balanced coefficients.

Change:

- `AutoRewardManager` calls the reward module's optional `configure(**kwargs)` when
  the worker starts, so invalid reward settings fail before any rollout. It exposes
  `reward_state_supported()`, `is_stateful()`, `get_reward_state()`, and
  `set_reward_state()`, which call the module's optional `get_state()`,
  `set_state()`, and `is_stateful()` hooks. Reward files without these hooks
  (including the reward files of the paper's runs) behave exactly as before.
- `RayPPOTrainer._save_checkpoint` writes `reward_state.json` (training reward) and
  `val_reward_state.json` (validation reward) into `global_step_*/` before updating
  `checkpoint_tracker.json`. Files are written atomically.
- `RayPPOTrainer._load_checkpoint` restores both states. If a stateful reward has no
  saved state, loading fails unless `trainer.allow_missing_reward_state=true`, which
  reproduces the original restart-from-scratch behaviour with a warning. A changed
  controller configuration is rejected unless `trainer.allow_reward_config_change=true`.
- `RayPPOTrainer.fit` rejects stateful training rewards together with
  `algorithm.online_filtering=true` or `algorithm.adv_estimator=remax`, because those
  call the training reward more than once per step (unfiltered generations or greedy
  baselines) and would change the controller update schedule. The paper's ATPO
  runs used neither option.

Training from scratch is unchanged: the controller update schedule (one update per
training step, computed over the full rollout batch in a single reward worker) is
the same as in the paper's runs.

Verification: `tests/test_easyr1_patch.py` runs real Ray reward workers against the
patched trainer methods (model workers replaced by fakes) and checks that a run
resumed from a saved step reproduces the uninterrupted run's rewards exactly. This
is CPU-level verification; the patch has not yet been exercised in a GPU training run.
