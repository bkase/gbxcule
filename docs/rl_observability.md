# RL Observability (train_log.jsonl)

This repo uses **structured JSONL logs** for RL training. The canonical schema is
enforced by `tools/validate_train_log.py` and is designed to be compact enough
for 8K–16K envs while still capturing the “why” of success/failure.

## Files

- `train_log.jsonl`: primary machine-readable log
- `checkpoint.pt`: model + optimizer + counters
- `rollout.mp4` + `rollout.jsonl`: optional single-env debug artifact

## Schema (v1)

`train_log.jsonl` is newline-delimited JSON:

- First line: `{ "meta": { ... }, "config": { ... }, "torch_version": "...", "warp_version": "..." }`
- Subsequent lines: one record per optimizer step

### Required `meta` fields

- `schema_version` (int)
- `run_id` (str)
- `rom_path`, `rom_sha256`
- `state_path`, `state_sha256`
- `goal_dir`, `goal_sha256`
- `action_codec_id`
- `frames_per_step`, `release_after_frames`, `stack_k`
- `num_envs`, `num_actions`
- `seed`

### Required record fields

- Identifiers: `run_id`, `trace_id`
- Counters: `env_steps`, `opt_steps`, `wall_time_s`
- Performance: `sps`
- Reward/outcomes: `reward_mean`, `done_rate`, `trunc_rate`, `reset_rate`
- Distance: `dist_p10`, `dist_p50`, `dist_p90`
- Policy/value: `entropy_mean`, `value_mean`, `value_std`
- Actions: `action_hist` (length = `num_actions`)

## Tools

Validate:

```bash
uv run python tools/validate_train_log.py --log path/to/train_log.jsonl
```

Plot:

```bash
uv run python tools/plot_train_log.py --log path/to/train_log.jsonl --out train.png
```

MP4 dump (single env):

```bash
uv run python tools/dump_policy_mp4.py \
  --rom red.gb --state states/pokemonred_pallet_house.state \
  --goal-dir states/rl_m5_goal_pallet_house \
  --checkpoint bench/runs/rl_m5_a2c/xxxx/checkpoint.pt \
  --output-dir bench/runs/rl_m5_a2c/xxxx
```

## Example (fixture)

See `tests/fixtures/train_log_example.jsonl` for a minimal valid log used by
tests to enforce schema stability.
