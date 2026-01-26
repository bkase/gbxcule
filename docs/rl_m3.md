# Simple-RL M3 (Goal Template + Done Detection)

This document covers **Milestone 3** for the pixels-only RL path: capturing a goal template and verifying goal detection via a replay gate.

## Action trace format (locked)

`actions.jsonl` must be **one JSON list per step**, where each list contains the action index for each env:

```
[3]
[3]
[5]
```

For multi-env traces, each line is e.g. `[a0, a1, a2, ...]`.

**No NOOP action** is added; use an inert input (e.g., a blocked direction or a button ignored on that screen) to represent waiting.

## Capture a goal template

```
uv run python -m gbxcule.rl.capture_goal_template \
  --rom /path/to/rom.gb \
  --state /path/to/start.state \
  --actions /path/to/actions.jsonl \
  --output-dir /path/to/goal_template \
  --frames-per-step 24 \
  --release-after-frames 8 \
  --stack-k 4 \
  --tau 0.05 \
  --k-consecutive 2
```

Outputs:

- `goal_template.npy` (uint8 shades)
- `goal_template.meta.json` (strict metadata)

## Replay gate (exit 0/1)

```
uv run python -m gbxcule.rl.replay_goal_template \
  --rom /path/to/rom.gb \
  --state /path/to/start.state \
  --actions /path/to/actions.jsonl \
  --goal-dir /path/to/goal_template \
  --output-dir /path/to/replay_receipt
```

- Exit **0** if `done` triggers within the trace, **1** otherwise.
- On failure, a minimal receipt is written (dist curve + last frame).

## Calibration (suggest tau)

```
uv run python -m gbxcule.rl.replay_goal_template \
  --rom /path/to/rom.gb \
  --state /path/to/start.state \
  --actions /path/to/actions.jsonl \
  --goal-dir /path/to/goal_template \
  --calibrate \
  --calibrate-output /path/to/calibration.json
```

This prints (or writes) a dist summary and a recommended `tau`.
