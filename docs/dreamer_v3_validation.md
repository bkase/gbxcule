# Dreamer v3 Validation + Regression Gates

This document defines **Milestone M8** validation and regression checks for the
Dreamer v3 pipeline. It assumes M0–M7 are complete and the Dreamer engine is
ready on CUDA.

## 1) Unified CLIs

### Train (GPU)
```
uv run python tools/rl_train_gpu.py --algo dreamer_v3 --mode full \
  --rom red.gb --state states/rl_stage1_exit_oak/start.state \
  --goal-dir states/rl_stage1_exit_oak --output-tag exit_oak_smoke
```

### Eval (GPU)
```
uv run python tools/rl_eval.py --algo dreamer_v3 --checkpoint <checkpoint.pt> \
  --rom red.gb --state states/rl_stage1_exit_oak/start.state \
  --goal-dir states/rl_stage1_exit_oak --episodes 8 --greedy
```

Optional trajectory JSONL (single-env only):
```
uv run python tools/rl_eval.py --algo dreamer_v3 --checkpoint <checkpoint.pt> \
  --rom red.gb --state states/rl_stage1_exit_oak/start.state \
  --goal-dir states/rl_stage1_exit_oak --episodes 1 \
  --trajectory bench/runs/rl/trajectory.jsonl
```

### Bench (GPU)
```
uv run python tools/rl_gpu_bench_dreamer.py --rom red.gb \
  --state states/rl_stage1_exit_oak/start.state \
  --goal-dir states/rl_stage1_exit_oak --iterations 10 --warmup 2
```

## 2) Validation scenarios

### A) Standing Still / Reconstruction
Goal: validate world model reconstruction before behavior learning can hide
issues.

```
uv run python tools/rl_train_gpu.py --algo dreamer_v3 --mode standing_still \
  --rom red.gb --state states/rl_stage1_exit_oak/start.state \
  --goal-dir states/rl_stage1_exit_oak --output-tag standing_still_smoke
```

Key expectations:
- Reconstruction losses improve quickly.
- `State/post_entropy` and `State/prior_entropy` remain close.
- No NaNs/Infs in losses or entropies.

### B) Exit Oak (stage1_exit_oak)
Goal: validate the full loop (replay → world model → imagination → actor/critic).

```
uv run python tools/rl_train_gpu.py --algo dreamer_v3 --mode full \
  --rom red.gb --state states/rl_stage1_exit_oak/start.state \
  --goal-dir states/rl_stage1_exit_oak --output-tag exit_oak_smoke
```

Key expectations:
- Non-zero success rate on greedy eval.
- Stable losses and entropies.
- Replay ratio near target.

## 3) Baseline capture and compare

Use `tools/rl_dreamer_regression.py` to freeze baselines and enforce regression
thresholds. It supports JSON (eval summaries) and JSONL (metrics logs).

### Capture baselines
Example (training metrics JSONL, last 25 records):
```
uv run python tools/rl_dreamer_regression.py capture \
  --input bench/runs/rl/<run_id>/metrics.jsonl \
  --output bench/baselines/dreamer_v3/standing_still.json \
  --last 25 --threshold-pct 0.15
```

Example (eval summary JSON):
```
uv run python tools/rl_eval.py --algo dreamer_v3 --checkpoint <checkpoint.pt> \
  --rom red.gb --state states/rl_stage1_exit_oak/start.state \
  --goal-dir states/rl_stage1_exit_oak --episodes 8 --greedy \
  --output bench/runs/rl/exit_oak_eval.json

uv run python tools/rl_dreamer_regression.py capture \
  --input bench/runs/rl/exit_oak_eval.json \
  --output bench/baselines/dreamer_v3/exit_oak_eval.json \
  --keys success_rate,mean_return,Game/ep_len_avg
```

### Compare against baselines
```
uv run python tools/rl_dreamer_regression.py compare \
  --baseline bench/baselines/dreamer_v3/standing_still.json \
  --input bench/runs/rl/<run_id>/metrics.jsonl
```

```
uv run python tools/rl_dreamer_regression.py compare \
  --baseline bench/baselines/dreamer_v3/exit_oak_eval.json \
  --input bench/runs/rl/exit_oak_eval.json \
  --threshold-pct 0.15
```

## 4) Gate matrix

### CPU daily gate
- `pytest -q tests/rl_dreamer`
- CPU dreamer smoke (tiny batch, 1–2 iterations)
- Sheeprl parity micro-checks (fixtures)

### GPU main gate (DGX)
- CUDA dreamer smoke (short run)
- Standing Still GPU gate
- Exit Oak GPU gate + greedy eval
- Replay/throughput sanity (`tools/rl_gpu_bench_dreamer.py`)

## 5) Metrics parity (sheeprl)
The Dreamer v3 metrics emitted in training/eval should align with the sheeprl
aggregator keys:
- `Loss/world_model_loss`, `Loss/observation_loss`, `Loss/reward_loss`,
  `Loss/state_loss`, `Loss/continue_loss`
- `State/kl`, `State/post_entropy`, `State/prior_entropy`
- `Loss/policy_loss`, `Loss/value_loss`
- `Grads/world_model`, `Grads/actor`, `Grads/critic`
- Eval summaries: `Rewards/rew_avg`, `Game/ep_len_avg`

Baseline thresholds should be applied to median values over a stable window.
