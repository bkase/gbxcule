# Dreamer v3 M8 Plan: Full System Validation + Regression Gates

This plan turns Milestone M8 (full-system validation and regression gates) into a
repeatable, automated release process. It assumes M0-M7 are complete and the
Dreamer v3 engine can run on both CPU (deterministic) and CUDA (async, zero-copy).

---

## 0) Objective and scope

### Objective
Prove Dreamer v3 is correct, stable, and performant on real tasks. Lock those
properties behind automated gates so regressions are caught immediately.

### In scope (M8 deliverables)
- Unified CLI entry points for train/eval/bench (Dreamer v3).
- Two validation scenarios with clear acceptance criteria:
  1) Standing Still / Reconstruction
  2) Exit Oak (stage1_exit_oak)
- Regression gates for CPU daily and GPU (DGX) main.
- Metrics, artifacts, and repro bundles for failures.
- A baseline run pack with frozen thresholds for ongoing regression checks.

### Out of scope
- New model architecture changes.
- New replay or engine refactors (M0-M7 only).
- New reward shaping or goal definitions beyond stage1_exit_oak.

---

## 1) Prerequisites (must already exist)

- M0-M7 code complete, tests green.
- Dreamer config includes beta_dyn, beta_rep, free_bits, ReturnEMA, and
  stable GRU FP32 internals.
- ReplayRingCUDA supports packed2 and commit markers.
- Engine has failfast hooks (non-finite detection + shape/device asserts).
- Golden Bridge fixtures for math, RSSM step, reconstruction loss.
- Packed2 render pipeline and goal templates for stage1_exit_oak.

If any prerequisite is missing, stop and fix the earlier milestone first.

---

## 2) Tooling and artifact contract

### 2.1 Unified CLIs
Create or extend these entry points:

- `tools/rl_train_gpu.py --algo dreamer_v3`
  - Runs Dreamer training on CUDA.
  - Supports `--mode full` and `--mode standing_still`.
  - Writes Experiment artifacts under `bench/runs/rl/`.

- `tools/rl_eval.py --algo dreamer_v3`
  - Greedy or stochastic evaluation.
  - Emits JSONL trajectories + summary.

- `tools/rl_gpu_bench_dreamer.py`
  - Measures throughput, replay ratio, and model step latency.
  - Emits JSON summary for regression tracking.

These should follow the existing Experiment harness conventions:
- `bench/runs/rl/<timestamp>__dreamer_v3__<rom>__<tag>/`
- `meta.json`, `config.json`, `metrics.jsonl`, `checkpoints/`, `failures/`

### 2.2 Metrics payload
Dreamer metrics must include (minimum list):

- Env and throughput: `env_steps`, `opt_steps`, `sps`, `train_sps`
- Replay: `replay_size`, `replay_ratio`, `commit_stride`, `ready_steps`
- Losses: `loss_model`, `loss_actor`, `loss_critic`
- Reconstruction: `obs_nll`, `reward_nll`, `continue_nll`
- KL terms: `kl_dyn`, `kl_rep`, `kl_total`, `free_bits_applied`
- Entropy: `prior_entropy`, `posterior_entropy`, `action_entropy`
- Value stats: `value_mean`, `value_std`, `adv_mean`, `adv_std`
- ReturnEMA: `ret_p05`, `ret_p95`, `ret_scale`

Add run metadata for: rom sha, state sha, goal sha, action codec, torch/warp
versions, GPU name, git commit + dirty flag.

### 2.3 Failure bundles
On any non-finite or shape/device error:
- Write a failure bundle via `Experiment.write_failure_bundle()`.
- Include last known metrics, config, and a repro script.
- Save small tensors: a tiny batch of obs/actions/replay indices.

---

## 3) Validation scenario A: Standing Still / Reconstruction

### Purpose
Verify the world model and reconstruction pipeline before behavior learning
can mask bugs. This is also where we track the M8 gotcha: prior vs posterior
entropy drift.

### Setup
- Environment: Pokemon Red with a static or near-static state.
- Use a fixed action sequence that yields minimal movement. Two options:
  1) Fixed action against a wall in stage1_exit_oak.
  2) A static micro-ROM (e.g., BG_STATIC) for a pure recon sanity run.
- Policy: force a constant action (no policy learning) for the data stream.
- Train: world model only (disable actor/critic) for a short window.

### Metrics to track
- Reconstruction losses should drop quickly.
- KL terms should stay above free_bits without exploding.
- `prior_entropy` and `posterior_entropy` should stay close
  (ratio in ~[0.7, 1.3] after warmup).
- `obs_nll` and `reward_nll` stable and decreasing.

### Acceptance criteria (smoke gate)
- Recon loss improves by >= 25% within the first N updates
  (choose N to fit < 5 minutes on GPU).
- No NaNs/Infs in any loss/entropy.
- `abs(prior_entropy - posterior_entropy) / max(1, prior_entropy) < 0.3`
  after warmup.

### Acceptance criteria (regression gate)
- After a baseline run is recorded, all key metrics must stay within
  +/- 15% of the baseline median at matching step ranges.

### Artifacts
- Short recon video (decoded frames) for visual inspection.
- A metrics plot: obs_nll, reward_nll, kl_dyn, kl_rep, entropies.
- Saved config + reproducible script.

---

## 4) Validation scenario B: Exit Oak (stage1_exit_oak)

### Purpose
Prove full Dreamer loop (replay -> world model -> imagination -> actor/critic)
learns a real task and does not collapse.

### Setup
- State: `states/rl_stage1_exit_oak/start.state`
- Goal: `states/rl_stage1_exit_oak/`
- Frames per step: 24 (release_after_frames=8)
- Max steps: 128 (from rl_stages.json)
- Action codec: `pokemonred_puffer_v1`

### Metrics to track
- Success rate and done rate over training.
- Distances to goal (p10/p50/p90) should trend down.
- Actor/critic losses stable (no explosion).
- Replay ratio and throughput stable.

### Acceptance criteria (smoke gate)
- Success rate > 0 and done_rate > 0 within a short run.
- No training collapse: losses finite, entropies non-zero.
- Replay ratio within config target (no starvation).

### Acceptance criteria (regression gate)
- After baseline run is recorded, require:
  - success_rate >= 0.6 * baseline success_rate at equivalent env_steps
  - dist_p50 <= 1.2 * baseline dist_p50
  - no sustained non-finite events

### Artifacts
- Greedy eval summary (JSON).
- Optional MP4 of best rollout.
- Baseline metrics plot and run config.

---

## 5) Regression gate matrix

### CPU daily gate (fast)
- `pytest -q tests/rl_dreamer` (fixtures, math, RSSM, WM)
- CPU dreamer smoke (tiny batch, 1-2 updates)
- Standing Still CPU fixture run (short, deterministic)

### GPU main gate (DGX)
- CUDA dreamer smoke: end-to-end train step (short)
- Standing Still GPU gate (short run, recon improvement)
- Exit Oak GPU gate (short run, non-zero success)
- memcpy gate: no host transfers in hot path
- pointer stability + replay commit fence check
- optional unpack performance threshold

Each gate must emit a JSON summary and a one-line pass/fail report.

---

## 6) Benchmarks and throughput targets

`tools/rl_gpu_bench_dreamer.py` should report:
- Env steps/s
- Train steps/s
- Replay sampling ms
- Model forward/backward ms
- End-to-end ms per update

Targets:
- Record a baseline once; enforce <= 15% regression.
- If unpack kernel is used, enforce a hard ceiling on unpack time.

---

## 7) Implementation steps (order)

1) **Unified CLI**
   - Add `tools/rl_train_gpu.py` with `--algo dreamer_v3`.
   - Route to Dreamer engine config; support `--mode standing_still`.

2) **Eval tool**
   - Add `tools/rl_eval.py --algo dreamer_v3`.
   - Reuse `gbxcule.rl.eval` if compatible or create Dreamer-specific eval.

3) **Bench tool**
   - Implement `tools/rl_gpu_bench_dreamer.py` using Experiment harness.

4) **Standing Still dataset and run mode**
   - Add fixed-action mode (or static ROM option).
   - Ensure deterministic action sequence with fixed RNG.

5) **Regression checks**
   - Implement baseline capture (store summary JSON).
   - Add compare script to enforce thresholds.

6) **Documentation**
   - Add `docs/dreamer_v3_validation.md` with exact commands and thresholds.
   - Update `history/dreamer-plan.md` or cross-link M8 doc.

7) **Baseline runs**
   - Run Standing Still (GPU) and Exit Oak (GPU) once.
   - Freeze baseline metrics for regression gates.

---

## 8) Definition of done

- All M8 CLIs exist and run without manual patching.
- Standing Still and Exit Oak validations pass on GPU.
- CPU daily gate green with deterministic results.
- Regression scripts catch intentional metric regressions.
- Artifacts + docs are committed.

---

## 9) Known risks and mitigations

- **Latent drift**: track `prior_entropy` vs `posterior_entropy` explicitly.
- **Replay starvation**: enforce commit_stride <= seq_len/2 and log ready_steps.
- **Hidden host transfers**: keep memcpy gate in M8 main gate.
- **Threshold tuning**: calibrate once, then lock baseline JSON.

---

## 10) Runbook (example commands)

Standing Still (GPU, short):
```
uv run python tools/rl_train_gpu.py --algo dreamer_v3 --mode standing_still \
  --rom red.gb --state states/rl_stage1_exit_oak/start.state \
  --goal-dir states/rl_stage1_exit_oak --output-tag standing_still_smoke
```

Exit Oak (GPU, short):
```
uv run python tools/rl_train_gpu.py --algo dreamer_v3 --mode full \
  --rom red.gb --state states/rl_stage1_exit_oak/start.state \
  --goal-dir states/rl_stage1_exit_oak --output-tag exit_oak_smoke
```

Eval:
```
uv run python tools/rl_eval.py --algo dreamer_v3 --checkpoint <path> \
  --rom red.gb --state states/rl_stage1_exit_oak/start.state \
  --goal-dir states/rl_stage1_exit_oak --episodes 8
```

Bench:
```
uv run python tools/rl_gpu_bench_dreamer.py --rom red.gb \
  --state states/rl_stage1_exit_oak/start.state
```
