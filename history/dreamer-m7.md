# Dreamer v3 M7 Plan: Async Engine Integration (commit_stride decoupled, starvation-proof)

This plan is derived from:

- `history/dreamer-plan.md` (Master Plan v2, M7 definition)
- `history/dreamer-gotchas.md` (async gotchas + sheeprl parity notes)
- `src/gbxcule/rl/async_ppo_engine.py` and `src/gbxcule/rl/async_ppo.py` (existing async patterns, stream/event choreography)
- `third_party/sheeprl/sheeprl/algos/dreamer_v3/*` (reference Dreamer v3 loop, PlayerDV3 state handling)
- `third_party/sheeprl/sheeprl/utils/utils.py` (`Ratio` scheduling logic from Hafner et al.)
- `third_party/sheeprl/howto/work_with_steps.md` (policy-step vs gradient-step semantics)

The goal is to turn the Dreamer v3 components from M0-M6 into a **two-loop async engine** (actor + learner) that is correct, stable, and does not starve the learner.

---

## 1) Objective

Implement **Milestone M7**: a **Dreamer v3 async engine** that:

- runs **actor env stepping + replay writes** on one stream
- runs **learner sampling + world model update + behavior update** on another stream
- uses **commit_stride** decoupled from `steps_per_rollout` and `seq_len`
- maintains a **replay-ratio scheduler** (train-to-data ratio) without starvation
- guarantees **weight sync ordering** (avoid stale policy race)
- integrates **failfast** checks and metrics

This is the first “real” Dreamer runtime. Correctness and observability are more important than peak speed in this milestone.

---

## 2) Non-goals (explicitly out of scope in M7)

- Full CLI training/eval tools (M8)
- Full task validation (Standing Still + Exit Oak) (M8)
- Distributed or multi-GPU training
- Advanced replay prioritization
- Any CPU <-> GPU copies in the hot path (outside test-only utilities)

---

## 3) Preconditions (must already exist from M0-M6)

M7 assumes these components are implemented and passing tests:

- `ReplayRing` / `ReplayRingCUDA` with:
  - `obs` packed2 storage, `is_first`, `continue`, `episode_id`
  - commit markers / committed_t for CUDA direct-write
- RSSM with **is_first masking**, FP32 GRU internals
- World model training step + stop-grad KL + free bits
- Behavior learning with ReturnEMA normalization
- M6 zero-copy ingestion from Warp into replay

If any of these are missing, pause and complete the prior milestone.

---

## 4) Hard constraints & gotchas (must enforce)

1) **Stale weights race (M7 gotcha)**
   - Copy learner -> actor weights **before** recording `policy_ready_event`.
   - Actor must wait on `policy_ready_event` before next rollout.

2) **is_first semantics**
   - Sampling is non-strict (is_first can appear mid-sequence).
   - RSSM scan must reset on `is_first[t]` inside the sequence.
   - If you follow the **transition-aligned schema** in `history/dreamer-plan.md`, do not shift data.
   - If you import or mirror **sheeprl-style sequences**, note they set `is_first[0]=1` and shift actions
     (`batch_actions = cat(zeros, actions[:-1])`) before RSSM scan.

3) **continue semantics**
   - `continue = 0.0` only for true terminal; `1.0` for truncation and normal steps.
   - Learner uses `gamma * continue` for returns; do not mix with reset logic.

4) **No implicit sync in hot loop**
   - No `.item()` / `.cpu()` / `.numpy()` or host-side branches inside actor loop.

5) **Deterministic CPU path**
   - No global RNG; use explicit `torch.Generator` for sampling.

---

## 5) Architecture Overview

### 5.1 Engine layout

Create a Dreamer async engine module under `src/gbxcule/rl/dreamer_v3/`:

- `async_dreamer_v3_engine.py` (or consistent name with existing `engine_cpu.py`/`engine_cuda.py`)
- Export a config dataclass and a main `AsyncDreamerV3Engine` class.

Two execution modes share the same functional core:

1) **CPU reference engine** (deterministic, sequential)  
   - Runs actor and learner in the same thread in a simple loop.
   - Uses CPU replay, CPU model forward, CPU updates.  
   - Used for fast correctness testing (< 2 minutes).

2) **CUDA async engine**  
   - Actor stream for env stepping + replay writes.
   - Learner stream for replay sampling + training.
   - Weight sync via device-to-device copy and CUDA event.

### 5.2 Data flow (actor)

At each actor step:

- `obs_t` is written **directly** into replay (`ReplayRingCUDA.obs_slot(t)`), packed2.
- Maintain **per-env RSSM state** (recurrent + stochastic + previous action), PlayerDV3-style:
  - encode current obs, update posterior, then sample action from actor on latent state
  - zero action and reset latent state on `is_first`/reset
- Compute `action_t` from actor policy (uses latent state).
- Step env; compute `reward_t`, `is_first`, `continue`, `episode_id`.
- Write transition fields into replay at time index `t`.
- Every `commit_stride` steps:
  - record a CUDA event
  - update `committed_t` (safe boundary for learner sampling)

### 5.3 Data flow (learner)

Learner loop repeatedly:

- Wait until **min_ready_steps** and stable committed region are available.
- Sample sequences of length `seq_len` with `ReplayRing.sample_sequences(...)`.
- **Action alignment** (choose one, consistent with replay schema):
  - **Transition-aligned** (our canonical schema): use `actions[t]` directly.
  - **Sheeprl-aligned** (obs-aligned rewards): set `is_first[0]=1` and shift actions
    `batch_actions = cat(zeros, actions[:-1])`.
- Run world model update (M4) then behavior update (M5).
- Update metrics; run failfast checks.
- Sync weights to actor and record `policy_ready_event`.

### 5.4 Concurrency and event ordering (CUDA)

**Required ordering** (avoid stale policy):

1) Learner finishes update.
2) Learner copies weights to actor.
3) Learner records `policy_ready_event`.
4) Actor waits on `policy_ready_event` before starting next rollout window.

Replay visibility:

- Actor records `commit_event` every `commit_stride` steps.
- Learner waits on `commit_event` and only samples from `<= committed_t - safety_margin`.
- `safety_margin` should be at least `seq_len` (configurable).

### 5.5 Replay-ratio scheduler (Hafner-style, starvation-proof)

Sheeprl uses the **Hafner Ratio** scheduler (`sheeprl.utils.utils.Ratio`) where
`replay_ratio` is **gradient steps per policy step** and **does not account for**
`action_repeat` or `learning_starts`.

Adopt the same behavior:

- Track **policy steps** (`env_steps_total`) as `num_envs` per env step (per rank).
- Use a `Ratio`-style accumulator:
  - `ratio = Ratio(replay_ratio, pretrain_steps)`
  - `prefill_steps = learning_starts - int(learning_starts > 0)` (match sheeprl)
  - `per_tick_updates = ratio(env_steps_total - prefill_steps)`
- Run exactly `per_tick_updates` learner updates when replay has sufficient data.

Guardrails:

- `min_ready_steps >= seq_len` before any learner step.
- `commit_stride <= seq_len / 2` recommended (warn or error otherwise).
- Optional `max_learner_steps_per_tick` to avoid long learner stalls.

### 5.6 Metrics + failfast

Integrate existing helpers:

- `gbxcule.rl.failfast.assert_finite`, `assert_device`, `assert_shape`
- `Experiment.log_metrics` for JSONL metrics

Track metrics per update:

- Timing: `actor_ms`, `learner_ms`, `overlap_eff`
- Losses: world model loss, behavior loss, total loss
- Replay: `replay_ratio_actual`, `replay_fill`, `committed_t`, `sampled_t_min/max`
- Policy lag: `actor_policy_version`, `learner_policy_version`

---

## 6) Config Surface (M7 additions)

Extend `dreamer_v3/config.py` with engine controls:

- `steps_per_rollout`: actor convenience window
- `commit_stride`: visibility cadence for replay
- `seq_len`, `batch_size`
- `replay_ratio`: gradient steps per **policy step** (per sheeprl/Hafner; not action_repeat-adjusted)
- `learning_starts`: number of policy steps before learning begins (prefill)
- `pretrain_steps`: number of gradient steps to perform on first update (Hafner Ratio)
- `min_ready_steps`: minimum committed steps before learning
- `safety_margin`: steps to keep away from write head
- `max_learner_steps_per_tick`: cap for per-iteration learning burst
- `log_every`, `profile_every`
- `device`, `precision_policy`

Ensure config validation enforces:

- `commit_stride >= 1`
- `seq_len >= 2`
- `min_ready_steps >= seq_len`
- `safety_margin >= seq_len`

---

## 7) Implementation Plan (phased)

### Phase 1: Engine skeleton + scheduler

- Add `async_dreamer_v3_engine.py` with:
  - config dataclass
  - replay scheduler logic (pure functions for unit tests)
  - CPU engine loop stub
- Implement replay-ratio accounting with explicit counters.
- Implement `Ratio`-style scheduler (match Hafner behavior from sheeprl).

### Phase 2: CPU reference engine (deterministic)

- Single-threaded loop:
  - actor steps -> replay writes
  - learner updates while scheduler allows
- Use toy env for tests (no ROM needed).
- Ensure deterministic sampling via `torch.Generator`.
- Mirror PlayerDV3-style actor state (recurrent + stochastic + previous action).

### Phase 3: CUDA async engine

- Add actor and learner CUDA streams.
- Integrate `ReplayRingCUDA` commit events + committed_t tracking.
- Implement weight sync ordering with `policy_ready_event`.
- Mirror async PPO engine patterns for timing and overlap metrics.
- Ensure actor state reset matches `is_first` and zero-action convention.

### Phase 4: Metrics + failfast

- Wire `Experiment` logging.
- Add non-finite checks around key tensors:
  - `loss_total`, `rssm_state`, `logits`
- Add device/shape asserts in debug mode.

### Phase 5: Tests + gates

- Unit tests for scheduler math (Ratio behavior) and config validation.
- CPU toy engine smoke test (fast, always-on).
- CUDA smoke test (skipped if no CUDA): ensure no deadlock and tensors stay on GPU.
- Starvation test: small `steps_per_rollout`, ensure learner still runs under valid `commit_stride`.

---

## 8) Test Plan (concrete)

### Unit tests (CPU, always-on)

1) `test_replay_ratio_scheduler`
   - deterministic counters; verify updates count vs target ratio.

2) `test_commit_stride_config_validation`
   - invalid values raise; `commit_stride > seq_len` warns or errors.

3) `test_cpu_async_engine_smoke`
   - toy env, small replay, 2-3 updates, assert finite losses.

### CUDA integration tests (skip if no GPU)

1) `test_dreamer_async_engine_cuda_smoke`
   - minimal env backend; actor/learner streams run 1-2 updates without deadlock.

2) `test_policy_ready_ordering`
   - assert `policy_ready_event` recorded **after** weight copy; actor waits.

3) Optional memcpy gate
   - use `torch.profiler` or a tiny harness to assert no HtoD/DtoH in hot path.

---

## 9) Risks & mitigations

- **Learner starvation**: use replay-ratio scheduler + `min_ready_steps` + `commit_stride` guard.
- **Policy staleness race**: enforce weight copy before `policy_ready_event`.
- **Hidden host sync**: avoid `.item()`/`.cpu()` in actor loop; use GPU counters and reduce at end.
- **Replay overwrite hazards**: sample only from `committed_t - safety_margin`.
- **Debug complexity**: failfast bundles + deterministic CPU engine.

---

## 10) Definition of Done (M7 gate)

- `AsyncDreamerV3Engine` runs on CPU (deterministic toy) and CUDA (smoke) without deadlock.
- Replay-ratio scheduler keeps replay ratio within tolerance on toy runs.
- Weight sync ordering is correct (no stale-policy race).
- CPU tests green; CUDA tests green when GPU available.
- Metrics and failfast bundles are emitted on failure.

---

## 11) Implementation Notes / Alignment with existing code

- Follow patterns in `src/gbxcule/rl/async_ppo_engine.py` for stream/event timing and overlap metrics.
- Reuse `gbxcule.rl.experiment.Experiment` for logging and failure bundles.
- Keep hot path GPU-only; use `torch.cuda.Event` timing for overlap metrics.
- Keep all new modules in `src/gbxcule/rl/dreamer_v3/` (consistent with M0 layout).

---

## 12) Acceptance Checklist

- [ ] Async engine config validated (commit_stride, seq_len, min_ready_steps)
- [ ] CPU toy smoke passes in < 2 minutes
- [ ] CUDA smoke passes (skipped if no GPU)
- [ ] Replay ratio metrics tracked and reported
- [ ] No host copies in actor/learner hot path
- [ ] Weight sync event ordering validated
