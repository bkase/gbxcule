Cool — “streaming A2C at 8K–16K envs” is exactly the regime where your **GPU-parallel emulator** becomes the _point_, and the learner becomes the _constraint_. The good news is: you can get there with **surgical changes** that mostly live in:

- `src/gbxcule/core/action_codec.py` (NOOP)
- `src/gbxcule/rl/pokered_pixels_goal_env.py` (stack=1 fast path, a couple knobs)
- **a new training script** (streaming A2C)
- eval/logging tooling

Below is a concrete, parallelizable set of workstreams, plus the minimal diffs you’ll need.

---

## What we’re building

**Target:** run stage-based navigation training with **N = 8,192 or 16,384 envs**, **T = 1 streaming updates** (or small grad-accum windows), and **pixels-only** reward/done via your goal templates.

**Key design choices:**

1. **NOOP action** (otherwise “waiting” is painful and learning destabilizes)
2. **`stack_k = 1`** for high-N (otherwise memory bandwidth and stack-shift overhead eat you)
3. **Streaming A2C / TD(0)** with optional **grad accumulation** (update every K steps)
4. Keep PPO around as a baseline at smaller N, but A2C becomes the “scale mode”.

---

## Surgical changes: the minimal set

### 1) Add NOOP action codec (v1)

**File:** `src/gbxcule/core/action_codec.py`

- Add `POKERED_PUFFER_V1_ID = "pokemonred_puffer_v1"`
- Action names become 8 actions: `("NOOP","A","B","START","UP","DOWN","LEFT","RIGHT")`
- For NOOP:
  - `_pyboy_buttons[NOOP] = None`
  - `_dpad_masks[NOOP] = 0`
  - `_button_masks[NOOP] = 0`

- Register it in `_REGISTRY`
- Add stable kernel id in `KERNEL_CODEC_IDS` (e.g. v0=0, v1=1)

**Also update callers that assume 7 actions.**
Right now `tools/rl_m5_train.py` hard-fails if `num_actions != 7`. That needs to go or be updated.

> Potential hidden dependency: Warp kernels / backend might rely on codec id → numeric mapping. You already have `KERNEL_CODEC_IDS` for that. After adding v1, search for any code that assumes action indices 0..6 map to a particular order.

---

### 2) Make the pixel env fast at `stack_k=1`

At 8K–16K, even small per-step extra copies hurt.

**Files:**

- `src/gbxcule/rl/pokered_pixels_env.py`
- `src/gbxcule/rl/pokered_pixels_goal_env.py`

Today both do a “shift stack” with a `clone()`:

```py
self._stack[:, :-1].copy_(self._stack[:, 1:].clone())
self._stack[:, -1].copy_(pix)
```

For `stack_k=1`, that should become **just**:

```py
self._stack[:, 0].copy_(pix)
```

Add a branch:

- if `stack_k == 1`: no shift, no clone

This is a pure perf win, and it’s one of the biggest “surgical” wins for 16K envs.

---

### 3) Teach `PokeredPixelsGoalEnv` to accept `action_codec` + `stack_k=1` cleanly

It already supports `action_codec` and optional `stack_k`, but you want to make sure:

- `load_goal_template(...)` is validating `action_codec_id`, `frames_per_step`, `release_after_frames`, and **`stack_k=1`**.
- your goal templates for stages are generated with that same metadata.

This is mostly “make sure the goal templates you create match the new run config”.

---

### 4) Add a new streaming A2C trainer script (no rollout buffer)

**New file:** `tools/rl_m5_train_a2c.py` (or similar)

Core loop is:

- `obs = env.reset(seed=...)`
- For each step:
  - `logits, v = model(obs)` (v shape `[N]`)
  - sample actions: `a ~ Categorical(softmax(logits))` (int64 → cast to int32)
  - `next_obs, r, done, trunc, info = env.step(actions_i32)`
  - bootstrap: `_, v_next = model(next_obs)` under `no_grad`
  - `not_done = ~(done|trunc)`
  - `target = r + gamma * not_done.float() * v_next`
  - `adv = (target - v)` (detach appropriately)
  - loss:
    - `policy = -(logp(a|s) * adv.detach()).mean()`
    - `value = value_coef * (target.detach() - v).pow(2).mean()`
    - `entropy = entropy_coef * entropy.mean()` (added as `-entropy_coef * H` in total)
    - `total = policy + value - entropy_coef*H`

  - backward
  - optimizer step every `update_every` steps (grad accumulation)

**Why grad accumulation matters**
At huge N, “optimizer step every env-step” can become overhead-heavy. A good starting point is:

- accumulate gradients for `update_every = 4` env-steps
- then `optimizer.step()`

(You can tune; 8 is also common.)

This script can reuse your existing:

- `PixelActorCriticCNN` model
- `logprob_from_logits` helper (already in `ppo.py`)
- action codec id passed into env

---

### 5) Fix eval for autoreset vector envs (must-have)

Your current eval breaks when **any env** finishes. For scale training you’ll want one of:

- **Simple:** eval with `num_envs=1` (fast and correct)
- **Proper vector eval:** keep stepping until you collect M episode terminations, then compute stats

I’d do **num_envs=1** for now. It’s “boring” but it eliminates a whole class of misleading metrics.

You can reuse `tools/rl_m5_eval.py` with minor tweaks:

- instantiate env with `num_envs=1`, `stack_k=1`, codec v1
- report success rate and steps-to-goal reliably

---

### 6) Expose the right knobs (reward + env + trainer)

For stage training you’ll want CLI flags in A2C trainer:

- Env:
  - `--num-envs` (8192 / 16384)
  - `--frames-per-step` (24)
  - `--release-after-frames` (8)
  - `--stack-k` (default 1 for scale)
  - `--action-codec` (default v1)
  - `--max-steps` (stage-dependent)

- Reward shaping:
  - `--step-cost`
  - `--alpha`
  - `--goal-bonus`
  - `--tau`
  - `--k-consecutive`

- Trainer:
  - `--lr`
  - `--gamma`
  - `--value-coef`
  - `--entropy-coef`
  - `--grad-clip`
  - `--update-every` (grad accumulation window)
  - `--total-env-steps` (not “updates”; count steps is clearer for streaming)
  - `--eval-every-steps`

---

## Recommended “first run” parameters (for Stage 1)

This is a good “boring but likely to work” starting config for Stage 1 (exit house), assuming goal matching calibrated reasonably:

**Scale mode**

- `num_envs`: 8192 (start), then 16384
- `stack_k`: 1
- `frames_per_step`: 24
- `release_after_frames`: 8
- `action_codec`: `pokemonred_puffer_v1` (with NOOP)
- `max_steps`: 96–128 for stage 1

**Reward shaping (start)**

- `step_cost = -0.01`
- `alpha = 10.0` _(often needs >1.0 for pixel-distance shaping to have bite)_
- `goal_bonus = 5.0` (stage 1 doesn’t need huge bonus)
- `k_consecutive = 4`
- `tau`: calibrate, but initial try `0.04` (then adjust based on measured dist-at-goal)

**A2C trainer**

- `gamma = 0.99`
- `lr = 1e-4` (big batches often want lower LR)
- `entropy_coef = 0.02` (keep exploration)
- `value_coef = 0.5`
- `grad_clip = 0.5`
- `update_every = 4` (accumulate 4 env-steps per optimizer step)
- `total_env_steps`: e.g. 5–20 million (you’ll measure by success, not by time)

**What you should watch**

- entropy: shouldn’t collapse immediately
- done-rate: should start above random quickly in stage 1 if shaping is sane
- dist median: should drift down over training

---

## Workstreams (parallelizable) with dependencies

Here’s a clean decomposition into “teams can work in parallel” units.

### Workstream A — Action space + NOOP (foundational)

**Goal:** add codec v1, make everything accept 8 actions.

Tasks:

1. Add `pokemonred_puffer_v1` codec with NOOP
2. Add stable kernel codec id mapping
3. Update any hard-coded `num_actions==7` checks (train/eval)
4. Verify backend handling of NOOP:
   - PyBoy path: `to_pyboy_button(None)` means “don’t press”
   - Warp joypad masks: (0,0)

5. Update docs / defaults (make v1 the default for RL)

**Can be done in parallel with B/C/D.**
Dependency: none.

---

### Workstream B — High-N env performance + stack_k=1 fast path

**Goal:** remove per-step stack shift copies at scale.

Tasks:

1. Add `stack_k==1` fast path in both pixel envs
2. Confirm env allocs and resets don’t scale with Python loops:
   - In `PokeredPixelsEnv.reset`, there’s a Python `for k in range(stack_k)`; with `stack_k=1` it’s trivial.

3. Confirm `ResetCache.apply_mask_torch(mask)` works efficiently at 16K
4. Optionally add a “subsampled info” mode so logging doesn’t carry giant tensors around

**Can be done in parallel with A/C/D.**
Dependency: none.

---

### Workstream C — Streaming A2C trainer (core new capability)

**Goal:** new `tools/rl_m5_train_a2c.py` using TD(0), no rollout buffer.

Tasks:

1. Implement trainer loop with:
   - action sampling
   - TD target + advantage
   - losses + grad accumulation

2. Add CLI knobs (env + reward + trainer)
3. Add checkpoint saving (model + optimizer + config + step count)
4. Add training log JSONL (one record per optimizer step)
5. Add “sanity smoke mode” with small N for debugging

**Can be done in parallel with A/B/D/E.**
Dependencies:

- wants action codec v1 eventually, but can be implemented with v0 initially.

---

### Workstream D — Eval correctness (stop lying to yourself)

**Goal:** correct evaluation metrics, ideally with num_envs=1 eval.

Tasks:

1. Create/modify eval script to run `num_envs=1`
2. Report:
   - success rate
   - median steps-to-goal
   - mean return

3. Optionally dump a trajectory (actions + dist) for a single run for debugging

**Can be done in parallel with A/B/C.**
Dependency: none.

---

### Workstream E — Goal templates for 3 stages (content pipeline)

**Goal:** produce `goal_dir_stage1/2/3` consistent with codec v1 + stack_k=1.

Tasks:

1. Decide the exact three goal screens (you already agreed on the stages)
2. Create a small goal-template creation script if you don’t have one:
   - load ROM + state
   - drive env with a short action trace to reach the goal
   - record the goal frame(s) into `goal_template.npy`
   - write meta json via `save_goal_template`

3. Calibrate `tau` for each stage by sampling dist at the goal

**Can be done in parallel with A/B/C/D.**
Dependencies:

- action codec and stack_k should be decided before finalizing templates (but you can prototype now).

---

### Workstream F — Observability (metrics + debugging artifacts)

**Goal:** metrics that scale to 16K envs without drowning you.

Tasks:

1. Extend training log records with:
   - `done_rate`, `trunc_rate`, `reset_rate`
   - `dist_{p10,p50,p90}` (computed from `info["dist"]`)
   - action histogram (8 bins)
   - entropy, value stats

2. Add a lightweight plotter:
   - reward_mean, done_rate, dist_median, entropy vs step

3. Add a single-env GIF dump tool (optional but extremely helpful)

**Can be done in parallel with everything.**
Dependency: none.

---

## The “merge plan” (how these fit together)

1. **A + B** land first (codec + env perf fast path)
2. **C + D** land next (A2C trainer + eval)
3. **E** produces templates (or in parallel, then validate meta)
4. **F** tightens debug loop

At that point you can run:

- Stage 1 at 8192 envs → verify success
- Stage 2 → verify success
- Stage 3 → verify success
- then try 16384 envs

---

## A couple of pitfalls to explicitly guard against

1. **Entropy collapse with huge N**
   If all envs reset identically, early gradients can push the same direction hard.
   Mitigations:

- slightly higher `entropy_coef` early (0.02–0.05)
- use NOOP to avoid “always doing something”
- consider adding small per-env stochasticity (e.g. random initial wait of 0–2 steps) _only if you can do it pixels-only_

1. **Goal match threshold too tight / too loose**

- too tight → never done
- too loose → false positives → training lies
  So: measure dist at goal and set `tau` empirically.

1. **Logging giant tensors**
   Don’t write `dist` vectors of length 16K into json. Log percentiles / means only.

---

## If you want, I can give you a “patch-plan” level checklist

If you tell me whether you want:

- `stack_k=1` **and no recurrence** (simplest), or
- `stack_k=1` + a **tiny GRU** on top of CNN features (still cheap, adds memory)

…I’ll lay out the exact trainer loss + model delta accordingly.

But even without that decision: the workstreams above can start immediately in parallel, and the core learning loop (streaming A2C with TD(0) + grad accumulation) will get you to “8K envs training” with minimal repo disturbance.
