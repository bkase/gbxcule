# m5 Plan: PPO-lite Training on GPU (Pixels-only, Pokémon Red, 24 frames/step)

This plan is derived directly from:

- `history/simple-rl.md` (RL direction + non-negotiable constraints + milestone definitions)
- `CONSTITUTION.md` (**the spec**: correctness, verifiability, speed, docs-as-code)

It also incorporates the explicit user constraint: **do not add a NOOP action**; keep actions as they are.

---

## m5 Objective

Implement **Milestone 5 — “PPO-lite training on GPU”** from `history/simple-rl.md`:

- Collect rollouts from **N parallel GPU envs** (Warp/CUDA stepping).
- Train a **pixels-only** policy/value network (PyTorch CUDA) using **PPO-lite**.
- Evaluate and report: **success rate** and **steps-to-goal** (goal defined by pixel-template similarity).

M5 is not “researchy RL”; it is a **minimal, verifiable, GPU-native loop** that can be iterated on.

---

## Spec (Non-Negotiables) To Enforce In m5

### From `history/simple-rl.md`

1) **Everything on GPU**

- Env stepping: Warp/CUDA.
- Policy/value forward+backward: PyTorch CUDA.
- Hot path must avoid host transfers (`.cpu()`, `.numpy()`) and global synchronizations.

2) **No RAM reads for the policy**

- The policy only consumes pixels (frame stacks).
- Reward/done are computed from pixels only (goal template similarity).
- Debug tools may read RAM, but training must not depend on it.

3) **24 frames per step**

- Default training config uses `frames_per_step = 24` and `release_after_frames = 8`.
- Treat `frames_per_step != 24` as an explicit config choice (logged); do not “accidentally drift”.

4) **Minimal + incrementally testable**

- Each sub-component has a deterministic “exit 0/1” gate (unit tests + smoke scripts).
- No “it trains on my machine after 3 hours” as the first validation loop.

5) **Action space is fixed**

- Keep `pokemonred_puffer_v0` as-is (7 actions): `A, B, START, UP, DOWN, LEFT, RIGHT`.
- Do **not** add a NOOP action.

---

## `CONSTITUTION.md` → m5 Concrete Commitments (Spec → Implementation → Gate)

### I. The Doctrine of Correctness

**Correctness by Construction**

- Implementation:
  - Define explicit configs: `EnvConfig`, `PPOConfig`, `GoalConfig`.
  - Make invalid states loud: assert tensor `device`, `dtype`, and `shape` at module boundaries.
  - Enforce the fixed action space (`num_actions == 7`) and action index range checks.
- Gates:
  - `uv run pyright` passes for RL modules.
  - Unit tests fail fast on wrong shapes/devices/dtypes.

**Functional Core, Imperative Shell**

- Implementation:
  - Pure-ish torch functions for PPO math: `compute_gae`, `ppo_losses`, `update_step`.
  - CLI scripts are the shell: argument parsing, checkpoint I/O, logging, and orchestration only.
- Gates:
  - PPO core module has no filesystem writes and no reliance on global state.

**Unidirectional Data Flow**

- Implementation:
  - Training dataflow is explicit and one-way:
    - `obs -> model -> action -> env.step -> (obs,reward,done) -> rollout -> ppo_update -> new params`
  - No side-channel state mutations inside the PPO core.
- Gates:
  - Rollout buffer is append-only during collection; consumed once per update.

**Verifiable Rewards**

- Implementation:
  - “Reward” here includes the PPO objective and the pixels-only environment reward/done:
    - PPO update must be testable: finite losses/grads; parameters change.
    - Pixels-only reward/done must be testable: goal replay triggers `done` reliably.
- Gates:
  - Deterministic PPO unit tests (exit 0/1).
  - A short RL smoke run that completes end-to-end in seconds/minutes.

**The AI Judge**

- Implementation:
  - Treat the automated suite (`ruff`, `pyright`, `pytest`) as the first reviewer for correctness/perf/security.
- Gates:
  - Standard check commands are green before a human reads diffs.

### II. The Velocity of Tooling

**Latency is Technical Debt (≤ 2 minutes full tests)**

- Implementation:
  - Keep always-on tests tiny (pure-torch math on small tensors).
  - Put GPU integration smokes behind skips/markers so CPU CI stays fast.
- Gates:
  - `make test` remains fast; RL tests skip if torch (or CUDA) isn’t installed.

**Native Speed / Density & Concision**

- Implementation:
  - Vectorize over `N` envs; minimize Python loops (only loop over `T` rollout steps).
  - Avoid per-step allocations; preallocate buffers on GPU.
  - Keep one canonical PPO implementation (no duplicated variants).
- Gates:
  - A simple perf counter: report `env_steps/s` and `updates/s` (low overhead).

**Code is Cheap, Specs are Precious**

- Implementation:
  - Keep this plan (`history/m5-plan.md`) up to date; treat it as the prompt for implementation.
- Gates:
  - If implementation changes, update acceptance criteria and gates in this doc.

### III. The Shifting Left (Test Pyramid Inverted)

Target validation order for m5:

1) **Types**: pyright catches contract mismatches.
2) **Unit tests**: PPO math, determinism, grad-finite.
3) **Integration**: tiny end-to-end rollout+update on a toy env.
4) **Golden/Snapshot**: freeze expected scalar losses for a fixed synthetic batch.
5) **Agentic E2E (optional)**: short GPU run on real env.

### IV. The Immutable Runtime (Infrastructure & Deps)

- Implementation:
  - Add torch as an **optional** dependency group (e.g. `uv sync --group rl`) so the core emulator remains usable without torch.
  - Record exact versions (torch/warp/cuda) in run manifests.
- Gates:
  - Base tests do not require torch; RL tests are conditionally skipped.

### V. Observability & Self-Healing

- Implementation:
  - Structured logs only: write JSONL with per-update metrics.
  - Crash-only: checkpoints are atomic; training supports resume.
  - On failure (NaNs/infs): dump a minimal “state snapshot” (config + last metrics + trace id).
- Gates:
  - Killing the process and resuming continues from the last checkpoint cleanly.

### VI. The Knowledge Graph (Documentation)

- Implementation:
  - Document exact run commands and configs in-repo (this plan + CLI `--help`).
- Gates:
  - Docs remain consistent with tooling conventions (avoid “undocumented magic”).

---

## Prerequisites (m5 depends on m1–m4 being in place)

m5 assumes these exist (from the earlier milestones in `history/simple-rl.md`):

1) **Multi-env downsampled pixel buffer on GPU**: `pix_uint8[N,H,W]` updated every step with no host copies.
2) **Torch wrapper over the Warp backend** that provides:
   - `obs_uint8[N,4,H,W]` frame stack on GPU
   - `reward_f32[N]`, `done_bool[N]`, `trunc_bool[N]`
   - per-env reset logic (`reset_mask`) and truncation (`max_steps`)
3) **Goal template** on disk and pixels-only goal detection:
   - distance metric (L1 or MSE) computed on GPU in torch
   - `done = dist < tau` for `K` consecutive steps

If any prerequisite is missing, stop and implement its milestone first (don’t hack around it in m5).

---

## Deliverables

1) **PPO-lite implementation (torch)**

- PPO core: GAE, clipped objective, value loss, entropy bonus, grad clip, metrics.
- Rollout buffer stored on GPU.

2) **Pixel policy/value model**

- Minimal Atari-ish CNN trunk:
  - input: `x: [N,4,H,W]` where pixels are shade indices (`0..3`)
  - normalize: `x = obs.float() / 3.0`
  - policy head: logits `[N,7]`
  - value head: scalar `[N]` or `[N,1]`

3) **Training CLI**

- One command runs:
  - env init (GPU), goal template load, rollout collection, PPO updates, periodic eval, logging, checkpointing.

4) **Evaluation CLI**

- Greedy eval episodes: report `success_rate`, `median_steps_to_goal`, `mean_return`.

5) **Verifiable gates**

- Unit tests (torch math).
- RL smoke test (tiny run).
- Optional CUDA integration smoke (skipped if no GPU).

---

## Implementation Steps (Detailed)

### 0) Track the work (beads_rust)

- `br create --title="m5: PPO-lite pixels-only training loop on GPU" --type=task --priority=2`
- `br update <id> --status=in_progress`

### 1) Dependencies & runtime shape (keep base repo lightweight)

- RL dependency group already exists in `pyproject.toml`:
  - `rl = ["torch==2.9.1", "torchvision==0.24.1"]` with Linux CUDA index.
  - Keep the default `make test` path not requiring torch.
  - RL tests use `pytest.importorskip("torch")`.
- Decide the “supported baseline”:
  - CUDA GPU available for real training.
  - CPU-only mode allowed only for unit tests / toy integration tests.

### 2) Define canonical module layout (functional core vs shell)

Suggested structure (names flexible; the split is not):

- `src/gbxcule/rl/ppo.py`:
  - `compute_gae(...)`
  - `ppo_losses(...)`
  - `ppo_update(...)`
- `src/gbxcule/rl/models.py`:
  - `PixelActorCriticCNN(num_actions=7, in_frames=4, h=72, w=80, ...)`
- `src/gbxcule/rl/rollout.py`:
  - `RolloutBuffer` with preallocated CUDA tensors
- `bench/rl/train_ppo_pixels.py` (shell):
  - wires env + model + PPO; logging/checkpoints
- `bench/rl/eval_ppo_pixels.py` (shell):
  - loads checkpoint; runs greedy eval; prints JSON

### 3) PPO core (math-first, deterministic)

Implement PPO-lite with explicit tensor shapes (GPU resident):

- Rollout tensors:
  - `obs_u8: [T, N, 4, H, W]` (store uint8 to reduce memory)
  - `actions: [T, N]` (int64)
  - `rewards: [T, N]` (float32)
  - `dones: [T, N]` (bool)
  - `values: [T, N]` (float32)
  - `logprobs: [T, N]` (float32)
- After rollout:
  - `last_value: [N]`
  - compute `advantages, returns: [T, N]`
- PPO losses:
  - clipped surrogate objective
  - value MSE loss (optionally clipped)
  - entropy bonus
  - metrics: approx_kl, clipfrac
- Optimization:
  - advantage normalization
  - grad norm clip
  - NaN/inf guard (fail with trace id + snapshot)

### 4) Rollout collection (GPU-only hot path)

- Loop `t in 0..T-1`:
  - `logits, value = model(obs)`
  - sample `action ~ Categorical(logits)` (GPU)
  - `next_obs, reward, done, trunc = env.step(action)` (GPU)
  - write all tensors into the rollout buffer (GPU)
  - `obs = next_obs`
- No `.cpu()`/`.numpy()` inside this loop.
- Logging happens only at update boundaries (after rollout), not per step.

### 5) Update schedule & defaults (first working config)

Initial config (tunable, but pick one canonical default):

- Env:
  - `frames_per_step = 24`
  - `release_after_frames = 8`
  - downsample: `H=72, W=80`
  - frame stack: `K=4`
- PPO:
  - `N = 256` (start small; scale later)
  - `T = 128` (or 64 for faster smoke)
  - `gamma = 0.99`, `gae_lambda = 0.95`
  - `clip = 0.1`
  - `lr = 2.5e-4`
  - `entropy_coef = 0.01`
  - `value_coef = 0.5`
  - `epochs = 2` (full-batch to start; minibatches optional later)

### 6) Memory plan (don’t accidentally OOM)

- Compute and print a single “rollout memory estimate” at startup:
  - `obs_u8` dominates: `T*N*4*H*W` bytes.
- Phase 1: keep `N` modest and validate correctness.
- Phase 2 (only if needed): add one of:
  - minibatch PPO update (reduce peak activation memory)
  - 2-bit packing for shades (advanced; only if rollout storage becomes a bottleneck)

### 7) Evaluation loop (verifies progress, not vibes)

- Every `eval_every` updates:
  - Run `E` eval episodes with greedy actions (`argmax(logits)`).
  - Report:
    - `success_rate`
    - `median_steps_to_goal`
    - `mean_return`
- Keep eval small and fast (it’s a gate, not a benchmark).

### 8) Logging & checkpointing (crash-only)

- Write JSONL logs:
  - one line per update: losses, entropy, approx_kl, clipfrac, throughput, eval metrics when run
  - include config + git SHA + version info at run start
- Checkpoint atomically:
  - `checkpoint.tmp` then rename to `checkpoint.pt`
- Support `--resume`:
  - restore model, optimizer, step/update counters, RNG seeds, and config

---

## Test Plan (Fast Gates; Optional CUDA)

### Unit tests (always fast; torch-optional)

1) **GAE correctness**

- Tiny hand-checkable tensors; compare against expected advantages/returns.

2) **PPO math smoke**

- Synthetic batch; assert losses finite and approx_kl/clipfrac within plausible bounds.

3) **Gradient smoke**

- One update step: grads finite; parameters change.

4) **Snapshot regression**

- Freeze expected scalar outputs for a fixed seed synthetic batch (golden numbers).

### Integration tests (toy env; always-on)

- A minimal toy “goal” env implemented in torch:
  - observation is a tiny pixel grid
  - reward/done from pixel similarity
  - ensures end-to-end training loop works without Warp/CUDA dependencies

### Optional CUDA integration smoke (skipped without GPU)

- A tiny real-env run:
  - `N` small (e.g. 8–32), `T` small (e.g. 32)
  - run 1 update; assert no NaNs and that rollout stays on CUDA

All gates must respect the **≤ 2 minute** suite budget.

---

## Acceptance Criteria (m5 Done Means)

1) **End-to-end PPO-lite runs on GPU**

- Rollout collection and PPO updates execute with CUDA tensors end-to-end.
- No action-space changes (still exactly 7 actions).

2) **Verifiable correctness gates exist and pass**

- PPO unit tests pass deterministically.
- RL smoke run completes quickly and produces logs + checkpoint.

3) **Learning signal is real (not necessarily SOTA)**

- Compared to random baseline:
  - success rate increases measurably, and/or
  - median steps-to-goal decreases over training.

---

## Non-Goals (Explicitly Out of Scope for m5)

- Adding a NOOP action (explicitly disallowed).
- RNN/LSTM policies (frame stack is the minimal approach).
- Intrinsic rewards (e.g., RND) — can be a follow-on once PPO works.
- Making the renderer scanline-perfect for all envs — training camera only needs to be consistent and useful.

---

## Session Closeout (Repo Protocol)

- `git status`
- `git add <files>`
- `br sync --flush-only`
- `git commit -m "m5: plan PPO-lite pixels-only training loop"`
- `git push`
- `br close <id> --reason="Completed"`
