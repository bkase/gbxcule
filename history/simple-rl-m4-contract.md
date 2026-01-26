## Simple-RL M4 Contract (Pixel Reward + Autoreset)

This document is the binding M4 contract. It must be read together with:
- `CONSTITUTION.md` (spec-level requirements)
- `history/simple-rl-m4-plan.md` (implementation plan)

Hard constraint: **Action space is unchanged** (no NOOP added). Action codec remains `pokemonred_puffer_v0` (`A, B, START, UP, DOWN, LEFT, RIGHT`).

---

# 1) Tensor Contracts (types, shapes, devices)

All hot-path tensors are on **CUDA** (torch) and **uint8** pixels are in `[0,3]`:

- `pixels`: `uint8[N, 72, 80]` (downsampled shades)
- `obs_stack`: `uint8[N, K, 72, 80]` (stacked frames)
- `reward`: `float32[N]`
- `done`: `bool[N]`
- `trunc`: `bool[N]`
- `episode_step`: `int32[N]`
- `prev_dist`: `float32[N]`
- `consec_match`: `int32[N]`

No silent dtype coercion is allowed in the functional core.

---

# 2) Step Semantics (Unidirectional data flow)

**Reducer-style pipeline** for each step:

1) `actions` → `backend.step_torch(actions)`
2) `render_pixels` → `pixels`
3) update `obs_stack` (shift + append latest frame)
4) compute `dist`, `done`, `reward`, `trunc`
5) `reset_mask = done | trunc`
6) apply resets (Warp) + reset torch-side trackers + obs stack

**Autoreset semantics:**
- `done/trunc` refer to the *transition that ended*.
- Returned `obs_stack` is **post-reset** for reset envs.

---

# 3) Goal Matching (pixel-only)

Distance (normalized L1):

```
dist = mean(|frame - goal|) / 3.0
```

Where `frame` is:
- the last frame if goal is 2D (`[H,W]`), or
- the full stack if goal is 3D (`[K,H,W]`).

**Done definition:**
- `done = (dist < tau)` for `K` consecutive steps.

---

# 4) Reward Shaping (pixel-only)

Per-step reward:

```
r = step_cost + alpha * (prev_dist - dist)
if done: r += goal_bonus
```

Default first-pass values:
- `step_cost = -0.01`
- `alpha = 1.0`
- `goal_bonus = 10.0`
- `tau = 0.05`, `K = 2`

---

# 5) Truncation + Reset

Truncation:
- `episode_step += 1`
- `trunc = (episode_step >= max_steps)`

Reset mask:
- `reset_mask = done | trunc`

**Reset must restore:**
- Emulator state (mem, regs, counters, cart state/ram, timers, PPU state)
- Torch-side trackers (`episode_step`, `prev_dist`, `consec_match`)
- Obs stack reset to the start stack (cached)
- Pixel buffer consistent with the start state (copy cached pixels or rerender)

---

# 6) GPU-only hot path

No `.cpu()` / host sync inside the step loop. Tests and tools may synchronize.

---

# 7) M4 Gates (Verifiable rewards)

To accept M4:

1) **Unit tests** for reward math and truncation (fast, CPU torch ok).
2) **Reset correctness** test (CPU backend) + optional CUDA variant.
3) **Determinism gate**: fixed seed + trace yields identical pixel hashes and
   `(reward, done, trunc)` sequences across runs.

---

# 8) Constitution alignment (explicit)

This contract enforces:
- **Functional Core, Imperative Shell**: reward/done/trunc are pure torch; reset is Warp.
- **Unidirectional Data Flow**: single step pipeline; no hidden side effects.
- **Verifiable Rewards**: deterministic tests as gates.
- **Density & Concision**: one canonical reward definition + one reset path.
