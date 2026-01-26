## Simple-RL Milestone 4 Plan: Pixel Reward + `reset_mask` (GPU-only)

This plan implements **Milestone 4** from `history/simple-rl.md`: **reward shaping from pixels + `reset_mask` + truncation**, while treating `CONSTITUTION.md` as the binding spec.

Hard constraint (per user): **do not add a NOOP action**; keep the action space exactly as-is (currently `A, B, START, UP, DOWN, LEFT, RIGHT` via `pokemonred_puffer_v0`).

---

# 1) Spec: `CONSTITUTION.md` → concrete M4 requirements

## I. The Doctrine of Correctness

- **Correctness by Construction**
  - Centralize invariants as code-level checks:
    - pixels are `uint8` in `[0,3]`
    - shapes are consistent (`[N,H,W]`, stacks `[N,K,H,W]`, masks `[N]`)
    - goal template matches render shape
  - Avoid ambiguous flags by representing episode bookkeeping explicitly:
    - `episode_step: int32[N]`
    - `prev_dist: float32[N]`
    - `consec_match: int32[N]`
    - `done: bool[N]`, `trunc: bool[N]`, `reset_mask: bool[N]`

- **Functional Core, Imperative Shell**
  - Keep reward/done/trunc math pure and side-effect-free (Torch only).
  - Keep side effects at the edges:
    - Warp step kernel launch
    - Warp render kernel launch
    - Warp reset kernels (apply `reset_mask`)
    - Torch buffer updates for stacks and per-episode state.

- **Unidirectional Data Flow**
  - Enforce a single-step reducer-style pipeline:
    1) `actions` → warp step
    2) warp render → `pixels`
    3) update stack → `stack`
    4) compute `dist/reward/done/trunc`
    5) `reset_mask = done | trunc`
    6) apply resets (warp) + reset episode trackers (torch)

- **Verifiable Rewards**
  - Every non-trivial component must have a deterministic gate (exit 0 / exit 1):
    - Reward math unit tests (no warp required)
    - Reset correctness tests (warp CPU required; CUDA optional)
    - End-to-end determinism test (fixed start state + action trace → same hashes)

- **The AI Judge**
  - Keep `pytest -q` green for the new M4 tests and existing suite.

## II. The Velocity of Tooling

- **Full suite < 2 minutes**
  - M4 tests must be small and fast:
    - small `N` (e.g., 4–16)
    - small `T` (e.g., 4–32 steps)
    - skip CUDA when not available (`GBXCULE_SKIP_CUDA=1` pattern already exists)

- **Density & Concision / One canonical way**
  - One reward definition (single implementation) and one reset path (single authority).

## III. The Shifting Left

- Prefer type/shape checks and unit tests for reward logic.
- Prefer hash/snapshot assertions for complex buffers (pixels/mem) rather than brittle per-element asserts.

## IV. Immutable Runtime

- No new heavy dependencies for M4.
- Reuse current tooling (warp + torch + pytest).

## V. Observability & Self-Healing

- Silence on success.
- On failure, log enough for an agent to reproduce:
  - env index
  - `dist`, `prev_dist`, `episode_step`, `consec_match`
  - whether `reset_mask` applied
  - pixel/mem hashes

## VI. Knowledge Graph

- Keep M4 semantics in-repo (this plan + a short implementation-spec doc next to the code).

---

# 2) What Milestone 4 means (from `history/simple-rl.md`)

## 2.1 Reward (pixel-only, GPU-friendly)

- Similarity metric (computed on GPU, in Torch):
  - `dist_t = mean(|frame - goal|) / 3.0` (normalized L1) **or**
  - `dist_t = mean((frame - goal)^2) / 9.0` (normalized MSE)
- Done condition:
  - `done = (dist < τ)` for `K` consecutive steps.
- Reward shaping:
  - `r = step_cost + α * (prev_dist - dist)`
  - If `done`: `r += goal_bonus`
- Suggested defaults (first pass):
  - `step_cost = -0.01`
  - `α = 1.0`
  - `goal_bonus = 10.0`
  - `τ = 0.05`, `K = 2`

## 2.2 Episode mechanics

- `episode_step += 1` each step
- `trunc = (episode_step >= max_steps)`
- `reset_mask = done | trunc`
- Applying `reset_mask` must:
  - restore the emulator state for those envs to the start snapshot
  - reset torch-side episode trackers for those envs
  - reset torch-side observation stack for those envs without host copies in the hot path

---

# 3) Preconditions (M4 assumes M1–M3 exist)

Before implementing M4, confirm these are already true (or explicitly finish them first):

1. Multi-env GPU pixel buffer exists: `pix: uint8[N,H,W]` updated every step.
2. Goal template exists and matches render shape (`uint8[H,W]` or `uint8[K,H,W]`).
3. Torch frame stack exists: `stack: uint8[N,K,H,W]` updated on each step.

If any are missing, M4 blocks because reward/done computation depends on them.

---

# 4) Implementation plan (phased, test-gated)

## Phase A — Write the M4 contract (spec-first)

Deliverable: a short implementation spec (docstring or small `docs/`/`history/` note) that pins down:

- `step()` return semantics with autoreset:
  - Recommended: `done/trunc` reflect the transition that ended; returned `stack` is **post-reset** for reset envs.
- Exact reward formula and default hyperparameters.
- “GPU-only hot path” definition (allowed host work in tests only).
- Explicitly state: **no action space changes** (no NOOP).

Acceptance:
- Spec is unambiguous enough to derive tests.

## Phase B — Implement reward/done/trunc as a pure Torch module (functional core)

Create a small module (e.g., `src/gbxcule/rl/reward_pixels.py`) with pure functions:

- `dist(frame_or_stack, goal) -> float32[N]`
- `update_consec(dist, consec_match, tau) -> (consec_match, done)`
- `reward(prev_dist, dist, done, step_cost, alpha, goal_bonus) -> float32[N]`
- `truncate(episode_step, max_steps) -> (episode_step, trunc)`

Unit tests (CPU torch ok):
- Tiny synthetic tensors with known `dist`.
- K-consecutive done behavior.
- Reward sign sanity (improving dist increases reward).
- Trunc boundary conditions.

Acceptance:
- Unit tests pass and are deterministic.

## Phase C — Implement `ResetCache` + `reset_mask` on GPU (imperative shell)

Use the existing stubs:

- `src/gbxcule/core/reset_cache.py` (currently empty stub)

Design:
- Capture a start snapshot once (likely from a `.state` file) and keep it resident for fast resets.
- Implement `reset_mask(mask)` to restore only the envs where `mask[i] == True`.
- Keep hot-path GPU-only:
  - do Warp device-to-device copies for `mem`, `cart_ram`, `cart_state`, and register arrays
  - clear serial buffers/lengths/overflow for reset envs

Integration tests:
- CPU backend reset correctness (always runs):
  - step once, reset half envs, verify “reset envs match baseline” via hashes or `save_state()` comparison.
- CUDA backend reset correctness (optional):
  - same test, but skip when CUDA unavailable.

Acceptance:
- Reset restores state exactly for selected envs and does not mutate others.

## Phase D — Integrate reward + reset into the RL wrapper

Add an RL env wrapper (if not already present) that owns:
- device pixel buffer
- device frame stack + ring update
- per-episode trackers (`prev_dist`, `consec_match`, `episode_step`)
- `ResetCache` instance

Step pipeline (single authority):
1) warp step
2) warp render → `pix`
3) update stack
4) compute `dist/reward/done/trunc`
5) `reset_mask = done | trunc`
6) apply warp resets
7) reset torch-side trackers + obs stacks for reset envs:
   - precompute `start_stack` once (repeat of start frame K times)
   - copy `start_stack` into reset slots without re-rendering

Acceptance:
- Short smoke run shows:
  - non-constant rewards
  - done/trunc causes episodes to recycle correctly
  - no host copies in the hot path (tests may still sync).

## Phase E — M4 end-to-end gates (exit 0 / exit 1)

Add one deterministic end-to-end test:
- Fixed start state + fixed action trace (small N, small T)
- Assert:
  - identical pixel hash sequence
  - identical `(reward, done, trunc)` sequences
  - resets happen on the same steps

Acceptance:
- M4 behavior is reproducible and verifiable with one command.

---

# 5) Risk register (M4-specific) + mitigations

- **Risk: reset snapshot incomplete**
  - Mitigation: tests compare `save_state()` output (or full mem/cart_state hashes) after reset to baseline.

- **Risk: done false positives due to transitions/fades**
  - Mitigation: keep `K>=2` and consider using a small goal-template set later (out of M4 scope).

- **Risk: performance regressions from implicit host sync**
  - Mitigation: keep tests small; avoid `.cpu()` in step wrapper; add a debug mode that asserts no unintended sync calls (optional).

---

# 6) “Definition of Done” for Milestone 4

Milestone 4 is complete when:

1) Pixel-only reward shaping works and is unit-tested.
2) `reset_mask` restores selected envs correctly (integration-tested).
3) Time-limit truncation works and triggers resets.
4) End-to-end determinism test passes for a fixed trace.
5) No action-space changes were made (no NOOP added).

