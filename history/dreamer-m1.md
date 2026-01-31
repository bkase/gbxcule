# Dreamer v3 — M1 Plan: ReplayRing (CPU-first) + Continuity Invariants

This plan derives from `history/dreamer-plan.md` and the M1 gotchas in `history/dreamer-gotchas.md`. It focuses on the ReplayRing backbone: time-major, packed2, deterministic sampling, and strict continuity validation via `episode_id`.

## M1 Objective

Build a **device-agnostic ReplayRing** (CPU now, CUDA-ready later) that stores packed2 observations and transition fields, supports **non-strict sampling**, and enforces **episode continuity invariants** using `episode_id` plus explicit `is_first` and `continue` semantics.

If M0 scaffolding is incomplete, we only add the minimal Dreamer v3 module scaffolding needed to host ReplayRing and its tests; no Golden Bridge or model code is part of M1.

## Non-Negotiable Constraints (from master plan + gotchas)

- **Time-major storage**: replay buffers are `[Tcap, N, ...]` (time, env). Sampling returns `[T, B, ...]`.
- **Packed2 only** for now: `obs` is `uint8[Tcap, N, 1, 72, 20]` (using `DOWNSAMPLE_H`, `DOWNSAMPLE_W_BYTES`).
- **Mandatory `episode_id`**: `int32[Tcap, N]`, strictly increasing on resets.
- **True `is_first`** flags in storage: ReplayRing must record real episode starts. Training may still override `is_first[0] = 1` (as sheeprl does), but the buffer must not fabricate boundaries.
- **`continue` semantics**: float32 target where `0.0` = true terminal, `1.0` = time-limit truncation or normal step. (Use name `continue_` or `continues` in code to avoid Python keyword.)
- **Non-strict sampling**: sequences are allowed to include `is_first` inside the window; **do not reject** them.
- **Deterministic sampling**: requires explicit `torch.Generator` (no global RNG).
- **Continuity invariants** are enforced in tests and via a debug checker:
  - For each env and adjacent time steps: `episode_id[t+1] == episode_id[t]` **OR** `is_first[t+1] == 1`.
  - If `is_first[t+1] == 1`, then `episode_id[t+1] == episode_id[t] + 1`.
  - `continue` values must be exactly (or very close to) `{0.0, 1.0}`.

## Sheeprl Reference Alignment (Replay-specific)

Key behaviors observed in `third_party/sheeprl` that inform this M1 plan:

- **Sequential, boundary-agnostic sampling**: DreamerV3 uses `SequentialReplayBuffer`, explicitly sampling **consecutive** sequences **without** filtering episode boundaries. We mirror this with non-strict sampling and rely on `is_first` in the RSSM scan.
- **Avoid the ring discontinuity**: When the buffer is full, sheeprl **avoids** start indices that would make a sequence cross the write pointer (the temporal discontinuity). Our ReplayRing should sample from a **chronological view** (oldest→newest) so sequences are contiguous in time and never wrap across the discontinuity.
- **One env per sequence**: sheeprl chooses a single environment for each sampled sequence; we do the same (env index per sequence, not per time step).
- **Training forces `is_first[0]=1`**: sheeprl sets `is_first` true at time index 0 for every sampled sequence (even if the buffer has `is_first=0`). We keep true `is_first` in storage, and treat this as a training-time option (M3+).
- **No stored `continue` in sheeprl**: they compute `continue = 1 - terminated` (timeouts are `truncated=1, terminated=0`). Our explicit `continue` field encodes the same semantics and should match this behavior.
- **Action alignment + shift**: sheeprl stores actions aligned with `obs[t]`, then **shifts** actions during training (prepend zero action, drop last) to match Dreamer’s dynamics update. ReplayRing should store raw actions; shifting happens in model code (M3).

## Deliverables

1. **ReplayRing implementation** (device-agnostic; CPU tests only for M1).
2. **ReplayRing invariant checker** (callable in tests and optional debug mode).
3. **Deterministic sampling API** returning time-major sequences.
4. **CPU pytest suite** covering wraparound, invariants, determinism, and truncation semantics.

## File/Module Layout

If `src/gbxcule/rl/dreamer_v3/` does not exist yet, create minimal scaffolding:

- `src/gbxcule/rl/dreamer_v3/__init__.py`
- `src/gbxcule/rl/dreamer_v3/replay.py` (or `replay_ring.py`)
- (Optional) `src/gbxcule/rl/dreamer_v3/rng.py` if M0 isn’t done; include `require_generator(gen)` utility to enforce explicit RNG.

Tests:

- `tests/rl_dreamer/test_replay_ring.py`

## ReplayRing Design (M1 Scope)

### Constructor

```
ReplayRing(
  capacity: int,
  num_envs: int,
  device: str = "cpu",
  obs_shape: tuple[int, int, int] = (1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES),
  debug_checks: bool = False,
)
```

- Allocates **time-major** tensors on `device`:
  - `obs`: `uint8[capacity, num_envs, 1, 72, 20]`
  - `action`: `int32[capacity, num_envs]`
  - `reward`: `float32[capacity, num_envs]`
  - `is_first`: `bool[capacity, num_envs]`
  - `continue_` (or `continues`): `float32[capacity, num_envs]`
  - `episode_id`: `int32[capacity, num_envs]`
- Tracks internal ring state:
  - `head` (next write index), `size` (# valid steps), `total_steps` (monotonic counter).

### Write API

Two options are compatible with later CUDA ingestion; pick one and keep it consistent:

1. **Implicit index** (simplest):
   - `push_step(obs, action, reward, is_first, continue_, episode_id)`
   - Writes to `head`, increments `head`/`size`/`total_steps`.

2. **Explicit t with guard**:
   - `push_step(t, ...)` where `t` must equal `total_steps`.
   - Allows future actor loops to pass global time explicitly without holes.

Either way, the ring must guard against invalid shapes/dtypes and **must not copy obs if caller uses a slot view**.

### Direct-write slot (for M6 readiness)

- `obs_slot(step_idx: int) -> Tensor` returning a **view** of `obs[step_idx]`.
- Guarantee slot is contiguous to support direct writes.

### Sampling API

```
sample_sequences(batch: int, seq_len: int, gen: torch.Generator) -> dict[str, Tensor]
```

- Requires `seq_len <= size`; else raise.
- Randomly sample `(env_idx, start_offset)` pairs with explicit `gen`.
- Build time indices from a **chronological view** (oldest→newest). This ensures:
  - sequences are contiguous in time,
  - sequences never cross the ring discontinuity when full (sheeprl behavior),
  - sequences are valid even when the ring wraps.
- output shapes: `[seq_len, batch, ...]` for each field.
- **No filtering** on `is_first` or `episode_id` (non-strict sampling).
- Optionally return `indices` for test/debug (e.g., `meta={"env_idx":..., "start_offset":...}`).
- (Optional, post-M1) Add `n_samples` dimension to match sheeprl’s `sample(..., n_samples=K)` → `[K, T, B, ...]`.

### Invariant Checker

A utility method (or standalone function) that walks the **chronological view** of the ring and verifies:

- `episode_id` continuity rules
- `continue_` in `{0.0, 1.0}` (tolerate tiny epsilon)
- Optional: `is_first` implies episode increment for all envs

Expose as:

- `check_invariants()` for tests
- `debug_checks` flag to run after each write in development builds

## Test Plan (CPU, fast, deterministic)

Create `tests/rl_dreamer/test_replay_ring.py` with the following unit tests:

1. **Shape + dtype sanity**
   - Create ring and assert all buffers have correct shapes and dtypes.

2. **Obs slot view**
   - `slot = ring.obs_slot(i); slot.fill_(...)` and assert underlying buffer changed.

3. **Wraparound correctness + no discontinuity crossing**
   - Capacity small (e.g., 5), write > capacity and ensure:
     - `size == capacity`
     - chronological order matches expected overwrite pattern
     - `sample_sequences` only draws from the most recent `capacity` steps
     - sampled windows never wrap across the oldest/newest boundary

4. **Episode continuity invariants**
   - Write two episodes with explicit `is_first` and `episode_id`.
   - Run `check_invariants()` and expect success.
   - Add a negative test that violates invariants (e.g., increment without `is_first`) and expect failure.

5. **Continue semantics (terminal vs truncation)**
   - Simulate a timeout reset: `continue=1.0` at last step, then `is_first=1` next step.
   - Simulate true terminal: `continue=0.0` at last step.
   - Assert stored values are exactly as written and pass invariant check.

6. **Deterministic sampling**
   - Use fixed `torch.Generator` seed; run sampling twice and assert identical indices and samples.

7. **Non-strict sampling across boundary**
   - Create a sample window that includes an `is_first` in the middle; ensure sampling succeeds and data is returned unchanged.

All tests must run without CUDA and complete in milliseconds.

## Implementation Steps

1. **Claim task in beads**
   - `br create --title="dreamer m1: replay ring + invariants" --type=task --priority=2`
   - `br update <id> --status=in_progress`

2. **Scaffold module if missing**
   - Create `src/gbxcule/rl/dreamer_v3/` with `__init__.py` and `replay.py`.
   - Add `_require_torch()` helper consistent with existing RL modules.

3. **Implement ReplayRing**
   - Allocate tensors with explicit dtypes.
   - Implement ring state (`head`, `size`, `total_steps`).
   - Add `obs_slot`, `push_step`, `sample_sequences`, `check_invariants`.
   - Keep code purely functional where possible (no IO, no env dependencies).

4. **Write tests**
   - Add `tests/rl_dreamer/test_replay_ring.py`.
   - Keep tests small and deterministic.

5. **Run CPU tests**
   - `uv run pytest -q tests/rl_dreamer/test_replay_ring.py`

6. **Close session per repo protocol**
   - `git status`
   - `git add <files>`
   - `br sync --flush-only`
   - `git commit -m "dreamer m1: replay ring + invariants"`
   - `git push`
   - `br close <id> --reason="Completed"`

## Acceptance Criteria (M1 Done Means)

- ReplayRing stores packed2 obs and transition fields with correct shapes/dtypes.
- Sampling is deterministic with an explicit `torch.Generator` and **does not reject** sequences with `is_first` inside.
- Continuity invariants are enforced and tested (including wraparound).
- `continue` semantics are preserved exactly as authored (0.0 terminal, 1.0 truncation/normal).
- All new tests pass on CPU; no CUDA required.

## Non-Goals for M1

- CUDA direct-write ingestion or zero-copy guards (M6).
- RSSM/world model/behavior learning (M3–M5).
- Golden Bridge fixtures (M0/M2/M4).
- Replay ratio scheduling / async engine integration (M7).
