# Simple RL — Milestone M2 Spec + Plan (Torch-only pixels env wrapper)

Date: 2026-01-25

Source context:
- `history/simple-rl.md` (GPU-only RL from pixels; milestone list and constraints)
- `CONSTITUTION.md` (**this is the spec**; engineering doctrine + validation philosophy)
- Repo reality checks:
  - Warp stepping backend exists: `src/gbxcule/backends/warp_vec.py`
  - Env0-only scanline-latched BG renderer exists: `src/gbxcule/kernels/ppu_step.py`
  - Multi-env downsampled snapshot renderer exists: `src/gbxcule/kernels/ppu_render_downsampled.py`
  - `WarpVecCudaBackend(..., render_pixels=True)` allocates `pix: uint8[N,72,80]` and `pixels_torch()` exposes a stable torch view
  - There is no RL wrapper module yet (`src/gbxcule/rl/` does not exist)

This document is the **spec** for Simple-RL Milestone **M2** and the execution plan. If behavior changes, update this document and the tests together.

M2 = “Torch-only env wrapper (no reward in Warp yet)” from `history/simple-rl.md`.

---

## Objective (M2)

Build a minimal, GPU-native stepping loop:

`pixels (uint8 on GPU) → torch policy/value forward → actions (GPU) → Warp step+render → pixels`

The wrapper is a thin PyTorch-facing layer around `WarpVecCudaBackend` that:
- returns **pixel-only** observations (no RAM features for the policy),
- maintains a **4-frame stack**, and
- keeps the hot path **GPU-only** and **stream-correct**.

### Non-goals (explicitly out of scope for M2)

- Goal template capture, done detection, or reward shaping (those are Milestone 3/4 in `history/simple-rl.md`).
- PPO implementation/training loop (Milestone 5).
- ResetCache optimization (Milestone 6).
- Changing the action space in any way (per user: **do not add NOOP**, keep actions as they are).

---

## Spec: The Engineer’s Constitution (mandatory constraints)

This section is a direct application of `CONSTITUTION.md` to this milestone.

### I. Doctrine of Correctness

- **Correctness by Construction**
  - Shapes/dtypes/devices are part of the wrapper’s invariants:
    - pixels are `torch.uint8` on `cuda`
    - stacked obs are `torch.uint8[N,4,H,W]`
    - actions are integer tensors on `cuda` with `0 <= a < num_actions`
  - Fail fast with clear error messages if an invariant is violated.
- **Functional Core, Imperative Shell**
  - Pure functions: frame-stack update, pixel hashing, optional normalization (`u8 → float`).
  - Imperative shell: calling Warp kernels, loading `.state` files, managing device buffers.
- **Unidirectional Data Flow**
  - Treat `step()` as a reducer:
    - Inputs: `prev_state`, `action`
    - Side-effectful edges: “step emulator + render pixels”
    - Outputs: `new_state` (stack, counters) and `obs`
- **Verifiable Rewards**
  - M2 must have deterministic exit-0/exit-1 gates (tests + smoke script).
- **AI Judge**
  - Before human review, run: ruff + pyright + pytest (CUDA tests must be skippable).

### II. Velocity of Tooling

- **Latency ≤ 2 minutes** for the full test suite
  - CUDA-only tests are **skipped by default** (consistent with `Makefile test` setting `GBXCULE_SKIP_CUDA=1`).
  - Keep CUDA tests tiny (few envs, few steps) and avoid long compile cascades.
- **Density & Concision**
  - Keep wrapper small; prefer simple tensor ops and minimal abstractions.

### III. Shifting Left (validation hierarchy)

Apply the inverted pyramid for M2:
1. **Types**: pyright enforces wrapper API expectations (torch optional import is handled cleanly).
2. **Unit tests**: determinism, zero-copy pointer stability, action validation.
3. **Integration tests**: smoke end-to-end stepping on CUDA.
4. **Snapshot/golden**: for M2, pixel-hash sequences act as lightweight snapshots.

### IV. Immutable Runtime (deps)

- Torch is required for RL, but it must be optional for base workflows:
  - add an **optional dependency group** (e.g., `uv sync --group rl`) instead of pulling torch into default deps.
  - wrapper should raise a clear “install torch” error if imported/used without torch.

### V. Observability & Self-Healing

- On failure (mismatch / nondeterminism), emit **structured** debug:
  - step index, env index, pixel-hash, and minimal backend counters if available.
  - avoid verbose logs on success (“silence on success”).

### VI. Knowledge Graph (docs-as-code)

- This plan lives in-repo (`history/`) and is updated when behavior or gates change.

---

## Spec: Simple RL constraints (from `history/simple-rl.md`)

### Hard constraints

1. **Everything on GPU**:
   - Warp/CUDA env stepping + torch forward/backward on CUDA.
2. **No RAM reads for the policy**:
   - the policy input is pixels only (renderer can read VRAM/OAM/IO internally to produce pixels).
3. **24 frames per step**:
   - action frequency is fixed at 24 frames/step for training.
4. **Minimal + incrementally testable**:
   - each milestone has a fast deterministic gate.

### Observation spec (“eyes”)

- 4-shade representation consistent with existing shade logic.
- Downsample recommendation: **80×72** (exact /2 from 160×144), stored as `uint8`.
- Frame stacking: keep last **4 frames**: `uint8[N, 4, 72, 80]`.
- Policy normalization: `x = obs_u8.float() / 3.0` (in torch).

### Warp ↔ Torch bridge requirements

- Zero-copy interop via Warp’s torch conversion helpers.
- Stream-correct sequencing:
  - Warp launches must use the **torch current stream** (no global synchronize in hot path).

### Action space (user override)

- Keep the current action codec behavior.
- Do not add NOOP or change action indices.

---

## Dependencies (M1 → M2 contract)

Milestone 1 is already implemented in-tree:
- snapshot downsampled renderer kernel: `src/gbxcule/kernels/ppu_render_downsampled.py`
- backend wiring: `src/gbxcule/backends/warp_vec.py` supports `render_pixels=True` and accessors:
  - `pixels_wp()` → Warp array `uint8[num_envs*72*80]`
  - `pixels_torch()` → stable torch view `torch.uint8[num_envs,72,80]` (zero-copy via `wp.to_torch`)

**What M2 still must ensure:**
- step+render launches are **ordered on the torch current stream** (no stale reads when the wrapper immediately consumes pixels in torch).
- actions can be provided from torch on CUDA without staging through numpy/host memory.

Notes:
- The existing renderer `ppu_render_bg_env0` is env0-only and full-res (`src/gbxcule/kernels/ppu_step.py`); it is useful for debug/validation but is not sufficient for multi-env RL.

---

## Definition of Done (M2)

### Core behavior

- A new module provides a torch-facing wrapper around `WarpVecCudaBackend`:
  - `obs_u8`: `torch.uint8[N,4,H,W]` on CUDA
  - `step(actions_cuda)` updates pixels and frame stack with **no host copies** in the hot path
  - `reset()` initializes stack deterministically from a fixed `.state`
- Action validation rejects out-of-range actions; action space unchanged.

### Determinism gate (verifiable reward)

- With a fixed:
  - start `.state`
  - action trace
  - `frames_per_step=24`
  - `release_after_frames=8`
- The pixel-hash sequence is identical across two runs (same GPU, same software stack).

### Zero-copy + stream gate

- Zero-copy proof:
  - the torch pixel tensor’s `data_ptr()` is stable across steps (no re-wrap allocating new tensors).
- Stream correctness proof:
  - immediately after `step()`, a torch op reading pixels on the **current stream** never sees stale data (no explicit global `wp.synchronize()` in the production step path).

### Tooling/latency

- Default `make test` remains fast and CPU-only (CUDA tests skipped by env var).
- CUDA tests exist but are tiny and skippable.

---

## Design (interfaces + invariants)

### Backend-facing contract (minimal required API)

We keep `WarpVecCudaBackend.step(np_actions)` for existing harness usage, and add GPU-friendly entry points for RL (CUDA-only):

- `step_torch(actions: torch.Tensor) -> None`
  - `actions` is `torch.int32[N]` on CUDA.
  - Uses Warp↔Torch interop to avoid host copies (device-to-device is ok; host staging is not).
  - Wrap launches in `wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream()))`.
- `render_pixels_snapshot() -> None`
  - Launches the snapshot downsampled renderer without advancing CPU state (used by wrapper `reset()` to build an initial frame stack without stepping).
- `pixels_torch() -> torch.Tensor`
  - Already exists and returns a stable `torch.uint8[N,H,W]` view (created once; reused).

Invariants:
- No `wp.synchronize()` inside these hot-path methods (tests may synchronize for assertions).

### Torch wrapper API

Create `src/gbxcule/rl/pokered_pixels_env.py` with:

- `class PokeredPixelsEnv` (name is flexible; keep it explicit)
  - `__init__(rom_path, state_path, num_envs, frames_per_step=24, release_after_frames=8, out_h=72, out_w=80, stack_k=4, action_codec=...)`
  - `reset(seed=None) -> torch.uint8[N,4,H,W]`
  - `step(actions: torch.Tensor) -> torch.uint8[N,4,H,W]` (optionally return `info` dict for hashing/counters)
  - `close()`

Wrapper-owned state (GPU tensors unless otherwise noted):
- `pix_t: torch.uint8[N,H,W]` (backend view)
- `stack_t: torch.uint8[N,4,H,W]`
- `episode_step: torch.int32[N]` (plumbing for later truncation/reset)

Functional-core helpers (pure):
- `update_stack_inplace(stack_t, frame_t) -> None`
- `hash_pixels_u64(frame_or_stack) -> torch.uint64[N]` (determinism gate)
- `normalize(obs_u8) -> torch.float16/float32` (policy convenience)

---

## Implementation Plan (workstreams)

### WS0 — Torch as optional dependency (keep base workflows clean)

Deliverables:
- Add `[dependency-groups].rl = ["torch"]` (or similar) in `pyproject.toml`.
- Ensure importing non-RL modules never requires torch.
- Wrapper module raises a clear error if torch is missing (actionable install hint).

Acceptance:
- `make test` passes in CPU-only environments without torch installed.

### WS1 — Backend GPU actions path (no host copies)

Deliverables:
- Add a device-actions stepping entry point to `WarpVecCudaBackend`:
  - accepts `torch.int32[N]` on CUDA
  - passes actions into the step kernel without staging through numpy
  - stream-scoped to torch current stream

Acceptance:
- A micro smoke run can step with torch-generated actions without `.cpu().numpy()` conversion.

### WS2 — Pixel buffer torch view (stable, zero-copy)

Deliverables:
- Confirm the existing `pixels_torch()` view is stable (pointer does not change across steps).
- Ensure step+render are ordered on torch stream (no stale reads).
- Add `render_pixels_snapshot()` so `reset()` can initialize stacks without stepping.

Acceptance:
- `pix_t.data_ptr()` stable across steps; contents change each step.

### WS3 — Implement the M2 wrapper (Torch-only env wrapper)

Deliverables:
- `src/gbxcule/rl/pokered_pixels_env.py`
  - `reset()`:
    - `backend.reset()`
    - load `.state` into envs (slow path acceptable for M2)
    - render one frame and initialize frame stack (repeat frame 4×)
  - `step(actions_cuda)`:
    - validate shape/dtype/device/range
    - call backend step+render on torch stream
    - update frame stack
    - increment `episode_step`

Acceptance:
- End-to-end loop: pixels→torch model→actions→pixels works for small `N` on CUDA.

### WS4 — Deterministic gates + minimal tests (verifiable reward)

Deliverables:
- `tools/rl_m2_smoke.py`:
  - runs two identical rollouts and compares pixel-hash sequences
  - prints JSONL lines with step + hash summaries (structured, terse)
- `tests/test_rl_m2_pixels_env.py` (CUDA-skipped by default):
  - determinism test (2 runs match)
  - zero-copy pointer stability (`data_ptr` unchanged)
  - stream correctness sanity (torch op after step sees updated pixels)

Acceptance:
- On a CUDA machine, smoke + tests pass; on CPU-only, tests skip cleanly.

---

## Validation commands (expected workflow)

- CPU-only fast gate: `make test` (skips CUDA tests by default).
- CUDA smoke (manual): `uv run python tools/rl_m2_smoke.py` (requires `uv sync --group rl` and a CUDA-capable torch).
- Lint/type: `make lint` and `make typecheck`.

---

## Risks & mitigations (M2-specific)

1. **Torch installation friction / huge dependency**
   - Mitigation: optional dependency group; RL code path errors clearly when torch missing.
2. **Warp↔Torch stream ordering bugs (stale reads)**
   - Mitigation: enforce `wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream()))` in the wrapper hot path; add explicit stream correctness test.
3. **Renderer fidelity / availability**
   - Mitigation: M2 treats the downsampled renderer as the “camera”; it must be deterministic and stable, not necessarily scanline-perfect. Env0-latched renderer remains a debug reference only.
