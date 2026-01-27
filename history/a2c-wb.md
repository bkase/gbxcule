# Workstream B Plan — High-N Env Performance + `stack_k=1` Fast Path

This plan is derived from `history/a2c-training.md` (Workstream B section) and is constrained by `CONSTITUTION.md` (the spec).

## Goal (Workstream B)

Make **8,192–16,384** env training viable by removing hidden per-step overhead in the torch-facing pixel envs:

- Eliminate per-step frame-stack shifting/allocations when `stack_k=1`.
- Ensure autoreset/reset and logging don’t introduce Python loops, sync points, or giant tensor churn at high N.
- Add automated verification (tests/bench) so performance + correctness are **verifiable** (exit 0/1).

Primary touchpoints:

- `src/gbxcule/rl/pokered_pixels_env.py` (stack update in `step()`, stack init in `reset()`)
- `src/gbxcule/rl/pokered_pixels_goal_env.py` (stack update in `step()`, stack init in `reset()`, reset-mask path)
- `src/gbxcule/core/reset_cache.py` (`apply_mask_torch()` / masked reset kernel launches)
- (new) a small bench tool under `tools/` to measure throughput + allocations

---

## Spec Constraints (from `CONSTITUTION.md`) → Requirements for Workstream B

### I. The Doctrine of Correctness

- **Correctness by Construction**
  - Keep invariants explicit (`num_envs >= 1`, `stack_k >= 1`, dtype/device/shape checks for tensors).
  - Add tests so `stack_k=1` fast path cannot silently diverge from the general path semantics.
- **Functional Core, Imperative Shell**
  - Isolate “stack update” into a small helper that is easy to test (pure-ish tensor transform).
  - Keep backend stepping (`backend.step_torch`, `render_pixels_snapshot_torch`) and mutation at the edges.
- **Unidirectional Data Flow**
  - Treat stack update as: `prev_stack + new_pix -> new_stack` (one-way). Logging must not mutate env state.
- **Verifiable Rewards**
  - Performance and correctness must be gated by automated checks (tests/bench) that fail loudly (exit 1).
- **The AI Judge**
  - Keep diffs minimal, invariants obvious, and validation easy to run/interpret.

### II. The Velocity of Tooling

- **Latency is Technical Debt (≤ 2 minutes full test suite)**
  - New tests must be small/targeted; CUDA tests must skip cleanly when CUDA is unavailable.
- **Native Speed**
  - Avoid per-step Python loops, per-step tensor allocations, and sync points.
- **Density & Concision**
  - One canonical implementation of stack update; avoid duplication across envs.
- **Code is Cheap, Specs are Precious**
  - Encode assumptions as assertions/tests/bench artifacts, not tribal knowledge.

### III. The Shifting Left

- **Types / unit tests / integration tests first**
  - Unit-test stack semantics on CPU tensors (fast).
  - Add minimal CUDA integration tests (skip when CUDA missing).
- **Golden & snapshot verification**
  - Bench/log output should be stable JSON/JSONL; avoid dumping huge tensors.
- **Agentic E2E**
  - Reuse existing smoke scripts (`tools/rl_m2_smoke.py`, `tools/rl_m4_smoke.py`) and run them with `stack_k=1`.

### IV. The Immutable Runtime (Infrastructure & Deps)

- Keep torch as an optional dependency (retain dynamic import pattern).
- No new heavy deps; use existing timing primitives (e.g., `torch.cuda.Event`) for perf.
- Bench results must record config so they’re reproducible and comparable.

### V. Observability & Self-Healing

- **Structured logs only**
  - Bench/perf outputs should be machine-readable (JSON/JSONL).
- **Crash-only**
  - Maintain clean resource teardown (`close()`); don’t leave persistent state.
- **Silence on success**
  - One summary record per run; on failure include enough context to repro.

### VI. The Knowledge Graph (Documentation)

- Document “scale mode” knobs and caveats in-repo (this file + related tooling).
- If new knobs are added (e.g., `info_mode`), they must appear in `--help` and docs.

---

## Work Items (B0–B5)

### B0 — Define “Done” + Baseline (Verifiable)

**Purpose:** Establish measurable success criteria and a stable way to validate improvements.

1. **Target scenarios**
   - `PokeredPixelsEnv`: `num_envs ∈ {8192, 16384}`, `stack_k=1`, plus a small sanity case (`num_envs=1/256`, `frames_per_step=1`).
   - `PokeredPixelsGoalEnv`: same, with trunc-based resets to exercise autoreset.
2. **Add a benchmark tool** (new script, e.g. `tools/rl_env_bench.py`)
   - Warmup (e.g., 50 steps), then measure (e.g., 500–2000 steps).
   - Use `torch.cuda.Event` timing to report:
     - `env_steps_per_sec`, `ms_per_step`, `num_envs`, `stack_k`, `frames_per_step`, `release_after_frames`.
   - Record allocation behavior (e.g., `torch.cuda.max_memory_allocated()` deltas) after warmup.
   - Emit a single JSON record to stdout (and optionally a JSONL log file).
   - Add `--max-mem-delta` to fail if allocations grow after warmup.
3. **Acceptance criteria (initial)**
   - `stack_k=1` must perform **no per-step CUDA allocations after warmup** (within small tolerance).
   - Throughput must be **non-regressing** vs pre-change baseline; expected improvement from removing `.clone()` + shift.

### B1 — Implement the `stack_k==1` Fast Path (Core Change)

**Purpose:** Remove the largest per-step overhead at high N.

**Files:** `src/gbxcule/rl/pokered_pixels_env.py`, `src/gbxcule/rl/pokered_pixels_goal_env.py`

1. **Reset path**
   - Special-case `stack_k==1` to initialize stack with a single `copy_` instead of a loop.
2. **Step path (big win)**
   - Replace:
     - `self._stack[:, :-1].copy_(self._stack[:, 1:].clone())`
     - `self._stack[:, -1].copy_(pix)`
   - With:
     - if `stack_k == 1`: `self._stack[:, 0].copy_(pix)`
     - else: keep current logic (for now).
3. **upgrade (recommended): remove per-step `.clone()` for `stack_k>1`**
   - Allocate a reusable scratch buffer once and implement overlap-safe shifting without allocating each step.
   - Benefit: improves both scale and non-scale cases; aligns with “latency is technical debt”.

### B2 — Reset/Autoreset Scaling Check (`ResetCache.apply_mask_torch`)

**Purpose:** Ensure autoreset isn’t quietly dominating time at high N.

**Files:** `src/gbxcule/core/reset_cache.py`, `src/gbxcule/rl/pokered_pixels_goal_env.py`

1. **Measure reset costs**
   - In the bench tool, include:
     - “no resets” run (mask all-false),
     - “forced resets” run (force trunc every step for a short run).
2. **If “no resets” is expensive**
   - Consider an early-exit when the mask is empty (but avoid host-side sync regressions).
   - Any “skip if no reset” optimization must be benchmarked: sometimes checking `any(mask)` can cost more than launching kernels.
3. **Verify no per-env Python work**
   - Confirm `reset()` paths don’t contain per-env Python loops at high N (other than unavoidable env0 snapshot capture).

### B3 — Optional: “Subsampled Info” Mode (Prevent 16K Tensor Logging)

**Purpose:** Make it hard to accidentally log giant tensors or flood JSONL logs.

**Option A (trainer-side only):**

- In training scripts, never write raw `dist` vectors; compute percentiles/means.

**Option B (env-supported):**

- Add an `info_mode` knob: `full | stats | none`.
- In `stats` mode, return only aggregates (GPU-computed), not full tensors.

### B4 — Tests: Correctness + Allocation Regression Guard

**Purpose:** “Verifiable rewards” — correctness/perf cannot be anecdotal.

1. **CPU unit tests**
   - Unit-test stack update semantics for `stack_k=1` and `stack_k>1` on CPU tensors (fast).
2. **CUDA integration tests (skip when CUDA absent)**
   - Extend `tests/test_rl_m2_pixels_env.py` to cover `stack_k=1`:
     - shape is `(N, 1, H, W)`
     - `obs[:, 0]` matches `env.pixels` after step.
   - Add a minimal `PokeredPixelsGoalEnv` CUDA test for `stack_k=1` (reward/done/trunc smoke).
3. **Allocation guard**
   - Add a CUDA test or bench assertion that max allocated memory is stable after warmup across many steps.

### B5 — Documentation (“Scale Mode” Knobs)

**Purpose:** Keep the repo as the source of truth.

1. Document recommended scale config:
   - `stack_k=1`, avoid per-step tensor logging, compute stats summaries.
2. If new flags exist (e.g. `info_mode`), ensure they show in `--help` and are referenced in this doc.
   - Current scale knobs: `stack_k=1`, `info_mode=stats|none`, `skip_reset_if_empty`, `--max-mem-delta`.

---

## Deliverables Checklist

- `stack_k=1` fast path in both pixel envs (no shift, no clone).
- (Optional) scratch-buffer shift for `stack_k>1` (eliminate per-step allocations).
- Bench tool that outputs structured JSON and can fail on regressions.
- Tests covering `stack_k=1` semantics and (optionally) allocation stability, skipping cleanly without CUDA.
- Clear documentation of “scale mode” defaults and logging constraints.
