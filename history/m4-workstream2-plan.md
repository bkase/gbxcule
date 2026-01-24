# Workstream 2 Plan — Fused Kernel Family (stage variants via templates)

> Assumption: “Workstream 2” refers to `history/m4-architecture.md` §10 item 2 (“Fused kernel family”).
> If instead you meant M3’s WS2 (“`warp_vec_cuda` explicit GPU backend”) from `history/m3-workstreams.md`,
> say so — much of that is already implemented in `src/gbxcule/backends/warp_vec.py`.

## Goal

Produce a single CPU-stepping kernel family with four compiled **stage variants**:

- `emulate_only`: advance CPU only
- `reward_only`: CPU + reward epilog
- `obs_only`: CPU + obs epilog
- `full_step`: CPU + obs + reward epilog

All stages must share one stable “superset” kernel ABI (signature) to avoid churn in later workstreams
(action schedule + JOYP + training integration).

## Scope / non-goals

**In-scope**

- Add a second injection site (`POST_STEP_DISPATCH`) to the Warp kernel generator.
- Implement minimal deterministic `reward_v0` and `obs_v0` epilogs.
- Compile/cache 4 kernels keyed by stage.
- Wire `bench/harness.py --stage ...` into Warp backends so it changes behavior (not metadata-only).
- CPU-first tests proving stage correctness and “no emulation divergence across stages”.

**Out-of-scope (explicitly)**

- Action codec standardization and Pokémon schedule parity.
- JOYP emulation and JOYP-driven micro-ROM(s).
- Pufferlib baselines.
- “Return host obs/reward” training integration (debug readback for tests is fine).

---

## 1) Lock the fused-kernel ABI (spec you implement against)

All 4 stage kernels use the same signature, even if some args are unused in WS2.

### Proposed kernel args

**Core state (existing)**

- `mem: uint8[num_envs * 65536]`
- `pc/sp/a/b/c/d/e/h/l/f: int32[num_envs]`
- `instr_count/cycle_count: int64[num_envs]`
- `cycle_in_frame: int32[num_envs]`
- `frames_to_run: int32`

**Reserved for later workstreams**

- `actions: int32[num_envs]`
- `joyp_select: uint8[num_envs]`
- `release_after_frames: int32`

**Outputs**

- `reward_out: float32[num_envs]`
- `obs_out: float32[num_envs * OBS_DIM]` (flat buffer; backend reshapes)

### OBS_DIM policy (for WS2)

To avoid a “compile per obs_dim” explosion during early bring-up:

- Support only `obs_dim=32` in WS2 (enforce in WarpVec backends).
- If/when needed later: extend kernel cache key to include `obs_dim`.

### Stage write semantics (make a decision now)

Pick one and then test it:

- **Preferred:** non-owning outputs are left untouched (lets tests catch accidental writes).
- Alternative: write zeros for non-owning outputs (simpler but hides bugs).

---

## 2) Extend the kernel builder with `POST_STEP_DISPATCH`

Target file:

- `src/gbxcule/kernels/cpu_step_builder.py`

### Changes

- Add `POST_STEP_DISPATCH` placeholder to `_CPU_STEP_SKELETON`, placed:
  - after the frame loop completes
  - before regs/counters are written back to arrays
- Extend the CST injector to replace `POST_STEP_DISPATCH` with a stage-specific statement block,
  similar to how `INSTRUCTION_DISPATCH` works today.

### Builder API shape

Either approach is fine; pick one and keep it minimal:

- **Minimal:** `build_cpu_step_source(..., post_step_body=...)`
- **Template-based:** add an `EpilogTemplate` dataclass and specialize it per stage

### Acceptance checks

- Generated source contains no remaining placeholder tokens.
- The module hash changes when stage changes (separate kernel cache entries).

---

## 3) Add post-step templates: `reward_v0` and `obs_v0`

New files:

- `src/gbxcule/kernels/cpu_templates/post_step.py`

### `reward_v0` (deterministic, small, real work)

Requirements:

- Deterministic given state.
- Cheap but not optimizable-away.
- Touches a tiny fixed set of bytes/regs.

Example spec:

- Read `mem[base + 0xC000]` and `mem[base + 0xC001]`.
- Compute `reward = float32((a_i ^ (m0 + 3*m1)) & 0xFF) / 255.0`.
- Write `reward_out[i] = reward`.

### `obs_v0` (writes exactly OBS_DIM floats per env)

Requirements:

- Writes exactly `OBS_DIM` floats to `obs_out[i*OBS_DIM + j]`.
- Includes a few reg features + a few tiny WRAM-derived features.
- Never writes out-of-bounds.
- Deterministic and stable across devices.

---

## 4) Compile/cache four kernels by stage

Target file:

- `src/gbxcule/kernels/cpu_step.py`

### Changes

- Change `get_cpu_step_kernel()` to accept `stage` (and optionally `obs_dim`):
  - `get_cpu_step_kernel(stage: Stage, obs_dim: int = 32)`
- Maintain a kernel cache keyed by `(stage, obs_dim)`.
- Build `post_step_body` per stage:
  - `emulate_only`: empty body
  - `reward_only`: reward template body
  - `obs_only`: obs template body
  - `full_step`: obs body + reward body (fixed ordering)

### Warmup

- Update warmup helpers to warm up the selected stage kernel (not just a single default kernel).

### Acceptance checks

- Stage-specific warmup compiles the intended kernel once.
- Subsequent `get_cpu_step_kernel(stage=...)` calls reuse the cached kernel.

---

## 5) Wire stage selection through Warp backends + harness

Target files:

- `src/gbxcule/backends/warp_vec.py`
- `bench/harness.py`

### Backend changes

- Add `stage: Stage = "emulate_only"` to WarpVec constructors.
- Store `self._stage` and select kernel accordingly.
- Allocate device buffers in `reset()`:
  - `actions_dev: int32[num_envs]` (reserved)
  - `joyp_select_dev: uint8[num_envs]` (reserved)
  - `reward_dev: float32[num_envs]`
  - `obs_dev: float32[num_envs * obs_dim]`
- `step(actions)` launches the stage kernel with the full signature.
- Preserve the current invariant: no unconditional `wp.synchronize()` in CUDA `step()`.

### Harness changes

- Plumb `--stage` into `create_backend(...)` so Warp backends receive it.
- Artifacts already record `"stage": args.stage`; WS2 makes it behaviorally true.

### Acceptance checks

- `uv run python bench/harness.py --backend warp_vec_cpu --stage full_step --rom ... --steps 10` runs and uses the stage kernel.
- CUDA path remains async-friendly; harness `--sync-every` still governs measurement correctness.

---

## 6) Tests + gates (CPU-first)

New test file (suggested):

- `tests/test_fused_kernels_ws2.py`

### CPU unit tests (fast)

- `emulate_only` does not write reward/obs (or writes zeros, depending on chosen policy).
- `reward_only` writes reward but not obs.
- `obs_only` writes obs but not reward.
- `full_step` writes both and matches `reward_only`/`obs_only` for the same input state.

### Invariant test

CPU emulation state must be identical across stages after a step:

- regs (`pc/sp/a/.../f`) match
- (optionally) a small WRAM hash matches

### Optional CUDA smoke (guarded)

- If CUDA is available, run 1–2 steps and validate readback via explicit synchronize + copy.

---

## Definition of Done (WS2)

- Kernel generator supports both `INSTRUCTION_DISPATCH` and `POST_STEP_DISPATCH`.
- `get_cpu_step_kernel(stage=...)` returns four distinct cached kernels.
- Warp backends accept `stage` and launch the correct stage kernel.
- `bench/harness.py --stage ...` affects actual Warp execution (not metadata only).
- CPU tests prove:
  - stage write behavior is correct
  - emulation core state is identical across stages

