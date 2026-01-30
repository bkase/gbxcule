# Dreamer v3 M6 Plan: GPU Ingestion + Zero-Copy Replay (Packed2 Direct-Write)

This plan is derived directly from `history/dreamer-plan.md` and the Dreamer gotchas. It focuses exclusively on **Milestone M6**: making **CUDA direct-write replay ingestion real** and enforcing **zero-copy gates**.

---

## M6 Objective (One Sentence)

Implement a **CUDA-native ReplayRing** that accepts **packed2 pixel frames written directly by Warp** into replay memory, with **commit-gated visibility** for the learner and **zero host transfers** in the hot path.

---

## Non‑Negotiables (Hard Constraints)

1) **Zero-copy hot path**
- No `.cpu()`, `.numpy()`, host staging buffers, or implicit syncs in the actor → replay path.
- All replay buffers live on CUDA; sampling happens on CUDA.

2) **Direct-write packed2**
- Replay stores packed2 (`uint8`) frames; **env writes straight into `ReplayRingCUDA.obs_slot()`**.
- No unpacking for ingestion; unpacking only inside the encoder (or benchmark-only).

3) **Commit-gated visibility**
- Learner can **only** sample from `committed_t` (with a safety margin).
- Commit happens **every `commit_stride` steps**; no per-slot fences.

4) **Pointer stability & alignment**
- Replay storage addresses remain stable across writes.
- Use `torch.empty(..., device='cuda')` to ensure alignment (256‑byte alignment matters for vectorized loads).

5) **Canonical time-major contract**
- Replay remains `[Tcap, N, ...]` time-major; do not introduce `[B, T]` storage internally.

---

## Dependencies / Preconditions

M6 assumes these are already in place from M0–M5:

- **DreamerV3 config + schemas** (including `commit_stride`, `replay_capacity`, `seq_len`, `num_envs`).
- **ReplayRing (CPU)** with correct schema (`obs`, `action`, `reward`, `is_first`, `continue`, `episode_id`).
- **Packed2 contract** (`[N, 1, 72, 20]` bytes), and encoder support for packed2 input.
- **World model / RSSM / behavior learning** not required for ingestion to function, but expected to exist as the learner target by M7.
- **Warp CUDA backend** already supports `render_pixels_snapshot_packed_to_torch(out)` for direct write.

If any precondition is missing, stop and backfill before implementing M6.

---

## Deliverables (What M6 Must Ship)

1) **`ReplayRingCUDA`**
- Same public API as `ReplayRing` but CUDA-resident.
- Allocates:
  - `obs: uint8[Tcap, N, 1, 72, 20]` (packed2)
  - `action: int32[Tcap, N]`
  - `reward: float32[Tcap, N]`
  - `is_first: bool[Tcap, N]`
  - `continue: float32[Tcap, N]`
  - `episode_id: int32[Tcap, N]`
- **`obs_slot(t)` returns a contiguous view** suitable for direct-write by Warp.

2) **Commit scheme (CUDA events + safe sampling window)**
- Actor writes continuously; **every `commit_stride` steps** it records a CUDA event and advances `committed_t`.
- Learner samples only from `[0, committed_t - safety_margin]` (wrap-aware), ensuring no read-before-write.
- Commit logic should be small, centralized, and testable (`ReplayCommitManager` or equivalent).

3) **Zero‑copy guards**
- Explicit runtime guard that checks for unexpected HtoD/DtoH transfers in the ingestion loop (using `torch.profiler`).
- Failfast assertion hook enabled in debug builds.

4) **Optional: alternate unpack kernel integration**
- Provide a stable `unpack_impl = "lut" | "triton" | "warp"` switch (default `lut`).
- Hook into encoder in a way that M6 can swap implementation without refactoring M4.

5) **CUDA tests / gates (skippable without GPU)**
- Pointer stability, event gating, and no-memcpy guard.
- Direct-write smoke test using the Warp backend.

---

## Design Notes (Key Decisions to Lock)

### A) ReplayRingCUDA API

- Should mirror `ReplayRing` to minimize changes elsewhere.
- Must expose:
  - `obs_slot(t) -> torch.Tensor` (contiguous, `uint8`, CUDA)
  - `push_step(t, action, reward, is_first, continue, episode_id)` for non-obs fields
  - `sample_sequences(B, T, gen, committed_t, safety_margin)` returning CUDA tensors

### B) Commit / Visibility Model

- Maintain **monotonic `write_t`** and **monotonic `committed_t`** (logical time, not modded index).
- Store a small ring buffer of `torch.cuda.Event` for commit points.
- Learner must `wait_event(commit_event)` before sampling past that commit.
- Safety margin (default: `seq_len`) avoids reading samples that overlap the just-written tail.

### C) Alignment + Contiguity

- Allocate with `torch.empty(..., device='cuda')`.
- Validate:
  - `obs.is_contiguous()`
  - `obs.data_ptr() % 256 == 0` (or at least 128; assert 256 when available)

### D) Integration with Warp Backend

- Use `WarpVecCudaBackend.render_pixels_snapshot_packed_to_torch(slot, base_offset=0)` to write packed2 bytes directly into `ReplayRingCUDA.obs_slot(t)`.
- No intermediate buffers; no `.copy_()` from a staging tensor.

---

## Implementation Plan (Step‑by‑Step)

### 0) Track the work (beads_rust)
- `br create --title="dreamer m6: cuda replay ingestion + zero-copy gates" --type=task --priority=2`
- `br update <id> --status=in_progress`

### 1) Add CUDA replay module

Create `src/gbxcule/rl/dreamer_v3/replay_cuda.py` (name flexible, API not):

- Class `ReplayRingCUDA`:
  - Allocates CUDA tensors for all fields (time-major).
  - Stores `capacity`, `num_envs`, `device`, `dtype` constants.
  - `obs_slot(t)` returns `[N, 1, 72, 20]` contiguous view.
  - `push_step(t, ...)` writes all non-obs fields.
  - Optional debug `check_invariants(t)` for `episode_id` and `continue`.

### 2) Commit manager

Add a small helper in `src/gbxcule/rl/dreamer_v3/replay_commit.py`:

- Maintains:
  - `write_t` (logical time)
  - `committed_t` (logical time)
  - `commit_stride`, `safety_margin`
  - `commit_events[slot]` (cuda events)
- Methods:
  - `record_commit(write_t, stream)`
  - `wait_for_commit(target_t, stream)`
  - `safe_max_t()` returns `committed_t - safety_margin`

### 3) Hook ingestion into actor loop

Add a CUDA ingestion function (likely in `engine_cuda.py` or a new `ingest_cuda.py`):

- On each env step:
  1. Obtain `slot = replay.obs_slot(write_t % capacity)`.
  2. `backend.render_pixels_snapshot_packed_to_torch(slot, 0)`.
  3. Fill action/reward/is_first/continue/episode_id for `write_t`.
  4. If `(write_t + 1) % commit_stride == 0`, record commit event and advance `committed_t`.

- Ensure **all of this happens on the actor stream**, and the learner waits on the commit event before sampling.

### 4) Sampling guardrail

- Modify `sample_sequences` to accept `max_t` (logical time) and ignore any indices past `max_t`.
- If `max_t - min_ready < seq_len`, return “not enough data yet” (learner should wait).

### 5) Zero-copy guard & profiler test

- Add a debug helper:
  - `assert_no_host_memcpy()` wraps a short ingestion loop in `torch.profiler` and fails if any HtoD/DtoH memcpy appears.
- Provide a tiny CLI (e.g., `tools/rl_gpu_bench_replay_ingest.py`) to run this check on demand.

### 6) Optional unpack kernel hook

- Extend the M4 unpack switch to allow `"warp"` or `"triton"` kernels.
- Keep the interface in the encoder stable (`unpack_impl` option only).
- Only implement alternative kernel if the M4 benchmark shows LUT unpacking is too slow.

---

## Test Plan (CUDA‑Optional, Fast by Default)

### Unit / Contract Tests (CPU, always-on)

These should be small and fast, using a CPU mock of the commit logic if needed:

- `test_commit_stride_logic()` — commit counter updates correctly.
- `test_safe_max_t()` — sampling never includes uncommitted data.
- `test_wraparound_indexing()` — ring indexing remains valid across wrap.

### CUDA Tests (skippable without GPU)

1) **Direct-write smoke**
- Allocate `ReplayRingCUDA`, call `obs_slot(0)`, write a pattern, ensure tensor reflects it (no reallocation).

2) **Pointer stability**
- Record `obs.data_ptr()` at init; after N writes, assert unchanged.

3) **Event gating**
- Write/commit on actor stream; on learner stream, wait for commit; read back and compare expected data.

4) **Memcpy profiler gate**
- Run ingestion loop under `torch.profiler` and assert no `MemcpyHtoD`/`MemcpyDtoH` events.

### Performance / Throughput (Optional DGX Gate)

- Measure env steps/s for packed2 ingestion.
- Ensure unpack throughput is below the M4 benchmark ceiling if enabled.

---

## Acceptance Criteria (M6 Done Means)

1) **Direct-write works**
- Warp writes packed2 frames into `ReplayRingCUDA.obs_slot()` with no staging buffers.

2) **Zero-copy verified**
- `torch.profiler` shows **no HtoD/DtoH memcpy** in the ingestion loop.

3) **Commit gating correct**
- Learner can only sample from `committed_t - safety_margin` and never reads uncommitted data.

4) **Pointer stability + alignment**
- `data_ptr()` stable across writes; alignment assertion passes.

5) **Skippable CUDA tests exist**
- CUDA smoke tests are present and pass on GPU; CPU tests remain fast.

---

## Risks & Mitigations

1) **Hidden host transfers**
- Mitigation: profiler gate + code review guardrails, explicit no‑CPU assertions in the hot path.

2) **Race conditions between actor/learner**
- Mitigation: commit events + safety margin, no per-slot fencing.

3) **Replay wraparound corruption**
- Mitigation: monotonic logical time, modulo indexing, tests for wrap correctness.

4) **Alignment‑sensitive kernels**
- Mitigation: `torch.empty` allocations + alignment asserts.

5) **Unpack bottleneck**
- Mitigation: keep LUT path + optional Triton/Warp kernel switch.

---

## Out of Scope (Explicitly Not M6)

- Full async Dreamer engine integration (M7).
- World model training or behavior learning changes (M4–M5 already cover).
- Multi-GPU sharding or distributed replay.
- Changing packed2 storage format.

---

## Session Closeout (Repo Protocol)

- `git status`
- `git add history/dreamer-m6.md`
- `br sync --flush-only`
- `git commit -m "dreamer: add M6 cuda ingestion plan"`
- `git push`
- `br close <id> --reason="Completed"`

