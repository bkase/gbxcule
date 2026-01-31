Yes — **R0+R1 should be explicitly built around (and validated by) that async PPO / multistream double-buffer design**, because that’s the “real” systems shape you’re targeting: **actor+env on one CUDA stream, learner on another, zero host staging, minimal sync, and predictable overlap**.

Below is the **updated plan in full** (R0 + R1) with the async PPO benchmark/script you pasted treated as the canonical reference workload and the place we prove performance.

---

# Updated Milestone R0 — Torch-required Experiment Harness + Async PPO Engine (CPU/GPU)

## R0 objective

Make a **single, reproducible RL experiment harness** that can run:

- **CPU**: correctness + determinism + fast dev loop
- **GPU**: the real target: **async PPO, multistream**, no CPU staging, preallocated buffers

…and produce consistent artifacts (meta/config/metrics/failure bundles) like the rest of the repo.

### Key constraints (for R0)

1. **Torch is required** (repo-wide).
2. **Same codepaths run on CPU and CUDA** (CUDA tests can skip, but the API doesn’t fork wildly).
3. **Async PPO is first-class**: we don’t bolt it on later; we make it the main execution shape.

---

## R0.0 Make torch required + split “CPU vs GPU expectation” cleanly

**Repo changes**

- Move torch into mandatory deps.
- Keep CUDA availability conditional (your pattern in tools already does this).

**Acceptance**

- `make test` works on CPU-only machines with torch (CPU build).
- `make test-gpu` runs on DGX/GB10 and fails if CUDA is unavailable.

---

## R0.1 Add a unified Experiment harness (artifacts + logging + failure bundles)

### New module: `src/gbxcule/rl/experiment.py`

This becomes the one thing all RL tools import.

**Responsibilities**

- Create `run_dir` (timestamp + algo + rom_id + tag)
- Write:
  - `meta.json` (repro + system info)
  - `config.json` (full config, no hidden defaults)
  - `metrics.jsonl` (streamed records)

- On failure (exception / NaN):
  - write a failure bundle under `failures/`
  - include minimal tensor snapshots when available

**Run directory layout**

```
bench/runs/rl/
  <ts>__<algo>__<rom_id>__<tag>/
    meta.json
    config.json
    metrics.jsonl
    checkpoints/
      ckpt_latest.pt
    failures/
      <ts>__<trace_id>__<kind>/
        failure.json
        tensors.pt        (optional)
        repro.sh          (optional)
```

**Atomicity**

- JSON: write `.tmp` then rename
- Checkpoints: `torch.save` to `.tmp` then rename
- Failure bundles: write to temp dir then rename

---

## R0.2 Define stable schemas (meta/config/metrics/failure)

### New module: `src/gbxcule/rl/schemas.py`

- `RL_RUN_SCHEMA_VERSION = 1`
- `RL_METRICS_SCHEMA_VERSION = 1`
- `RL_FAILURE_SCHEMA_VERSION = 1`

**`meta.json` minimum**

- `run_id`, `timestamp_utc`, `schema_version`
- rom: `rom_path`, `rom_sha256`
- state (optional): `state_path`, `state_sha256`
- env: `num_envs`, `frames_per_step`, `release_after_frames`, `stack_k`
- pipeline: `obs_format` (`u8`/`packed2`), `action_codec_id`
- algo: `algo_name`, `algo_version`
- code: `git_commit`, `git_dirty`
- system: `platform`, `python`, `torch_version`, `warp_version`, `cuda_available`, `gpu_name`

**`metrics.jsonl` required per record**

- `run_id`, `trace_id`, `schema_version`
- `wall_time_s`, `env_steps`, `sps`
- plus: actor/learner timings when relevant (see R0.5)

---

## R0.3 Centralize fail-fast checks (NaNs, shape drift, device drift)

### New module: `src/gbxcule/rl/failfast.py`

Functions like:

- `assert_finite(t, name, exp, trace_id)`
- `assert_device(t, device, ...)`
- `assert_shape(t, expected, ...)`

**Critical for async PPO**: if anything silently falls to CPU (e.g., accidental `.cpu()`), we detect it.

On failure:

- write failure bundle with:
  - traceback
  - small tensors: `obs[0:8]`, `actions[0:8]`, `logits[0:8]`, loss scalars, etc.

---

## R0.4 Turn async PPO into a reusable engine (not a one-off tool)

Your `tools/rl_gpu_bench_async.py` is a great starting point, but it needs to become a library component so we can reuse it for:

- benchmarking sweeps
- regression tests
- later: “real training” (not just bench)

### New module: `src/gbxcule/rl/async_ppo_engine.py`

A single class that encapsulates:

- actor backend + reset cache
- actor model + learner model
- rollout buffers (double-buffered)
- streams + events + synchronization rules
- a `run(update_count)` method that returns metrics

**Important fixes we must incorporate (these matter for correctness + performance):**

#### (A) Fix stream correctness for policy weight sync

In your script, `_sync_actor_weights()` runs on the learner stream, but the actor stream never explicitly waits for that copy before using `actor_model`.

**Plan**: add a `policy_ready_event`:

- After copying weights on learner stream: `policy_ready_event.record(learner_stream)`
- Before starting next rollout on actor stream: `actor_stream.wait_event(policy_ready_event)`

This is the minimum to avoid races / undefined behavior.

#### (B) Fix “double render” semantics

Right now you call `backend.render_pixels_snapshot_torch()` after `backend.step_torch(actions)`.
But your backend currently renders pixels inside `_launch_step()` when `render_pixels=True`.

That likely means you are doing **two render passes per step**.

**Plan**: decide and enforce exactly one of these two models:

1. **Render-on-step** (default for RL):
   - `step_torch()` includes render
   - env wrappers / async loop must NOT call `render_pixels_snapshot_*` after step

2. **Manual render** (better for explicit pipelines):
   - `step_torch()` does _not_ render
   - caller explicitly calls render into desired buffer

Because you want “render directly into rollout buffer” in R1, **manual render is the more powerful choice** (and avoids hidden work). If we choose manual render, we add a flag to backend (`render_on_step`) or we simply set `render_pixels=False` and call explicit render kernels.

Either way, in R0 we must make behavior **unambiguous and benchmarked**.

---

## R0.5 Instrumentation for overlap (actor vs learner timing)

Async PPO only matters if we can prove we’re actually overlapping.

### Add timing utilities in engine

Use CUDA events per update to measure:

- `t_actor_rollout_ms`
- `t_learner_update_ms`
- `t_total_update_ms`
- derived: `overlap_efficiency = (t_actor + t_learner) / t_total`

Log these into `metrics.jsonl`.

**Acceptance**

- On GPU, overlap_efficiency should be > 1.0 (otherwise you’re not overlapping).
- In the benchmark sweep, we should see overlap improve at larger env sizes.

---

## R0.6 CPU compatibility mode

We don’t need CPU async streams. We need CPU to validate:

- correctness of rollout plumbing
- determinism / reproducibility
- schema logging

### CPU mode behavior

- Use `WarpVecCpuBackend`
- Use a single “stream” (no streams, no events)
- Execute “async PPO engine” in sequential order:
  - rollout then learn

- Still emits identical artifacts

**Acceptance**

- Engine runs on CPU for small env counts and produces artifacts.

---

## R0.7 Refactor the existing tool to use the harness + engine

### Update `tools/rl_gpu_bench_async.py`

- Replace ad-hoc printing/output writing with `Experiment`
- Use `AsyncPPOEngine` to run
- Emit one JSON record per run with the standard schema

---

## R0 tests (must-have)

1. **Policy sync correctness test (GPU-skip if no CUDA)**
   - Run 2 updates with forced weight change in learner
   - Assert actor sees updated weights (e.g., compare a checksum / small param slice on actor stream after wait)

2. **Rollout buffer lifecycle test**
   - Ensure buffer free/ready event protocol is obeyed (no use-before-ready)

3. **Double-render detection**
   - Add a debug counter or timing test that flags if render kernel launches twice per step in the default RL path

---

## R0 acceptance checklist

- ✅ Torch is required; CPU and GPU runs both work.
- ✅ Async PPO engine exists and is used by the benchmark tool.
- ✅ Proper CUDA event sync for policy weights (no races).
- ✅ Artifacts are stable: meta/config/metrics/failure bundles.
- ✅ Timing shows meaningful overlap on GPU.

---

---

# Updated Milestone R1 — Packed2 Pipeline + Zero-copy Rollout Writes + Benchmark Sweep (async-first)

## R1 objective

Make the async PPO system **maximally GPU efficient** by:

- switching to **packed2 pixels** end-to-end,
- eliminating unnecessary copies,
- rendering **directly into rollout storage** when possible,
- and ending with a **beautiful scaling sweep** (JSONL + summary.md + plots) that demonstrates improvements vs your current numbers.

This is where we “lock in” the high-performance stack.

---

## R1.0 Decide the canonical rendering contract for RL

To enable zero-copy writes into rollout storage, we want:

- **No implicit render hidden in `step_torch`**
- Explicit render calls that can target:
  - a small “current obs” buffer, and/or
  - the exact rollout buffer slot

### R1 decision (recommended)

**Manual render for RL**:

- Create backend with `render_pixels=False` and `render_pixels_packed=False`
- Use explicit render calls:
  - packed: `render_pixels_snapshot_packed_to_torch(out, base_offset_bytes)`
  - (optional) add an unpacked equivalent later

This makes the pipeline predictable and avoids accidental duplicate render work.

---

## R1.1 Packed2 becomes first-class (policy + rollout + goal)

### Update core RL code to support packed2 everywhere

- Observations stored as `uint8` packed2: `shape [N, 1, 72, 20]` (stack_k=1 for the async path initially)
- Goal template stored as packed2 too
- Distance computation uses packed-friendly ops (see R1.4)

You already have:

- packed render kernel
- unpack LUT utilities
- diff LUT utility

---

## R1.2 Zero-copy(ish) rollout filling: render directly into rollout buffer slots

This is the main performance win.

Right now the hot path often does:

1. render into backend pixel buffer
2. read pixels tensor
3. `rollout.add()` copies obs into rollout storage

Instead, we want:

- render kernel writes **directly** into `rollout.obs[...]` storage at the correct offset

### Update `RolloutBuffer` to expose “slots”

Add APIs that let the engine avoid copying obs:

- `obs_slot(step_idx) -> torch.Tensor` view to the underlying storage for that step
- `set_step_fields(step_idx, actions, rewards, dones, values, logprobs)` (small copies are fine)
- Or: add `add_no_copy_obs(step_idx, actions, ...)` where obs is assumed already written

### How to render into the slot (packed2)

For stack_k=1, each step needs:

- `frame_bytes = num_envs * 72 * 20`

The slot `rollout.obs_u8[step_idx]` is contiguous; flatten it and pass it to:

- `backend.render_pixels_snapshot_packed_to_torch(out_flat, base_offset_bytes)`

Where `base_offset_bytes` points to the correct step slot in the flattened rollout buffer.

This removes the heavy obs copy from the loop — the render kernel writes the bytes exactly where they need to be stored.

---

## R1.3 Async PPO engine updated to use packed2 + direct writes

Modify the engine loop:

**Warm start**

- Render initial obs into `rollout.obs_slot(0)` (or a dedicated `obs_current` tensor).
- Set `obs = rollout.obs_slot(0)` as the current observation.

**For each step t**

- Compute action from `obs`
- Step backend
- Render next obs into `rollout.obs_slot(t+1)` (or `obs_next`)
- Compute reward/done/trunc
- Store small fields (actions/rewards/values/logprobs)
- Apply `ResetCache.apply_mask_torch` for done/trunc
- If reset happened: re-render obs for those envs into the _same_ next slot (so next obs is correct)

This keeps everything on GPU, avoids big copies, and preserves correctness.

---

## R1.4 Make reward shaping packed-friendly (optional but high ROI)

Your current distance metric:

```python
diff = torch.abs(next_obs.float() - goal_f)
curr_dist = diff.mean(...) / 3.0
```

For packed2, you can use the existing `get_diff_lut()`:

- compute per-byte L1 distance between packed bytes (sum of per-pixel abs diffs)
- reduces compute and bandwidth

This matters at 8k–16k envs.

**Deliverable**

- `gbxcule/rl/packed_metrics.py` with `packed_l1_distance(packed_obs, packed_goal)`

---

## R1.5 Eliminate any remaining host staging / implicit copies

Add “guardrails”:

- Assert all core tensors in async loop are CUDA:
  - obs buffers
  - rollout buffers
  - actions
  - model params

- Ensure backend step uses `warp.from_torch` / stream interop (already does)

Optional: add a profiler tool:

- `tools/rl_profile_one_update.py` that runs one update under `torch.profiler` and confirms there are no `memcpyDtoH`/`memcpyHtoD` events in the hot path.

---

## R1.6 “Nice benchmark sweep” — async-first, with reports + plots

This is the endcap and should be treated like a product deliverable.

### New tool: `tools/rl_bench_sweep_async_ppo.py`

Runs a sweep and emits:

- `results.jsonl` (one record per config)
- `summary.md` (human-readable)
- `plots/*.png` (SPS curves and comparisons)

**Default sweep parameters (match your comparability set)**

- `--steps-per-rollout 64`
- `--updates 2`
- `--ppo-epochs 1`
- `--minibatch-size 16384`

**Sweep axes**

- `num_envs ∈ {1024, 2048, 4096, 8192, 16384}`
- `obs_format ∈ {u8_baseline, packed2_directwrite}`
- (optional) `mode ∈ {sync, async}` for proving async benefit

**What each record contains**

- env config: envs, frames_per_step, release_after_frames
- pipeline config: packed/unpacked, directwrite yes/no, render mode
- PPO config: rollout, updates, epochs, minibatch
- results:
  - `elapsed_s`, `env_steps`, `sps`
  - actor/learner timings + overlap efficiency
  - GPU mem allocated / reserved
  - policy sync strategy (event-based)

**summary.md should include**

- a table like:

| envs | SPS u8 async | SPS packed2 async | speedup | overlap_eff |
| ---- | ------------ | ----------------- | ------- | ----------- |

- a short narrative: where throughput saturates, what bottleneck likely is (render, model, optimizer, reset)

**Plots**

- SPS vs envs for each pipeline
- packed speedup vs envs
- overlap efficiency vs envs

---

## R1 acceptance checklist

- ✅ Async engine runs packed2 end-to-end.
- ✅ Render is unambiguous (no accidental double render).
- ✅ Obs are written directly into rollout slots (no large obs copy in the hot loop).
- ✅ No host staging in the hot path (validated by assertions and/or profiler spot check).
- ✅ Benchmark sweep produces:
  - structured artifacts,
  - summary.md,
  - plots,
  - and shows improvement vs the baseline numbers you posted.

---

# How this connects to the script you pasted (explicit mapping)

Your `tools/rl_gpu_bench_async.py` becomes:

1. **The reference implementation** for the async engine semantics
2. The “before” baseline for SPS
3. The thing we systematically improve by:
   - fixing policy sync correctness (event)
   - fixing render contract (no duplicate renders)
   - switching to packed2
   - rendering directly into rollout slots
   - optimizing reward distance computation (optional)

Your posted sweep results (SPS scaling from 1024→16384) become the baseline curve we want to **beat** once we remove redundant work and reduce bandwidth.

---

If you want one concrete “first surgical move” before any refactors: **audit and fix render duplication** in the async benchmark (because it’s the most likely reason the curve flattens early). After that, packed2 + direct-write should move the curve materially, and the sweep/report will make that obvious.
