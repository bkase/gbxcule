## Workstream 0 — M3 spec + gates (the “contract” stream)

**Purpose:** lock exactly what M3 means and make it executable in `make` targets.

**Deliverables**

- Make targets:
  - `make verify-gpu` (M3 must-pass: ALU_LOOP + MEM_RWB with `C000:C100`, `frames_per_step=1`, `verify_steps=1024`, `compare_every=1`)
  - `make bench-gpu` (scaling sweeps)
  - `make check-gpu` (fast-ish DGX gate)

- Harness/README updated so the commands are canonical.

**Dependencies:** none (but will be updated as other streams land).

**DoD**

- A DGX user can run `make verify-gpu` and get a pass _once the CUDA backend is correct_.
- A DGX user can run `make bench-gpu` and get scaling artifacts.

---

## Workstream 1 — DGX + packaging: force CUDA 13 wheel via uv

**Purpose:** make the environment deterministic and non-negotiable: Linux uses **cu13 wheel**, macOS stays PyPI.

**Deliverables**

- `pyproject.toml` `[tool.uv.sources]` entry for `warp-lang` URL with `marker="sys_platform == 'linux'"` (cu13 wheel URL pinned)
- Updated `uv.lock`
- Optional `docs/DGX_SETUP.md` or README section

**Dependencies:** none.

**DoD**

- `uv sync` on DGX installs the cu13 wheel (provable via `python -c "import warp; print(warp.__version__)"` + provenance capture).
- `uv sync` on macOS still works without special steps.

---

## Workstream 2 — Backend architecture: `warp_vec_cuda` (explicit GPU backend)

**Purpose:** introduce an explicit CUDA backend without duplicating all CPU code.

**Deliverables**

- `WarpVecBaseBackend` refactor (shared logic)
- `WarpVecCudaBackend` (new) with explicit `name="warp_vec_cuda"` and device selection `cuda:0`
- Kernel warmup for CUDA (compile once) and consistent initialization lifecycle

**Dependencies**

- Workstream 1 strongly recommended first (so the backend is built/tested on the right wheel).

**DoD**

- `uv run python bench/harness.py --backend warp_vec_cuda --rom ... --steps 10` runs on DGX.
- No unconditional `wp.synchronize()` in `step()` (benchmark path must be async-friendly).

---

## Workstream 3 — CUDA-safe memory reads for verify (`--mem-region`)

**Purpose:** make verify mode viable on GPU without catastrophic host copies.

**Core issue**

- On CUDA, calling `.numpy()` on a big Warp array implies a device→host copy. For M3’s memory hashing region, you must copy **just the region**.

**Deliverables**

- `WarpVecCudaBackend.read_memory(env_idx, lo, hi)` implemented as **slice-copy** (copy only `[lo:hi)` for that env) using Warp copy primitives
- Keep CPU path simple (can still use `.numpy()` on CPU)

**Dependencies**

- Workstream 2 (needs a CUDA backend to implement).

**DoD**

- `make verify-gpu` can execute `--mem-region C000:C100` without O(num_envs \* 64KB) copies.
- Unit test on DGX that reads a small range and matches expected bytes.

---

## Workstream 4 — Harness sync policy for GPU benchmarking (`--sync-every`)

**Purpose:** make benchmark numbers _true_ under async CUDA launches.

**Deliverables**

- Add `--sync-every K` behavior:
  - default for CUDA benchmarks = **64**
  - always sync at end of measurement window

- Record `sync_every` in artifacts
- Ensure verify mode remains correct (it syncs implicitly when reading state/memory)

**Dependencies**

- Workstream 2 (needs CUDA backend in registry), but can be started earlier in harness with stubs.

**DoD**

- Running `bench` with CUDA produces stable, believable timing (no “too fast because we never synced”).
- Artifacts include sync policy fields.

---

## Workstream 5 — Correctness: CUDA verify passes for micro-ROMs

**Purpose:** the actual “emulator correctness on GPU” work.

This is where you close the loop between:

- `kernels/cpu_step.py` (opcode semantics, cycle/frame stepping)
- ABI buffers (regs/mem)
- Verify sampling boundary (after each `step`)

**Deliverables**

- Fix any CPU-vs-CUDA differences (type widths, masking, memory writes, etc.)
- Ensure both must-pass profiles succeed:
  - ALU_LOOP step-exact
  - MEM_RWB step-exact + mem-region hash

**Dependencies**

- Workstream 2 (CUDA backend)
- Workstream 3 (mem reads)
- Workstream 4 (not strictly required for correctness, but helps for reproducible runs)

**DoD**

- `make verify-gpu` passes on DGX repeatedly.
- Mismatch bundles still work (for when you intentionally break it).

---

## Workstream 6 — Observability + provenance in mismatch bundles

**Purpose:** when CUDA goes wrong, bundles must tell you _exactly what environment produced the mismatch_.

**Deliverables**

- Extend `get_system_info()` + mismatch bundle metadata to include:
  - GPU name
  - driver version
  - warp version
  - warp “direct URL” provenance if present (PEP 610 direct_url.json), or at least “wheel source unknown”

- Tighten `MAX_MEM_DUMP_BYTES` (I’d set to **1024**)

**Dependencies:** none (can land anytime), but most useful once verify is running on CUDA.

**DoD**

- A mismatch bundle contains enough info to answer: “Which GPU/driver/wheel produced this?” without guesswork.

---

## Workstream 7 — Scaling reports + analysis scripts

**Purpose:** produce the deliverable M3 explicitly promises: scaling reports.

**Deliverables**

- `bench/analysis/summarize.py`:
  - loads scaling artifacts
  - prints speedup table vs CPU baseline (`pyboy_vec_mp`)

- `bench/analysis/plot_scaling.py`:
  - plots total SPS vs env count (and optionally speedup curve)

- `make bench-gpu` runs:
  - scaling sweep for `warp_vec_cuda`
  - scaling sweep for `pyboy_vec_mp`
  - writes artifacts + optionally plots into a `bench/runs/reports/<timestamp>/`

**Dependencies**

- Workstream 4 (sync semantics) recommended
- Workstream 5 (correctness) not required for plotting, but practically you’ll want correctness first.

**DoD**

- One command produces:
  - scaling JSON artifacts
  - a human-readable summary
  - a plot image

- You can attach the plot + summary in a report / devlog.

---

# Suggested parallelization map (minimal blocking)

**Start immediately (parallel):**

- WS0 (gates/Makefile wiring)
- WS1 (cu13 wheel via uv)
- WS6 (bundle provenance / metadata)

**Then:**

- WS2 (CUDA backend skeleton) depends on WS1 ideally
- WS4 (sync policy) can proceed once WS2 lands or in parallel in harness

**Then:**

- WS3 (CUDA memory slice-copy) depends on WS2
- WS5 (CUDA correctness passes) depends on WS2+WS3

**Finally:**

- WS7 (scaling reports) depends on WS4 and ideally WS5
