# Milestone 3 Architecture Plan (M3)

## M3 Definition of Done (locked)

### Correctness (must pass)

On DGX (CUDA available), both must pass:

- **ALU_LOOP**
  - `--verify --ref-backend pyboy_single --dut-backend warp_vec_cuda`
  - `frames_per_step=1`
  - `verify_steps=1024`
  - `compare_every=1`

- **MEM_RWB**
  - same as above **plus**
  - `--mem-region C000:C100` (mandatory)

_(Scope of comparison remains: registers + flags + optional counters; memory hashing only for specified regions.)_

### Scaling reports (must ship)

Produce **scaling sweep artifacts** for the example env counts:

- `env_counts = 1, 8, 64, 512, 2048, 8192`

…and a summarized report (CLI summary + one plot).

### Packaging (must)

DGX must install **Warp CUDA 13 wheel** (not PyPI’s CUDA 12 runtime wheel). Warp’s own PyPI page explicitly states PyPI binaries are CUDA 12 runtime and CUDA 13 wheels are on GitHub releases. ([PyPI][1])

---

## 1) Dependency + lock strategy: enforce CUDA 13 wheel on Linux only (uv-native)

### Goal

- macOS dev stays simple (PyPI wheel)
- DGX/Linux uses **cu13 wheel**
- **single `uv.lock`**, no manual post-installs

### Approach

Use **uv platform-specific sources** with markers. uv supports per-platform sources directly in `pyproject.toml` via `[tool.uv.sources]` and a `marker`. ([Astral Docs][2])

#### Concrete design

- Keep `dependencies = ["warp-lang>=…"]` as-is.
- Add:

```toml
[tool.uv.sources]
# On Linux (DGX), force cu13 wheel from GitHub release URL.
warp-lang = { url = "<CU13_WHEEL_URL>", marker = "sys_platform == 'linux'" }
```

This ensures:

- macOS resolves `warp-lang` normally (PyPI)
- Linux resolves `warp-lang` from the cu13 wheel URL
- `uv lock` and `uv sync` remain the single source of truth. ([Astral Docs][2])

**Notes**

- You’ll pin `<CU13_WHEEL_URL>` to an exact release asset (and therefore an exact Warp version). Warp’s PyPI page shows the intended naming convention and installation model for cu13 wheels. ([PyPI][1])

---

## 2) Backend architecture: explicit CUDA backend, no accidental `.numpy()` on GPU

### 2.1 Explicit backend names (locked)

Add **`WarpVecCudaBackend`** and register it as:

- `warp_vec_cpu` (existing)
- `warp_vec_cuda` (new)
- keep `warp_vec` as alias if you want, but **M3 gating uses explicit names only**.

### 2.2 Refactor structure (recommended)

In `src/gbxcule/backends/warp_vec.py`:

- `class WarpVecBaseBackend`: owns **ABI allocation**, ROM/bootrom load, shared step plumbing
- `class WarpVecCpuBackend(WarpVecBaseBackend)`: `device="cpu"`
- `class WarpVecCudaBackend(WarpVecBaseBackend)`: `device="cuda:0"`

This avoids copy/paste drift between CPU and CUDA paths.

### 2.3 Critical rule: memory reads must be _slice-copy_, not full-buffer `.numpy()`

Right now, `WarpVecCpuBackend.read_memory()` does `self._mem.numpy()` and slices on host. That’s fine on CPU, but on GPU it implies a synchronous device→host copy of the _entire_ array, which is catastrophic at scale.

#### M3 requirement

Implement device-aware memory reads:

- CPU: continue using `.numpy()` (cheap enough)
- CUDA: **copy only the requested region** into a small host buffer

Warp itself uses `wp.copy(..., src_offset=..., count=...)` patterns for reading small slices. ([GitLab][3])

**Implementation sketch**

- Compute `base = env_idx * MEM_SIZE`
- For `lo:hi`, copy `count = hi-lo` from `src_offset = base+lo`
- Copy into a CPU `wp.array(dtype=wp.uint8, device="cpu")`
- Then return `dest.numpy().tobytes()`

This makes `--mem-region C000:C100` feasible on CUDA verify without dragging `num_envs * 64KB` across PCIe every compare.

### 2.4 Write semantics

Keep `write_memory()` primarily as a testing/debug affordance. For CUDA:

- either:
  - host staging + device upload for the affected bytes, or
  - allow it only on CPU and raise on CUDA (acceptable if tests don’t require CUDA writes)

For M3, CUDA tests only need `read_memory()`.

---

## 3) Kernel layer: add CUDA warmup and make compilation/launch deterministic

### 3.1 Device-aware warmup

You already have `warmup_warp_cpu()`.

Add:

- `warmup_warp_cuda(device="cuda:0")`

It should:

- allocate minimal buffers on CUDA
- launch `cpu_step` once
- synchronize

This ensures benchmarks don’t accidentally measure “first-time compile” and avoids noisy perf curves.

### 3.2 Synchronization discipline

For CUDA timing correctness, you must define a sync policy; otherwise measured time can undercount queued GPU work.

Warp docs emphasize PyPI wheels’ CUDA runtime, and in practice Warp kernel launches on CUDA are async relative to Python. Your harness must enforce a sync boundary for benchmark windows. ([NVIDIA GitHub Pages][4])

---

## 4) Harness semantics: choose the “best” sync policy (your Q6)

You said “unsure — do what’s best.” Here’s the policy I recommend for M3:

### 4.1 Verify mode (correctness)

- **Always synchronize at compare points**, implicitly, by the act of:
  - calling `get_cpu_state()` (device→host reads), and/or
  - calling `read_memory()` for mem hashes

- This keeps verify strict and deterministic.

### 4.2 Benchmark mode (performance)

Introduce and implement a real `--sync-every K`:

- **Default on CUDA: `sync_every = 64`**
  - Rationale: stable measurements, bounded queueing, and still amortized overhead.

- **Always sync at the end of the measurement window**, regardless of `sync_every`.
- Allow:
  - `--sync-every 0` meaning “sync only at end” (max throughput, least fidelity).

Record `sync_every` into artifacts for auditability.

---

## 5) Mismatch bundle upgrades: include device/driver and wheel provenance (locked)

You want mismatch bundles to record:

- device/driver
- warp version
- and “which wheel”

### 5.1 System info capture

In `bench/harness.py::get_system_info()`:

- Add CUDA fields when available:
  - `gpu_name`, `driver_version`, `cuda_visible_devices`
  - prefer `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` when present (DGX will have it)

- Record:
  - `warp.__version__`
  - `importlib.metadata.version("warp-lang")`

### 5.2 Wheel provenance

When installed from a URL, Python packaging often drops a `direct_url.json` in the `.dist-info` directory (PEP 610). Practical approach:

- best-effort find and parse `.../warp_lang-*.dist-info/direct_url.json`
- record `"warp_direct_url": "<url>"` if present; else `None`

This gives you a factual “this was the cu13 wheel URL” breadcrumb.

---

## 6) Tighten memory dump policy (your Q9)

I recommend tightening:

- `MAX_MEM_DUMP_BYTES = 1024` (down from 4096)

Why:

- CUDA correctness iteration can produce frequent mismatches; smaller dumps reduce bundle size + IO churn.
- For `C000:C100`, you’re only dumping 256 bytes anyway, so you lose nothing.

---

## 7) Tests + Makefile: keep CPU gate ≤2 minutes, add DGX-only GPU gates

### 7.1 Tests

Add:

- `tests/test_warp_vec_cuda_backend.py`
  - skipped if CUDA not available
  - `reset`, a few `step`s, `get_cpu_state`, `read_memory(C000:C100)` sanity

- `tests/test_verify_cuda_micro_roms.py`
  - a short verify (e.g. 64 steps) against `pyboy_single` for both ROMs on CUDA
  - keep this _short_ to avoid turning GPU tests into long-running suites

### 7.2 Makefile targets

Add targets (DGX runs these; other machines can skip):

- `check-gpu`: ruff/pytest + GPU tests + very small CUDA verify smoke
- `verify-gpu`: the full M3 must-pass verify profile (1024 steps for both ROMs)
- `bench-gpu`: scaling sweep runs + artifacts

And keep:

- `check` unchanged and CPU-fast.

---

## 8) Scaling reports deliverables (M3)

### 8.1 Data

Run scaling sweeps for:

- `pyboy_vec_mp` baseline
- `warp_vec_cuda` DUT

Same ROM, same `frames_per_step`, same `sync_every`, same warmup/steps.

### 8.2 Analysis scripts

Implement:

- `bench/analysis/summarize.py`
  - load scaling JSON(s)
  - print table-like summary + speedup vs baseline at each env_count

- `bench/analysis/plot_scaling.py`
  - matplotlib plot of `env_count → total_sps`
  - optionally a second plot: `env_count → speedup`

(Keep these out of the precommit gate.)

---

# PR slicing (how I would ship M3 without drama)

1. **PR1 — uv cu13 pinning (Linux-only)**
   - `pyproject.toml` `[tool.uv.sources]` + marker
   - update `uv.lock`
   - doc snippet for DGX setup
   - smoke: `uv sync` on macOS + DGX

2. **PR2 — `warp_vec_cuda` backend skeleton**
   - base class refactor
   - CUDA warmup + device selection
   - benchmark path works (even if slow)

3. **PR3 — CUDA-safe `read_memory()`**
   - `wp.copy` slice-copy path for CUDA
   - tests for `--mem-region` path on CUDA

4. **PR4 — Harness sync policy**
   - implement `--sync-every`
   - enforce end-of-window sync
   - record into artifacts

5. **PR5 — GPU verify + scaling reports**
   - `make verify-gpu`, `make bench-gpu`, analysis scripts
   - produce sample scaling artifacts on DGX

---

## Web-grounded constraints we’re explicitly honoring

- Warp PyPI wheels are CUDA 12 runtime; CUDA 13 wheels are distributed via GitHub releases. ([PyPI][1])
- uv supports **URL sources** and **platform-specific sources via markers**, so we can pin cu13 only on Linux while keeping macOS dev sane. ([Astral Docs][2])
- Warp usage patterns show `wp.copy(... src_offset=..., count=...)` style small transfers, which is exactly what we need for `--mem-region` on CUDA. ([GitLab][3])

---

If you want one final “lock it in” decision from me: **I’d set the default CUDA benchmark `sync_every=64`** (configurable), and I’d keep verify-mode comparisons step-exact with sync inherently forced by the state/memory reads. That gives you _credible_ scaling curves without sacrificing throughput exploration.

Want me to also propose the exact new Makefile targets + the canonical DGX commands for producing the M3 scaling report bundle (including filenames + where to put plots)?

[1]: https://pypi.org/project/warp-lang/?utm_source=chatgpt.com "warp-lang"
[2]: https://docs.astral.sh/uv/concepts/projects/dependencies/ "Managing dependencies | uv"
[3]: https://git.ias.informatik.tu-darmstadt.de/cai/warp/-/blob/b0880cbf777c96d03e49145a9cee5a6e907cc408/warp/fem/utils.py?utm_source=chatgpt.com "warp/fem/utils.py"
[4]: https://nvidia.github.io/warp/?utm_source=chatgpt.com "NVIDIA Warp Documentation — Warp 1.11.0"
