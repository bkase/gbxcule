# M2 Engineering Plan (Principal Engineer / Architect view)

## M2 objective (PRD-aligned)

**Milestone M2 = Warp CPU-debug correctness on micro-suite.**
Deliver a real `warp_vec_cpu` backend (no longer a stub) that passes verification against `pyboy_single` for `ALU_LOOP` and `MEM_RWB`, using the harness’ deterministic verify loop and mismatch bundles.

This must work in:

- macOS CPU-only dev
- cloud CPU-only dev
- DGX Spark CPU mode (GPU mode is not part of M2)

## M2 definition of done (tight, testable)

### Correctness (core)

- `bench/harness.py --verify` passes for:
  - `ALU_LOOP.gb`: `ref=pyboy_single`, `dut=warp_vec_cpu`, `compare_every=1`, `verify_steps >= N`
  - `MEM_RWB.gb`: same, **plus** memory signal if we enable `--mem-region` (see below)

**Recommended N (DoD):**

- `frames_per_step=1`, `verify_steps=1024`, `compare_every=1`
- Plus a smoke: `frames_per_step=24`, `verify_steps=16`, `compare_every=1`

### Deterministic repro (bundle quality)

- Any mismatch produces a bundle that is **self-contained**:
  - includes `rom.gb` embedded
  - `repro.sh` uses the embedded ROM path
  - action replay uses `actions.jsonl`
  - bundle is written atomically (already true)

### Tooling latency

- CPU gate (`make check`) remains ≤2 minutes.
  (Warp compilation is the main risk; mitigate below.)

---

## Architectural shape (what changes, where)

### A) Make mismatch bundles hermetic by embedding ROM (you requested this)

**Changes:**

- `bench/harness.py::write_mismatch_bundle`:
  - copy ROM bytes into `rom.gb` inside the temp bundle dir
  - `metadata.json` includes `rom_filename="rom.gb"` (and optionally `rom_size`)
  - `repro.sh` uses `--rom "$(dirname "$0")/rom.gb"` rather than original path

**Why now:** this pays off immediately once Warp is real and mismatches start happening (bundles become portable artifacts).

### B) Introduce `warp_vec_cpu` backend (real Warp, CPU device)

**Changes:**

- `bench/harness.py` backend registry:
  - register `"warp_vec_cpu"` (and keep `"warp_vec"` aliasing CPU for now)

- `src/gbxcule/backends/warp_vec.py`:
  - convert from stub → real implementation using Warp
  - ensure Warp import is lazy and module compilation is controlled

**Important dependency note:** Warp supports CPU on macOS/Linux/Windows, but its PyPI wheels are built with CUDA 12 runtime for GPU; NVIDIA provides CUDA 13 runtime wheels on GitHub releases. ([GitHub][2])
(M2 is CPU-only; this becomes critical in M3 when we turn on CUDA.)

### C) ABI v0 becomes real (not comments)

**Changes:**

- `src/gbxcule/core/abi.py` should stop being a comment stub and become the authoritative layout module:
  - constants: `ABI_VERSION`, `MEM_SIZE = 65536`
  - typed layout decisions: how registers and memory are represented in Warp arrays
  - helper functions for index calculations (env, addr → offset)
  - clear versioning notes

### D) Warp CPU stepping kernel (micro-ROM instruction subset)

**Changes:**

- `src/gbxcule/kernels/cpu_step.py` becomes the center of gravity:
  - Warp kernels and Warp user functions for:
    - fetch/decode/execute of the minimal opcode set
    - flag updates (INC, ADD)
    - memory reads/writes (LD (HL),A / LD B,(HL))
    - cycle accounting per instruction

**Key design constraint:** we need _repeatable_ stepping that matches PyBoy at the sampling points we compare. PyBoy advances in **frames** (`tick(n, render)` advances n frames). ([GitHub][1])
So Warp must implement a comparable “advance for K frames” notion.

---

## Workstreams and sequencing

### Workstream 0 — Mismatch bundle upgrade: embed ROM (fast, low risk)

**Deliverables**

- `rom.gb` inside mismatch bundle
- `repro.sh` uses embedded ROM
- Add/extend tests in `tests/test_harness.py::TestMismatchBundle`:
  - assert `rom.gb` exists
  - assert `repro.sh` references `rom.gb`

**Acceptance**

- Existing mismatch bundle tests pass + new assertions.

---

### Workstream 1 — Warp bring-up on CPU with a trivial kernel (E0 CPU plumbing)

Before emulation correctness, ensure the Warp pipeline is real and stable.

**Deliverables**

- `warp_vec_cpu` backend can:
  - `reset()`
  - `step()` increments a per-env counter via a Warp kernel
  - `get_cpu_state()` returns predictable state derived from that counter

- Add a benchmark run entry in README later (optional in M2, but good sanity)

**Acceptance**

- `bench/harness.py --backend warp_vec_cpu --rom ALU_LOOP.gb --steps 10` runs and emits artifacts
- Determinism: same seed/config → same state progression

This is the “we can actually execute kernels” checkpoint.

---

### Workstream 2 — ROM loading + memory model (ABI v0 real)

**Deliverables**

- Per-env 64KB memory buffer exists (flat)
- ROM is loaded into `mem[0:rom_len]` at reset
- Writes to ROM region can be ignored or allowed (micro-ROMs won’t write there)
- `get_cpu_state()` reads regs from ABI buffers (PC/SP/A/F/B/C/D/E/H/L)

**Acceptance**

- Unit test: after reset, memory prefix equals ROM bytes (for at least one env)
- `MEM_RWB` stepping actually mutates WRAM region in the Warp backend (even if we don’t validate yet)

---

### Workstream 3 — CPU interpreter subset + cycle accounting

This is the core M2 work.

**Deliverables**

- Implement the minimal opcode set (listed above), including correct flags:
  - F lower nibble masked out (matches typical GB behavior; PyBoy’s register file masks it too) ([PyBoy Documentation][3])

- Implement a “frame stepping loop”:
  - advance CPU in a cycle budget loop, using instruction cycle table
  - decide and document the policy for overshoot vs exact cutoff (the emulation policy must be stable; we’ll align it to whatever matches PyBoy for these ROMs)

**Acceptance**

- A new integration test suite:
  - `pyboy_single` vs `warp_vec_cpu` passes on `ALU_LOOP` for the chosen verify config
  - then passes on `MEM_RWB`

**Debug ergonomics requirement**

- When it fails, the mismatch bundle must be enough to reproduce and reason:
  - embedded ROM
  - action trace
  - states/diff
  - include Warp counters (`instr_count`, `cycle_count`) even if ref lacks them (harness already ignores counters if ref is None)

---

### Workstream 4 — Optional memory hashing in verification (recommended)

If you accept the default I suggested, we add a **memory signal** for MEM_RWB.

**Design**

- Add `--mem-region` flag to `bench/harness.py` verify mode (e.g., `"C000:C100"`)
- Backends provide a uniform way to read a slice for env 0:
  - simplest: add `read_memory(env_idx, lo, hi) -> bytes` to `VecBackend` contract
  - `pyboy_single` uses `pyboy.memory[lo:hi]` ([PyBoy Documentation][3])
  - `warp_vec_cpu` copies that window from its mem buffer
  - `pyboy_vec_mp` can implement via worker RPC (only needed if you ever want it)

**Acceptance**

- `MEM_RWB` verify passes with `--mem-region C000:C100` and hash compare enabled
- Default verify remains register-only unless flag is set (keeps fast gate)

---

### Workstream 5 — Gates, targets, and doc alignment

Once M2 passes, `make verify` should stop being “expected to fail”.

**Deliverables**

- Makefile targets:
  - `make verify` becomes “ref vs warp_vec_cpu should PASS”
  - optionally `make verify-mismatch` runs `write_mismatch_bundle` unit path (or uses a tiny fake backend) so we always exercise failure path in CI-as-hook

- README updated:
  - verification is now expected to pass
  - mismatch bundles are hermetic (embedded ROM)

**Acceptance**

- `make check` still ≤2 minutes on CPU-only dev machines

---

## Risk register (the real ones)

1. **Warp JIT compile cost can blow the 2-minute gate**
   - Mitigation: preload Warp module in the backend constructor or first reset (and ensure tests don’t compile repeatedly).
   - Warp provides explicit module loading controls for reducing JIT surprises. ([NVIDIA GitHub][4])

2. **PyBoy sampling point ambiguity**
   - PyBoy docs warn `register_file` is best used in hooks because `tick()` doesn’t return at a “specific point.” ([PyBoy Documentation][3])
   - Mitigation: keep compare quantum small (`frames_per_step=1`) and rely on “match at the same operational boundaries” (after each tick call) rather than assuming sub-frame determinism. If this proves flaky, we add a deterministic sampling hook for micro-ROMs (address-based hook) as a fallback.

3. **Cycle cutoff / overshoot policy mismatch**
   - Mitigation: treat this as a first-class spec decision in `kernels/cpu_step.py` and add a regression test that locks the policy once it matches PyBoy for both ROMs.

[1]: https://github.com/Baekalfen/PyBoy/wiki/Migrating-from-v1.x.x-to-v2.0.0?utm_source=chatgpt.com "Migrating from v1.x.x to v2.0.0 · Baekalfen/PyBoy Wiki · GitHub"
[2]: https://github.com/NVIDIA/warp?utm_source=chatgpt.com "GitHub - NVIDIA/warp: A Python framework for accelerated simulation, data generation and spatial computing."
[3]: https://docs.pyboy.dk/?utm_source=chatgpt.com "pyboy API documentation"
[4]: https://nvidia.github.io/warp/modules/runtime.html?utm_source=chatgpt.com "Python Reference — Warp 1.10.1"
