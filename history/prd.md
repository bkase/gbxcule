Absolutely — here’s an updated **full PRD draft** with the exact repo structure you specified, **no goldens for now**, and marketing starting with a **devlog tweet** (why + story + initial tests). I’ve preserved the hypothesis-driven gates, mismatch automation, DGX Spark default, and commit-hook CI.

---

# PRD Draft: GBxCuLE Learning Lab

GPU-Native Many-Env Game Boy Runtime (Warp→CUDA) + Benchmark/Verification Harness

## 0) Document purpose

Define product requirements for a GPU-native Game Boy runtime and supporting tooling that:

- enables incremental correctness against a trusted reference (PyBoy),
- provides rigorous benchmarking vs CPU baselines,
- and quickly verifies/falsifies the core hypothesis that GPU stepping achieves meaningful speedup over CPU multiprocessing.

This PRD omits timelines.

---

## 1) Executive summary

We will build:

1. A **backend-driven benchmark + verification harness** that can run multiple backends with consistent measurement.
2. A **Warp-based Device Under Test (DUT)** runtime that steps many environments on GPU (and has a CPU-debug Warp target).
3. A **micro-ROM suite** to validate CPU core correctness early, and stage toward more realistic workloads later.
4. A **hypothesis experiment ladder** (E0–E4) that forces early learning on whether GPU stepping can beat CPU multiprocessing on a DGX Spark-class GPU.

We will also ship a **public devlog tweet** early explaining why we’re doing this and showing the first baselines/tests to build external excitement.

---

## 2) Hypothesis (project thesis)

> **Hypothesis H:** A GPU-native multi-environment Game Boy runtime can achieve **meaningful steady-state throughput speedups** relative to CPU multiprocessing baselines (PyBoy + vectorized/multiprocess stepping) on a **moderately powerful NVIDIA GPU (DGX Spark class)** for emulator stepping workloads representative of RL training loops.

This PRD requires early experiments intended to **verify or falsify** H before substantial investment in game-specific complexity.

---

## 3) Goals

### Product goals

G1. **Correctness automation that scales with team size**
Any mismatch between reference and DUT automatically produces a deterministic **repro bundle**.

G2. **Backend-driven harness with fair benchmarking**
A single harness runs reference and DUT backends with consistent step definitions, warmup discipline, and structured outputs.

G3. **Hypothesis-driven progress**
Evidence (not vibes) guides whether we continue investing; experiments E0–E4 must be runnable and reported early.

G4. **Path to device-resident RL stepping**
Stage toward device-resident state and reward (and later obs), minimizing per-step host involvement.

G5. **Brand building**
Ship approachable dev updates for curious developers, emphasizing rigor and fun, not CUDA/RL expertise.

### Engineering goals

E1. Deterministic reproducibility (action sequences, seeds, ROM identity).
E2. Clean ABI boundaries for state/obs/reward buffers.
E3. Fast local gating via commit-hook CI.
E4. Avoid premature coupling to Pokémon-specific code; keep it as a reference baseline.

---

## 4) Non-goals

NG1. Pokémon Red boot correctness as an early requirement.
NG2. Audio emulation.
NG3. Cycle-perfect PPU early.
NG4. Full MBC coverage immediately.
NG5. Shipping copyrighted ROMs/states in the repo.
NG6. Full end-to-end training system product (training integration is staged).

---

## 5) Personas

P1. **Kernel engineer**: wants deterministic correctness diffs and scaling curves.
P2. **Systems engineer**: wants stable harness outputs, reproducible runs, profiling hooks.
P3. **RL engineer**: wants a stable env-like API and eventual device-resident tensors.
P4. **Curious developer audience**: wants an understandable story and compelling demos.

---

## 6) Repository structure (REQUIRED; must match exactly)

The repo must use the following structure and conventions:

```text
gbxcule-learning-lab/
├── README.md
├── pyproject.toml                  # single source of deps + tooling
├── uv.lock / poetry.lock           # choose one; lock it
├── configs/
│   ├── default.yaml
│   └── debug.yaml
├── src/
│   └── gbxcule/
│       ├── __init__.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── pyboy_single.py     # minimal single-env reference runner
│       │   ├── pyboy_vec_mp.py     # puffer/mp baseline adapter
│       │   ├── warp_vec.py         # main backend wrapper (CPU-debug + GPU)
│       │   └── common.py           # VecBackend protocol + data classes
│       ├── core/
│       │   ├── abi.py              # state/obs/reward buffer layouts (authoritative)
│       │   ├── signatures.py       # hashing / trace comparison utils
│       │   └── reset_cache.py      # cached reset states (CuLE-style)
│       └── kernels/
│           ├── __init__.py
│           ├── cpu_step.py         # warp kernels: CPU step
│           ├── ppu_step.py         # later
│           ├── reward.py           # minimal reward first
│           └── utils.py
├── bench/
│   ├── harness.py                  # master CLI: benchmark + verify + report
│   ├── roms/
│   │   ├── build_micro_rom.py      # generate valid test ROMs
│   │   ├── suite.yaml              # list of test ROMs
│   │   └── out/                    # generated roms go here (gitignored)
│   └── analysis/
│       ├── plot_scaling.py
│       └── summarize.py
├── tools/
│   ├── gen_trace.py                # generate traces from PyBoy (optional early)
│   └── check_trace.py              # replay + compare (optional early)
├── third_party/
│   └── pokemonred_puffer/          # submodule or vendored copy (read-only)
└── tests/
    ├── test_micro_roms.py
    └── test_signatures.py
```

**Note:** We are **not** requiring a `golden/` directory or golden artifacts yet.
If/when we add frame comparison or long-run trace stability needs, we can introduce goldens later.

---

## 7) System overview

### Components

1. **Backends (swappable via harness)**

- `pyboy_single`: simplest “trusted reference” runner for micro-ROMs and sanity baselines.
- `pyboy_vec_mp`: CPU multiprocessing baseline (puffer/mp style) where applicable.
- `warp_vec`: DUT backend supporting:
  - Warp CPU device (debug)
  - Warp CUDA device (performance)

1. **Harness (`bench/harness.py`)**

- single CLI entrypoint for:
  - correctness verify (ref vs dut)
  - benchmarks (single, scaling, steady-state)
  - experiment ladder E0–E4

1. **Micro-ROM suite**

- generated locally (license-safe)
- used to validate CPU correctness and stress divergence/memory early

---

## 8) Backend contract (REQUIRED)

In `src/gbxcule/backends/common.py`, define a backend interface such that harness can do:

- `reset(seed: int | None) -> obs, info`
- `step(actions) -> obs, reward, done, trunc, info`
- `get_cpu_state(env_idx: int) -> dict` (required for verification)
- `close()`

Backends must declare:

- `name`
- `device` (`cpu`, `cuda`)
- `num_envs`
- action/observation specs (at least shape/dtype)

---

## 9) Device ABI (REQUIRED)

`src/gbxcule/core/abi.py` is the authoritative contract for buffers used by Warp kernels and downstream consumers.

### ABI v0 requirements (CPU-only stepping)

- CPU regs: PC, SP, A, F, B, C, D, E, H, L
- optional counters: instruction_count or cycle_count (recommended)
- minimal memory model sufficient for micro-ROMs (may begin with flat 64KB model)

### ABI change policy

- ABI is versioned (simple integer in `abi.py`).
- ABI changes require updating tests that assume layouts.

---

## 10) Correctness verification requirements (no goldens yet)

### 10.1 What we verify (initial)

For micro-ROM suite:

- regs: PC, SP, A, F, B, C, D, E, H, L
- plus one memory window hash (e.g., WRAM[0:256] or a selected region appropriate to the ROM)

Comparisons:

- step-exact for first N steps (configurable)
- then interval compare every K steps (configurable)

### 10.2 Mismatch automation (REQUIRED)

Any mismatch must:

- exit non-zero
- automatically emit a **repro bundle**
- print a short mismatch summary to console

#### Repro bundle contents (REQUIRED)

Write to: `bench/runs/mismatch/<timestamp>_<rom_id>_<backend>/`

- `metadata.json`
  - rom path + SHA256
  - backend names + devices
  - num_envs, stage, steps, warmup
  - seed + action generator spec/version
  - mismatch step index + env id
  - git commit SHA (if available)
  - GPU name (if cuda) / CPU summary

- `ref_state.json`, `dut_state.json`
- `diff.json` (field-level diffs)
- `repro.sh` (one-command reproduction)

**Optional but recommended early:** `trace_tail.txt` if you implement trace mode.

### 10.3 No “goldens” yet

- We do **not** store step-by-step golden traces/signatures as repo artifacts at this stage.
- Reference is computed live (PyBoy) during verify runs.
- Tools `gen_trace.py` / `check_trace.py` are optional early, and intended to reduce iteration cost later, not a gating requirement now.

---

## 11) Benchmarking requirements

### 11.1 Metrics (REQUIRED)

- total steps/sec (SPS) across all envs
- per-env SPS
- scaling efficiency vs linear
- effective FPS (if step maps to a fixed tick count)
- steady-state vs warm-start throughput

### 11.2 Measurement protocol (REQUIRED)

- fixed `--warmup-steps` excluded from measurement
- measurement uses either fixed `--steps` or duration with measured step count recorded
- `--sync-every` governs GPU synchronization (avoid per-step sync unless asked)
- results must be emitted as structured JSON to:
  - `bench/runs/<timestamp>/<run_id>.json`

### 11.3 Stage separation (REQUIRED)

Harness must support:

- `--stage emulate_only` (CPU core stepping only)
- `--stage full_step` (as soon as reward exists; initially may equal emulate_only)

Soon after:

- `reward_only`, `obs_only` (not required immediately, but the CLI should allow staged addition without breaking)

---

## 12) Hypothesis experiment ladder (REQUIRED; E0–E4)

To rapidly verify/falsify Hypothesis H, the harness must support a standardized experiment ladder:

### E0: launch/plumbing ceiling

Warp kernel that does trivial per-env work (e.g., increment counter).
Goal: measure overhead scaling and establish ceiling.

### E1: ALU_LOOP micro-ROM

Deterministic tight loop with a small instruction mix.

### E2: divergence stress micro-ROM

Branching workload that causes env divergence.

### E3: memory stress micro-ROM

Load/store heavy workload stressing access patterns.

### E4: minimal reward on device

Compute a small fixed feature vector and reward on GPU without per-step host copies.

### Hypothesis evaluation outputs (REQUIRED)

For each experiment:

- DGX Spark results
- CPU baseline results
- steady-state throughput reported separately

### Speedup targets (committed)

- For emulate-only workloads: demonstrate ≥ **1.5×** throughput vs best CPU multiprocessing baseline at some meaningful env scale (measured steady-state).
- For emulate + minimal reward: demonstrate ≥ **1.2×** throughput vs CPU baseline, or parity with profiling-backed evidence for how it will exceed.

If targets fail materially, this is treated as hypothesis-negative evidence and triggers an architecture review before proceeding into deeper game-specific complexity.

---

## 13) Hardware policy

### Default iteration target

- DGX Spark is the canonical platform for benchmarks and iteration.

### Blackwell rentals (conditional)

Allowed only when justified by profiling evidence, such as:

- compute-bound saturation where more SMs should help
- bandwidth-bound behavior where Blackwell changes the ceiling
- VRAM capacity constraints preventing required env scaling tests

Rental requirements:

- explicit experiment plan (subset of E0–E4)
- post-run conclusion: does it change hypothesis confidence?

---

## 14) CI / commit hook requirements (no timelines)

CI runs as a commit hook. It must be fast, deterministic, and correctness-focused.

### Commit hook gates (REQUIRED)

- format + lint
- unit tests for ABI/signatures determinism
- micro-ROM generator sanity
- correctness smoke verify:
  - run verify on 1–2 micro-ROMs for a small step count
  - mismatches must fail and emit repro bundle

GPU smoke tests are optional depending on local availability, but must never silently pass if configured.

---

## 15) Marketing plan (REQUIRED)

### Audience

Curious developers; not necessarily experts in CUDA/RL.

### Primary narrative

“We’re building a GPU-native Game Boy runtime capable of stepping thousands of worlds, and we’re rigorously testing whether GPU parallelism can beat CPU multiprocessing for emulator workloads.”

### Initial marketing deliverable (explicit requirement)

**First deliverable:** a **devlog tweet post** (and optionally cross-post) including:

- Why we’re doing this (the hypothesis + constraints)
- Where we’re at today (baselines + early micro-ROM tests)
- What “success” looks like (steady-state scaling + correctness automation)
- A simple visual (one chart or one terminal snippet)
- A reproducible command (e.g., how to run E0 or run a baseline benchmark)

This devlog is the starting point of the project’s public narrative.

### Ongoing marketing deliverables (required)

- Each major milestone must produce a shareable artifact:
  - chart (scaling curve), or
  - correctness screenshot (mismatch bundle summary), or
  - short demo clip

- README must support “run the benchmark” instructions for curious developers.

### Constraints

- No copyrighted ROMs/states distributed.
- Micro-ROMs generated by our tool are allowed and should be used in demos.

---

## 16) Milestones and Definition of Done (no timelines)

### M0: Repo + harness foundation

**DoD**

- Repo installs cleanly (`pip install -e .`)
- `bench/harness.py` can run:
  - PyBoy single baseline
  - outputs JSON run artifacts

- micro-ROM builder works and produces valid ROM(s)
- devlog tweet drafted and ready (see marketing requirement)

### M1: Automated mismatch triage

**DoD**

- `--verify` mode runs ref vs dut (even if dut stubbed)
- on mismatch:
  - exits non-zero
  - writes repro bundle with required files

### M2: Warp CPU-debug correctness on micro-suite

**DoD**

- micro-ROM suite passes step-exact or interval checks for required steps under Warp CPU device

### M3: Warp GPU correctness + scaling reports

**DoD**

- micro-ROM suite passes under Warp CUDA device
- scaling reports produced (steady-state vs warm-start)

### M4: Minimal reward on GPU + “zero/near-zero copy” proof path

**DoD**

- E4 runs with reward computed on device
- no per-step host copy required for reward pipeline

(Additional milestones can be appended later when Pokémon realism becomes relevant.)

---

## 17) Legal and licensing

- No distribution of copyrighted ROMs or proprietary assets.
- Pokémon integration assumes “user provides ROM locally.”
- Public demos should use micro-ROMs and license-safe tests.

---

## 18) Risks and mitigations (focused on falsification)

- Divergence kills GPU advantage → E2 required early.
- Memory bandwidth dominates → E3 required early.
- Warp limitations → keep ABI stable so kernels can migrate later.
- Incorrect benchmarks (apples-to-oranges) → strict measurement protocol and steady-state reporting.
- Premature Pokémon complexity → gating criteria requires hypothesis evidence first.

---

## 19) Appendix: minimal CLI expectations for `bench/harness.py` (non-binding but recommended)

- `--backend {pyboy_single, pyboy_vec_mp, warp_vec_cpu, warp_vec_gpu}`
- `--stage {emulate_only, full_step, reward_only, obs_only}`
- `--rom <path>` or `--suite bench/roms/suite.yaml`
- `--verify` (ref vs dut)
- `--steps N`, `--warmup-steps W`
- `--sync-every K`
- `--output-dir bench/runs/...`

---

## Marketing deliverable: devlog tweet (draft)

Here’s a draft you can hand to the PM/architect to edit (no links; you can add them):

We’re starting a new build: a GPU-native Game Boy runtime (Warp → CUDA) designed to step _thousands_ of emulator worlds in parallel.

**The hypothesis:** for emulator-style workloads, a moderately powerful GPU can beat “just add more CPU processes” — similar in spirit to CuLE, but for Game Boy.

**Why now:** RL + emulation throughput is the bottleneck. CPU multiprocessing helps, but hits overhead + scaling limits. We want a device-resident stepping loop (and later reward/obs) that scales cleanly.

**Where we are today:**
✅ clean benchmark harness + CPU baselines (PyBoy single + multiprocess)
✅ micro-ROM generator (tiny deterministic tests)
✅ correctness automation plan: mismatches auto-dump a repro bundle

Next up: GPU “E0” ceiling test (kernel launch/plumbing), then CPU-core micro-ROM correctness on Warp CPU, then Warp CUDA.

If you’re curious, follow along — we’ll share scaling curves, debugging artifacts, and what we learn as we try to prove (or falsify) the hypothesis.

If you want, I can now produce a “PM companion” version of this PRD with a one-page executive summary + acceptance checklist for each milestone (still without timelines).
