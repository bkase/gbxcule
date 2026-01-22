# GBxCuLE Learning Lab — Engineering Architecture and Tech Stack

Audience: **kernel + systems engineers**.
Style: **hybrid** — hard constraints (**MUST**) plus preferred defaults (**SHOULD**).
Scope: architecture + stack for the repo described in the PRD (same structure), emphasizing correctness automation, reproducible benchmarking, and a fast dev loop across **macOS / cloud CPU-only** and **DGX Spark (CUDA 13) GPU**.

---

## 1) Architectural goals and invariants

### 1.1 Hard constraints (MUST)

**Correctness & determinism**

- **Every verification run MUST be reproducible**: given `(ROM SHA256, backend version, seeds, step quantum, action generator spec, configs)`, we can rerun and get identical mismatch (or no mismatch).
- **Mismatches MUST fail fast** (`exit != 0`) and emit a **repro bundle** containing enough state to reproduce without human forensics.
- **State comparisons MUST be stable** across platforms (ordering, numeric types, JSON encoding).

**Benchmark integrity**

- Benchmarks MUST:
  - exclude warmup,
  - clearly record synchronization policy,
  - emit structured JSON with a stable schema version,
  - report **steady-state throughput** separately from warm-start.

**Tooling latency**

- The “full” local gate (**CPU-only**) MUST complete in **≤ 2 minutes** on a typical dev machine.
- GPU gates MUST exist and MUST be what “main” is held to, even if most dev machines can’t run them.

**Repo is the source of truth**

- Architecture decisions, schemas, and reproducibility rules belong in-repo and are validated by tests (doc-checking can be added later, but schema correctness is non-negotiable).

### 1.2 Preferred defaults (SHOULD)

- **Functional core / imperative shell**: pure logic in `src/gbxcule/core/*`, effects in `backends/*` and `bench/harness.py`.
- **Unidirectional “step” model**: `actions → step(state) → new state (+ reward/obs/info)`.
- **Supply-chain minimalism**: prefer stdlib + a few well-chosen deps; avoid sprawling frameworks.

---

## 2) System context and high-level component model

### 2.1 External dependencies and trust boundaries

- **Trusted reference backend**: `pyboy_single` (PyBoy) is the initial oracle.

- **DUT backends**:
  - `warp_vec_cpu`: Warp on CPU (debug/correctness first),
  - `warp_vec_gpu`: Warp on CUDA (performance).

- **CPU baseline backend**: `pyboy_vec_mp` (multiprocess/vector baseline).

- **ROM source**: Micro-ROMs are generated locally; no copyrighted ROMs are shipped.

Trust boundary: only PyBoy is “oracle”; everything else is compared to it.

### 2.2 Component diagram (conceptual)

```
                ┌───────────────────────────────────────────┐
                │                bench/harness.py            │
                │  - CLI / configs                           │
 User ─────────▶│  - orchestration                            │─────────┐
                │  - benchmarking + verification              │         │
                └───────────────┬───────────────────────────┘         │
                                │ uses VecBackend protocol            │
     ┌──────────────────────────┼──────────────────────────┐          │
     │                          │                          │          │
┌────▼─────┐              ┌─────▼──────┐              ┌────▼────────┐│
│ pyboy_*  │              │  warp_vec  │              │ future tools ││
│ backends │              │ backend(s) │              │ (trace/etc.) ││
└────┬─────┘              └─────┬──────┘              └────┬────────┘│
     │                          │                          │         │
     │                          │ uses ABI + signatures    │         │
┌────▼──────────────────────────▼──────────────────────────▼─────┐   │
│                      src/gbxcule/core                           │   │
│  abi.py (authoritative layouts)                                 │   │
│  signatures.py (hashes/diffs/canonicalization)                  │   │
│  reset_cache.py (CuLE-style cached reset states)                │   │
└─────────────────────────────────────────────────────────────────┘   │
                                                                        │
                                                structured JSON artifacts
                                             bench/runs/* and mismatch/*
```

---

## 3) Execution environments and workflow

### 3.1 Supported environments

**macOS (dev)**

- CPU-only: can run `pyboy_single`, `pyboy_vec_mp` (if deps allow), and `warp_vec_cpu`.
- GPU code paths are importable but must gracefully skip/disable.

**Cloud “Codex/Claude”-style environments (dev)**

- CPU-only. Same expectations as macOS.

**DGX Spark (canonical benchmark + main gate)**

- Ubuntu + CUDA 13 (adapting as the box updates).
- This machine is the canonical source of “real” benchmark numbers and GPU correctness.

### 3.2 Main branch quality policy (MUST)

Because there is no centralized CI and GPU is only available on DGX:

- **All commits that land on `main` MUST be produced on the DGX Spark** after running the GPU-inclusive gate (details below).
- Other machines can develop freely on branches, but those branches are not considered “green” for main until validated on DGX.

This is a _process constraint_ enforced by:

- repo documentation,
- a git hook that runs different targets depending on branch + GPU availability,
- and (most importantly) team discipline: **merging to main happens on DGX**.

---

## 4) Canonical runtime semantics: what is a “step”?

This choice drives correctness and benchmark comparability.

### 4.1 Step quantum: “frames per step” (MUST)

We standardize the environment step in terms of **Game Boy emulator frames**:

- `frames_per_step` (default **24**) defines how many emulator frames elapse per `step()`.
- `frames_per_step` MUST be recorded in every run artifact.
- Benchmarks MUST report:
  - `env_steps_per_sec` (SPS),
  - and `frames_per_sec = env_steps_per_sec * frames_per_step`.

This lets you keep step semantics consistent with RL conventions (frameskip / action frequency) while preserving apples-to-apples comparisons via **frames/sec**.

This aligns with common Pokémon RL environment structure: send an input, then tick the emulator for `action_freq` frames before sampling state. ([Drubinstein][1])

### 4.2 Action semantics (SHOULD match pokemonred_puffer conventions)

We adopt an action model compatible with the Pokémon Red RL environment style:

- `action: int` indexes into a small action set (e.g., noop, up, down, left, right, A, B, start, select, and combinations if desired).
- Step applies:
  1. button press,
  2. scheduled release after `release_after_frames` (default 8),
  3. advance emulator by remaining frames: `frames_per_step - 1` (or equivalently the net frames_per_step). ([Drubinstein][1])

For micro-ROMs, actions can be **noop** by default, but the machinery must exist and be deterministic.

### 4.3 Observation semantics (your choice: small feature vector)

- `obs` is a small, fixed feature vector (early: minimal, later: richer).
- In early stages, `obs` can be a minimal feature vector derived from CPU state and a few memory reads; later it becomes device-resident features for E4.

Canonical early shape (recommended):

- `obs: float32[num_envs, obs_dim]` with `obs_dim` small (e.g., 16–128).
- The exact content is versioned as part of the backend contract (schema versioning below).

---

## 5) Backend contract and typing strategy

### 5.1 VecBackend protocol (MUST)

In `src/gbxcule/backends/common.py` define:

- `reset(seed: int | None) -> (obs, info)`
- `step(actions) -> (obs, reward, done, trunc, info)`
- `get_cpu_state(env_idx: int) -> dict`
- `close()`

Plus declared properties:

- `name: str`
- `device: Literal["cpu", "cuda"]`
- `num_envs: int`
- `action_spec`: (shape, dtype, meaning)
- `obs_spec`: (shape, dtype, meaning)

### 5.2 Make invalid states unrepresentable (practical Python)

Python isn’t a proof assistant, but we can still “type-drive” correctness:

- Use `dataclasses(frozen=True)` for **pure** result types (`StepOutput`, `RunConfig`, `RunResult`).
- Use `typing.Protocol` for the backend interface.
- Use `pyright` in strict mode for **core modules**:
  - `src/gbxcule/core/*`
  - `src/gbxcule/backends/common.py`
  - `bench/harness.py` (at least the config/result dataclasses)

We treat “types align” as an early check, not as the only correctness mechanism.

---

## 6) ABI v0: authoritative device buffers

### 6.1 ABI v0 decision (recommended): flat 64KB memory

**ABI v0** is defined in `src/gbxcule/core/abi.py` and is authoritative for Warp kernels and any future “native CUDA” kernels.

Recommended ABI v0 buffers per env:

**Registers (u16/u8)**

- `pc: u16`, `sp: u16`
- `a,f,b,c,d,e,h,l: u8`

**Flags**

- Store `f: u8` in ABI.
- Derive flag bits (`z,n,h,c`) in `get_cpu_state()` and in comparison helpers for stable diffs.

**Counters**

- `instr_count: u64` (recommended: always present)
- `cycle_count: u64` (optional early; helpful later)

**Memory**

- `mem: u8[65536]` flat 64KB per env.

Rationale:

- Flat 64KB keeps early kernel work straightforward.
- It minimizes ABI churn when moving from micro-ROMs to real ROM behaviors.
- It’s expensive, but manageable for “meaningful env scales” on modern GPUs; and we can later evolve memory representation once we have evidence about bottlenecks.

### 6.2 ABI versioning policy (MUST)

- ABI has an integer `ABI_VERSION`.
- Any ABI change requires:
  - version bump,
  - updated tests,
  - and a note in `abi.py` describing the change and its migration impact.

---

## 7) Correctness verification architecture

You asked for **no goldens for now**, but correctness must still scale.

### 7.1 Canonical CPU state schema (MUST)

`get_cpu_state(env_idx)` returns a dict with stable keys and JSON-friendly scalars.

Recommended schema:

```json
{
  "pc": 4660,
  "sp": 65535,
  "a": 12,
  "f": 176,
  "b": 0,
  "c": 1,
  "d": 2,
  "e": 3,
  "h": 4,
  "l": 5,
  "flags": { "z": 1, "n": 0, "h": 1, "c": 0 },
  "instr_count": 12345,
  "cycle_count": 98765
}
```

Notes:

- Always include raw `f`; flags are derived for diffs and readability.
- Include counters if available; if not, omit or set to `null` (but prefer consistent presence).

### 7.2 Verification compare policy (best default)

Given the 2-minute gate, the simplest and strongest early default is:

- **step-exact comparison for the entire verify run**, but keep `verify_steps` modest in precommit (e.g., 512–4096 steps).
- Support a configurable “interval compare” mode for longer runs:
  - `compare_every = 1` default (step exact),
  - allow `compare_every = K` for longer stress tests.

This is both easy to reason about and cheap enough at small step counts.

### 7.3 Memory hashing (your current preference: off by default)

You selected “no memory hash” initially. Here’s how we reconcile that with correctness needs:

- **M0 default**: regs + flags (+ counters if present).
- Architecture MUST support memory hashing from day 1 (because it will matter), but:
  - the default can be off in CPU-only dev and in early precommit,
  - and can become on-by-default when Warp CPU correctness work begins.

Implementation approach in `core/signatures.py`:

- Provide `hash_state(cpu_state, *, mem_region=None)` where `mem_region` is optional.
- When enabled, `mem_hash = blake2b(mem[lo:hi], digest_size=16)` (stdlib).

### 7.4 Mismatch automation and repro bundle (MUST)

On mismatch, harness MUST:

- `exit != 0`
- write bundle to:
  `bench/runs/mismatch/<timestamp>_<rom_id>_<ref>_vs_<dut>/`

Bundle contents (JSON-only, per your preference):

- `metadata.json`
- `ref_state.json`
- `dut_state.json`
- `diff.json`
- `repro.sh`
- **action trace** for the run (required): e.g. `actions.jsonl` or `actions.npy` (but if strictly JSON-only, use JSONL).

**Design details**

- `metadata.json` includes:
  - ROM path + SHA256
  - backend names/devices
  - env count, steps, warmup, stage
  - seeds, action generator name/version
  - mismatch step index + env id
  - git commit SHA (if available)
  - system info (CPU, GPU name/driver, Warp version+provenance)
  - `frames_per_step`, `release_after_frames`, `sync_every`

**Crash-only / idempotence**

- Bundle writes should be atomic:
  - write files to a temp dir,
  - then rename into final path.

---

## 8) Benchmarking architecture

### 8.1 Measurement protocol (MUST)

Harness benchmarking mode must implement:

- Warmup:
  - run `warmup_steps` steps not included in measurement

- Measurement window:
  - either fixed step count (`--steps`) or fixed duration (optional later)

- GPU sync:
  - `--sync-every K` controls explicit sync
  - default avoids per-step sync; instead sync every K steps and at end

- Output:
  - write JSON to `bench/runs/<timestamp>/<run_id>.json`

### 8.2 Metrics (MUST)

Report:

- `total_env_steps_per_sec` (SPS aggregated across envs)
- `per_env_steps_per_sec`
- `frames_per_sec = total_env_steps_per_sec * frames_per_step`
- scaling efficiency vs linear
- warm-start vs steady-state throughput

### 8.3 Run artifact schema (MUST, versioned)

Introduce `RESULT_SCHEMA_VERSION` (int) and include it in every artifact.

Recommended top-level JSON:

```json
{
  "schema_version": 1,
  "run_id": "2026-01-21T12:34:56Z__E1__pyboy_single__rom_ALU_LOOP__envs_64",
  "timestamp_utc": "2026-01-21T12:34:56Z",
  "config": {...},
  "system": {...},
  "results": {
    "warmup_steps": 512,
    "measured_steps": 8192,
    "seconds": 1.234,
    "total_env_steps_per_sec": 530000.0,
    "frames_per_step": 24,
    "frames_per_sec": 12720000.0
  }
}
```

---

## 9) Backend implementations: design notes per backend

### 9.1 `pyboy_single` (trusted oracle)

Purpose:

- correctness reference
- baseline single-env perf

Key design points:

- Headless execution, no rendering.
- Deterministic seeding where possible (note: PyBoy determinism can be subtle; for micro-ROMs it’s usually fine).
- Implements the canonical step semantics (frames per step).

`get_cpu_state(0)` must normalize:

- register ints as Python ints
- consistent keys and flag derivation.

### 9.2 `pyboy_vec_mp` (CPU throughput baseline)

Purpose:

- best available “CPU multiprocessing baseline” early

Constraints:

- Must be deterministic enough for benchmarking.
- Verification mode should **not** use MP as oracle (you selected `pyboy_single` only).

Recommended implementation strategy (minimize overhead):

- `multiprocessing` with `spawn` (macOS compatibility) but tuned for Linux too.
- Each worker hosts N envs; actions are batched per worker.
- Communication:
  - keep it minimal during benchmarking (avoid returning large obs every step),
  - but the backend still must satisfy the interface.

Pragmatic compromise:

- In benchmark mode, allow obs to be a small fixed vector (zeros or minimal features) to avoid high IPC costs.
- Keep `info` small; record detailed info only in artifacts at the harness layer.

### 9.3 `warp_vec` (DUT backend)

Purpose:

- shared wrapper that supports Warp CPU and Warp CUDA with identical interface.

Responsibilities:

- allocate ABI buffers
- translate actions → kernel inputs
- run kernels
- expose `get_cpu_state` by reading ABI buffers (and potentially copying from device)

For performance:

- In CUDA mode, avoid per-step host copies:
  - only copy to host when:
    - verification needs it,
    - mismatch occurs,
    - or `sync_every` boundary requires it.

---

## 10) Warp kernels: staging plan (E0–E4 aligned)

Your PRD’s ladder is the forcing function; architecture should make each stage isolated and measurable.

### 10.1 E0: plumbing ceiling

Kernel does trivial work:

- increment a counter per env
- maybe do a tiny amount of arithmetic to avoid dead-code elimination

This establishes:

- kernel launch overhead
- scaling vs env count
- sync policy effects

### 10.2 E1–E3: micro-ROM-driven CPU stepping

Kernel coverage grows:

- E1: ALU loop
- E2: divergence stress
- E3: memory stress

Architecture requirement:

- the harness must be able to swap ROM suites and keep configs stable.

### 10.3 E4: minimal reward and features on device

Goal:

- reward computed in kernel without per-step host copies.

Architecture requirement:

- `abi.py` includes reward/obs buffer layouts (or a parallel `RewardAbiV0`).
- `warp_vec` returns reward/obs without forcing host transfer (unless caller requests).

---

## 11) Tooling and developer workflow

### 11.1 Package management (MUST): uv-only

- `pyproject.toml` is single source of deps + tooling config.
- `uv.lock` is committed and used everywhere.

### 11.2 Formatting, linting, typing (your chosen stack)

- **ruff**: formatting + linting
- **pyright**: type checking
- Policy:
  - strict typing in core modules; gradual elsewhere.
  - don’t rewrite third-party code to satisfy types.

### 11.3 Tests

- pytest
- pytest-xdist (parallel CPU tests where useful)
- hypothesis (property tests, especially for signatures/determinism)

### 11.4 Makefile as the canonical interface (MUST)

Targets (recommended):

- `make setup` → `uv sync`
- `make fmt` → `uv run ruff format` + `uv run ruff check --fix`
- `make lint` → `uv run ruff check`
- `make type` → `uv run pyright`
- `make test` → `uv run pytest -q`
- `make roms` → generate micro-ROMs
- `make bench` → run a small baseline bench and emit JSON
- `make verify` → run verify mode (CPU-only by default)
- `make check` → **CPU gate** (≤2 min)
- `make check-gpu` → GPU-inclusive gate (DGX)
- `make check-main` → alias for “what main requires” (on DGX: includes GPU; elsewhere: refuses or warns)

### 11.5 Git hooks: “CI as precommit” (MUST)

Implement a simple hook installer (no extra framework required):

- `tools/install_git_hooks.py` writes `.git/hooks/pre-commit` that runs `make check`.

Branch/device behavior:

- On macOS/cloud CPU-only:
  - `pre-commit` runs `make check` (CPU gate)

- On DGX:
  - if on `main` (or preparing a merge-to-main):
    - `pre-commit` runs `make check-main` (includes GPU)

  - otherwise:
    - default to CPU gate, with an easy opt-in to GPU gate

This matches your “branches can skip GPU, main cannot” requirement.

---

## 12) Tech stack (pinned, minimal, fast)

### 12.1 Language/runtime choices

- **Python 3.11** (target)
- Warp for kernels (CPU debug + CUDA performance)
- Minimal optional Rust tooling allowed (but not required early)

### 12.2 Python dependencies (recommended shape)

**Runtime deps (keep small)**

- `pyboy` (reference emulator)
- `warp-lang` (Warp)
- `numpy`
- `PyYAML` (configs/suite yaml)

**Dev deps**

- `ruff`
- `pyright`
- `pytest`
- `pytest-xdist`
- `hypothesis`
- type stubs as needed (e.g., `types-PyYAML`)

**Analysis deps (optional extra)**

- `matplotlib`
- `pandas`

### 12.3 Node/bun stance

- bun is allowed **only if clearly justified** for a specific job.
- Default: **no Node toolchain required** for day-to-day work.

### 12.4 OS-level deps

**macOS**

- keep zero-to-one minimal: python + uv.
- avoid requiring SDL windows; use headless PyBoy mode.

**DGX (Ubuntu CUDA 13)**

- apt installs allowed if needed (e.g., system libraries PyBoy might require).
- we adapt to CUDA/driver updates; we pin Python deps via uv.

---

## 13) Container strategy (for “future big GPU rentals”)

You want bare-metal for day-to-day, but docker as a portability check before renting big hardware.

Architecture recommendation:

- Add a `docker/` directory later (or root `Dockerfile`) with:
  - a CUDA base image matching the DGX Spark’s CUDA major
  - `uv` install + `uv sync`
  - an entrypoint for `make check-main` and `bench/harness.py`

This container is not the primary dev path; it is a **portability harness**.

---

## 14) Observability and self-healing

### 14.1 Structured logs only

- Logging output should be:
  - minimal on success,
  - JSON-ish structured on failure (or at least structured fields).

- Every run has a `run_id` and every mismatch has a `trace_id`.

### 14.2 Crash-only posture

- Bench runs and mismatch runs write into unique timestamped dirs.
- Restarting a run does not corrupt shared state; worst case you get another run directory.

---

## 15) Documentation-as-code (knowledge graph)

Minimum docs to keep in repo (recommended):

- `README.md`: story + setup + “run E0/E1” commands
- `ARCHITECTURE.md`: this document
- `SCHEMAS.md`: JSON schema versions and field semantics (or embed in `ARCHITECTURE.md`)

Optional later:

- doc-linting and doc-tests once the code is stable enough.

---

## 16) Specific decisions made from your answers

Here are the decisions I locked in (so the repo can move without ambiguity):

1. **Step = frames-based quantum**, default **24 frames per step**, recorded in artifacts; report frames/sec as the cross-backend comparable metric. ([Drubinstein][1])
2. **Action semantics** match the “press, delayed release, tick frames” model; `release_after_frames=8` default. ([Drubinstein][1])
3. **Verification** uses `pyboy_single` as the reference oracle only (no MP oracle).
4. **Mismatch bundle** is JSON-only and includes an **action trace**.
5. **Memory hashing** supported but **off by default in M0**; architecture provides an easy switch to enable later.
6. **ABI v0** recommended as flat 64KB memory + CPU regs + flags + instr counter (cycle counter optional).
7. **Main branch guarantee**: merges to main happen on DGX after GPU-inclusive `make check-main`.

---

## 17) What this enables next (epic-by-epic planning readiness)

With the above architecture, each epic can be planned as:

- a small number of pure-core modules (`core/*`) + thin shell adapters (`backends/*`, `bench/harness.py`)
- tests that prove determinism and schema stability
- performance measurements that are comparable and reproducible

When you’re ready, I can turn this into:

- a repo-ready `ARCHITECTURE.md`,
- plus an “engineering plan per epic” where each epic has:
  - design intent,
  - module-level tasks,
  - invariants/tests to write first,
  - and acceptance checks tied directly to `make check` / `make check-main`.

If you want one tiny follow-up _without reopening a big Q&A_: tell me whether you prefer `obs_dim = 16`, `32`, or `64` for the initial fixed feature vector, and I’ll lock the contract + artifact schema around that.

[1]: https://drubinstein.github.io/pokerl/docs/chapter-2/env-setup/ "The Environment | Pokémon RL"
