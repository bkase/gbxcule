Below is the updated **M0 Epic + Story backlog** with your decisions applied:

- Include **both** `pyboy_single` and `pyboy_vec_mp`
- Use **uv as the single way** to install/run everything (one workflow)
- Use a **Makefile** for common commands / commit-hook entrypoints
- Use suggested answers for the rest (artifacts under `bench/runs/`, 2 micro-ROMs minimum in M0, etc.)

Per your request, I’m keeping **engineering architecture decisions out** (no kernel mapping, SoA/AoS, internal implementation approaches, etc.). This is phrased as PM-ready epics/stories with acceptance criteria.

---

# M0 Epics & Stories (Detailed)

## M0 Definition of Done (sign-off checklist)

M0 is complete when all are true:

- Repo matches agreed directory structure and is runnable via **uv** only
- `make setup` / `make test` / `make bench` work on a fresh machine
- Micro-ROM builder generates at least **two valid ROMs** into `bench/roms/out/`
- Harness runs:
  - `pyboy_single` benchmark on a micro-ROM
  - `pyboy_vec_mp` benchmark on a micro-ROM

- Harness emits **structured JSON** artifacts under `bench/runs/`
- Verification scaffolding exists and mismatch emits a **repro bundle**
- A **devlog tweet draft** exists (why we’re doing this + initial tests + where we’re at)
- Commit-hook checks exist via Makefile target(s)

---

## Epic 1 — Repo scaffolding and developer workflow (uv-only)

**Goal:** A clean, reproducible repo that new engineers can set up and run with one workflow.

### Story 1.1 — Create repository structure exactly as spec

**Acceptance criteria**

- Repo matches the agreed structure:
  - `configs/`, `src/gbxcule/...`, `bench/...`, `tools/`, `third_party/`, `tests/`

- Import path is clean (no runtime path hacks)
- Placeholder modules exist so imports resolve after installation

### Story 1.2 — Establish uv-based dependency management

**Acceptance criteria**

- `pyproject.toml` defines runtime + dev dependencies
- `uv.lock` exists and is used as the single source of truth
- README contains a single install method using `uv` (no alternate paths)

### Story 1.3 — Provide uv-only run interface (no “multiple ways”)

**Acceptance criteria**

- All documented commands use either:
  - `uv run ...` or
  - `make <target>` that internally uses `uv run ...`

- No docs suggest `pip install` or `python` directly without `uv run`

---

## Epic 2 — Makefile + commit-hook automation (local CI-as-hook)

**Goal:** Standardized team workflow: one set of commands, fast local gates, consistent outputs.

### Story 2.1 — Add Makefile with core targets

**Acceptance criteria**

- Makefile includes at least:
  - `make setup` (installs deps via uv)
  - `make fmt` (format/lint)
  - `make test` (unit tests)
  - `make roms` (generate micro-ROMs)
  - `make bench` (run baseline benchmark)
  - `make verify` (run verification mode scaffold)
  - `make check` (fmt + test + roms + a small smoke bench)

- Each target uses `uv run ...` internally

### Story 2.2 — Add commit-hook(s) that call Makefile targets

**Acceptance criteria**

- A documented setup step installs git hooks that run `make check` (or equivalent)
- Hook execution is fast and deterministic
- Hook failure output is actionable (shows the failing command and artifact paths)

### Story 2.3 — “Smoke” target for developer sanity

**Acceptance criteria**

- `make smoke` builds micro-ROMs and runs a short `pyboy_single` benchmark
- Writes JSON output to `bench/runs/`
- Returns non-zero exit code on failure

---

## Epic 3 — Backend contract and shared types

**Goal:** A stable interface so the harness can run any backend consistently.

### Story 3.1 — Define backend interface contract

**Acceptance criteria**

- `src/gbxcule/backends/common.py` defines:
  - a backend interface (protocol or base class) for `reset`, `step`, `close`, `get_cpu_state`
  - common output container types (e.g., StepOutput) used by harness

- Contract supports:
  - `num_envs`
  - `device` (cpu/cuda)
  - action spec and minimal env metadata for reporting

### Story 3.2 — Standardize run metadata schema

**Acceptance criteria**

- Harness emits a JSON schema containing:
  - backend name/device, ROM id/path/sha
  - steps/warmup, seeds, env count, worker count (if applicable)
  - SPS + per-env SPS
  - host/system info fields (basic)

---

## Epic 4 — Micro-ROM generation + suite definition

**Goal:** Deterministic, license-safe ROMs for correctness + performance baselines.

### Story 4.1 — Implement micro-ROM builder

**Acceptance criteria**

- `bench/roms/build_micro_rom.py` generates at least 2 ROMs:
  - `ALU_LOOP.gb`
  - `MEM_RWB.gb`

- Output location: `bench/roms/out/`
- ROMs load and execute under PyBoy headless without crashing for a small number of steps

### Story 4.2 — Add micro-ROM suite file

**Acceptance criteria**

- `bench/roms/suite.yaml` exists and includes at least the 2 ROMs
- Each entry includes:
  - id/name
  - relative path to ROM
  - step configuration fields needed for consistent benchmarking (e.g., ticks/step or equivalent)
  - short description (“what it tests”)

### Story 4.3 — Tests for ROM generation

**Acceptance criteria**

- `tests/test_micro_roms.py`:
  - runs ROM generator
  - asserts ROM files exist
  - instantiates PyBoy headless and steps for a small count (sanity check)

---

## Epic 5 — Reference baseline backend: `pyboy_single`

**Goal:** A trusted baseline and a correctness oracle for micro-ROMs.

### Story 5.1 — Implement `pyboy_single` backend

**Acceptance criteria**

- `src/gbxcule/backends/pyboy_single.py` implements the backend contract
- Runs headless
- Provides `get_cpu_state(env_idx=0)` returning a consistent register set (as available)
- Can execute N steps with minimal overhead suitable for benchmark loops

### Story 5.2 — Add minimal “state snapshot” normalization

**Acceptance criteria**

- Output of `get_cpu_state` has stable key names and types
- If a field cannot be read, backend fails with a clear error message (not silent wrongness)

---

## Epic 6 — CPU multiprocessing baseline backend: `pyboy_vec_mp`

**Goal:** Establish “best available CPU baseline” early for hypothesis comparisons.

### Story 6.1 — Implement `pyboy_vec_mp` backend

**Acceptance criteria**

- `src/gbxcule/backends/pyboy_vec_mp.py` implements the backend contract for N envs
- Supports:
  - `num_envs`
  - `num_workers`

- Returns aggregate SPS and per-env SPS in reports
- Produces deterministic run artifacts (same config → comparable results)

### Story 6.2 — Basic reliability checks for MP backend

**Acceptance criteria**

- A test or smoke run demonstrates:
  - startup succeeds
  - shutdown/cleanup succeeds
  - failures propagate (no hanging processes)

---

## Epic 7 — Harness CLI for benchmark + reporting

**Goal:** One CLI to run baselines and produce consistent structured results.

### Story 7.1 — Implement harness CLI skeleton

**Acceptance criteria**

- `bench/harness.py` supports:
  - `--backend {pyboy_single, pyboy_vec_mp}`
  - `--rom <path>` and `--suite <suite.yaml>`
  - `--stage emulate_only`
  - `--steps N`, `--warmup-steps W`
  - `--output-dir` defaulting to `bench/runs/`

- Writes one JSON artifact per run with a stable schema

### Story 7.2 — Add scaling run support (data-only)

**Acceptance criteria**

- Harness supports `--env-counts a,b,c` and runs sequential benchmarks
- Emits a single JSON report containing an array of results by env_count
- (Plotting can be deferred; the data format must be stable)

### Story 7.3 — Deterministic action generation (even if unused by micro-ROMs)

**Acceptance criteria**

- Harness accepts `--actions-seed`
- Records action generator name/version + seed in JSON output
- Actions can be “no-op” for micro-ROMs but must be reproducible

---

## Epic 8 — Verification scaffold + mismatch repro bundle

**Goal:** Correctness debugging scales from day 1, even before DUT exists.

### Story 8.1 — Add `--verify` mode scaffold

**Acceptance criteria**

- Harness supports `--verify` with:
  - `--ref-backend pyboy_single`
  - `--dut-backend <stub>` (can be placeholder in M0)

- Verification loop runs for N steps and compares states
- On mismatch:
  - exits non-zero
  - writes a repro bundle

### Story 8.2 — Implement mismatch repro bundle writer

**Acceptance criteria**

- Repro bundle is written to:
  - `bench/runs/mismatch/<timestamp>_<rom_id>_<ref>_vs_<dut>/`

- Bundle includes:
  - `metadata.json`
  - `ref_state.json`
  - `dut_state.json`
  - `diff.json`
  - `repro.sh`

- Console output prints the mismatch summary + path to bundle

> Note: No goldens are required or used in M0.

---

## Epic 9 — Documentation + initial marketing devlog tweet draft

**Goal:** Make the project understandable and exciting to outsiders immediately.

### Story 9.1 — README: “why + how to run”

**Acceptance criteria**

- README includes:
  - project premise + hypothesis (plain language)
  - how to set up via uv
  - how to generate micro-ROMs
  - how to run `pyboy_single` and `pyboy_vec_mp` baselines
  - where artifacts are written (`bench/runs/`)
  - how to interpret SPS at a high level

### Story 9.2 — Devlog tweet draft (first post)

**Acceptance criteria**

- A draft tweet exists (in a `docs/` note or README section) covering:
  - why we’re doing this (hypothesis)
  - initial tests/baselines now working
  - where we’re at + what’s next
  - one reproducible command

---

# M0 Suggested top-level command inventory (what PM expects)

All of these must work via uv-only workflow (either direct `uv run` or Makefile targets):

- `make setup`
- `make roms`
- `make bench` (runs both `pyboy_single` and `pyboy_vec_mp` at least once)
- `make check` (commit-hook gate)
- `make verify` (scaffold, expected to fail until DUT exists, but produces repro bundle)

---

If you want, I can also provide:

- a priority ordering and dependency graph between these stories (still without engineering decisions), and
- a PM-ready “M0 acceptance checklist” formatted for a ticketing system.
