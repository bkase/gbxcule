Below is an **engineering plan for Epic 2 (Makefile + commit-hook automation)** for M0. This is written so an engineer can pick it up and implement it directly, with clear deliverables, file changes, commands, and validation steps.

I’m intentionally keeping emulator/runtime architectural decisions out; this is purely **repo automation + local gates**.

---

# Epic 2 Engineering Plan

## Epic 2 — Makefile + commit-hook automation (local CI-as-hook)

### Goals

1. **One workflow**: Everything runs via `uv` (directly or wrapped by `make`).
2. **Fast local gates**: Pre-commit hook runs a deterministic check suite quickly.
3. **Standard entrypoints**: Common commands are discoverable and consistent (`make setup`, `make test`, etc.).
4. **Actionable failures**: When a hook fails, devs get a short explanation and the exact next command to run.

### Non-goals (M0)

* No remote CI integration.
* No GPU benchmarking in commit hook.
* No expensive multi-process performance runs in commit hook (those can exist as explicit `make bench`).

---

## Scope breakdown (mapped to stories)

### Story 2.1 — Add Makefile with core targets

#### Deliverables

* `Makefile` at repo root
* Targets:

  * `setup`
  * `fmt`
  * `lint`
  * `test`
  * `roms`
  * `bench`
  * `smoke`
  * `verify`
  * `check`
  * `hooks`
  * `help` (prints available commands)

#### Engineering decisions (tooling)

* Use **ruff** for lint + formatting (single tool, fast).
* Use **pytest** for tests.
* All python invocations go through **`uv run ...`** to guarantee the locked environment is used.

#### Implementation details

* The Makefile should:

  * avoid touching/formatting `third_party/` and generated directories (`bench/roms/out`, `bench/runs`)
  * provide sane defaults but allow overrides via env vars

**Recommended Makefile variables**

* `PY := uv run python`
* `RUFF := uv run ruff`
* `PYTEST := uv run pytest`
* `ROM_OUT := bench/roms/out`
* `RUNS_OUT := bench/runs`

**Target behaviors**

* `make setup`

  * installs deps from lock file
  * creates common output dirs if needed
* `make fmt`

  * applies formatting + autofix lint (developer convenience)
* `make lint`

  * checks formatting (no changes) + lint (no fix), fails if not clean
  * this is what the commit hook should run, to avoid surprise file modifications during commit
* `make test`

  * runs unit tests
* `make roms`

  * generates micro-ROMs
* `make smoke`

  * very small, deterministic “is the world broken?” run
  * should run **pyboy_single only**, with very small step count
  * writes output artifact in a stable location (preferably overwriting `bench/runs/smoke.json` so it doesn’t spam the repo with timestamps)
* `make bench`

  * runs the fuller baseline benchmark(s) (can include `pyboy_vec_mp` too)
  * writes timestamped run artifacts
* `make check`

  * fast gate used by commit hook
  * recommended composition:

    * `lint` → `test` → `roms` → `smoke`
  * do **not** include multiprocess benchmarking here

---

### Story 2.2 — Add commit-hook(s) that call Makefile targets

#### Deliverables

* `.githooks/` directory tracked in repo
* `.githooks/pre-commit` script that runs `make check`
* `make hooks` target that installs hooks via git config (recommended approach)

#### Why `core.hooksPath`

Git hooks in `.git/hooks/` aren’t tracked, so teams drift. The clean solution is:

* Put scripts in a tracked `.githooks/`
* Run: `git config core.hooksPath .githooks`

That’s reproducible and works per-repo.

#### `pre-commit` hook requirements

* Must fail fast and print:

  * which step failed (`lint`, `test`, etc.)
  * how to fix (usually `make fmt` or `make test`)
* Must detect missing `uv` and provide a clear message:

  * “Install uv, then run `make setup`”
* Must run using bash (Linux/macOS); if Windows support matters later, we can add a `.cmd` wrapper (out of M0 unless requested).

---

### Story 2.3 — “Smoke” target for developer sanity

#### Deliverables

* `make smoke` exists and is fast (<10–20s on a typical dev machine)
* Runs something deterministic and meaningful:

  * generates ROMs first (or asserts they exist)
  * runs `pyboy_single` on `ALU_LOOP.gb` (or equivalent) for a small number of steps
  * writes a JSON artifact to `bench/runs/smoke.json` (overwrite)
* Returns non-zero exit code on failure

#### Guardrails

* Smoke should **not** require:

  * GPU
  * multiprocessing
  * external services
* Smoke should be stable and not flaky.

---

## Files to add / modify

### 1) `Makefile` (new)

Minimum recommended content outline:

* `help` target (prints targets)
* `setup` → `uv sync`
* `fmt` → `ruff format` + `ruff check --fix`
* `lint` → `ruff format --check` + `ruff check`
* `test` → `pytest`
* `roms` → `python bench/roms/build_micro_rom.py`
* `smoke` → `python bench/harness.py ...` (pyboy_single, small steps)
* `bench` → `python bench/harness.py ...` (optionally suite runs)
* `verify` → `python bench/harness.py --verify ...` (scaffold)
* `check` → `lint test roms smoke`
* `hooks` → `git config core.hooksPath .githooks`

Also include:

* `.PHONY` for all targets
* `set -euo pipefail` behavior inside scripts (hook script) rather than Makefile (Makefile’s shell semantics are quirky, but we can enforce with `SHELL := /bin/bash` and `.ONESHELL:` if desired)

### 2) `.githooks/pre-commit` (new)

A bash script that:

* checks `uv` exists
* runs `make check`
* prints short, friendly output

### 3) `.gitignore` (may already exist; ensure these are included)

To avoid commit-hook noise and accidental commits:

* `bench/roms/out/`
* `bench/runs/`
* `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.venv/` (if applicable)

### 4) `README.md` (modify)

Add a small “Developer workflow” section:

* `make setup`
* `make hooks`
* `make check`
* `make fmt`
* `make bench`

---

## Concrete target specifications (what each must do)

### `make setup`

**Command**

* `uv sync --dev` (or plain `uv sync` depending on how you declare dev deps in pyproject)

**Acceptance**

* `uv run python -c "import gbxcule"` works afterward

---

### `make hooks`

**Command**

* `git config core.hooksPath .githooks`

**Acceptance**

* `git config --get core.hooksPath` outputs `.githooks`
* A subsequent `git commit` runs the pre-commit hook

---

### `make lint`

**Commands**

* `uv run ruff format --check src bench tools tests`
* `uv run ruff check src bench tools tests`

**Acceptance**

* Fails if formatting is required or lint issues exist
* Does not modify files

---

### `make fmt`

**Commands**

* `uv run ruff format src bench tools tests`
* `uv run ruff check --fix src bench tools tests`

**Acceptance**

* Applies formatting and autofix where safe
* Leaves repo in a clean state such that `make lint` passes

---

### `make test`

**Commands**

* `uv run pytest -q`

**Acceptance**

* Returns non-zero on failure

---

### `make roms`

**Commands**

* `uv run python bench/roms/build_micro_rom.py`

**Acceptance**

* Produces expected ROMs under `bench/roms/out/`
* Idempotent (rerunning overwrites/updates)

---

### `make smoke`

**Commands (example intent)**

* ensures ROMs exist (either depends on `make roms` or checks)
* runs: `uv run python bench/harness.py --backend pyboy_single --rom bench/roms/out/ALU_LOOP.gb --stage emulate_only --warmup-steps <small> --steps <small> --output <fixed path>`

**Acceptance**

* Produces a JSON artifact at a stable path (e.g. `bench/runs/smoke.json`)
* Fast enough for commit hook usage
* Deterministic across runs

---

### `make check` (commit-hook gate)

**Composition**

* `make lint`
* `make test`
* `make roms`
* `make smoke`

**Acceptance**

* If it passes, commit proceeds
* If it fails, hook blocks commit and prints next action

---

## Hook script behavior (requirements)

### `.githooks/pre-commit`

**Must:**

* run `make check`
* print a clear header (e.g., “Running local checks…”)
* on failure, print:

  * “Run `make fmt` to auto-fix formatting/lint”
  * “Then re-run `make check`”
* exit code reflects pass/fail

**Should:**

* skip running if `SKIP_HOOKS=1` env var is set (useful for emergency situations)

  * but don’t advertise it as the default workflow

---

## Validation plan (how we confirm Epic 2 is done)

1. Fresh clone on a dev machine:

   * `uv --version` works
   * `make setup`
   * `make hooks`
2. Make a trivial code change and commit:

   * pre-commit runs `make check`
   * passes or fails meaningfully
3. Introduce a lint violation and commit:

   * hook fails
   * message instructs `make fmt`
4. Run:

   * `make fmt`
   * `make check`
   * commit succeeds
5. Confirm repo cleanliness:

   * generated files land only in gitignored paths

---

## Risk list + mitigations

* **Risk:** Commit hook too slow → devs bypass it
  **Mitigation:** keep `make check` small; reserve heavy benchmarking for `make bench`.
* **Risk:** Hook modifies files (surprising)
  **Mitigation:** hook runs `make check` (non-mutating); `make fmt` is explicit.
* **Risk:** Generated artifacts pollute git status
  **Mitigation:** `.gitignore` includes generated directories; `smoke` overwrites a single file instead of timestamp spam.
* **Risk:** `uv` not installed across all machines
  **Mitigation:** hook prints clear install instructions and fails fast.

---

If you want, I can also provide a **ready-to-paste Makefile + pre-commit hook script** aligned with the exact CLI flags your harness will expose in M0 (so smoke/bench targets won’t drift).

