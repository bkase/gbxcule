# Engineering plan: Epic 1 — Repo scaffolding and developer workflow (uv-only)

Epic 1 is about making the repo _real_ on day 1: correct structure, installable package, and a single reproducible developer workflow that works on:

- **macOS** (CPU-only; GPU parts don’t run)
- **cloud dev envs** (CPU-only)
- **DGX Spark (Ubuntu, CUDA 13)** (CPU+GPU; main branch is gated here)

This plan stays within Epic 1’s scope: **structure + uv workflow + canonical commands**. (Makefile + hooks is Epic 2.)

---

## Epic 1 Definition of Done

Epic 1 is done when all are true:

1. **Repo structure exists** exactly as required in the PRD (all directories + placeholder modules).
2. `uv sync` completes on **macOS** and **Ubuntu** and creates a local environment. In uv projects, syncing creates/updates the `.venv` and installs the project in editable mode by default. ([docs.astral.sh][1])
3. `uv run python -c "import gbxcule"` succeeds (no import path hacks).
4. There is **one canonical way** documented to install + run: **uv-only** (`uv sync`, `uv run ...`). ([docs.astral.sh][2])
5. There is a minimal “smoke” verification (even if tiny) that yields an **automated pass/fail** signal (e.g., `pytest` running at least an import test).

---

## Scope boundaries

### In-scope (Epic 1)

- Directory tree + placeholder modules so imports resolve.
- Python packaging metadata (`pyproject.toml`) and **`uv.lock`**.
- Dependency grouping (dev tooling vs runtime) using dependency groups. ([docs.astral.sh][3])
- uv settings that reduce surprises (platform environments, lock behavior guidance). ([docs.astral.sh][4])
- Minimal docs for **uv-only** setup/run.

### Explicitly out-of-scope (later epics)

- Makefile targets (Epic 2).
- Bench harness behavior (Epic 7).
- Backend contract, ABI, kernels (Epics 3+).

---

## Story 1.1 — Create repository structure exactly as spec

### Deliverable

All required paths exist and are importable as a package (even if implementations are stubs).

### Tasks

1. **Create the directory tree exactly as in the PRD**
   - Create every folder and file path listed (including all `__init__.py` locations).
   - Add placeholder files even where content will come later (e.g., `bench/harness.py`, backend modules, kernel modules).

2. **Add minimal placeholder content (import-safe)**
   - Goal: `import gbxcule` must not implicitly import heavy optional subsystems.
   - Concrete rules for placeholders:
     - Top-level package `gbxcule/__init__.py` should not import `pyboy` or `warp`.
     - Backend modules can import dependencies lazily (inside functions) to avoid import-time failures.
     - Kernel modules should not compile kernels at import time.

3. **Add a minimal `__main__.py`**
   - `python -m gbxcule` should print a short help message like:
     - where the harness lives (`bench/harness.py`)
     - how to run commands using `uv run`

   - This is a cheap “it installed correctly” check.

4. **Add `.gitignore`**
   - Must ignore:
     - `.venv/`
     - `__pycache__/`, `.pytest_cache/`
     - `bench/roms/out/`
     - `bench/runs/`

   - Keep it minimal and deterministic.

### Acceptance checks (verifiable)

Run these from repo root:

- `uv run python -c "import gbxcule; print('ok')"`
- `uv run python -c "import gbxcule.backends; import gbxcule.core; import gbxcule.kernels; print('ok')"`

---

## Story 1.2 — Establish uv-based dependency management

### Deliverable

`pyproject.toml` + `uv.lock` committed, and the environment can be created reproducibly using uv.

### Key decisions for Epic 1 (based on your constraints)

- **uv is the only workflow**.
- Use **dependency groups** for dev tooling (pytest/ruff/pyright/etc). uv reads dev deps from `[dependency-groups]`, and the `dev` group is special-cased and installed by default. ([docs.astral.sh][5])
- Prefer **lockfile-driven reproducibility**:
  - By default, `uv run` and other commands may lock/sync automatically.
  - Use `--locked` or `--frozen` in automation to avoid mutating lock/environment. ([docs.astral.sh][5])
  - (Epic 2 will enforce this via Makefile/hook.)

### Tasks

1. **Pick packaging build backend and configure src-layout**
   - Use **hatchling** as build backend (simple, modern).
   - Configure wheel packages for src layout with:
     - `packages = ["src/gbxcule"]` (this is the documented pattern). ([hatch.pypa.io][6])

2. **Write `pyproject.toml`**
   - Must include:
     - `[build-system]`
     - `[project]` metadata (name, version, requires-python)
     - runtime deps: at least `pyboy`, `warp-lang`, `pyyaml`, `numpy`
     - `[dependency-groups].dev`: `pytest`, `pytest-xdist`, `hypothesis`, `ruff`, `pyright` (per your decision set)

   Notes grounded in sources:
   - PyBoy current latest on PyPI is **2.6.1** (released Nov 20 2025) and requires Python >= 3.9. ([PyPI][7])
   - Warp runs on CPU across macOS/Linux/Windows; GPU requires CUDA-capable NVIDIA GPU. ([PyPI][8])
   - Warp wheels on PyPI are built with CUDA 12 runtime (doc note); that’s likely fine on a CUDA 13 driver stack, but it’s a known integration risk to validate on DGX Spark. ([nvidia.github.io][9])

3. **Configure uv “environments” for lock resolution**
   - You explicitly care about **macOS + Linux**; not Windows.
   - uv supports restricting the environments it resolves against to avoid unsatisfiable branches and speed up lock. ([docs.astral.sh][4])
   - Set:
     - `environments = ["sys_platform == 'darwin'", "sys_platform == 'linux'"]`

4. **Generate and commit `uv.lock`**
   - Use `uv lock` (or `uv sync`, which will lock/sync automatically).
   - Commit `uv.lock` as the reproducible source of truth.

5. **Document lock discipline**
   - Add a short note in README (or `docs/dev.md` if you prefer) that:
     - `uv sync` is how you set up / update.
     - CI/hook will later use `uv run --locked` or `--frozen` so it fails rather than silently rewriting lock. ([docs.astral.sh][5])

### Acceptance checks (verifiable)

- Fresh clone:
  - `uv sync`
  - `uv run python -c "import gbxcule; print(gbxcule.__version__)"`

- Lock discipline sanity:
  - `uv lock --check` should pass once lock exists. ([docs.astral.sh][5])

---

## Story 1.3 — Provide uv-only run interface

### Deliverable

There is exactly one “way of working” documented and used in examples: `uv run ...` / `uv sync ...`.

### Tasks

1. **Canonical commands**
   - Decide and document the canonical commands (these will later be wrapped by Makefile):
     - Setup: `uv sync`
     - Run tests: `uv run pytest`
     - Run formatting/linting: `uv run ruff check .` and `uv run ruff format .`
     - Typecheck: `uv run pyright`

   - Note: `uv run` will create/update the environment automatically when run inside a project. ([docs.astral.sh][5])

2. **Add a minimal README “Developer Quickstart”**
   - Keep it tiny for Epic 1; Epic 9 will do full narrative.
   - Include:
     - `uv sync`
     - `uv run pytest`
     - `uv run python -m gbxcule`

3. **Minimal test to guarantee importability**
   - Add `tests/test_imports.py`:
     - imports `gbxcule` and a couple subpackages
     - asserts `__version__` exists

   - This is the “verifiable reward” for Epic 1: it produces an immediate `exit 0` / `exit 1`.

### Acceptance checks (verifiable)

- `uv run pytest -q` passes (with the import test).
- `uv run python -m gbxcule` prints guidance.

---

## Recommended `pyproject.toml` skeleton (high-signal, Epic 1 level)

This is a _template_, not the only solution—but it encodes the decisions above and aligns with uv + hatchling conventions.

```toml
[build-system]
requires = ["hatchling>=1.0"]
build-backend = "hatchling.build"

[project]
name = "gbxcule-learning-lab"
version = "0.0.0"
description = "GPU-native many-env Game Boy runtime (Warp→CUDA) + benchmark/verification harness"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "numpy",
  "pyyaml",
  "pyboy>=2.6.1",
  "warp-lang>=1.11.0",
]

[dependency-groups]
dev = [
  "pytest",
  "pytest-xdist",
  "hypothesis",
  "ruff",
  "pyright",
]

[tool.uv]
# Resolve lock for the platforms we actually support in dev.
# This can improve lock performance and avoid unsatisfiable branches.
environments = ["sys_platform == 'darwin'", "sys_platform == 'linux'"]

[tool.hatch.build.targets.wheel]
# src-layout packaging: ship the package from src/gbxcule
packages = ["src/gbxcule"]
```

Notes:

- The `packages = ["src/gbxcule"]` pattern is the hatch config for “package lives under src/”. ([hatch.pypa.io][6])
- The PyBoy version and release info is taken from PyPI. ([PyPI][7])
- Warp’s latest version on PyPI is 1.11.0 (released Jan 2 2026). ([PyPI][8])

---

## Risks & mitigations (Epic 1 relevant)

1. **PyBoy / warp import-time failures**
   - Mitigation: keep imports lazy; do not import these at `gbxcule/__init__.py` import time.

2. **Warp CUDA runtime mismatch**
   - Warp docs note PyPI binaries are built with CUDA 12 runtime. ([nvidia.github.io][9])
   - On DGX Spark (CUDA 13), this should be okay if the driver is new enough, but we should treat it as a first-class integration check as soon as Epic 2/3 introduce GPU smoke gates.

3. **Lockfile drift / “it works on my machine”**
   - Mitigation: commit `uv.lock`; later enforce `uv run --locked`/`--frozen` in hooks and any container flow. ([docs.astral.sh][5])

---

## What I would implement first (order of operations)

1. Story 1.1: filesystem tree + placeholder modules + `.gitignore`
2. Story 1.2: `pyproject.toml` + `uv lock` + commit `uv.lock`
3. Story 1.3: README quickstart + `tests/test_imports.py`

That gives us a tight loop: **clone → uv sync → uv run pytest**.

If you want, I’ll now write Epic 1 as a PR-ready checklist (commit-by-commit) so it’s easy to land cleanly, or we can go straight to the engineering plan for Epic 2 (Makefile + hook gating including “GPU-only on DGX Spark”).

[1]: https://docs.astral.sh/uv/concepts/projects/config/?utm_source=chatgpt.com "Configuring projects | uv - Astral Docs"
[2]: https://docs.astral.sh/uv/guides/projects/?utm_source=chatgpt.com "Working on projects | uv - Astral Docs"
[3]: https://docs.astral.sh/uv/concepts/projects/dependencies/?utm_source=chatgpt.com "Managing dependencies | uv - Astral Docs"
[4]: https://docs.astral.sh/uv/reference/settings/ "Settings | uv"
[5]: https://docs.astral.sh/uv/concepts/projects/sync/?utm_source=chatgpt.com "Locking and syncing | uv - Astral Docs"
[6]: https://hatch.pypa.io/1.9/config/build/ "Build - Hatch"
[7]: https://pypi.org/project/pyboy/ "pyboy · PyPI"
[8]: https://pypi.org/project/warp-lang/?utm_source=chatgpt.com "warp-lang"
[9]: https://nvidia.github.io/warp/?utm_source=chatgpt.com "NVIDIA Warp Documentation — Warp 1.11.0"
