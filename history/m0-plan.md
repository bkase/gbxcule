# m0 Plan: Env0 Pixel Smoke (Pokemon Red, 24 frames/step)

This plan is derived directly from:

- `history/simple-rl.md` (project RL direction + non-negotiable constraints)
- `CONSTITUTION.md` (engineering spec: correctness, verifiability, speed, docs-as-code)

It also incorporates the user constraint: **do not add a NOOP action**; keep actions as they are.

## m0 Objective

Prove we can produce a meaningful **4-shade** (`uint8` values `0..3`) **160x144** env0 framebuffer **every step** at **24 frames/step**, and do it with a fast, deterministic verification gate.

This is explicitly the “Milestone 0 — Single-env pixel smoke” from `history/simple-rl.md`.

## Spec (Non-Negotiables) To Enforce In m0

### From `history/simple-rl.md`

- **Everything on GPU**: env stepping (Warp/CUDA) + policy/value forward/backward (PyTorch CUDA).
  - m0 is a smoke milestone, so we can allow CPU for a deterministic CI gate, but m0 must also support running on CUDA for the real intended path.
- **No RAM reads for the policy**: the policy must not consume RAM features (only pixels).
  - m0 should not introduce any “RAM-to-policy” pathway. Debugging scripts may read memory, but m0 should not depend on it.
- **24 frames per step**: action frequency is fixed for the project.
  - m0 scripts/tests must treat `frames_per_step != 24` as a configuration error (hard fail), not “best effort”.
- **Minimal + incrementally testable**: every milestone has a fast deterministic gate.
- **Observation**: a **4-shade** screen representation consistent with the existing renderer/harness logic.
  - For m0 we can use the existing env0 renderer output: `read_frame_bg_shade_env0()` which returns a `SCREEN_H * SCREEN_W` buffer of shade indices.

### From `CONSTITUTION.md`

- **Correctness by construction**: validate inputs and make invalid states unrepresentable where practical.
  - Validate action range `[0, 7)` and `frames_per_step == 24` early.
- **Functional core, imperative shell**:
  - Pure helper functions for hashing/quantizing/PNG conversion; CLI does IO and orchestration.
- **Unidirectional data flow**: keep state updates explicit and testable (Action -> Step -> Frame -> Artifact).
- **Verifiable rewards**: automated checks with exit 0/1 immediately after code is written.
  - For m0 this means a pytest gate (and optionally a CUDA parity gate).
- **Latency budget**: keep the m0 gate fast (seconds); do not add slow end-to-end checks.
- **Supply chain minimalism**: prefer existing deps/utilities already in the repo.
- **Docs-as-code**: the plan and its acceptance criteria live in-repo and stay consistent with tooling.

## User Constraint

- Keep the action space exactly as-is:
  - `pokemonred_puffer_v0` codec with 7 discrete actions:
    - `0=A, 1=B, 2=START, 3=UP, 4=DOWN, 5=LEFT, 6=RIGHT`
  - Do **not** add a NOOP action.

## Deliverables

1. **CLI smoke tool** that:

- Loads a known `.state` for Pokemon Red (`red.gb`).
- Steps env0 at **24 frames/step** using only existing action indices `[0..6]`.
- Renders env0 BG shades each step (`read_frame_bg_shade_env0()`).
- Writes deterministic artifacts:
  - per-step frame hashes,
  - the exact action trace used,
  - optional PNGs for quick human sanity.

1. **Fast deterministic test gate** (pytest) that:

- Runs a short fixed action trace from a fixed state.
- Asserts framebuffer invariants (shape, shade range, non-constant).
- Asserts deterministic frame-hash sequence (“snapshot test” for hashes).

1. (Optional but recommended) **CUDA parity smoke** that:

- If CUDA is available, runs the same trace on `WarpVecCudaBackend` and compares frame bytes/hashes against CPU for the same steps.
  (run this to make sure it works -- you'll need a long timeout since the compilation is slow, but don't add it to make check because the GPU tests are too slow)

## Implementation Steps

### 0) Track the work (beads_rust)

- Create and claim an issue:
  - `br create --title="m0: env0 pixel smoke (BG shades @24f/step) + deterministic gate" --type=task --priority=2`
  - `br update <id> --status=in_progress`

### 1) Choose canonical ROM/state inputs for determinism

- ROM: `red.gb` (repo root).
- State: select a repo-committed Pokemon Red `.state` under `states/` that is stable and renders correctly.
  - Candidate: `states/pokemonred_bulbasaur_warp.state` (or the most reliable `.state` already used for renderer verification).
- Action trace:
  - Prefer a short fixed list of action indices in `[0..6]` that causes some screen changes.
  - If using RNG, persist the generated actions to JSONL and replay them by default.

### 2) Add the CLI smoke script

Add: `tools/pokered_pixel_smoke.py`

Minimal CLI flags (repro-oriented):

- `--backend warp_vec_cpu|warp_vec_cuda` (default `warp_vec_cpu` for broad compatibility)
- `--rom red.gb`
- `--state <path>`
- `--steps <int>` (default ~64)
- `--frames-per-step 24` (hard fail if not 24)
- `--release-after-frames <int>` (default 8; logged)
- `--actions-file <jsonl>` (replay exact actions) OR `--actions-seed <int>` (generate then persist)
- `--output-dir <dir>` and `--save-every <N>` (PNG cadence)

Behavior:

- Instantiate backend with `render_bg=True`, `frames_per_step=24`, `release_after_frames=8`.
- `reset(); load_state_file(...)`.
- For each step:
  - pick action in `[0..6]`,
  - `step()`,
  - read `frame_bg_shade_env0`,
  - compute hash (e.g. blake2b digest),
  - optionally write a PNG using a fixed palette mapping of shades to grayscale.
- Emit structured run artifacts:
  - `meta.json` (config + ROM/state hash)
  - `actions.jsonl` (or JSON list)
  - `frames.jsonl` with `{step, action, frame_hash}`

### 3) Add the pytest “verifiable gate”

Add: `tests/test_m0_pixel_smoke.py`

CPU test (always runs; fast):

- Setup `WarpVecCpuBackend(..., frames_per_step=24, release_after_frames=8, render_bg=True)`.
- Load canonical state and run a fixed trace (8–16 steps).
- Assert:
  - frame byte length == `160 * 144`
  - all shades in `0..3`
  - frame not constant (e.g. `unique_count >= 2`)
  - frame-hash list equals committed snapshot (golden).

CUDA parity test (skip if CUDA unavailable):

- Run the same trace on `WarpVecCudaBackend` and compare frames/hashes to CPU.
- Keep this a smoke test (few steps) to respect the latency budget.

Snapshot update policy:

- If the renderer changes intentionally, update golden hashes deliberately and in the same commit as the renderer change.

### 4) Optional: Makefile convenience target

Add a target such as `m0` (and optionally `m0-gpu`) to run the smoke tool with canonical inputs and write artifacts under `bench/runs/...`.

If referenced in `README.md`, ensure it exists (docs tests check that).

### 5) Verification loop (keep it fast)

- CPU gate:
  - `uv run pytest -q tests/test_m0_pixel_smoke.py`
- Full CPU suite sanity:
  - `make test` (already runs with `GBXCULE_SKIP_CUDA=1`)
- CUDA smoke on GPU machines:
  - `make check-gpu` (existing)
  - `uv run python tools/pokered_pixel_smoke.py --backend warp_vec_cuda ...`

### 6) Close out the session (repo protocol)

- `git status`
- `git add <files>`
- `br sync --flush-only`
- `git commit -m "m0: env0 pixel smoke + deterministic gate"`
- `git push`
- `br close <id> --reason="Completed"`

## Acceptance Criteria (m0 Done Means)

- Running the smoke tool produces recognizably valid Game Boy frames (not blank/noise) from the chosen Pokemon Red state.
- The CPU pytest gate is deterministic and fast, and passes reliably.
- On CUDA machines, the CUDA smoke/parity path runs successfully (and ideally matches CPU for the same trace).
- No changes to the action space: still exactly 7 actions (A, B, START, UP, DOWN, LEFT, RIGHT), and no NOOP added.

## Non-Goals (Explicitly Out of Scope for m0)

- Multi-env renderer (that is later milestones).
- Downsampled 80x72/84x84 buffers for training.
- Torch bridge, PPO loop, goal template capture, reward shaping.
