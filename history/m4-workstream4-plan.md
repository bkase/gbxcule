# M4 Workstream 4 Plan — Real PufferLib baselines (PyBoy)

> Source spec: `history/m4-architecture.md` (sections 6 and 10; WS4 = “Pufferlib baselines”).
>
> This rewrite assumes the explicit decision: **Yes, we use real PufferLib** even if it’s harder to install on DGX Spark.

## 0) Scope & intent (what WS4 is for)

WS4 exists to produce **honest, reproducible CPU baselines** that:

- match the Pokémon RL community’s **press → delayed release → tick** semantics,
- use **PufferLib’s vectorization** for the scaling baseline (not homegrown MP),
- and integrate into the existing `bench/harness.py` artifact pipeline so we can compare against `warp_vec_cuda` without “apples-to-oranges” ambiguity.

Non-goals for WS4:

- Not responsible for Warp kernel correctness (that’s other M4 workstreams).
- Not responsible for full Pokémon environment logic (we use micro-ROMs and “user provides ROM” paths).
- Not responsible for making puffer installs “pleasant” on all platforms — but we must make it **work** on DGX Spark with explicit docs.

## 1) Primary reference implementation (local, working)

We will treat `~/Documents/pokemonred_puffer` as the **reference for “how puffer + PyBoy actually works in practice”**.

Key facts we must match (from that repo):

### 1.1 PufferLib pin that works

`~/Documents/pokemonred_puffer/pyproject.toml` uses:

- `pufferlib @ git+https://github.com/thatguy11325/PufferLib.git@1.0`

And the installed package in `~/Documents/pokemonred_puffer/.venv` is:

- `pufferlib==1.0.1`

### 1.2 Canonical action indices (important)

`~/Documents/pokemonred_puffer/pokemonred_puffer/environment.py` defines:

- `VALID_ACTIONS_STR = ["down", "left", "right", "up", "a", "b", "start"]`
- `VALID_ACTIONS = [PRESS_DOWN, PRESS_LEFT, PRESS_RIGHT, PRESS_UP, PRESS_A, PRESS_B, PRESS_START]`
- `VALID_RELEASE_ACTIONS` in the same order
- `ACTION_SPACE = spaces.Discrete(len(VALID_ACTIONS))` ⇒ **7 actions**

So the canonical action index order is:

0. Down
1. Left
2. Right
3. Up
4. A
5. B
6. Start

No Select. No Noop.

### 1.3 Canonical step schedule (press/release/tick)

`~/Documents/pokemonred_puffer/pokemonred_puffer/environment.py:run_action_on_emulator` does:

- `pyboy.send_input(VALID_ACTIONS[action])`
- `pyboy.send_input(VALID_RELEASE_ACTIONS[action], delay=8)`
- `pyboy.tick(self.action_freq - 1, render=False)`
- later it does a final `pyboy.tick(1, ...)` (“one last tick”) plus extra ticks for Pokémon-specific determinism loops.

For **our WS4 baselines**, we will:

- match the core schedule (press + delayed release + tick) exactly,
- and explicitly **not** implement Pokémon’s extra control-return loop (that depends on Pokémon RAM symbols like `wJoyIgnore`) for micro-ROM baselines.

## 2) WS4 contract (must be explicit + testable)

### 2.1 Terminology mapping

- `frames_per_step` in GBxCuLE ⇔ `action_freq` in pokemonred_puffer.
- `release_after_frames` in GBxCuLE ⇔ the `delay=` argument passed to `send_input` (default 8).

### 2.2 Step schedule (canonical)

For a single env, one `step(action)` must do:

1. `send_input(PRESS[action])`
2. `send_input(RELEASE[action], delay=release_after_frames)`
3. advance exactly `frames_per_step` frames total, implemented as:
   - `tick(frames_per_step - 1, render=False)` (if `frames_per_step > 1`)
   - `tick(1, render=False)`

Why the split?

- It matches the reference shape (`tick(action_freq - 1)` + final `tick(1)`), and it avoids the edge case where `frames_per_step=1` would otherwise mean “tick(0) and do nothing”.

Constraints:

- `frames_per_step >= 1`
- `0 <= release_after_frames` (if larger than `frames_per_step`, it is still well-defined for `send_input(..., delay=N)` but we should **reject** it for now to avoid undefined semantics across steps).

### 2.3 Action mapping authority (dependency on WS1)

WS4’s baseline must consume actions using the canonical order above.

Long-term: the mapping should live in WS1’s `ActionCodec` (so all backends share one truth).

Short-term (if WS1 isn’t done yet): WS4 can implement a local mapping for puffer backends, but:

- it must be recorded in artifacts (`action_codec` placeholder with `name="pokemonred_puffer_compat"` and `version="draft"`),
- and WS1 must later replace it.

### 2.4 Benchmark honesty fields (artifacts)

Every artifact that uses puffer baselines must record:

- `pufferlib_version`
- `pufferlib_direct_url` (PEP 610 if present)
- `vec_backend`: `puffer_serial` | `puffer_mp_sync` | (optional later) `puffer_mp_async`
- `action_schedule`: `{frames_per_step, release_after_frames, tick_split: [frames_per_step-1, 1]}`
- `action_codec`: `{name, version, num_actions, mapping_order}`

## 3) Dependency strategy (real pufferlib, DGX-friendly)

### 3.1 The compatibility problem we must solve up front

GBxCuLE currently locks to `numpy==2.4.1` (see `uv.lock`), while the known-good pufferlib build in the reference stack is `pufferlib==1.0.1`, which requires `numpy==1.23.3`.

This means WS4 must resolve one of:

- **Option 1: project-wide downgrade** (make GBxCuLE compatible with `numpy==1.23.3`).
- **Option 2: isolate puffer baselines in a separate environment** (recommended if Option 1 breaks Warp/CUDA or other core deps).

WS4 should explicitly decide this by a quick experiment:

1. add pufferlib pin
2. run `uv lock`
3. run `make check` and `make verify` on CPU

If it passes, Option 1 is acceptable. If not, we implement Option 2.

### 3.2 Pin the same pufferlib as the working reference

Use the same known-good pin as `pokemonred_puffer`:

- `pufferlib @ git+https://github.com/thatguy11325/PufferLib.git@1.0`

Then (after verifying) upgrade it to a **commit SHA pin** for reproducibility:

- `pufferlib @ git+https://github.com/thatguy11325/PufferLib.git@<sha>`

### 3.3 Make puffer optional (but first-class)

Implementation approach:

- Add a new uv dependency group, e.g. `[dependency-groups] puffer = [...]`.
- Gate puffer backends behind lazy imports so `make check` doesn’t require puffer installed.
- Add a DGX-oriented setup target:
  - `make setup-puffer` → installs puffer group (and documents any OS deps).

### 3.4 DGX Spark prerequisites (document explicitly)

Based on pufferlib’s dependency surface (opencv, pygame, cython extension build), document a minimal DGX setup checklist:

- toolchain: `gcc/g++`, `python3-dev`/headers, `make`
- runtime libs often needed by opencv/pygame wheels:
  - `libgl1`, `libglib2.0-0` (opencv)
  - `libsdl2-2.0-0` (pygame), plus common X/alsa stubs if required

WS4 deliverable includes: a short `docs/DGX_PUFFER_SETUP.md` (or README section) listing the exact commands.

## 4) Implementation plan (what we build in GBxCuLE)

### 4.1 Add a tiny Gymnasium env wrapper for PyBoy micro-ROMs

Goal: pufferlib expects `GymnasiumPufferEnv(env=some_gym_env)`; we need a minimal gym env that wraps a PyBoy instance.

Add a module like:

- `src/gbxcule/backends/pyboy_gym_env.py` (name bikeshed)

Requirements:

- Implements `gymnasium.Env` (`reset`, `step`, `close`).
- Uses our ROM path + boot ROM (`bench/roms/bootrom_fast_dmg.bin`).
- Observation space: keep it small and cheap:
  - either zeros (fastest baseline) or the same register-feature vector we already compute in `pyboy_single` (more “apples-to-apples” if we later compare obs stages).
- Action space: `gymnasium.spaces.Discrete(7)` with the canonical pokemonred_puffer order.
- Step semantics: use the canonical schedule in section 2.2 using `PyBoy.send_input` + `tick(count=...)`.

**DoD**

- Single env can step a micro-ROM deterministically for N steps without leaks/crashes.

### 4.2 Implement `pyboy_puffer_single` (trusted oracle backend)

Add:

- `src/gbxcule/backends/pyboy_puffer_single.py`

Behavior:

- Single env oracle (like `pyboy_single`) but:
  - uses the canonical puffer schedule (send_input with delayed release)
  - uses canonical action mapping order
- Must support:
  - `get_cpu_state(0)`
  - `read_memory(0, lo, hi)`

Why this backend exists:

- It gives us a correctness oracle whose input timing matches the future “real baseline”.

**DoD**

- `bench/harness.py --verify --ref-backend pyboy_puffer_single ...` works on micro-ROMs (including `frames_per_step=1`).

### 4.3 Implement `pyboy_puffer_vec` (PufferLib vectorized scaling baseline)

Add:

- `src/gbxcule/backends/pyboy_puffer_vec.py`

Behavior:

- Uses real pufferlib:
  - constructs `env_creator` that returns `pufferlib.emulation.GymnasiumPufferEnv(env=PyBoyMicroRomGymEnv(...))`
  - vectorizes via `pufferlib.vector.make(...)`
- Default vectorization mode for benchmark parity:
  - `backend=pufferlib.vector.Multiprocessing`
  - `batch_size=num_envs` (synchronous stepping; matches existing harness semantics)
  - `zero_copy=True`
- Worker selection:
  - choose `num_workers` such that `num_envs % num_workers == 0` and `num_workers <= os.cpu_count()`
  - record the chosen `num_workers` into artifacts

Interface to GBxCuLE harness (`VecBackend`):

- `reset(seed)` forwards to vecenv.reset/async_reset as appropriate.
- `step(actions)` calls `vecenv.step(actions)` (sync semantics).
- `get_cpu_state/read_memory`:
  - can be `NotImplementedError` for `pyboy_puffer_vec` (benchmark-only), by design.

Optional extension (clearly separated + recorded):

- Add a mode for “async ceiling”:
  - `batch_size < num_envs`, and benchmark loop uses `send/recv` with `env_ids`.
  - This requires harness metric changes (see 4.5).

**DoD**

- Scaling sweeps run successfully and produce believable SPS (no hidden sync bugs).

### 4.4 Wire into harness CLI + backend registry

Update `bench/harness.py`:

- register backends:
  - `pyboy_puffer_single`
  - `pyboy_puffer_vec`
- extend CLI choices for:
  - `--backend`
  - `--ref-backend`
  - `--dut-backend` (if we ever want to compare against warp)

Update artifact writing:

- extend `get_system_info()` to include pufferlib dist info, similar to existing warp provenance capture:
  - `pufferlib_version`
  - `pufferlib_dist_name`
  - `pufferlib_direct_url` (if present)

### 4.5 Benchmark protocol & SPS math (sync vs async)

For WS4 baseline parity, the default is **sync**:

- `total_env_steps = steps * num_envs`
- `total_sps = total_env_steps / seconds`

If we add an async mode later:

- `total_env_steps` becomes `sum(mask_count_per_recv)` (pufferlib returns masks)
- artifacts must record `vec_backend=puffer_mp_async` and the counting method

This prevents accidentally comparing “async partial stepping” to “sync full stepping”.

### 4.6 Make targets & docs

Add Make targets (DGX-oriented):

- `make setup-puffer` (installs puffer group; may include doc link)
- `make bench-cpu-puffer` (runs scaling sweeps with `pyboy_puffer_vec`)

Add docs:

- `docs/DGX_PUFFER_SETUP.md` (or README section) with:
  - OS package prerequisites
  - `uv sync` commands
  - smoke commands to verify pufferlib import and a 1-env run

## 5) Tests (shift-left; avoid flaky perf assertions)

### 5.1 Unit test: schedule correctness (no PyBoy required)

Create a “fake PyBoy” object that records calls:

- `send_input(event, delay=...)`
- `tick(count, render=...)`

Assert that our schedule function:

- sends press and release with correct delay
- ticks exactly `frames_per_step` total frames via split ticks
- handles edge cases (`frames_per_step=1`, `release_after_frames=0`, etc.)

### 5.2 Integration tests (guarded)

If pufferlib is installed:

- `pyboy_puffer_single` can reset + step micro-ROM for a few steps.
- `pyboy_puffer_vec` Serial backend (`pufferlib.vector.Serial`) can run 1–2 envs for a few steps (avoid multiprocessing flake in CI).

If pufferlib is not installed:

- tests should skip cleanly.

## 6) WS4 Definition of Done (explicit)

WS4 is done when:

- `pyboy_puffer_single` exists and works as a verify oracle on micro-ROMs.
- `pyboy_puffer_vec` exists and can run scaling sweeps via pufferlib Multiprocessing with synchronous semantics.
- Artifacts record enough metadata to keep comparisons honest:
  - pufferlib version/provenance
  - action mapping + schedule
  - vectorization backend + chosen worker counts
- DGX setup for puffer baselines is documented and repeatable.

