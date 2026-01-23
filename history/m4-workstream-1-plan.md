# M4 Workstream 1 Plan (Spec Phase)

## What WS1 Covers (From `history/m4-architecture.md` + repo reality)

- Today, action semantics live in `src/gbxcule/backends/common.py` as a **9-action** mapping (includes `NOOP` + `SELECT`), and `bench/harness.py` hardcodes `0..9`.
- PyBoy backends implement press->tick->release timing inline (`src/gbxcule/backends/pyboy_single.py`, `src/gbxcule/backends/pyboy_vec_mp.py`).
- WS1 (M4) wants an explicit, versioned **action codec** + a single canonical **press/release schedule**, and to make the harness use `codec.num_actions`.
- Constraint from `CONSTITUTION.md`: spec first, deterministic/reproducible artifacts, cheap verification (tests fast), and "make invalid states unrepresentable" (don't hide action-space truth in random constants/strings).

## Workstream 1 Goals (Definition of Done)

- `src/gbxcule/core/action_codec.py` exists, is pure (no PyBoy/Warp imports), typechecked, and defines **versioned** action codecs.
- PyBoy backends validate and interpret actions exclusively via a codec (no more "action truth" in `backends/common.py`).
- Harness action generation uses `codec.num_actions` (no more hard-coded `9`).
- Every run artifact + mismatch bundle records `action_codec: {name, version, action_names}` (or equivalent) so replays are auditable.
- CPU gate stays fast: `make check` remains comfortably under the 2-minute rule.

## Key Spec Decisions to Lock Before Coding

### 1) Codec identity and backwards-compat strategy

- Recommendation: implement **two codecs** immediately:
  - `legacy_v0` (the current 9-action mapping) so existing mismatch bundles / historical runs remain interpretable.
  - `pokemonred_puffer_v0` (7 actions: `A, B, START, UP, DOWN, LEFT, RIGHT`) as the M4 canonical mapping.
- Don't silently change meaning of old `actions.jsonl`: if a replay depends on old indices, it must specify `--action-codec legacy_v0`.

### 2) Step schedule semantics (PyBoy contract in this repo)

- Keep "one step = exactly `frames_per_step` ticks" because `frames_per_step=1` is used for step-exact verify today (Warp does 1 frame).
- Define pressed timing as: press at step start; remain pressed for `release_after_frames` ticks; release; tick remaining frames.
- Enforce `0 <= release_after_frames <= frames_per_step` everywhere (PyBoy MP already enforces; PyBoy single currently doesn't).

### 3) JOYP/Warp forward-compat in the codec (even though WS3 implements JOYP)

- Define `to_joypad_mask(action) -> (dpad_mask, button_mask)` now, with a clear bit convention (e.g., 4-bit masks where bit=1 means pressed).
- This avoids later refactors when Warp JOYP lands.

## Detailed Plan (Execution Order)

### 1) Add the action codec module (pure core)

- Create `src/gbxcule/core/action_codec.py`.
- Define:
  - `ActionCodec` Protocol (or small abstract base) with `name`, `version`, `num_actions`, `action_names`, `to_pyboy_button(action) -> str | None`, `to_joypad_mask(action) -> tuple[int, int]`, and `validate_action(action)`.
  - `LegacyV0Codec` implementing the current mapping (including `NOOP=None`, includes `select`).
  - `PokemonRedPufferV0Codec` implementing 7-action mapping (no select; no noop).
  - `get_action_codec(codec_id: str) -> ActionCodec` with a fixed registry (no dynamic imports).
- Add unit tests: `tests/test_action_codec.py` asserting: stable ordering, `num_actions` correctness, `action_names` length matches, `to_pyboy_button` returns expected strings, and out-of-range actions raise.

### 2) Centralize schedule math (avoid duplicated/buggy timing)

- Add a tiny helper in core (recommended): `src/gbxcule/core/action_schedule.py` with something like `compute_press_ticks(frames_per_step, release_after_frames) -> int` and/or `pressed_at_frame(frame_idx, release_after_frames) -> bool`.
- Unit tests: boundary cases (0, 1, equal, etc).
- Goal: PyBoy single + MP workers share exactly one definition.

### 3) Refactor backend "action truth" to use codec

- Update `src/gbxcule/backends/common.py`:
  - Stop defining canonical mappings as module-level constants. Keep legacy constants only if you need them temporarily, but make codec the source of truth.
  - Add/standardize a backend-visible property: `num_actions: int` (and ideally `action_codec: dict[name, version]`) so tests don't need to import global constants.
- Update `src/gbxcule/backends/__init__.py` exports accordingly (prefer exporting codec helpers over old constants).

### 4) Update PyBoySingleBackend to use codec + schedule

- Edit `src/gbxcule/backends/pyboy_single.py`:
  - Constructor accepts `action_codec: str = "legacy_v0"` (or `"pokemonred_puffer_v0"` if you're ready to flip defaults) and resolves it once.
  - Validate `release_after_frames <= frames_per_step` at init (fail fast).
  - `action_spec.meaning` becomes something like `action index [0, {num_actions}) ({codec.name}@{codec.version})`.
  - In `step()`: validate action via codec; map to PyBoy button; apply schedule using the shared helper.

### 5) Update PyBoyVecMpBackend similarly (including worker process)

- Edit `src/gbxcule/backends/pyboy_vec_mp.py`:
  - Add `action_codec` to `PyBoyMpConfig` (validated as known string).
  - Pass codec id into `_worker_main(...)` and resolve codec inside the worker (avoid pickling codec objects).
  - Replace `action_to_button(action)` usage with `codec.to_pyboy_button(action)`.

### 6) Update harness: generator range + artifact recording + replay

- Edit `bench/harness.py`:
  - Add CLI flag `--action-codec` with explicit choices (`legacy_v0`, `pokemonred_puffer_v0`).
  - Resolve codec once per run; pass it into `create_backend(...)` and into action generation.
  - Change `generate_actions(...)` signature to accept `num_actions` (or codec id) so it can generate in `[0, num_actions)`. Update `tests/test_harness.py` accordingly.
  - Record in artifacts (single-run + scaling + mismatch): `action_codec: {name, version, action_names}`.
  - Update mismatch repro generation to include `--action-codec ...` so bundles remain hermetic even after defaults change.

### 7) Update tests and Make targets to match the new contract

- Update `tests/conftest.py` compliance suite:
  - Stop importing `ACTION_NOOP`, `NUM_ACTIONS`.
  - Use `backend.num_actions` for range tests; use actions like `0` and `min(1, num_actions-1)` for "valid action" tests.
  - For "too large", use `backend.num_actions` directly.
- Update `tests/test_pyboy_vec_mp_backend.py` to avoid hardcoded action indices like `7` if `num_actions` might be 7; derive from codec/backends.
- Update Makefile verify targets if you decide to switch default codec away from legacy (otherwise old verify flows can break if `NOOP` disappears). At minimum, enforce `--release-after-frames 0` for any `--frames-per-step 1` verify profile if the action space has no true noop.

## Validation Checklist (What You Run to Prove WS1 Is Done)

- `make check` (must remain fast).
- `make verify` and `make verify-smoke` (ensures step semantics didn't drift).
- One mismatch bundle smoke (`make verify-mismatch`) and confirm bundle metadata includes `action_codec` and `repro.sh` includes `--action-codec`.

## Main Risks / Footguns (Callouts)

- Default-changing without recording codec will break old replays; mitigate by always recording and always including `--action-codec` in repro scripts.
- `release_after_frames > frames_per_step` currently silently misbehaves in `pyboy_single`; WS1 should make it impossible (raise early).
- If you remove `NOOP` entirely, `--action-gen noop` stops being semantically "noop"; that's fine if documented, but don't let it create hidden frame-count changes (validate timing).

