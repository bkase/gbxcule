# M4 Workstream 3 Plan — Joypad Emulation + `JOY_DIVERGE_PERSIST`

This plan is derived from:

- `history/m4-architecture.md`
- `CONSTITUTION.md` (spec-first, functional core + verifiable checks)

## Purpose

Make Warp stepping “feel like RL”: per-env inputs via JOYP (`0xFF00`) with the same press→delayed-release timing as the PyBoy backends, and a micro-ROM that **sustains** divergence so E4 benchmarks aren’t fake-convergent.

## Dependencies (Blockers / Assumptions)

- **WS2** provides the stepping-kernel ABI surface needed for input: per-env `actions`, `release_after_frames`, and persistent per-env IO state (at least `joyp_select`).
- **WS1** (action codec) is strongly recommended so JOYP mapping is canonical/versioned, but WS3 can start with the current action IDs in `src/gbxcule/backends/common.py`.

## Spec Decisions (lock before coding)

### JOYP semantics (match PyBoy/hardware)

- Writes to `0xFF00` update selection lines **P14/P15** (bits 4/5, active-low).
- Reads from `0xFF00`:
  - return bits 0–3 **active-low** for the selected group (dpad vs buttons)
  - bits 6–7 read as `1`
- Define exact behavior when **both** or **neither** group is selected (pick one consistent with hardware/PyBoy and freeze it in tests).

### Step timing contract for input

- “Pressed” is true iff:
  - `frame_idx_in_step < release_after_frames`, and
  - the env’s action corresponds to a button action (not noop)
- `frame_idx_in_step` is derived from the kernel’s `frames_done` counter (present in the `cpu_step_builder.py` skeleton).

### Action → joypad bit mapping

- Dpad bits: Right/Left/Up/Down
- Button bits: A/B/Select/Start
- Decide what to do with:
  - `NOOP` (none pressed)
  - unsupported combos (for now: no combos; treat as none pressed)
- If WS1 lands first: prefer a codec-layer `to_joypad_mask(action)`.

## Implementation Plan (Kernel + Backend)

### 1) Kernel-side JOYP state + helpers

- Add a per-env `joyp_select` buffer (u8) persisted across steps (init to `0x30`).
- Implement Warp-safe functions:
  - `joyp_write(value, joyp_select)`
  - `joyp_read(action_i, frame_idx, release_after_frames, joyp_select)`
- Implement action→(dpad_mask, button_mask) mapping:
  - either hard-coded temporarily (current actions),
  - or via WS1 codec.

### 2) Hook JOYP in load/store templates (localized)

Update `src/gbxcule/kernels/cpu_templates/loads.py`:

- In `template_ld_r8_hl`:
  - if `hl == 0xFF00`, return `joyp_read(...)` instead of `mem[...]`
- In `template_ld_hl_r8`:
  - if `hl == 0xFF00`, call `joyp_write(SRC_i, joyp_select[i])` instead of writing `mem[...]`

### 3) Warp backend plumbing (`warp_vec_*`)

Update `src/gbxcule/backends/warp_vec.py` so actions actually matter:

- Allocate device buffers:
  - `actions` (int32)
  - `joyp_select` (uint8)
- Each `step(actions)`:
  - copies actions to device
  - passes `release_after_frames` through to the kernel
- Keep CUDA path async-friendly:
  - **no sync** in `step()` unless verify/mem-read requires it.

## Implementation Plan (Micro-ROM)

### 4) Add `JOY_DIVERGE_PERSIST.gb`

Add generator in `bench/roms/build_micro_rom.py`.

ROM behavior (goal: sustained divergence + memory divergence + deterministic signature):

1. write JOYP select (read buttons + dpad)
2. read JOYP, update a persistent `mode` (keep in a register or in WRAM)
3. branch on `mode & 3` into one of 4 inner loops:
   - loop0: ALU-heavy
   - loop1: stride-1 WRAM writes
   - loop2: stride-17 WRAM writes (pattern stress)
   - loop3: branchy loop
4. write a small signature into WRAM `0xC000:0xC010` each outer iteration

Notes:

- Keep opcode surface minimal.
- Keep deterministic behavior (no timers/PPU dependence).
- Ensure the ROM actually reads `0xFF00` and branches so divergence is real.

### 5) Add to suite

Update `bench/roms/suite.yaml`:

- add a `roms:` entry for `JOY_DIVERGE_PERSIST`
- default `frames_per_step: 24`, `release_after_frames: 8`

## Likely Required CPU Opcode Additions

Add only what the ROM needs, but keep cycle-correct (frame stepping depends on cycles):

- `LD A,(HL)` (`0x7E`) so JOYP reads don’t clobber the mode register
- `AND d8` (`0xE6`) for `mode & 3`
- `JR NZ,r8` (`0x20`) (and optionally `JR Z,r8` `0x28`) for branching/loops

Important:

- Conditional jumps require correct flag semantics **and** variable cycle counts (taken vs not taken), or verify drift will occur vs PyBoy.

## Verification & Tests (fast, automated)

### Micro-ROM generation tests

Extend `tests/test_micro_roms.py`:

- deterministic bytes
- valid header/global checksums
- PyBoy headless smoke run

### Joypad correctness receipt (new verify test)

Add a verification test that runs `pyboy_single` vs `warp_vec_cpu` on `JOY_DIVERGE_PERSIST` with:

- `frames_per_step=24`, `release_after_frames=8`
- `verify_steps ≈ 16–64`
- `--mem-region C000:C010`
- deterministic non-noop actions:
  - either add a harness generator like `striped`, or
  - use `--actions-file` in the test

### Multi-env divergence proof (warp-only)

- Run `warp_vec_cpu` with `num_envs=4`
- Fixed actions: `[UP, DOWN, LEFT, RIGHT]`
- After N steps, assert `C000:C010` hashes differ across envs
  - proves per-env JOYP state + per-env actions work.

## Definition of Done

- `make roms` emits `bench/roms/out/JOY_DIVERGE_PERSIST.gb`
- `uv run pytest -q` passes (including new ROM tests)
- Deterministic verify profile passes:
  - `pyboy_single` vs `warp_vec_cpu` on `JOY_DIVERGE_PERSIST`
  - `--mem-region C000:C010`
  - RL-ish timing (`frames_per_step=24`, `release_after_frames=8`)
- Warp multi-env test shows distinct signatures per env under different actions

## Suggested PR Slicing (minimize blocking)

- **PR1:** Add missing opcodes + unit tests (no JOYP yet)
- **PR2:** JOYP emulation in templates + backend buffers (keep existing ROMs as regression)
- **PR3:** `JOY_DIVERGE_PERSIST` ROM + `suite.yaml` + verification tests (+ optional harness `striped` action generator)

