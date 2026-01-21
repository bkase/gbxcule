# Epic 3 — Backend contract and shared types

## Epic intent

Create a **single, strict backend interface** and shared types so the harness can run any backend (PyBoy single, PyBoy MP, Warp CPU/GPU) uniformly, with:

- stable shapes/dtypes for arrays,
- stable CPU-state schema for verification,
- JSON-serializable specs for run artifacts and mismatch bundles.

This epic is the “type system” of the project.

## Architecture constraints and invariants

### MUST

1. **Backend interface is stable and minimal**:
   - `reset(seed)`, `step(actions)`, `get_cpu_state(env_idx)`, `close()`.

2. **Unidirectional step**: `actions → step → outputs` (no hidden mutation exposed).
3. **Batch semantics**:
   - For vector backends, `actions` is batched over `num_envs`.
   - For `pyboy_single`, batch size is effectively 1, but _types still behave like batch_.

4. **Shapes and dtypes are explicit and validated at boundaries** (cheap checks):
   - actions: `int32[num_envs]`
   - obs: `float32[num_envs, obs_dim]` (obs_dim configurable; default can be 32)
   - reward: `float32[num_envs]`
   - done/trunc: `bool[num_envs]`

5. **CPU state schema is canonical**:
   - regs: `PC, SP, A, F, B, C, D, E, H, L`
   - derived flags: `{z,n,h,c}`
   - counters: optional (present if available; never silently wrong)

### SHOULD

- Common types are **pure** (no side effects), live in `src/gbxcule/backends/common.py`.
- JSON serialization uses stdlib (`json`) with explicit conversions (no pydantic).
- Validation is “cheap and clear”: raise `ValueError` with actionable messages.

## Deliverables (files)

### Create/modify

- `src/gbxcule/backends/common.py` (**main deliverable**)
- `src/gbxcule/backends/__init__.py` (export protocol/types)
- `tests/test_backends_common.py` (new)
- (Optional but recommended) `src/gbxcule/core/signatures.py` minor additions:
  - stable hashing helper for CPU state dict (not goldens; just determinism checks)

## Story 3.1 — Define backend interface contract

### Design: `common.py` contents

Implement:

#### 1) Core literals and aliases

- `Device = Literal["cpu", "cuda"]`
- `Stage = Literal["emulate_only", "full_step", "reward_only", "obs_only"]`
- `NDArrayF32`, `NDArrayI32`, etc (via `numpy.typing.NDArray`)

#### 2) ArraySpec and BackendSpec dataclasses (pure + JSON-friendly)

```python
@dataclass(frozen=True)
class ArraySpec:
    shape: tuple[int, ...]
    dtype: str  # e.g. "float32"
    meaning: str  # short human description

@dataclass(frozen=True)
class BackendSpec:
    name: str
    device: Device
    num_envs: int
    action: ArraySpec
    obs: ArraySpec
```

Include `to_json_dict()` methods (or a separate `as_json()` function) that convert tuples to lists and keep dtype as string.

#### 3) StepOutput dataclass (internal convenience)

Even if the protocol returns tuples, having a dataclass makes validation and testing cleaner.

```python
@dataclass(frozen=True)
class StepOutput:
    obs: NDArray[np.float32]
    reward: NDArray[np.float32]
    done: NDArray[np.bool_]
    trunc: NDArray[np.bool_]
    info: dict[str, Any]
```

#### 4) CPU state schema typing

Define a `TypedDict` to standardize keys.

```python
class CpuFlags(TypedDict):
    z: int; n: int; h: int; c: int

class CpuState(TypedDict, total=False):
    pc: int; sp: int
    a: int; f: int; b: int; c: int; d: int; e: int; h: int; l: int
    flags: CpuFlags
    instr_count: int | None
    cycle_count: int | None
```

Rationale: `total=False` allows counters to be omitted or `None` without breaking callers.

#### 5) The Protocol

```python
class VecBackend(Protocol):
    name: str
    device: Device
    num_envs: int
    action_spec: ArraySpec
    obs_spec: ArraySpec

    def reset(self, seed: int | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]: ...
    def step(self, actions: NDArray[np.int32]) -> tuple[
        NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_], dict[str, Any]
    ]: ...
    def get_cpu_state(self, env_idx: int) -> CpuState: ...
    def close(self) -> None: ...
```

#### 6) Boundary validation helpers (fast)

- `validate_actions(actions, num_envs)`:
  - ensures `ndim==1`, `len==num_envs`, dtype is integer, cast to int32 if safe.

- `validate_step_output(output, num_envs, obs_dim)`:
  - used by tests; backends can call it in debug mode.

Keep these helpers in `common.py` so every backend is consistent.

### Implementation tasks (ordered)

1. **Write `common.py` skeleton** with literals, dataclasses, protocol.
2. Add helper functions:
   - `as_i32_actions(actions, num_envs) -> np.ndarray[int32]`
   - `empty_obs(num_envs, obs_dim) -> np.ndarray[float32]`

3. Add `__init__.py` exports so imports are clean:
   - `from .common import VecBackend, BackendSpec, ArraySpec, StepOutput, CpuState, Device, Stage`

4. Run `pyright` over `src/gbxcule/backends/common.py` in strict mode.

## Story 3.2 — Standardize run metadata schema

Even though harness implementation is in another epic, **the schema must be defined here** so all downstream code shares one contract.

### Design: RunArtifact dataclasses

Add in `common.py` (or `src/gbxcule/core/signatures.py` if you prefer “core owns schema”; I’d keep it in `common.py` for now because it’s harness-facing).

Define:

```python
@dataclass(frozen=True)
class SystemInfo:
    platform: str
    python: str
    numpy: str
    pyboy: str | None
    warp: str | None
    cpu: str | None
    gpu: str | None

@dataclass(frozen=True)
class RunConfig:
    backend: str
    device: Device
    rom_path: str
    rom_sha256: str
    stage: Stage
    num_envs: int
    frames_per_step: int
    release_after_frames: int
    steps: int
    warmup_steps: int
    actions_seed: int
    sync_every: int | None

@dataclass(frozen=True)
class RunResult:
    measured_steps: int
    seconds: float
    total_sps: float
    per_env_sps: float
    frames_per_sec: float
```

Plus:

- `RESULT_SCHEMA_VERSION = 1`
- `def run_artifact_to_json_dict(...) -> dict` (pure conversion).

### Implementation tasks (ordered)

1. Add dataclasses + schema version constant.
2. Add `to_json_dict()` conversions (no fancy serialization magic).
3. Add a tiny unit test proving JSON roundtrip works and keys are stable.

## Tests for Epic 3

### `tests/test_backends_common.py`

Keep these fast (no PyBoy import required).

1. **ArraySpec / BackendSpec JSON serialization**
   - `json.dumps(spec.to_json_dict())` succeeds.

2. **Action validation**
   - wrong shape raises
   - float dtype raises
   - int64 casts safely to int32

3. **CpuState schema sanity**
   - a helper `flags_from_f(f)` (can live in `common.py`) is tested with hypothesis:
     - random `f` → `flags` bits match `f & 0xF0` expectation.

### Exit conditions

- `pytest` must fail with clear messages when a backend returns wrong dtype/shape.

## Acceptance checklist for Epic 3

- `common.py` defines the protocol + shared dataclasses + validation helpers.
- All types pass `pyright` strict (at least for `common.py`).
- Tests verify JSON-serializability and basic boundary validation.
- No new heavy dependencies introduced.

---

# Epic 4 — Micro-ROM generation + suite definition

## Epic intent

Create **license-safe, deterministic micro-ROMs** that:

- are generated locally,
- run in PyBoy headless,
- stress targeted behaviors (ALU loop, WRAM read/write loop),
- and can be referenced by `suite.yaml`.

This epic creates the “known-good workloads” for correctness + perf.

## Architecture constraints and invariants

### MUST

1. ROMs are **deterministic**: same generator version → identical bytes.
2. ROM headers are **valid**:
   - correct Nintendo logo bytes and checksums so the ROM is legitimate. ([gbdev.io][1])

3. Output is written to `bench/roms/out/` by default (gitignored).
4. There are at least two ROMs:
   - `ALU_LOOP.gb`
   - `MEM_RWB.gb`

### SHOULD

- ROM builder is a pure library with a CLI wrapper.
- Tests verify:
  - files exist,
  - headers checksums are correct,
  - PyBoy can step them without crashing in `window="null"` mode. ([docs.pyboy.dk][2])

## Deliverables (files)

### Create/modify

- `bench/roms/build_micro_rom.py`
- `bench/roms/suite.yaml`
- `tests/test_micro_roms.py` (already listed; implement fully)

## Story 4.1 — Implement micro-ROM builder

### ROM format decisions (minimal, correct)

- ROM size: 32 KiB (`rom_size_code = 0x00`, “2 banks no banking”). ([gbdev.io][1])
- Cartridge type: `0x00` (ROM ONLY).
- Entry point at `0x0100`: `NOP; JP 0x0150` (standard practice). ([gbdev.io][1])
- Program bytes placed at `0x0150`.
- Nintendo logo bytes at `0x0104–0x0133`. ([gbdev.io][1])
- Header checksum at `0x014D`: computed per Pan Docs algorithm. ([gbdev.io][1])
- Global checksum at `0x014E–0x014F`: big-endian sum of all bytes except those two. ([gbdev.io][1])

### Implementation design (`build_micro_rom.py`)

Provide:

#### 1) Low-level helpers (pure)

- `compute_header_checksum(rom: bytes) -> int`
- `compute_global_checksum(rom: bytes) -> int`
- `sha256_bytes(b: bytes) -> str`

#### 2) `build_rom(title: str, program: bytes) -> bytes`

- Constructs a `bytearray(32768)`.
- Writes header + program.
- Computes and patches checksums.
- Returns immutable `bytes`.

#### 3) Two program payloads (deterministic)

**ALU_LOOP** (ALU-heavy + branch)

- Byte sequence (at `0x0150`) that loops forever doing ALU + `JR NZ`.
- Keep it small and deterministic.

**MEM_RWB** (WRAM read/write heavy)

- Loop that writes A to `(HL)` where `HL` ranges over `0xC000..0xC0FF`, reads back, does arithmetic, repeats forever.

#### 4) Atomic write

- Write to `tmp` then `rename` into final path so partial writes never exist.

#### 5) CLI wrapper

- `python bench/roms/build_micro_rom.py --out-dir bench/roms/out`
- Prints: paths + sha256 + size.

## Story 4.2 — Add micro-ROM suite file

### `bench/roms/suite.yaml` design

Keep it explicit and stable:

```yaml
suite_version: 1
defaults:
  frames_per_step: 24
  release_after_frames: 8
roms:
  - id: ALU_LOOP
    path: bench/roms/out/ALU_LOOP.gb
    description: "Deterministic ALU-heavy tight loop (E1-ish)."
    frames_per_step: 24
    release_after_frames: 8
  - id: MEM_RWB
    path: bench/roms/out/MEM_RWB.gb
    description: "WRAM read/write heavy loop (E3-ish)."
    frames_per_step: 24
    release_after_frames: 8
```

Even if harness later enforces global step config, include these fields now so the suite is self-describing.

## Story 4.3 — Tests for ROM generation

### `tests/test_micro_roms.py`

Tests must be fast and robust on macOS/cloud CPU-only.

Recommended test structure:

1. **Build ROMs into a temp dir**
   - Call `build_all(out_dir=tmp_path)` (you should implement this helper).
   - Assert both files exist and are non-zero.

2. **Validate header checksums**
   - Recompute header checksum and assert the stored byte matches.
   - Recompute global checksum and assert matches. ([gbdev.io][1])

3. **PyBoy headless smoke**
   - `pyboy = PyBoy(path, window="null", sound_emulated=False, log_level="ERROR")` ([docs.pyboy.dk][2])
   - `pyboy.set_emulation_speed(0)` (no speed limit) ([PyPI][3])
   - `assert pyboy.tick(120, False)` (step 120 frames without rendering) ([docs.pyboy.dk][4])
   - `pyboy.stop(False)` (don’t write save files) ([docs.pyboy.dk][4])

## Acceptance checklist for Epic 4

- `build_micro_rom.py` generates `ALU_LOOP.gb` and `MEM_RWB.gb` deterministically.
- `suite.yaml` exists and references those ROMs with frames/action parameters.
- Tests validate:
  - file existence,
  - checksum correctness,
  - PyBoy can tick them headless without crashing.

---

# Epic 5 — Reference baseline backend: `pyboy_single`

## Epic intent

Implement the trusted single-env PyBoy backend that:

- conforms to `VecBackend`,
- is used as the oracle for early correctness,
- and provides a reliable single-env baseline for benchmarks.

## Architecture constraints and invariants

### MUST

1. `pyboy_single` implements the backend protocol exactly.
2. Runs headless by default (`window="null"`) and can tick many frames without rendering. ([docs.pyboy.dk][2])
3. `get_cpu_state(0)` returns a stable register schema (PC/SP/A/F/B/C/D/E/H/L + flags).
4. Output arrays are batched:
   - `obs.shape == (1, obs_dim)`
   - `reward/done/trunc.shape == (1,)`

5. On any inability to read a required field, **raise a clear error** (no silent fallback).

### SHOULD

- Use PyBoy v2+ API intentionally; pin `pyboy>=2,<3` in deps to avoid API drift. ([GitHub][5])
- Disable speed limit and avoid unnecessary features:
  - `set_emulation_speed(0)` ([PyPI][3])
  - `sound_emulated=False`
  - `tick(count, render=False, sound=False)` ([docs.pyboy.dk][4])

## Deliverables (files)

### Create/modify

- `src/gbxcule/backends/pyboy_single.py`
- `tests/test_pyboy_single_backend.py` (new)
- (Optional) a tiny helper in `src/gbxcule/backends/common.py` for action mapping constants

## Story 5.1 — Implement `pyboy_single` backend

### API and config decisions

#### Constructor / init config

Even if harness later owns config parsing, implement a clean constructor:

```python
class PyBoySingleBackend:
    def __init__(
        self,
        rom_path: str,
        *,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        obs_dim: int = 32,
        log_level: str = "ERROR",
    ): ...
```

- `obs_dim` is configurable; default 32 is a safe placeholder until you lock it.

#### Action representation

- Input: `actions: np.ndarray[int32]` of shape `(1,)`.
- Define a deterministic mapping from int → button string:
  - `0: noop`
  - `1: up`, `2: down`, `3: left`, `4: right`
  - `5: a`, `6: b`, `7: start`, `8: select`

- If action is out of range: raise `ValueError` (fail loud).

#### Step semantics (24 frames; release after 8)

Implementation options:

**Option A (fast + simple)**: use `pyboy.button(button, delay=release_after_frames)` then one `tick(frames_per_step, render=False, sound=False)`. PyBoy’s `button` supports delayed release and `tick(count, render=...)` advances by `count` frames. ([docs.pyboy.dk][2])

**Option B (max explicit)**: `button_press`, tick 8 frames, `button_release`, tick remaining frames. This is slightly more overhead but extremely explicit. ([docs.pyboy.dk][2])

For M0/M1 correctness and clarity, I recommend **Option B** unless profiling shows it matters.

### Implementation tasks (ordered)

1. **Backend skeleton**
   - Add class with required attributes:
     - `name = "pyboy_single"`
     - `device = "cpu"`
     - `num_envs = 1`
     - `action_spec`, `obs_spec` from `common.py`

2. **Emulator lifecycle**
   - `_make_pyboy()` helper:
     - `PyBoy(rom_path, window="null", sound_emulated=False, log_level=...)` ([docs.pyboy.dk][2])
     - `set_emulation_speed(0)` ([PyPI][3])

   - `close()` calls `pyboy.stop(False)` to discard saves. ([docs.pyboy.dk][4])

3. **`reset(seed)`**
   - Stop previous emulator (if any), create a fresh one.
   - Return `(obs, info)` where:
     - `obs` is `(1, obs_dim)` float32
     - `info` includes `{"seed": seed}` and optionally ROM sha.

4. **`step(actions)`**
   - Use `as_i32_actions()` from `common.py`.
   - Apply action semantics (noop or button press/release).
   - Tick exactly `frames_per_step` frames with `render=False` and `sound=False`. ([docs.pyboy.dk][4])
   - Return:
     - `obs`: float32 (1, obs_dim)
     - `reward`: float32 (1,) (zeros for now)
     - `done`: bool (1,) (True only if PyBoy returns False)
     - `trunc`: bool (1,) (False for now)
     - `info`: minimal dict

5. **`get_cpu_state(0)`**
   - Read via `pyboy.register_file` (PC, SP, A, F, B, C, D, E, HL). ([docs.pyboy.dk][2])
   - Derive H/L from HL.
   - Derive flags from F (Z=0x80, N=0x40, H=0x20, C=0x10).
   - Return dict matching `CpuState` schema.

6. **Observation encoding**
   You chose “small fixed feature vector”.
   For now, keep it minimal and stable:
   - first N slots: normalized registers (e.g., pc/65535, sp/65535, a/255, …)
   - remaining slots: zeros
   - dtype float32

This avoids coupling obs to Pokémon-specific memory early while still meeting the contract.

## Story 5.2 — State snapshot normalization

The goal is: `get_cpu_state` output is stable across time and machines.

### Requirements

- Always return integer values (Python ints).
- Always include the same keys for regs + flags.
- If you can’t provide counters, either:
  - omit them entirely, or
  - include them as `None` (but be consistent).

Recommended: include `instr_count: None`, `cycle_count: None` explicitly so later code doesn’t branch on key presence (but either is fine as long as the verification code handles it).

## Tests for Epic 5

### `tests/test_pyboy_single_backend.py` (fast, CPU-only)

1. **Smoke: init/reset/close**
   - Build ROMs into temp dir using the Epic 4 builder.
   - Create backend, call reset, close.

2. **Step shapes/dtypes**
   - `obs.dtype == float32`, shape `(1, obs_dim)`
   - `reward.dtype == float32`, shape `(1,)`
   - `done.dtype == bool`, shape `(1,)`

3. **CPU state keys and flag derivation**
   - Ensure keys exist: pc/sp/a/f/b/c/d/e/h/l/flags
   - Ensure `flags` matches bits in `f`.

4. **Determinism smoke**
   - Run two fresh resets on the same ROM and do:
     - 10 noop steps
     - compare `get_cpu_state()` at end

   - If PyBoy introduces nondeterminism here, keep the test but only assert _basic invariants_ (keys present, types correct) and move strict determinism checks to micro-ROM + Warp comparisons later.

## Acceptance checklist for Epic 5

- `pyboy_single` conforms to the backend protocol.
- Runs headless with `window="null"` and can tick without rendering. ([docs.pyboy.dk][2])
- `get_cpu_state(0)` returns stable regs + flags.
- Tests verify shapes/dtypes and state schema.
- No save files are written during tests (`stop(False)`). ([docs.pyboy.dk][4])

---

## Practical “done means done” gates for these three epics

When Epics 3–5 are complete, you should be able to run (conceptually; actual make targets are from other epics):

- `pytest -q` on macOS/cloud CPU-only:
  - validates shared types,
  - generates micro-ROMs in temp,
  - loads them into PyBoy headless and ticks frames,
  - exercises `pyboy_single`.

And every failure should:

- be a crisp exception (shape mismatch, dtype mismatch, missing key),
- not require reading logs for 5 minutes.

If you want, next I can produce:

- the exact `common.py` API layout (with function signatures + minimal code skeletons),
- and the precise byte-level program payloads for the two ROMs (ALU_LOOP and MEM_RWB) aligned with the plan above.

[1]: https://gbdev.io/pandocs/The_Cartridge_Header.html "The Cartridge Header - Pan Docs"
[2]: https://docs.pyboy.dk/ "pyboy API documentation"
[3]: https://pypi.org/project/pyboy/ "pyboy · PyPI"
[4]: https://docs.pyboy.dk/?utm_source=chatgpt.com "pyboy API documentation"
[5]: https://github.com/Baekalfen/PyBoy/wiki/Migrating-from-v1.x.x-to-v2.0.0 "Migrating from v1.x.x to v2.0.0 · Baekalfen/PyBoy Wiki · GitHub"
