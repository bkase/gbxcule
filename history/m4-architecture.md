## 1) Canonical “PokemonRedPuffer” step semantics (what we must match)

The Pokémon RL docs are explicit about their step core:

1. send press event
2. schedule release after 8 frames
3. tick `action_freq - 1` frames (and sometimes extra ticks for determinism in Pokémon) ([Drubinstein][1])

So for **our harness contract**, we should define:

- `frames_per_step` ≡ `action_freq`
- `release_after_frames` default = `8`
- _pressed state_ applies for frames `[0, release_after_frames)` of the step

For micro-ROMs we will **not** reproduce Pokémon’s extra “wait for control back” loop (that depends on Pokémon RAM like `wJoyIgnore`), but we _will_ match the **press→delayed release→tick** schedule exactly.

**Implication for Warp**: in the kernel, “pressed_now” is a pure function of:

- `action_i` (the env’s action for this step)
- `frame_idx` inside the step (derived from `frames_done` in your existing frame loop)
- `release_after_frames`

---

## 2) Action representation: make it match the Pokémon RL stack

### 2.1 What we can assert from public sources

The paper describing Pokémon Red RL (authors include Rubinstein) describes a discrete action space that is the **Game Boy buttons** `A, B, Start, Up, Down, Left, Right` and explicitly omits Select. ([arXiv][2])

That’s consistent with the “send_input(VALID_ACTIONS[action])” style you’re aiming for. ([Drubinstein][1])

### 2.2 How I’d implement this in _your_ repo (clean + future-proof)

Introduce an explicit action codec layer, so you can:

- lock the “PokemonRedPuffer” mapping as canonical
- keep micro-ROM / toy mappings if you ever want them
- record the codec version in artifacts (so benchmarks are auditable)

**New module**: `src/gbxcule/core/action_codec.py`

- `ActionCodec` protocol:
  - `name: str`
  - `version: str`
  - `num_actions: int`
  - `to_pyboy_press(action: int) -> Any`
  - `to_pyboy_release(action: int) -> Any`
  - `to_joypad_mask(action: int) -> (dpad_mask, button_mask)` for Warp JOYP

- `PokemonRedPufferCodec` implements the mapping:
  - `A, B, START, UP, DOWN, LEFT, RIGHT` (7 actions)
  - No NOOP; “no input” is represented by choosing something benign in downstream stacks, but for _our_ benchmark we’ll drive divergence via mixed actions anyway.

Then:

- `gbxcule/backends/common.py` stops being “the action truth” and instead imports the codec’s `num_actions` and mapping.
- `bench/harness.py` records `action_codec={name, version}` in artifacts and mismatch bundles.

This makes it impossible to silently drift from the Pokémon action meaning.

---

## 3) Fused kernel architecture: stage variants via templates

### 3.1 The key design: **one stepping kernel family, four compiled variants**

You want:

- `emulate_only`: just advance CPU
- `reward_only`: CPU + reward
- `obs_only`: CPU + obs/features
- `full_step`: CPU + obs + reward

And you want to generate these by template application. Great: we already have a libcst template pipeline for opcode dispatch; we extend it with a second injection site for “epilog work”.

### 3.2 Concrete changes to `cpu_step_builder.py`

Add **one new placeholder** in the skeleton:

- `POST_STEP_DISPATCH`

…and optionally a `STAGE_CONSTANTS` placeholder if you want to inject compile-time flags.

In `_CPU_STEP_SKELETON`, after the frame loop finishes and before writing regs back, insert:

```python
POST_STEP_DISPATCH
```

Then extend the builder:

- Add `EpilogTemplate` dataclass similar to `OpcodeTemplate`:
  - `name: str`
  - `template: Callable[..., Any]`
  - `replacements: dict[str,str]`

- Add a second injector that replaces `POST_STEP_DISPATCH` with a stage-specific CST block:
  - `emulate_only`: empty block (or `pass`)
  - `reward_only`: reward template body
  - `obs_only`: obs template body
  - `full_step`: obs template body + reward template body

### 3.3 Signature shape: keep a **superset signature**

To keep backends simple and to avoid signature churn in Warp, all 4 kernels share the same args:

- CPU state buffers (mem, regs, counters)
- **actions** buffer (int32)
- **joyp_select** buffer (uint8) if you emulate selection writes
- `frames_to_run` (int32)
- `release_after_frames` (int32)
- output:
  - `reward: float32[num_envs]`
  - `obs: float32[num_envs, obs_dim]` (flat buffer ok)

Even `emulate_only` accepts reward/obs outputs but does not write them. That’s the cleanest “one ABI; multiple kernels” model.

### 3.4 Templates for reward and obs (keep code simple)

Add:

- `src/gbxcule/kernels/cpu_templates/post_step.py`
  - `template_reward_v0(...)`
  - `template_obs_v0(...)`

These templates should use only:

- a few regs
- a small fixed WRAM slice (or a couple addresses)
- simple math (sum, xor, bit tests)

So the “E4 compute” is real but not dominated by a giant reduction.

---

## 4) Joypad support: use the same action schedule + JOYP reads

### 4.1 Warp must see divergence

To get **real divergence** the ROM must actually read JOYP (0xFF00) and branch; and the Warp emulator must produce different JOYP values per env based on action.

Implementation rule (minimal IO):

- writes to `0xFF00` update selection bits (P14/P15)
- reads from `0xFF00` return low nibble based on pressed buttons + selection

Pressed state is derived from `action_i` and `frame_idx < release_after_frames` (the “delayed release” schedule). ([Drubinstein][1])

### 4.2 Where to hook JOYP in your current kernel

You already route memory reads/writes through templates:

- `template_ld_r8_hl` (reads)
- `template_ld_hl_r8` (writes)

Extend those templates:

- if `hl == 0xFF00`:
  - reads call `joyp_read(action_i, frames_done, release_after_frames, joyp_select_i)`
  - writes call `joyp_write(value, joyp_select[i])` rather than writing mem

This keeps JOYP logic localized and keeps the opcode dispatch tree clean.

---

## 5) Make the divergence micro-ROM _actually_ realistic

Your initial JOY_DIVERGE (“if pressed, loop longer”) is directionally correct, but it can still “reconverge” quickly if the divergence happens once and both paths fall back into the same tight loop.

What we want instead is **sustained divergence** that mirrors real RL workloads:

- divergence persists across many instructions
- divergence impacts memory access patterns (not just loop count)
- divergence changes which basic blocks execute repeatedly

### 5.1 Proposed micro-ROM: `JOY_DIVERGE_PERSIST.gb`

Core idea: each env maintains a **mode byte** in WRAM, updated from JOYP each step. That mode controls which inner loop runs. Different envs → different modes → sustained divergence even after button release.

High-level behavior per outer iteration:

1. read JOYP
2. update `mode` in WRAM (e.g. XOR in bits depending on which button)
3. branch on `mode & 3` into one of 4 inner loops:
   - loop0: ALU-heavy
   - loop1: stride-1 WRAM writes
   - loop2: stride-17 WRAM writes (cache/pattern stress)
   - loop3: branchy loop (more control-flow divergence)

4. write a short signature to WRAM (`C000:C010`) so verification can hash it

This is “realistic” because:

- env divergence persists across steps (stateful mode)
- memory access diverges as well as control flow

### 5.2 Opcode budget

You’ll need to expand opcode support slightly beyond your current micro-suite. Keep it minimal and purposeful:

- `LD A,(HL)` (0x7E)
- `AND d8` (0xE6)
- `XOR d8` (0xEE) (or `XOR A` patterns)
- `JR Z/NZ,r8` (0x28 / 0x20)
- `DEC r8` (0x05)
- potentially `LD (HL+),A` / `LD A,(HL+)` if you want cheap strides (optional)

This is still small enough to remain “micro-ROM scope”, but large enough to create real divergence.

---

## 6) “Realistic PyBoy comparison”: pufferlib + pyboy as baselines

There are two different needs here:

1. **Correctness oracle** must be introspectable (`get_cpu_state`, `read_memory`) → keep it single-process / single-env.
2. **Performance baseline** should use pufferlib’s fast vectorization → use pufferlib VecEnv for scaling runs.

### 6.1 Add a “puffer-style single env” oracle backend

Add backend:

- `pyboy_puffer_single` (name bikeshed)
- Implements `VecBackend`
- Uses the **exact** schedule from the Pokémon RL docs: press, delayed release, tick `frames_per_step - 1` ([Drubinstein][1])
- Uses the **PokemonRedPufferCodec** mapping
- Provides `get_cpu_state` + `read_memory` for verify

This replaces `pyboy_single` as your oracle for M4 onward (or at least for JOYP-driven ROMs).

### 6.2 Add a pufferlib-vectorized CPU baseline backend for benchmarks

Add backend:

- `pyboy_puffer_vec` (benchmark-only)
- Internally uses pufferlib’s vectorization API:
  - `pufferlib.vector.make(...)` then `vecenv.step(actions)` for sync baseline
  - optionally `vecenv.send(actions)` / `vecenv.recv()` for async ceiling measurements ([Puffer][3])

**Important constraint**: this backend does _not_ need `get_cpu_state` for many envs. For verify you use the single-env oracle; for perf you only need SPS.

This cleanly separates concerns and avoids “introspecting across processes”.

### 6.3 Artifact honesty

Every benchmark artifact should record:

- `backend`: `pyboy_puffer_vec` vs `warp_vec_cuda`
- `action_codec`: name/version
- `action_schedule`: `{frames_per_step, release_after_frames}`
- `vec_backend`: serial / mp_sync / mp_async (whatever pufferlib backend you choose)
- `stage`: emulate_only/reward_only/obs_only/full_step

That makes it impossible to accidentally compare apples to oranges.

---

## 7) Warp backend updates to support fused kernels + action codec

### 7.1 Backend owns

- device buffers for:
  - `actions_dev (int32)`
  - `joyp_select_dev (uint8)`
  - `reward_dev (float32)`
  - `obs_dev (float32)`

- kernel selection:
  - `get_cpu_step_kernel(stage)` returns one of the 4 compiled kernels

### 7.2 `step(actions)`

1. validate action range using codec’s `num_actions`
2. copy actions to device (host pinned → device copy)
3. launch the stage kernel:
   - includes JOYP logic
   - includes reward/obs epilog if stage demands

4. **no implicit synchronize** in CUDA path (benchmark uses `sync_every`, as you already do)

### 7.3 Return values

For benchmark integrity you **do not** want per-step device→host copies of obs/reward.

So:

- by default, return cheap host placeholders (zeros) to satisfy API
- optionally add a `--return-host-obs` flag or “training integration mode” later

But record in artifacts that reward/obs were computed on device (stage + kernel id), even if not copied out.

---

## 8) Verification plan for the new ROM + fused stages

### 8.1 Verify profile for `JOY_DIVERGE_PERSIST`

- `ref=pyboy_puffer_single`
- `dut=warp_vec_cpu` and `warp_vec_cuda`
- `frames_per_step=1` for step-exact comparisons
- `--mem-region C000:C010` (or `C000:C100`) required

Also run one “RL-ish” smoke:

- `frames_per_step=24`, `release_after_frames=8`, `verify_steps=16`

### 8.2 Stage correctness (sanity)

For correctness we don’t need obs/reward equality yet (since Warp won’t copy to host), but we _do_ need:

- `reward_dev` is deterministic given actions + ROM
- `obs_dev` writes are in-bounds and deterministic

Add small Warp-only unit tests that:

- run 2 steps
- copy reward/obs once (explicit debug method)
- assert they changed and are deterministic across instances

---

## 9) Benchmark protocol for E4 (what we’ll actually measure)

### 9.1 Baselines

- CPU baseline: `pyboy_puffer_vec` (sync and/or async variants) ([Puffer][3])
- GPU DUT: `warp_vec_cuda` with stage kernel variants

### 9.2 Benchmark suites

Run scaling sweeps on:

- `ALU_LOOP` (mostly non-divergent)
- `MEM_RWB` (memory-heavy)
- `JOY_DIVERGE_PERSIST` (divergence + memory divergence)

For `JOY_DIVERGE_PERSIST`, force divergence with an action generator like:

- `striped`: env_idx mod 4 chooses among `UP/DOWN/LEFT/RIGHT` or `A/B/START` (depending on codec)

### 9.3 Divergence “receipt”

Add one cheap end-of-window signal:

- copy PC once and compute `unique_pc_count` at end, or
- maintain a device-side `mode_histogram` / `branch_taken` counter and copy only a few scalars

This is not about perfect warp-level divergence metrics — it’s just an integrity receipt that the benchmark actually diverged.

---

## 10) Workstream / PR slicing (minimal blocking)

1. **Action codec + puffer-style schedule**
   - `core/action_codec.py`
   - update existing PyBoy backend to use codec + schedule
   - update harness action generators to use codec’s `num_actions`

2. **Fused kernel family**
   - extend `cpu_step_builder.py` skeleton with `POST_STEP_DISPATCH`
   - add `cpu_templates/post_step.py`
   - compile 4 kernels by stage

3. **Joypad emulation + new micro-ROM**
   - JOYP handling in load/store templates
   - add `build_joy_diverge_persist()` + suite.yaml entry + tests

4. **Pufferlib baselines**
   - `pyboy_puffer_single` oracle backend (verify)
   - `pyboy_puffer_vec` benchmark backend (scaling)

5. **E4 scaling reports**
   - `make bench-e4-gpu` / `make bench-e4-cpu` or fold into existing `bench-gpu` with `--stage full_step` and the new ROM

---

## One important note: “pufferlib dependency” without blowing up the core loop

PufferLib is excellent, but it can be heavy depending on how you install it. The clean compromise is:

- keep core correctness + Warp development independent
- add pufferlib-backed backends behind a lazy import path
- gate pufferlib backends under DGX-only targets (so the 2-minute CPU gate stays tight)

This still fulfills “rely on pufferlib+pyboy” for the _real_ baseline runs, while not turning every laptop `uv sync` into a mini-distro.

(And it aligns with the Pokémon RL team’s own emphasis that vectorization/backends matter enormously to SPS.) ([Drubinstein][4])

---

[1]: https://drubinstein.github.io/pokerl/docs/chapter-2/env-setup/ "The Environment | Pokémon RL"
[2]: https://arxiv.org/pdf/2502.19920?utm_source=chatgpt.com "Pokémon Red via Reinforcement Learning"
[3]: https://puffer.ai/docs.html "PufferLib Docs"
[4]: https://drubinstein.github.io/pokerl/docs/chapter-3/running/ "Running | Pokémon RL"
