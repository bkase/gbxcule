# Game Boy Emulator — Milestone D Spec + Plan (PPU timing + VBlank + background rendering)

Date: 2026-01-24

Source context:
- `history/gameboy-emu.md` → “Milestone D — PPU timing + VBlank + background rendering”
- `CONSTITUTION.md` → correctness-by-construction, functional core/imperative shell, verifiable rewards, structured mismatch artifacts

This document is the **spec** for Milestone D and the execution plan. If behavior changes, update this document and the tests together.

Goal: implement a minimal DMG (original Game Boy) PPU in this repo such that:
- timing is **scanline-accurate** (not dot/t-cycle accurate) and still drives **VBlank interrupts**, and
- background rendering produces deterministic frames whose **hash matches PyBoy** for 1–2 repo-owned graphics micro-ROMs.

Non-goals (Milestone D):
- Window, sprites, STAT interrupts (beyond updating STAT mode/coincidence bits) (Milestone E).
- DMA (Milestone E).
- MBC bring-up for real cartridges (Milestone F).
- Sound.

---

## Definition of Done (DoD)

- **Timing:** `LY` advances at scanline granularity; `IF.VBlank` is requested on entry to `LY=144`, once per frame.
- **Rendering:** background-only rendering produces a framebuffer (or hash) that matches PyBoy for 1–2 deterministic ROMs.
- **Verification:** a fast, automated check produces an exit-0/exit-1 “verifiable reward” on frame-hash match/mismatch.
- **Artifacts:** mismatches dump enough structured state (PPU regs + optional images) to reproduce deterministically.

---

## Assumptions / Dependencies

Recommended prerequisites:
- Milestone B: CPU ISA is complete and cycle counts are correct (this repo’s stepping is cycle-sensitive).
- Milestone C: interrupts and timers are correct enough for `IF/IE/IME` gating (PPU requests VBlank via `IF`).
- Milestone A: an MMIO boundary exists (`read8/write8`) so LCD/PPU registers don’t become scattered across instruction templates.

Notes:
- This milestone keeps a **separate renderer kernel** (`src/gbxcule/kernels/ppu_step.py`), but integrates the scanline scheduler into the CPU step kernel so interrupts (`IF`) are visible during CPU execution (see D2/D3).

---

## Implementation Plan

### D0 — Spec (Integrated): scanline-accurate “PPU v0” (make invalid states unrepresentable)

Intent: “Spec-driven development” + “correctness by construction.”

Milestone D’s PPU is **scanline-accurate**, not dot-accurate. Concretely:
- We advance time in t-cycles (same unit as the CPU kernel), but we only model **scanline boundaries**.
- We do **not** model Mode 2/3/0 dot timing inside a scanline.
- We define a single, deterministic “sampling point” per scanline for rendering inputs.

This choice is informed by CuLE’s core finding (see `docs/cule_paper.md` §3): emulator subsystems with very different execution and memory-write profiles should be **decoupled into separate kernels** (CPU vs renderer) to reduce divergence and register pressure, and rendering work should be performed only when needed and parallelized across rows/pixels.

#### D0.1 Units + constants

- Kernel “cycles” are DMG **t-cycles** (NOP = 4), consistent with Milestone C.
- PPU time advances in the same unit, but Milestone D only observes **scanline boundaries**.
- Constants (DMG):
  - `CYCLES_PER_SCANLINE = 456`
  - `VISIBLE_LINES = 144` (LY 0–143)
  - `VBLANK_LINES = 10` (LY 144–153)
  - `LINES_PER_FRAME = 154` (0–153)
  - `CYCLES_PER_FRAME = CYCLES_PER_SCANLINE * LINES_PER_FRAME = 70224`

#### D0.2 State (PPU reducer state)

Per environment, the PPU state is:
- `scanline_cycle`: integer in `[0, CYCLES_PER_SCANLINE)` (t-cycles elapsed within the current scanline)
- `ly`: integer in `[0, LINES_PER_FRAME)` (current scanline)

Invariant:
- `scanline_cycle` is only used to decide when the next scanline boundary occurs; the PPU does not expose dot-level timing.

Derived mode bits for `STAT` (Milestone D coarse model):
- If `lcd_enabled == 0`: `mode = 0`
- Else if `ly >= 144`: `mode = 1` (VBlank)
- Else: `mode = 0` (treat visible scanlines as “HBlank” in this model)

Rationale:
- Milestone D does not implement STAT interrupts or access blocking; mode bits are maintained only as a stable, low-cost signal (VBlank vs non-VBlank).

#### D0.3 MMIO register contract (reads/writes + bit behavior)

The PPU observes and/or updates these registers in `mem`:

- `LCDC (0xFF40)`:
  - bit 7: LCD enable (1=on, 0=off)
  - bit 4: tile data select (1=0x8000 unsigned, 0=0x8800 signed)
  - bit 3: BG tile map select (1=0x9C00, 0=0x9800)
  - bit 0: BG enable (1=render BG, 0=BG outputs color 0)
  - All other LCDC bits are read (preserved) but have **no behavioral effect** in Milestone D.

- `STAT (0xFF41)`:
  - bits 1–0 (mode) are **PPU-controlled** (read-only from the CPU’s perspective).
  - bit 2 (LYC==LY) is **PPU-controlled**.
  - bits 6–3 are CPU-writable “enable bits” but are **not acted upon** until Milestone E.
  - bit 7 is treated as “unused”; for Milestone D reads, we define:
    - `STAT[7]` is preserved from the last value in `mem[0xFF41]` (no forced constant).
  - Update rule (at each scanline boundary, and on `LYC` writes if your MMIO layer supports it):
    - `stat_keep = mem[0xFF41] & 0xF8` (preserve bits 7–3 exactly)
    - `stat_out = stat_keep | (coincidence<<2) | mode`
    - `mem[0xFF41] = stat_out`

- `LY (0xFF44)`:
  - PPU-controlled readback of `ly`.
  - Update rule: after any `ly` change, `mem[0xFF44] = ly`.
  - Write behavior is out-of-scope for Milestone D; tests/ROMs must not rely on writes to `LY`.

- `LYC (0xFF45)`:
  - CPU writable; PPU reads it for coincidence.
  - Coincidence rule: `coincidence = (ly == mem[0xFF45]) ? 1 : 0`.

- `SCY (0xFF42)`, `SCX (0xFF43)`:
  - CPU writable; PPU reads them for BG scroll.

- `BGP (0xFF47)`:
  - CPU writable; PPU reads it to map BG color indices 0–3 to shade indices 0–3.
  - Mapping rule: `shade = (BGP >> (color*2)) & 0b11`.

- `IF (0xFF0F)`:
  - PPU sets bit 0 on VBlank entry; must preserve all other bits.
  - Update rule: `mem[0xFF0F] |= 0x01` when VBlank is requested.

#### D0.4 Timing reducer: `advance_scanlines(scanline_cycle, ly, cycles_elapsed)`

Let `lcdc = mem[0xFF40]` and `lcd_enabled = (lcdc >> 7) & 1`.

If `lcd_enabled == 0` (LCD off):
- Set `scanline_cycle = 0`, `ly = 0`
- Update `mem[LY] = 0`
- Update `STAT` using the rule in D0.3 with `mode=0` and `coincidence = (0 == LYC)`
- Do **not** request VBlank

If `lcd_enabled == 1`:
- Advance by `cycles_elapsed` t-cycles, but only act on **scanline boundaries**:
  1. `total = scanline_cycle + cycles_elapsed`
  2. `lines = total // CYCLES_PER_SCANLINE`
  3. `scanline_cycle = total % CYCLES_PER_SCANLINE`
  4. For each scanline boundary crossed (repeat `lines` times):
     - `prev_ly = ly`
     - `ly = (ly + 1) % LINES_PER_FRAME`
     - If `prev_ly == 143` and `ly == 144`: request VBlank (`mem[IF] |= 0x01`)
     - Update `mem[LY]` and `mem[STAT]` (D0.3)

Notes:
- Because the CPU kernel’s instruction cycle counts are small (≪ 456), `lines` is almost always `0` and scanline changes are rare and cheap to detect.
- “Access blocking” (VRAM/OAM restrictions during modes) is explicitly out-of-scope for Milestone D.

#### D0.5 Scanline latch + background rendering contract (BG only)

Rendering is defined only when `lcd_enabled == 1`.

CuLE-style decoupling for efficiency:
- The **CPU kernel** is responsible for scanline timing and for recording a compact per-scanline “render command” (latches).
- A separate **renderer kernel** consumes those latches and writes pixels/hashes.
- Rendering can be skipped entirely when not needed (e.g., training phases without pixel observations), just like CuLE avoids calling the TIA kernel when rendering is unnecessary.

Sampling point (deterministic; scanline-accurate):
- For each visible scanline `y = LY ∈ [0,143]`, the renderer must use a single set of “latched” values:
  - `LCDC_latch[y]`, `SCX_latch[y]`, `SCY_latch[y]`, `BGP_latch[y]`
- Latch time: **at scanline start** (when `LY` becomes `y`).
- Additionally: on LCD enable (`LCDC bit 7` transitions `0→1`), treat that moment as “scanline 0 start” and produce latches for `y=0` (so the first rendered frame has defined inputs).
- Consequence: CPU writes to SCX/SCY/BGP/LCDC *during* scanline `y` are defined to take effect at the next scanline in this model.

Inputs used by the renderer:
- Per-scanline latches for `y` (above).
- VRAM (`0x8000..0x9FFF`) read at render time.

VRAM timing restriction (Milestone D):
- For determinism (and because we are not dot-accurate), Milestone D assumes chosen ROMs do not rely on VRAM writes during visible scanlines.
- Micro-ROMs should write VRAM only while LCD is off or during VBlank.
Similarly, register timing restrictions for Milestone D validation ROMs:
- Validation ROMs must not rely on mid-scanline effects from `SCX/SCY/BGP/LCDC` writes.
- Any `SCX/SCY/BGP/LCDC` changes intended to affect rendering should occur while LCD is off, or during VBlank, or at a scanline boundary in a way that does not require cycle-perfect alignment.

Per visible scanline `y ∈ [0,143]`, using the latched values for that `y`:
- If `LCDC_latch[y] bit0 (BG enable) == 0`: output shade `0` for all x.
- Else, for each `x ∈ [0,159]`:
  - `sx = (x + SCX_latch[y]) & 0xFF`, `sy = (y + SCY_latch[y]) & 0xFF`
  - Tile map base:
    - `0x9800` if `LCDC_latch[y] bit3 == 0`, else `0x9C00`
  - `tile_id = vram[map_base + (sy>>3)*32 + (sx>>3)]`
  - Tile data base:
    - If `LCDC_latch[y] bit4 == 1` (unsigned): `tile_addr = 0x8000 + tile_id*16`
    - Else (signed): `tile_addr = 0x9000 + int8(tile_id)*16`
  - Row: `row = sy & 7`, fetch `lo = vram[tile_addr + row*2]`, `hi = vram[tile_addr + row*2 + 1]`
  - Bit: `bit = 7 - (sx & 7)`, `color = (((hi>>bit)&1)<<1) | ((lo>>bit)&1)` in `0..3`
  - Shade: `shade = (BGP_latch[y] >> (color*2)) & 0b11` in `0..3`
  - Store `shade` to output framebuffer at `(x, y)`

Framebuffer representation (Milestone D decision):
- Store **shade indices** (0..3), not RGBA, for deterministic hashing and to avoid palette RGB mismatches.
- Layout for env0: `frame_bg_shade_env0[(y*160) + x] = shade`.

#### D0.6 Frame boundary + hashing contract (for verification)

Frame “visible output” is the set of rendered scanlines `LY 0..143`.

For Milestone D verification:
- Preferred: run graphics ROMs with `frames_per_step = 1` so each step produces one full frame and one latch set.
- Compare by hashing the `frame_bg_shade_env0` buffer after each step (or compare a stored `frame_hash` if you implement on-device hashing).
- PyBoy oracle: quantize `pyboy.screen.ndarray` into 4 shade indices for the same ROM and hash the resulting 160×144 indices.

#### D0.7 Explicit non-goals (Milestone D)

- Dot-level/mode-level timing (Mode 2/3/0 dot counts, mid-scanline effects).
- Window rendering and window timing (`WX/WY`).
- Sprites (OBJ), OAM scan correctness, sprite priority.
- STAT interrupt requests (bit 1 in IF) and edge-trigger semantics (Milestone E).
- Bus conflicts / access restrictions (VRAM/OAM blocking).

### D1 — Extend ABI with explicit PPU state + frame output buffers

Intent: keep the “functional core / imperative shell” separation by making all persistent state explicit in buffers (no hidden globals).

Deliverables:
- ABI additions in `src/gbxcule/core/abi.py` (bump `ABI_VERSION` and document migration):
  - `SCREEN_W=160`, `SCREEN_H=144` constants.
  - PPU per-env arrays (scanline-accurate; see D0.2/D0.4):
    - `ppu_scanline_cycle: i32[num_envs]` (0..455)
    - `ppu_ly: i32[num_envs]` (0..153)
  - Env0 per-scanline BG latches (see D0.5; size `SCREEN_H`):
    - `bg_lcdc_latch_env0: u8[SCREEN_H]`
    - `bg_scx_latch_env0: u8[SCREEN_H]`
    - `bg_scy_latch_env0: u8[SCREEN_H]`
    - `bg_bgp_latch_env0: u8[SCREEN_H]`
  - Frame output (Milestone D decision; see D0.5/D0.6):
    - `frame_bg_shade_env0: u8[SCREEN_W*SCREEN_H]` (env0 only; values 0..3)
    - Optional (later): `frame_hash_env0: u64[1]` or `frame_hash_u64[num_envs]` for on-device hashing at scale.

Why env0-only framebuffer is acceptable for D:
- Milestone D’s oracle is frame hashes vs PyBoy; scaling to many envs is Milestone E+ territory.

### D2 — Implement scanline scheduler + VBlank request + LY/STAT maintenance + latch capture

Intent: make interrupts/timing correct at scanline granularity and capture per-scanline render inputs without doing pixel writes in the CPU kernel (CuLE-style decoupling).

Deliverables:
- Implement the scanline-accurate timing reducer inside the CPU stepping kernel (because `IF` must change during CPU execution):
  - Track `ppu_scanline_cycle` and `ppu_ly` in ABI buffers.
  - After each instruction, advance `ppu_scanline_cycle += cycles` and on scanline boundary:
    - increment `ppu_ly` (wrap 153→0),
    - write `mem[LY] = ppu_ly`,
    - update `mem[STAT]` (coarse mode bits + coincidence) per D0.3,
    - when `ppu_ly` transitions `143 → 144`, request VBlank (`mem[IF] |= 0x01`),
    - if `env_idx == 0` and `ppu_ly < 144`, latch `LCDC/SCX/SCY/BGP` into the env0 latch arrays at index `ppu_ly` (scanline start latch).
  - LCD-off semantics per D0.4:
    - force `ppu_scanline_cycle=0`, `ppu_ly=0`, `mem[LY]=0`, update `STAT`, and do not request VBlank.
  - (Optional but recommended) Update coincidence on `LYC` writes in the MMIO layer, since it is cheap and avoids “stale STAT” debugging.

Verification (cheap, inverted pyramid):
- Unit-style test that advances exactly `CYCLES_PER_FRAME` t-cycles and asserts:
  - `ly` wraps 0..153 correctly,
  - exactly one VBlank request is emitted per frame (for deterministic setup),
  - `STAT` mode bits match the coarse model (VBlank vs non-VBlank).

### D3 — Implement BG renderer kernel (consumes scanline latches)

Intent: keep CPU stepping free of pixel writes; render on-GPU using a dedicated kernel that can be parallelized across pixels/rows (CuLE-style).

Deliverables:
- Implement `src/gbxcule/kernels/ppu_step.py` as a renderer kernel module (repurposing the stub):
  - `ppu_render_bg_env0(mem, bg_*_latch_env0, frame_bg_shade_env0)`:
    - renders all `SCREEN_H * SCREEN_W` pixels for env0 using the per-scanline latches and VRAM reads described in D0.5.
  - Preferred launch shape: 2D grid over `(y, x)` so one thread = one pixel (parallelizable like CuLE’s row-parallel TIA idea).
- Optional: compute a frame hash on-device (either in the renderer kernel or a follow-up reduction kernel).

### D4 — Backend wiring + capture policy (“render only when needed”)

Intent: wire the scanline scheduler + latches into the Warp backend, and only launch the renderer kernel when required (CuLE-style).

Deliverables:
- In `src/gbxcule/backends/warp_vec.py`:
  - allocate PPU state buffers (`ppu_scanline_cycle`, `ppu_ly`) and env0 latch + framebuffer buffers in `reset()`.
  - pass these buffers into the generated `cpu_step` kernel so latches and `IF/LY/STAT` are updated during CPU execution.
  - after `cpu_step`, conditionally launch the renderer kernel (`ppu_render_bg_env0`) only when needed:
    - always in verification/graphics tests,
    - optionally gated in training/bench mode (no reason to pay for pixel writes if the agent doesn’t consume pixels).
- Define capture policy:
  - For Milestone D graphics verification, run ROMs with `frames_per_step=1` and always capture all 144 scanline latches.
  - For larger `frames_per_step`, capture either the last frame only or every Nth frame (explicit knob), to avoid the “render everything” tax.

### D5 — Harness support: frame-hash compare + mismatch bundles

Intent: “verifiable rewards” + “structured logs only.”

Deliverables in `bench/harness.py` verify mode:
- Flags:
  - `--compare-frame` (enable frame hash compare)
  - `--dump-frame-on-mismatch` (write `ref.png`/`dut.png` via Pillow for env0; render from 0..3 shade indices)
- On compare boundaries:
  - compute `ref_frame_hash` and `dut_frame_hash`
  - include PPU register snapshots (`LCDC/STAT/SCX/SCY/LY/LYC/BGP/IF`) in the mismatch `diff` payload
- Mismatch bundle:
  - add optional `ref.png`/`dut.png`
  - add a `ppu_state.json` (or embed in existing metadata) so a repro run can display “why” without manual stepping

### D6 — Add 1–2 deterministic, repo-owned graphics micro-ROMs

Intent: keep the oracle deterministic and license-safe.

Deliverables:
- Extend `bench/roms/build_micro_rom.py` and `bench/roms/suite.yaml` with:
  1) `BG_STATIC.gb`
     - LCD off
     - write known tile data + BG map + BGP
     - set `SCX/SCY=0`
     - enable LCD
     - infinite loop
  2) `BG_SCROLL_SIGNED.gb` (or similar)
     - exercises `SCX/SCY` scrolling and/or signed tile addressing (`LCDC bit4=0`)
     - writes VRAM only while LCD is off or during VBlank (per D0.5 restriction)
  - For these graphics ROMs, set `frames_per_step=1` in the suite to simplify “one frame per step” hashing (D0.6).
- Add explicit ROM design rules for PyBoy-oracle validation under a scanline-accurate PPU:
  - No mid-scanline register tricks: ROM must not change `SCX/SCY/BGP/LCDC` during visible scanlines expecting per-dot effects.
  - No dot-synchronized polling: ROM must not attempt to time behavior by counting cycles inside a scanline (no “wait N cycles then write SCX” patterns).
  - No reliance on access blocking: ROM must not depend on VRAM/OAM access restrictions or mode-specific behavior (out-of-scope for D).
  - VRAM writes only while LCD is off or during VBlank (already required above).
  - Prefer “static frame” or “VBlank-updated frame” patterns; if the ROM updates anything per frame, do it during VBlank and don’t assume an exact cycle offset within VBlank.
- Add a fast integration test (PyBoy vs Warp) that:
  - runs a small number of frames (e.g., 4),
  - optionally warms up 1–2 frames after LCD enable,
  - compares frame hashes on the subsequent frames,
  - fails fast with a mismatch bundle.

---

## Verification Loop (the milestone’s “reward function”)

For each sub-milestone (D2→D6):
- Implement the feature.
- Add the cheapest possible check first (unit/property-style invariants).
- Add the next cheapest integration check (PyBoy oracle + frame hash).
- Only then expand ROM coverage or add dumps.

Keep default logs silent on success; on failure, emit a mismatch bundle with enough state to reproduce.

---

## Suggested “file touch map” (for later implementation)

- PPU kernel + wiring:
  - `src/gbxcule/kernels/ppu_step.py`
  - `src/gbxcule/backends/warp_vec.py`
  - `src/gbxcule/kernels/cpu_step_builder.py` / `src/gbxcule/kernels/cpu_step.py` (scanline scheduler + env0 latch capture)
  - `src/gbxcule/core/abi.py` (ABI bump + screen/PPU buffer definitions)

- Harness + artifacts:
  - `bench/harness.py`

- Micro-ROM generation:
  - `bench/roms/build_micro_rom.py`
  - `bench/roms/suite.yaml`

- Tests:
  - New: `tests/test_ppu_timing.py` (scanline timing invariants + VBlank request)
  - New: `tests/test_ppu_bg_hash_vs_pyboy.py` (frame hash compare for new ROMs)
