# Game Boy Emulator — Milestone E Plan (Window + Sprites + STAT IRQ + DMA)

Date: 2026-01-24

Source context:
- `history/gameboy-emu.md` → “Milestone E — Window + sprites + STAT interrupts + DMA”
- `CONSTITUTION.md` → correctness-by-construction, functional core/imperative shell, verifiable rewards, inverted test pyramid

Goal: extend the DMG PPU from Milestone D (timing + VBlank + BG rendering) to a correctness-verified Milestone E implementation:
- **Window rendering**
- **Sprite rendering (OBJ)**
- **LCD STAT interrupts** (mode + LYC)
- **OAM DMA** (`0xFF46`)

Oracle/verification: PyBoy is the reference; Warp CPU is the debug target; Warp CUDA is the perf target. Only debug one delta at a time.

Non-goals (Milestone E):
- Sound (APU)
- MBC bring-up for commercial ROMs (Milestone F)
- Cycle-perfect DMA/OAM bus blocking (we can start with “copy occurs” correctness; timing refinements can come later)

---

## Definition of Done (DoD)

- **DMA correctness:** writing `0xFF46` copies exactly 160 bytes into `0xFE00..0xFE9F`, verified vs PyBoy and a repo-owned self-check ROM.
- **Window correctness:** deterministic frame hashes match PyBoy on a window-focused micro ROM (and at least one mixed BG+window ROM).
- **Sprite correctness:** deterministic frame hashes match PyBoy on sprite-focused ROMs covering flips, palettes, priority, and 8x16 mode.
- **STAT interrupt correctness:** mode interrupts and LYC=LY interrupt are edge-triggered (no “spam every cycle”), with counters matching PyBoy for a fixed run.
- **Mismatch artifacts:** failures are quiet by default; on mismatch, dump a structured bundle sufficient for fast repro (frame hashes, key registers, recent PPU events).
- **Fast feedback loop:** tests remain cheap (aim for the “≤ 2 minutes full suite” spirit from `CONSTITUTION.md`).

---

## Entrance Criteria / Dependencies

Milestone E assumes Milestone D is effectively complete:
- CPU, interrupts, and timers are good enough to run deterministic graphics ROMs for many frames.
- A real MMIO boundary exists (`read8/write8`) so PPU+DMA side effects are centralized and testable.
- PPU stepping exists (separate `ppu_step` kernel + backend integration) producing:
  - either a framebuffer (env0) or per-frame hashes (preferred for vectorization),
  - correct-ish LY/mode timing + VBlank interrupt.

If any of the above is missing, do not “power through” Milestone E: fix the prerequisite first or you’ll drown in false diffs.

---

## Implementation Plan

### E0 — Lock contracts (spec first; make invalid states unrepresentable)

Intent: “Code is cheap, specs are precious.” Make the behaviors and edges explicit before you implement them.

Deliverables:
- A small “Milestone E spec” section (either in this doc or a sibling doc) specifying:
  - PPU timing model: what advances per cycle / per dot / per phase; what is considered stable for verification.
  - STAT IRQ triggering semantics (edge-trigger points; what counts as a transition).
  - Sprite priority rules (BG/Window interaction; OBJ-vs-OBJ tie-break behavior you will implement).
- Formalize per-env PPU state fields (not just raw bytes in mem):
  - `ly`, `mode`, `dot_in_line` (or equivalent)
  - `window_line` (only increments on lines where window is active)
  - `stat_prev_conditions` (or a compact edge-detector state) so STAT IRQ behavior is intentional and testable

Testing:
- Pure unit tests for the edge detector and rendering helpers (no Warp, no PyBoy).

Verifiable reward:
- `pytest -k ppu_unit` is fast and deterministic.

---

### E1 — DMA: OAM DMA via `0xFF46`

Implement `write8(0xFF46, v)`:
- `src = (v << 8) & 0xFFFF`
- copy `mem[src:src+160]` → `mem[0xFE00:0xFEA0]`
- start with “copy occurs immediately” correctness (timing/bus blocking can be refined later).

Repo-owned test ROM:
- `DMA_OAM_COPY.gb`:
  - writes a known pattern to a source page (WRAM is easiest),
  - triggers DMA,
  - verifies OAM bytes and reports PASS (serial) or writes a signature in WRAM for harness checks.

Verification:
- PyBoy vs Warp CPU memory compare on `0xFE00..0xFEA0` after N frames.
- Warp CPU vs Warp CUDA compare once Warp CPU matches PyBoy.

---

### E2 — Window rendering

Implement window selection and addressing:
- LCDC window enable (bit 5) and window tilemap select (bit 6).
- WX/WY behavior:
  - window X starts at `WX - 7`,
  - window becomes active when `LY >= WY` and the scan reaches the start X.
- Maintain `window_line` correctly (increment once per scanline when window is active).

Rendering integration:
- Decide BG vs Window source per pixel before sprite overlay.
- Preserve “BG/Win color index” (0–3) until after sprite overlay so you can enforce priority rules correctly.

Repo-owned test ROM:
- `PPU_WINDOW.gb`:
  - deterministic tiles + BG map + window map,
  - non-trivial WX/WY values,
  - avoids mid-frame register changes initially.

Verification:
- Frame-hash snapshots vs PyBoy at selected frame indices.

---

### E3 — Sprite rendering (OBJ)

Implement sprite evaluation (per scanline):
- Iterate OAM entries in order, select up to **10 sprites** for the current LY (DMG behavior).
- Correct coordinate offsets:
  - sprite Y stored as `y+16`,
  - sprite X stored as `x+8`.
- Handle 8x8 vs 8x16 via LCDC bit 2.

Implement sprite pixel composition:
- Attributes: priority (behind BG), xflip, yflip, palette (OBP0/OBP1).
- OBJ color 0 is transparent.
- Priority rule (DMG):
  - if “OBJ behind BG”, sprite pixel is hidden unless BG/Window color index is 0.
- OBJ-vs-OBJ tie-break:
  - enforce a deterministic rule and document it; test it (X then OAM index is a common approach).

Repo-owned test ROM:
- `PPU_SPRITES.gb`:
  - uses DMA to populate OAM (exercises E1 + sprites together),
  - includes cases for flips, palettes, priority, and 8x16.

Verification:
- Frame-hash snapshots vs PyBoy on 2–3 frames per ROM.

---

### E4 — LCD STAT interrupts (mode + LYC)

Implement STAT sources with edge-triggering:
- Maintain STAT mode bits and coincidence flag (LYC==LY).
- Request IF bit 1 when:
  - entering Mode 0/1/2 and the corresponding STAT enable bit is set, and/or
  - LYC==LY becomes true and LYC interrupt enable is set.
- Edge-triggered means:
  - do **not** request every cycle the condition remains true,
  - do request on the transition into the true condition.

Repo-owned test ROM:
- `PPU_STAT_IRQ.gb`:
  - installs a STAT ISR,
  - counts interrupts by source (or by inferred mode),
  - reports counts deterministically (serial or WRAM signature).

Verification:
- Compare interrupt counters/serial output vs PyBoy for a fixed frame window.

---

### E5 — Harness + mismatch artifacts (silent on success; rich on failure)

Add structured artifacts for graphics verification:
- Frame hash history (N recent hashes).
- Snapshot of key PPU/MMIO registers on mismatch:
  - `LCDC/STAT/LY/LYC/SCX/SCY/WX/WY/BGP/OBP0/OBP1`
  - OAM bytes and relevant VRAM regions (bounded; avoid giant dumps by default).
- Minimal “recent events” ring buffer:
  - mode transitions, VBlank enter/exit, STAT IRQ edges, DMA trigger events.

Keep it fast:
- Hash by default; gate PNG dump behind a flag and only dump env0.

---

### E6 — Close the oracle ladder (PyBoy → Warp CPU → Warp CUDA)

Do not debug CPU and CUDA at the same time.

Order:
1. Make PyBoy vs Warp CPU pass for all new Milestone E ROMs (hash + targeted memory regions).
2. Make Warp CPU vs Warp CUDA match on the same ROMs.
3. Add CUDA tests guarded by availability checks; keep them short.

---

## Verification Strategy (inverted test pyramid)

1. **Pure unit tests** for helpers (edge detector, palette mapping, sprite compose).
2. **Repo-owned micro-ROMs** that are deterministic and license-safe (generated).
3. **Step-exact and region comparisons**:
   - For DMA: direct memory region equality (`0xFE00..0xFEA0`).
   - For PPU: frame-hash snapshots at selected frame indices.
4. **CUDA parity** once CPU is correct.

Rules of thumb:
- Prefer many small ROMs to isolate failures quickly.
- Always add a test ROM when adding a hardware feature (DMA/window/sprites/STAT).
- Default logs quiet; only emit structured mismatch bundles on failure (`CONSTITUTION.md`).

---

## Suggested File Touch Map (for later implementation)

PPU kernel + integration:
- `src/gbxcule/kernels/ppu_step.py` (implement; currently stub)
- `src/gbxcule/backends/warp_vec.py` (integrate PPU stepping and frame-hash buffers)
- (optional) `src/gbxcule/core/abi.py` (new buffers for PPU state / hashes / artifacts)

MMIO and DMA:
- `src/gbxcule/kernels/cpu_step_builder.py` (centralize `read8/write8`; implement DMA trigger/write path)
- `src/gbxcule/kernels/cpu_templates/*` (only if templates still write mem directly; migrate to MMIO helpers)

Micro-ROMs + tests:
- `bench/roms/build_micro_rom.py` (add `DMA_OAM_COPY`, `PPU_WINDOW`, `PPU_SPRITES`, `PPU_STAT_IRQ`)
- `tests/test_micro_roms.py` (ROM build + PyBoy smoke)
- `tests/test_warp_vec_ws3_verify.py` (PyBoy vs Warp CPU verification on new ROMs)
- `tests/test_verify_cuda_micro_roms.py` (CUDA parity, guarded)

