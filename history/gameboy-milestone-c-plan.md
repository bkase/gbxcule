# Game Boy Emulator — Milestone C Spec + Plan (Interrupts + Timers)

Date: 2026-01-24

Source context:
- `history/gameboy-emu.md` → “Milestone C — Interrupts + timers correct”
- `CONSTITUTION.md` → correctness-by-construction, functional core/imperative shell, verifiable rewards, inverted test pyramid

This document is the **spec** for Milestone C and the execution plan. If behavior changes, update this document and the tests together.

Goal: implement **DMG interrupt + timer behavior** accurately enough to (a) stay step-exact vs PyBoy for repo ROMs and (b) pass at least one external serial self-test ROM, without blowing the “fast feedback” loop.

Non-goals (Milestone C):
- PPU timing/rendering (Milestone D/E).
- DMA / STAT intricacies beyond what’s strictly needed for timer/interrupt correctness (Milestone E).
- Cartridge MBC bring-up (Milestone F).

---

## Definition of Done (DoD)

- **Interrupts:** `IME`/`IE` gating works; correct priority + vectors; `RETI` restores `IME`.
- **Timers:** `DIV/TIMA/TMA/TAC` behave correctly enough for tests; TIMA overflow requests `IF(Timer)` and reloads from `TMA` with correct timing for chosen tests.
- **Serial→IRQ:** serial transfer completion requests `IF(Serial)` (bit 3).
- **Verification:** repo-owned timer/interrupt micro-suite passes vs PyBoy; at least one external serial self-test ROM passes (skip-if-missing in CI).

---

## Assumptions / Dependencies

Milestone C assumes:
- **Milestone A:** centralized `read8/write8` MMIO boundary in the kernel (not scattered per-opcode special-cases) and Warp-side serial capture (write-to-buffer, not `print`).
- **Milestone B:** CPU ISA is complete enough to support `DI/EI/HALT/RETI`, stack pushes/pops, and cycle counts are stable.

If either is missing:
- Implement the MMIO boundary + serial capture first; otherwise timers/interrupts will be untestable and brittle.

---

## C0 — Spec (Integrated)

### C0.1 Cycle model

- Kernel `cycles` are DMG **t-cycles** (NOP = 4 cycles).
- `CYCLES_PER_FRAME = 70224` remains the canonical “frame” unit for stepping, but timer/interrupt logic updates on **cycles**, not frames.

### C0.2 Interrupt system

Registers and bits:
- `IE` at `0xFFFF` (only bits 0–4 are interrupt enables)
- `IF` at `0xFF0F` (only bits 0–4 are interrupt request flags)
- `IME` is an internal flag (not memory-mapped)

Interrupt lines (bit → name → vector):
- bit 0: VBlank → `0x0040`
- bit 1: LCD STAT → `0x0048`
- bit 2: Timer → `0x0050`
- bit 3: Serial → `0x0058`
- bit 4: Joypad → `0x0060`

Priority order:
- Lowest-numbered bit wins (0 highest priority, 4 lowest).

Servicing rule (instruction boundary):
- After each instruction completes, compute `pending = IE & IF & 0x1F`.
- If `IME == 1` and `pending != 0`, service exactly one interrupt:
  1. select the highest-priority bit set
  2. clear `IME = 0`
  3. clear that bit in `IF`
  4. push current `PC` to stack (via `write8`, little-endian; correct SP decrement order)
  5. set `PC = vector`
  6. add `20` cycles of service cost

IME mutation rules:
- `DI`: clears `IME` immediately and clears any “EI pending enable”.
- `EI`: sets `IME` **after the following instruction completes** (EI delay).
- `RETI`: sets `IME = 1` immediately after returning (same step as RET semantics).

HALT interaction (minimal required):
- `HALT` enters a halted state where the CPU stops fetching instructions.
- While halted, the system still advances time (timers run).
- CPU exits HALT when `pending = (IE & IF & 0x1F) != 0` (wake even if `IME == 0`).
- “HALT bug” edge cases are explicitly **out of scope** unless a chosen external ROM forces it; if it does, codify the exact behavior in this spec and add a micro-ROM reproducer.

### C0.3 Timers (DIV/TIMA/TMA/TAC)

Registers:
- `DIV`  at `0xFF04` (divider register)
- `TIMA` at `0xFF05` (timer counter)
- `TMA`  at `0xFF06` (timer modulo / reload value)
- `TAC`  at `0xFF07` (timer control)

Timer approach (Milestone C decision):
- **Edge-based** timer driven by `DIV` bit transitions. No accumulator-based “period elapsed” timer is used in Milestone C.
- Rationale: edge-based semantics are required to model `DIV` reset and `TAC` toggle “glitch” behavior, which shows up in real test ROMs.

TAC semantics:
- bit 2: timer enable
- bits 1–0: input clock select

Divider semantics:
- Maintain an internal `div_counter` in t-cycles and increment it by `cycles`.
- Reading `DIV` returns `(div_counter >> 8) & 0xFF`.
- Writing any value to `DIV` resets `div_counter = 0` (and must be routed through MMIO).

TIMA increment semantics (target behavior):
- Define `timer_in = (timer_enabled && selected_div_bit)`; TIMA increments on the **falling edge** of `timer_in`.
- This means writes to `DIV` (reset) and `TAC` (enable/clock select) can create edges and must be modeled (at least to the extent required by the chosen tests).
- Divider bit selection:
  - `TAC & 0b11 == 0b00` → use divider bit 9 (period 1024 cycles)
  - `TAC & 0b11 == 0b01` → use divider bit 3 (period 16 cycles)
  - `TAC & 0b11 == 0b10` → use divider bit 5 (period 64 cycles)
  - `TAC & 0b11 == 0b11` → use divider bit 7 (period 256 cycles)

Implementation contract (what must be true, regardless of optimization strategy):
- `timer_tick(cycles)` advances time by exactly `cycles` t-cycles and produces the same TIMA/IF results as if those `cycles` were simulated one at a time.
- MMIO writes to `DIV` and `TAC` are instantaneous and must:
  - recompute `timer_in` before/after the write, and
  - if the write causes a `1 -> 0` transition of `timer_in`, immediately apply one TIMA increment (the “glitch” falling-edge).
- A later performance optimization (event-based counting) is allowed only if it preserves the above semantics.

Overflow + interrupt request:
- When TIMA increments from `0xFF -> 0x00`, an overflow occurs.
- After **4 cycles**, TIMA reloads from `TMA` and `IF` bit 2 (Timer) is set.

Write interactions during overflow (Milestone C policy):
- This milestone implements only what is required by the chosen tests.
- If a timer/interrupt ROM fails due to overflow-write window behavior, add the exact rule here and ship it with a micro-ROM reproducer.

### C0.4 Serial (SB/SC subset for Milestone C)

Registers:
- `SB` at `0xFF01` (serial data)
- `SC` at `0xFF02` (serial control)

Behavior (simplified, internal clock only):
- On write to `SB`: store the byte.
- On write to `SC` where `(val & 0x80) != 0` (start) and `(val & 0x01) != 0` (internal clock):
  - the transfer completes immediately:
    - append the current `SB` byte to the per-env serial buffer
    - clear bit 7 in `SC`
    - set `IF` bit 3 (Serial)

### C0.5 MMIO boundary requirement

- Reads/writes to `IF/IE/DIV/TIMA/TMA/TAC/SB/SC` MUST be routed through a centralized in-kernel MMIO layer (`read8/write8` or equivalent helpers).
- Instruction templates must not re-encode these side effects ad-hoc.

### C0.6 Acceptance checklist (maps to tests; exit 0/1)

- `IF/IE` reads/writes behave consistently (bit masking policy stable).
- Timer micro-ROM suite:
  - validates DIV increments and DIV reset behavior
  - validates TIMA increment rate for each TAC clock select
  - validates TIMA overflow → reload from TMA + `IF(Timer)` request
- Interrupt micro-ROM suite:
  - validates priority ordering and vector dispatch
  - validates EI delay and RETI IME restore
  - validates HALT wake + timer interrupt delivery
- External serial ROM:
  - serial output is captured in Warp
  - ROM “pass token” appears in captured serial stream

---

## Implementation Plan

### C1 — Add required per-env internal state (ABI + backend plumbing)

You’ll need per-env state that is not “just bytes in mem”:

Interrupt core:
- `ime: u8[num_envs]` (internal master enable)
- `ime_pending: u8[num_envs]` (EI delay)
- `halted: u8[num_envs]` (HALT needs explicit state)

Timer core:
- `div_counter: u32[num_envs]` (DIV is derived from this)
- `timer_prev_in: u8[num_envs]` (previous `timer_in` for edge detection; see **C0.3**)
- overflow reload: `tima_reload_pending: u8[num_envs]` + small `tima_reload_delay: u8[num_envs]`

Plumbing tasks:
- Extend the Warp kernel signature in `src/gbxcule/kernels/cpu_step_builder.py`’s skeleton.
- Update kernel launch inputs in `src/gbxcule/kernels/cpu_step.py` warmup and `src/gbxcule/backends/warp_vec.py`.
- If this counts as an ABI change per `ARCHITECTURE.md`, bump `ABI_VERSION` in `src/gbxcule/core/abi.py` with a short migration note.

### C2 — Consolidate MMIO semantics in-kernel (single source of truth)

In the generated kernel (skeleton in `src/gbxcule/kernels/cpu_step_builder.py`), ensure all register side effects route through MMIO helpers:

- `read8(addr)` / `write8(addr, val)` handles:
  - `IF/IE` reads/writes (mask to bits 0–4; define upper bits policy and keep it consistent)
  - `DIV` read from `div_counter`; `DIV` write resets divider and participates in edge detection (see **C0.3**)
  - `TIMA/TMA/TAC` reads/writes with required side effects
  - Serial `SB/SC`: on “internal clock start”, append byte to serial buffer and request `IF(Serial)`

This keeps the instruction templates minimal and preserves the repo’s “functional core / imperative shell” boundary.

### C3 — Timer implementation (cheap-first, with a path to edge-correct)

Implement the timer exactly as specified in **C0.3** (edge-based).

Work items:
- Implement `timer_tick(cycles)` that:
  - increments `div_counter` in t-cycles
  - computes `timer_in` before/after the tick and detects falling edges
  - applies the correct number of TIMA increments for the tick (handle multi-edge ticks)
- Implement `DIV` reset via MMIO as `div_counter = 0`, and ensure it participates in edge detection (because it can create a falling edge).
- Implement TAC writes via MMIO and ensure enable/clock-select changes participate in edge detection.
- Implement TIMA overflow:
  - detect increment from `0xFF -> 0x00`
  - schedule a 4-cycle delayed reload from `TMA`
  - request `IF(Timer)` at reload
- Keep overflow-write-window behavior minimal until forced by a failing ROM, then:
  - codify the exact rule in **C0.3**
  - add a micro-ROM reproducer

### C4 — Interrupt controller (IE/IF/IME + servicing)

Service interrupts at instruction boundaries and while halted.

Milestone C also completes the missing CPU control opcodes needed for interrupts:
- Add ISA specs + templates for `DI` (`0xF3`) and `EI` (`0xFB`) (EI delay per **C0.2**).
- `RETI` must restore `IME` (and clear any EI pending state).

Core loop behavior:
- Compute `pending = IE & IF & 0x1F`.
- If `pending != 0`:
  - If `halted`: clear `halted` (wake even when `IME=0`).
  - If `ime == 1`: service highest priority pending interrupt.

Servicing sequence:
- `ime = 0`
- Clear selected bit in `IF`
- Push current `PC` to stack (via `write8`, correct SP decrement + little-endian)
- `PC = vector`
- Add interrupt service cycles (target: 20 t-cycles)

IME rules:
- `DI`: clears `ime` immediately (also clear `ime_pending`).
- `EI`: sets `ime_pending=1`; after *the next instruction completes*, set `ime=1`.
- `RETI`: sets `ime=1` immediately after return.

### C5 — HALT support (required for timer/interrupt ROMs)

Minimal viable HALT behavior:
- `HALT` sets `halted=1`.
- While halted:
  - do not fetch/execute opcodes
  - advance time in small quanta (e.g. `cycles=4`), tick timers, and re-check interrupts
  - if an interrupt is pending, wake; if `ime==1` service immediately

(HALT-bug edge cases can be deferred until a test forces it.)

Milestone C also completes the missing CPU control opcode:
- Add ISA spec + template for `HALT` (`0x76`) that sets `halted=1` and consumes 4 cycles.

### C6 — Verification (verifiable rewards, inverted pyramid)

Repo-owned micro-suite (fast, deterministic):
- Extend `bench/roms/build_micro_rom.py` with 2–3 ROMs and register them in `bench/roms/suite.yaml`, e.g.:
  - `TIMER_DIV_BASIC.gb`: configure timer, run fixed cycles, write `DIV/TIMA` snapshots into WRAM.
  - `TIMER_IRQ_HALT.gb`: set up a timer ISR at `$0050` that writes a signature then `RETI`; main enables timer+interrupts then `HALT`.
  - `EI_DELAY.gb`: asserts EI delay semantics (EI; NOP; assert IME on next boundary).
- Add pytest verify tests comparing `pyboy_single` vs `warp_vec_cpu`, similar to:
  - `tests/test_joy_diverge_persist.py`
  - `tests/test_warp_vec_ws3_verify.py`
- Enable memory-region hashing for IO/HRAM during these tests when useful (e.g. `FF00:FFFF`) to catch IF/IE/TIMA drift cheaply.

External serial self-test ROM (DoD requirement, but keep CI hermetic):
- Add a skip-if-missing test that runs a ROM from a repo-local external path (not committed) and asserts the captured serial stream contains a known “pass” token.
- Record the serial buffer in mismatch bundles to preserve repro quality.

---

## Suggested file touch map (when implementing)

Kernel + ABI plumbing:
- `src/gbxcule/core/abi.py` (ABI version bump if needed)
- `src/gbxcule/kernels/cpu_step_builder.py` (MMIO helpers, new state arrays, timer/interrupt logic)
- `src/gbxcule/kernels/cpu_step.py` (warmup inputs updated)
- `src/gbxcule/backends/warp_vec.py` (allocate/init new buffers; pass into kernel)

Micro-ROMs + verification:
- `bench/roms/build_micro_rom.py` (add timer/interrupt ROM builders)
- `bench/roms/suite.yaml` (register new ROM IDs)
- `tests/` (new verify tests for the new ROMs; follow existing patterns)
