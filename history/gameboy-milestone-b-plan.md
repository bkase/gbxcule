# Game Boy Emulator — Milestone B Plan (CPU ISA Complete)

Date: 2026-01-24

Source context:
- `history/gameboy-emu.md` → “Milestone B — CPU ISA complete (unprefixed + CB)”
- `CONSTITUTION.md` → correctness-by-construction, verifiable rewards, inverted test pyramid

Goal: implement the full SM83/LR35902 CPU instruction set (**256 unprefixed + 256 CB-prefixed opcodes**) with correct flags and cycle counts, keeping the repo’s fast, step-exact verification loop against PyBoy.

Non-goals (Milestone B):
- Timers, interrupts, HALT/IME edge cases beyond what’s needed for ISA bring-up (these are Milestone C).
- PPU rendering (Milestone D/E) and MBC/DMA (Milestone F/E).

---

## Definition of Done (DoD)

- **100% opcode coverage:** every opcode in `0x00..0xFF` and every CB opcode `0x00..0xFF` maps to an explicit implementation (no “default handler” for correctness runs).
- **Correctness ladder (repo-owned):** a growing suite of **repo-generated micro-ROMs** pass step-exact verification vs PyBoy (family-by-family).
- **Fail loud:** “unknown opcode” becomes a **trap** in verify/debug mode (no silent `pc+1`), with enough state recorded to reproduce.
- **Fast feedback:** verification remains cheap (aim for the “≤ 2 minutes full suite” spirit from `CONSTITUTION.md`).

---

## Assumptions / Dependencies

Recommended prerequisite (Milestone A):
- A real MMIO boundary (`read8/write8`) and serial capture. This isn’t strictly required for ISA completion if you lean on PyBoy state diffs, but it makes debugging + future external self-test ROMs dramatically easier.

If Milestone A is not done yet:
- Keep using **step-exact PyBoy vs Warp** as oracle.
- Prefer micro-ROMs that validate by writing signatures into WRAM (and use `--mem-region` hashing or direct state comparisons).

---

## Implementation Plan

### Template organization rule (maintainability)

Do **not** create one giant “all CPU templates” file. Keep templates split by opcode family so they stay readable and reviewable:

- Keep existing modules focused: `cpu_templates/loads.py`, `cpu_templates/alu.py`, `cpu_templates/jumps.py`, `cpu_templates/misc.py`.
- Add new modules as scope expands (recommended for Milestone B):
  - `cpu_templates/stack.py` (PUSH/POP/CALL/RET/RST helpers)
  - `cpu_templates/bitops.py` (CB RLC/RRC/RL/RR/SLA/SRA/SWAP/SRL + BIT/RES/SET + unprefixed rotates)
  - optionally `cpu_templates/control.py` (DI/EI/HALT/STOP if you pull any into B)
- The ISA table should reference templates from these modules; each module should stay “one family, one file”.

### B0 — Create an ISA “spec table” + coverage gate

Intent: “Code is cheap, specs are precious.” The ISA table becomes the single authoritative mapping from opcode → semantics/cycles.

Deliverables:
- Add a canonical opcode inventory for both tables:
  - Unprefixed: 256 entries
  - CB-prefixed: 256 entries
- Each entry includes:
  - opcode (byte)
  - mnemonic/name (string, human)
  - length (bytes)
  - cycles (and conditional cycles when applicable)
  - template function reference + replacement map
  - tags/group (loads/alu/branch/cb-bit/etc.)
- Add a hard unit test:
  - fails on missing/duplicate opcodes
  - fails if any opcode is mapped to a “default” template
  - optionally reports coverage percentage and the missing list for fast iteration

Suggested files:
- New: `src/gbxcule/core/isa_sm83.py` (or similar)
- Tests: add/extend `tests/test_cpu_step_builder.py` or a new `tests/test_isa_coverage.py`

### B1 — Kernel support for full ISA (CB prefix + scalable dispatch + trap)

Intent: avoid “giant if/elif” compile pain and make wrong opcodes unmistakable.

Work items:
- Add CB-prefixed control flow in the generated kernel:
  - fetch `opcode`
  - if `opcode == 0xCB`: fetch `cb = read8(pc+1)`, advance `pc += 2`, dispatch CB table, add CB cycles
  - else: normal dispatch (unprefixed)
- Replace the linear dispatch chain with a two-level dispatch:
  - e.g., bucket by `opcode >> 4` (16 buckets), then dispatch within bucket
  - keeps generated code depth manageable for Warp compilation
- Add a verify/debug “trap” mechanism:
  - records `trap_pc`, `trap_opcode`, `trap_kind` (unknown opcode / unimplemented / invariant broken)
  - stops stepping that env deterministically
  - surfaced via `get_cpu_state()` and mismatch bundle metadata

Suggested files:
- `src/gbxcule/kernels/cpu_step_builder.py` (dispatch/trap injection)
- `src/gbxcule/kernels/cpu_step.py` (wiring tables into kernel generation)
- Potential ABI additions in `src/gbxcule/core/abi.py` if you need new per-env arrays for trap state

### B2 — Implement “loads & moves” family (keeps everything runnable)

Order matters: implement families that let you run more ROM code quickly and reduce false mismatches.

Opcode groups:
- Register-to-register `LD r,r`
- `LD r,d8`, `LD rr,d16`
- `(HL)` addressing forms: `LD r,(HL)` and `LD (HL),r`
- Absolute/IO forms: `LD (a16),A`, `LD A,(a16)`, `LDH (a8),A`, `LD A,(a8)`
- Increment/decrement HL forms: `LDI/LDD` variants (if you choose to include them here)
- Stack moves: `PUSH/POP` (can be deferred to B5, but early support helps more tests)

Testing:
- Add multiple micro-ROMs that exercise a tight subset each and write known WRAM signatures.
- Keep each ROM short and deterministic.

Suggested files:
- Extend templates under `src/gbxcule/kernels/cpu_templates/loads.py`
- Extend ROM generator `bench/roms/build_micro_rom.py` + register in `bench/roms/suite.yaml`
- Add/extend verify tests similar to `tests/test_warp_vec_ws3_verify.py` for each new ROM

### B3 — Implement “ALU8 + flags” family (most bug-prone)

Intent: “Correctness by construction.” Centralize flag logic so every opcode uses the same truth.

Opcode groups:
- `ADD/ADC/SUB/SBC/AND/OR/XOR/CP` (r, (HL), d8)
- `INC/DEC r` and `INC/DEC (HL)`
- `DAA`, `CPL`, `SCF`, `CCF`

Implementation guidance:
- Add shared helpers for add/sub/adc/sbc to compute Z/N/H/C correctly and reuse everywhere.
- Confirm carry/half-carry edge cases via micro-ROMs and PyBoy diffs.

Testing:
- “flags torture” ROMs (half-carry boundaries, carry boundaries, zero results, DAA cases).

Suggested files:
- Extend templates `src/gbxcule/kernels/cpu_templates/alu.py`
- Add new ROM builders in `bench/roms/build_micro_rom.py`

### B4 — Implement “ALU16 + SP” family

Opcode groups:
- `ADD HL,rr`
- `INC/DEC rr`
- `ADD SP,e8` and `LD HL,SP+e8` (signed add; special flag rules)

Testing:
- Micro-ROM that deterministically walks SP/HL through signed offsets and validates results in WRAM.

### B5 — Implement “control flow + stack” family

Opcode groups:
- `JP/JR` (unconditional + all conditions)
- `CALL/RET` (unconditional + all conditions)
- `RST`
- `PUSH/POP` (AF/BC/DE/HL) if not already done

Implementation guidance:
- Make conditional cycle counts correct; cycle bugs will show up as drift vs PyBoy in this repo’s step framing.
- Ensure stack push/pop order matches SM83 (low/high byte order correctness matters).

Testing:
- Micro-ROMs that:
  - build a small call tree and verify stack contents
  - exercise each conditional branch direction deterministically

### B6 — Implement “rotates/shifts + CB bitops” family

Opcode groups:
- Unprefixed: `RLCA/RLA/RRCA/RRA`
- CB: `RLC/RRC/RL/RR/SLA/SRA/SWAP/SRL`
- CB: `BIT/RES/SET` for all registers and `(HL)`

Testing:
- Separate ROMs for:
  - shift/rotate algebra validation
  - BIT/RES/SET correctness on both regs and `(HL)`

---

## Verification Loop (“verifiable rewards”)

For each sub-milestone (B2→B6):
- Add/extend templates + opcode table entries
- Add at least one micro-ROM that targets the new behavior
- Add a small step-exact verify test vs PyBoy for that ROM
- Only then proceed to the next family

Safety rails:
- Keep trap-on-unknown enabled in verify/debug (never silently advance `pc`).
- Prefer many small ROMs over one giant one (faster to isolate regressions).
- Keep the test pyramid inverted:
  - unit-level coverage tests (ISA table + mapping completeness)
  - micro-ROM step-exact diffs
  - only later: heavier external ROM suites once serial/MMIO/interrupts/PPU exist

---

## Suggested “file touch map” (for later implementation)

- ISA spec/coverage:
  - `src/gbxcule/core/isa_sm83.py` (new)
  - `tests/test_isa_coverage.py` (new) or extend `tests/test_cpu_step_builder.py`
- Kernel generation:
  - `src/gbxcule/kernels/cpu_step_builder.py` (CB dispatch + scalable dispatch + trap)
  - `src/gbxcule/kernels/cpu_step.py` (consume spec table; build templates list)
- Templates:
  - `src/gbxcule/kernels/cpu_templates/loads.py`
  - `src/gbxcule/kernels/cpu_templates/alu.py`
  - `src/gbxcule/kernels/cpu_templates/jumps.py`
  - `src/gbxcule/kernels/cpu_templates/stack.py` (new; suggested)
  - `src/gbxcule/kernels/cpu_templates/bitops.py` (new; suggested)
  - `src/gbxcule/kernels/cpu_templates/misc.py` (trap/default semantics)
- Micro-ROMs:
  - `bench/roms/build_micro_rom.py` (new ROMs)
  - `bench/roms/suite.yaml` (register new ROM IDs)
  - `tests/test_warp_vec_ws3_verify.py` (or new verify tests for new ROMs)
