Below is a concrete, end-to-end plan to get from **where this repo is today** to a **working DMG (original Game Boy) emulator** with **full CPU**, **interrupts + timers**, **PPU rendering + LCD STAT/VBlank interrupts**, **DMA**, **MBC**, and a **tight incremental correctness loop** using **PyBoy as oracle** plus **test ROMs**. (No sound.)

---

## 0) Define “working emulator” for this repo

A reasonable “done” definition (DMG-only) that matches your goals:

1. **CPU:** Implements the full LR35902/SM83 instruction set (including CB-prefixed), correct flags, correct cycle counts, correct edge cases (HALT/IME/RETI behavior).
2. **Memory map:** Correct enough for real ROMs:
   - ROM banking (MBC1 at minimum; add MBC3 next)
   - VRAM, WRAM, OAM, HRAM
   - IO registers with side effects (timers, LCD, joypad, serial, DMA, boot ROM disable)

3. **Interrupts:** IE/IF, IME, priority order, vectors, timing.
4. **Timers:** DIV/TIMA/TMA/TAC and IF(Timer) behavior.
5. **PPU:** Timing (modes/LY/STAT), VBlank, background rendering at minimum; then window + sprites.
6. **Output for debugging:** serial output and/or framebuffer dumps, plus good mismatch artifacts.
7. **Correctness harness:** passes a ladder of tests: micro-ROMs → blargg/mooneye → frame hash comparisons vs PyBoy → run a couple of homebrew ROMs.

---

## 1) Debug output first: **do serial out before PPU**

You asked: “serial out or implement enough PPU for visual feedback?”

**Do serial out first.** It’s vastly cheaper and unlocks the whole ecosystem of CPU/timer/interrupt test ROMs that report PASS/FAIL over the serial registers.

### Why it’s feasible even in Warp

Warp kernels can’t `print`, but they _can write bytes to a buffer_. We emulate the classic serial behavior: ROM writes a byte to **SB (0xFF01)** then writes **SC = 0x81** (start + internal clock). Typical behavior: bit 7 clears when done and the serial interrupt may be requested. ([Retrocomputing Stack Exchange][1])

### Concrete design (Warp + PyBoy)

Add per-env buffers:

- `serial_buf: u8[num_envs * SERIAL_MAX]`
- `serial_len: u32[num_envs]`
- (optional) `serial_overflow: u8[num_envs]`

Implement in the memory write path:

- On write to `0xFF01` → store SB
- On write to `0xFF02`:
  - if `(value & 0x80) != 0` and internal clock selected (`value & 0x01`) → “complete” transfer immediately:
    - append SB byte to `serial_buf`
    - clear bit 7 in SC (SC becomes `value & 0x7F` or just `0x01`)
    - set IF bit for Serial (request interrupt) (ties into your interrupt controller) ([Retrocomputing Stack Exchange][1])

**This single feature gives you immediate “printf debugging” and makes blargg/mooneye-style tests practical.**

---

## 2) Restructure now: introduce a real **MMIO layer** in the kernel

Right now templates directly poke `mem[...]` and special-case JOYP at `0xFF00`. That won’t scale.

### New rule

All templates should go through:

- `read8(addr)` / `write8(addr, val)`
- `read16(addr)` / `write16(addr, val)` (composed of byte ops)

…and those functions encode:

- ROM vs VRAM vs WRAM vs OAM vs IO semantics
- special registers (JOYP, serial, timers, LCD/STAT, DMA, IF/IE, boot disable, etc.)
- MBC “writes” to ROM region (no longer ignored!)

**Implementation approach in this repo**
Because you already generate a full kernel source string, put these helpers _into the skeleton_ in `cpu_step_builder.py` and have templates call them.

This also creates a clean “functional core / imperative shell” inside the kernel: instruction semantics remain mostly pure, and MMIO side effects are centralized.

---

## 3) CPU completion plan (using your template system)

You’re correct: “next step is do all CPU instructions, test thoroughly.” This is the largest chunk; make it systematic.

### 3.1 Build an instruction inventory + coverage gate

Create a single authoritative table:

`src/gbxcule/core/isa_sm83.py` (or `isa.py`) containing for each opcode:

- mnemonic
- length
- cycles (and conditional cycles)
- template function + replacement mapping
- group tags (ALU, load, branch, CB-bit, etc.)

Then add a unit test:

- **“no unknown opcodes”**: once you flip into “complete CPU mode”, your default handler should become “trap” (or `halt` + set an error flag) rather than silently `pc+1`. This prevents accidentally “working” while being wrong.
- **coverage metric**: report % of (unprefixed + CB) mapped to non-default templates.

### 3.2 Fix your dispatch structure before it gets huge

Your builder currently generates a deep `if/elif/elif/...` chain. At 256 + 256 CB ops this will likely become unpleasant to compile.

Upgrade `cpu_step_builder._build_dispatch_tree` to a **two-level dispatch**:

- dispatch on `opcode >> 4` (16 buckets)
- inside each bucket dispatch on low nibble or explicit opcode checks

This keeps depth small and compilation stable.

### 3.3 Add CB-prefixed instruction support cleanly

In the kernel skeleton:

- Fetch opcode
- If opcode == 0xCB:
  - read `cb = read8(pc+1)`
  - `pc += 2`
  - dispatch CB templates
  - cycles add CB cycles
    Else:
  - normal dispatch uses `pc` increment rules inside templates

### 3.4 Implement instructions by **families** (order matters)

A workable progression that keeps you runnable continuously:

1. **Core loads**
   - LD r,r / LD r,(HL) / LD (HL),r
   - LD r,d8
   - LD rr,d16
   - LD (a16),A / LD A,(a16)
   - LDH (a8),A / LD A,(a8)
   - PUSH/POP (AF, BC, DE, HL)

2. **8-bit ALU**
   - ADD/ADC/SUB/SBC/AND/OR/XOR/CP (all addressing modes)
   - INC/DEC r and (HL)
   - DAA, CPL, SCF, CCF

3. **16-bit + SP**
   - ADD HL,rr
   - INC/DEC rr
   - ADD SP,e8 / LD HL,SP+e8

4. **Control flow**
   - JP/JR (conditional/unconditional), CALL/RET/RETI, RST

5. **Rotates/shifts + bit ops**
   - RLCA/RLA/RRCA/RRA
   - CB: RLC/RRC/RL/RR/SLA/SRA/SWAP/SRL, BIT/RES/SET

6. **CPU control**
   - DI/EI
   - HALT/STOP (and the HALT edge case once interrupts exist)

At the end of each family, you add a **micro-ROM** plus a **PyBoy differential test** that is family-specific.

---

## 4) Testing strategy: incremental, brutal, and cheap

You already have the right harness primitives: step-exact verify, memory region hashing, mismatch bundles, replayable actions.

Now add three new “tiers” of tests.

### Tier A — Generated micro-ROMs (repo-owned, deterministic)

Extend `bench/roms/build_micro_rom.py` with additional ROMs:

- **Opcode micro-suites**: small ROMs each stressing a tight subset (e.g., “CB_BIT”, “STACK_CALL_RET”, “DAA_EDGE”, “HALT_IME”).
- **MMIO ROMs**: JOYP/serial/timer/LCD register behavior.
- **Interrupt ROMs**: set IE/IF and check correct vector + stack push, etc.

Use your existing verify tests:

- `pyboy_single` vs `warp_vec_cpu` step-exact
- later `warp_vec_cpu` vs `warp_vec_cuda`
- optionally enable `--mem-region` for IO/WRAM windows.

### Tier B — Serial-output “self-check” ROMs (external + internal)

Once serial capture exists, you can use well-known test ROMs that print progress/results to the serial port by writing SB/SC. ([Retrocomputing Stack Exchange][1])

Add a harness mode:

- `--until-serial "Passed"` or `--serial-timeout-frames N`
- dumps captured serial bytes into the artifact/mismatch bundle

This enables:

- CPU instruction tests
- timer/interrupt tests
- many “mooneye-style” ROMs
  without needing PPU at all.

### Tier C — Framebuffer comparison vs PyBoy (PPU stage)

Once PPU exists, compare frames via hashes:

- PyBoy exposes `pyboy.screen.ndarray` as RGBA `(144, 160, 4)`. ([Pyboy][2])
- Hash RGBA bytes per frame (blake2b) and compare.
- Store last N frame hashes in artifacts; on mismatch dump the first differing frame.

Also: PyBoy warns that `tick()` doesn’t return at a specific point; use hooks for precise sampling when needed. ([Pyboy][2])

---

## 5) Interrupts + timers: implement _before_ PPU correctness

PPU correctness depends on correct timing and interrupt behavior, so wire the CPU-side pieces first.

### 5.1 Interrupt controller (IE/IF/IME)

Implement:

- `IE` at `0xFFFF`
- `IF` at `0xFF0F`
- `IME` internal flag (not memory-mapped)

Service rules + vectors + priority are well specified:

- IF bits request interrupts; servicing requires IME + IE bit. ([gbdev.io][3])
- Handler call pushes PC and jumps to vectors `$40,$48,$50,$58,$60`. ([gbdev.io][3])

Kernel changes:

- Track `ime: u8[num_envs]`
- Each instruction boundary: if `ime && (IE & IF) != 0`:
  - clear IME
  - clear IF bit for the selected interrupt
  - push PC to stack (via write8)
  - PC = vector
  - add the interrupt service cycles

### 5.2 Timer (DIV/TIMA/TMA/TAC)

Add per-env timer state (explicit arrays, not “just bytes in mem”):

- `div_counter: u16 or u32`
- `tima/tma/tac: u8`
- `timer_accum: u32` or edge-based logic

Update on **CPU cycles**, not frames:

- increment divider continuously
- timer increments according to TAC frequency when enabled
- on TIMA overflow: reload TMA and request Timer interrupt (IF bit 2)

**Test it via serial-output ROMs** (mooneye-style), and a repo-generated micro-ROM that checks known increments.

---

## 6) PPU plan: build it in stages, validate with hashes

Once CPU + interrupts + timer are solid enough, start PPU.

### 6.1 Decide architecture: separate kernel first, then fuse

For correctness and debuggability:

1. Implement `ppu_step` as a separate Warp kernel (`src/gbxcule/kernels/ppu_step.py`), advancing PPU state by cycles (or by a line quantum).
2. Integrate it in the backend step loop:
   - CPU runs and produces “cycles elapsed”
   - PPU consumes cycles elapsed and updates LY/mode/framebuffer/IF flags

3. Once correct, optionally fuse CPU+PPU later for performance.

### 6.2 Minimal PPU correctness sequence (DMG)

1. **Timing only**
   - dot counter, LY, mode transitions, STAT bits, VBlank period
   - request VBlank interrupt at LY=144 entry (IF bit 0)
   - request STAT interrupts for enabled conditions (optional initially, but eventually required)

2. **Background rendering**
   - Implement tile fetch from VRAM, SCX/SCY, BGP palette
   - Render to an RGBA or indexed framebuffer:
     - For debug: store only env0, or store per-env hashes to avoid huge memory.

3. **Window**

4. **Sprites**
   - OAM scan rules, priority, palettes OBP0/OBP1

### 6.3 Frame output in harness

Add flags:

- `--dump-frame-every K` (or `--dump-frame-at N`)
- Write PNG for env0 to `bench/runs/...`
- Also store `frame_hashes` in artifacts

Use PyBoy’s screen buffer as oracle: `pyboy.screen.ndarray` gives RGBA frames. ([Pyboy][2])

---

## 7) Memory map + MBC + DMA: required for real ROMs

### 7.1 Boot ROM disable

Implement `0xFF50` write: when written non-zero, boot ROM is unmapped and reads from `0x0000..` come from cartridge ROM.

### 7.2 MBC

Stop ignoring writes to ROM space in `write8`. For real ROMs, writes to `0x0000–0x7FFF` often control banking.

Implement in order:

1. **ROM-only (no MBC)** (already basically works)
2. **MBC1** (most common)
3. **MBC3** (RTC can be stubbed initially; but banking must work)

Add per-env cart state arrays:

- `rom_bank`, `ram_bank`, `ram_enable`, `bank_mode`, etc.

### 7.3 DMA (OAM DMA)

Implement write to `0xFF46`:

- triggers copy `160 bytes` from `(val << 8)` to `0xFE00..0xFE9F`
- block timing accuracy can be simplified at first, but copy must occur

This is required for sprites in most games and many tests.

---

## 8) How to keep correctness sane while you scale complexity

These are the guardrails that prevent “it boots sometimes” syndrome:

1. **Three-oracle ladder**
   - PyBoy = external oracle
   - Warp CPU = debug oracle
   - Warp CUDA = perf target
     You only debug one delta at a time.

2. **Every new hardware feature ships with a test ROM**
   - If you implement “DMA”, you also add a ROM that triggers DMA and validates OAM contents (via WRAM writes or serial out).

3. **Mismatch bundles get richer, not louder**
   - Add: serial buffer snapshot, last N opcodes/PCs trace buffer, last N interrupt events
   - Keep default logs quiet; dump structured state only on failure.

4. **Sampling boundary discipline**
   - PyBoy warns `tick()` doesn’t return at a specific point; if you need instruction-exact sampling, use hooks. ([Pyboy][2])
     For most regression tests, prefer “self-check ROMs via serial” and “frame hash at VBlank”.

---

## 9) Suggested concrete milestone ladder (from today → “plays games”)

### Milestone A — Serial output + MMIO layer

**DoD**

- Warp backends capture serial bytes (SB/SC semantics).
- Harness can print/dump serial output.
- New micro-ROM: `SERIAL_HELLO.gb` prints “OK”.

### Milestone B — CPU ISA complete (unprefixed + CB)

**DoD**

- Opcode coverage gate hits 100% mapped templates.
- A family of repo-generated ROMs pass vs PyBoy.
- “Default” opcode handler becomes a trap in debug mode.

### Milestone C — Interrupts + timers correct

**DoD**

- Timer overflow requests IF(Timer), IME/IE gating works, RETI restores IME.
- Serial interrupt requested on transfer completion (ties serial out to IF). ([gbdev.io][3])
- Pass a timer/interrupt micro suite + at least one external serial self-test ROM.

### Milestone D — PPU timing + VBlank + background rendering

**DoD**

- LY/modes correct enough to drive VBlank interrupts.
- Framebuffer hash matches PyBoy for 1–2 deterministic graphics ROMs using `pyboy.screen.ndarray`. ([Pyboy][2])

### Milestone E — Window + sprites + STAT interrupts + DMA

**DoD**

- Sprites appear correctly on known test ROMs.
- DMA works.
- Frame hashes match PyBoy on a broader set.

### Milestone F — MBC1 + MBC3 + “real ROM” bring-up

**DoD**

- You can run user-provided ROMs that use MBC1/MBC3 and reach gameplay.
- Deterministic stepping still holds under harness capture (same seed/actions, same hashes).

---

If you want, I can turn this into a **repo-native engineering plan** in your existing “workstream” style (WS0…WSN) with:

- exact file-level diffs you’ll need,
- the precise new kernel ABI fields (serial buffers, IME, IE/IF, timer state, PPU state, framebuffer),
- and the exact test ROMs to generate next (with the assembly-bytes sketches).

[1]: https://retrocomputing.stackexchange.com/questions/12549/how-does-the-gameboy-link-cable-work "emulation - How does the Gameboy Link Cable work? - Retrocomputing Stack Exchange"
[2]: https://docs.pyboy.dk/ "pyboy API documentation"
[3]: https://gbdev.io/pandocs/Interrupts.html "Interrupts - Pan Docs"
