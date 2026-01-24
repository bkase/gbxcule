# Game Boy Emulator — Milestone F Plan (MBC1 + MBC3 + Real ROM Bring-Up)

Date: 2026-01-24

Source context:
- `history/gameboy-emu.md` → “Milestone F — MBC1 + MBC3 + ‘real ROM’ bring-up”
- `CONSTITUTION.md` → correctness-by-construction, functional core / imperative shell, verifiable rewards, inverted test pyramid, ≤2min suite spirit

Goal: implement **cartridge banking** (MBC1 + MBC3) and the supporting harness/debugging needed to run **real DMG ROMs** deterministically, while keeping the repo’s **fast correctness loop** (PyBoy as oracle, artifact bundles on mismatch).

Non-goals (Milestone F):
- Sound (APU).
- CGB-only features (double speed, CGB palettes/VRAM banking, etc.).
- Perfect cycle-exact “DMA blocks CPU” timing. (Copy correctness first; timing refinements later.)
- RTC wall-clock correctness for MBC3. (RTC must be deterministic; “real time” can be a later story.)

---

## Definition of Done (DoD)

- **MBC1:** ROM + RAM banking works for typical commercial ROMs; no ignored “writes to ROM” that should be bank control.
- **MBC3:** ROM + RAM banking works; RTC behavior is either fully implemented deterministically or explicitly stubbed deterministically (no host-time dependencies).
- **Boot ROM disable (0xFF50):** correctly switches mapping so cartridge vectors/code run when expected.
- **Harness determinism:** same ROM + same action trace + same seed ⇒ identical hashes/artifacts across runs.
- **Verifiable rewards:** adds a repo-owned suite of **license-safe, generated MBC micro-ROMs** that pass:
  - `pyboy_single` vs `warp_vec_cpu` (step-exact)
  - and `warp_vec_cpu` vs `warp_vec_cuda` where CUDA is available
- **Performance budget:** default CI-ish suite remains fast (aim at the “≤ 2 minutes” spirit from `CONSTITUTION.md`).

---

## Dependencies / Preconditions (Milestones A–E “done enough”)

Milestone F becomes dramatically easier if these earlier architectural constraints are already in place:

- **Central MMIO boundary:** instruction templates call `read8/write8` (and `read16/write16` composed of byte ops), not direct `mem[...]` pokes. This is required so “ROM writes” can become MBC control writes rather than ignored writes.
- **Stable timing-ish core:** enough CPU + interrupts + timers + PPU to reach a steady “gameplay loop” and to generate meaningful, deterministic hashes.

If the repo isn’t there yet, do the minimal refactor required to centralize ROM/RAM behavior before implementing MBC logic; otherwise you will end up with scattered banking logic across templates.

---

## High-Level Design Decision (Correctness by Construction)

### Stop treating ROM as “just bytes in the 64KB mem array”

Today, `warp_vec` loads ROM bytes into the 64KB `mem` slice per env. This works only for ROM-only micro-ROMs and breaks down immediately with banking.

For Milestone F, the “cartridge” should be modeled explicitly:

- **ROM**: immutable byte buffer, shared across envs (one copy).
- **Cart RAM**: explicit per-env byte backing store.
- **Cart state**: explicit per-env registers and flags (MBC kind, banking regs, RAM enable, bootrom enabled, RTC state).

This aligns with `CONSTITUTION.md`:
- invalid states become unrepresentable (you can’t “accidentally write to ROM bytes”),
- functional core (pure mapping rules) + imperative shell (MMIO side effects),
- tests and artifacts provide verifiable rewards.

---

## F.0 — Cartridge Header Parsing (Host-Side, Pure)

Implement a small, pure cartridge header parser:

- Read and validate:
  - `0x0147` cart type → `ROM_ONLY | MBC1 | MBC3` (+ flags: RAM/BATTERY/RTC)
  - `0x0148` ROM size code → derive ROM bank count, total ROM bytes
  - `0x0149` RAM size code → derive RAM bank count and RAM bytes
- Precompute invariants for fast runtime mapping:
  - `rom_bank_count`, optional `rom_bank_mask` (only if power-of-two; otherwise clamp/mod)
  - `ram_bank_count`, `ram_bytes`
  - `has_rtc`, `has_battery`
- Fail fast on unsupported cart types (exit 1 with explicit message).

Deliverables:
- A small struct/dataclass (or TypedDict) that becomes the single “cartridge spec” source of truth used by backends.

---

## F.1 — ABI / Buffer Layout Upgrade (Warp Backends)

### Required new buffers

Add buffers that make banking explicit:

- `rom`: `u8[rom_len]` (shared, read-only)
- `cart_ram`: `u8[num_envs * cart_ram_len]` (per-env)
- `cart_state`: small per-env arrays (or a few arrays) for:
  - `mbc_kind` (ROM_ONLY/MBC1/MBC3)
  - `ram_enable`
  - `bootrom_enabled`
  - `rom_bank_lo`, `rom_bank_hi_or_mode`, `bank_mode` (MBC1)
  - `mbc3_rom_bank`, `mbc3_ram_or_rtc_sel`, latch state (MBC3)
  - optional deterministic RTC counters + latched snapshot

### Reset wiring

Update `WarpVec*Backend.reset()` semantics:

- ROM loaded once, not duplicated into per-env `mem[0..0x8000)`.
- Boot ROM overlay behavior remains supported via explicit bootrom mapping (not by mutating ROM bytes).
- Per-env state initialized deterministically:
  - RAM disabled by default
  - banking regs set to power-on defaults
  - bootrom enabled (until `0xFF50` is written)

Constraints:
- CUDA readback limitations still apply; don’t add features that require `write_memory` on CUDA.
- Keep CPU-only debugging ergonomics (state + memory reads) strong.

---

## F.2 — MMIO Routing for Banked Regions (Kernel-Side)

Implement `read8/write8` rules for cartridge regions:

### `read8(addr)`

- `0x0000–0x3FFF` (fixed ROM region):
  - if bootrom enabled and addr < 0x0100: read boot ROM
  - else: read ROM bank 0 mapping (may be affected by MBC1 mode)
- `0x4000–0x7FFF` (switchable ROM region):
  - read ROM using the active ROM bank number for the current MBC
- `0xA000–0xBFFF` (external RAM / RTC):
  - if RAM not enabled: return open bus (conventionally `0xFF`)
  - else if MBC3 and RTC reg selected: read RTC register (deterministic)
  - else: read external RAM bank

### `write8(addr, val)`

- `0x0000–0x7FFF`:
  - ROM_ONLY: ignore
  - MBC1/MBC3: treat as MBC control register write (banking, RAM enable, RTC latch)
- `0xA000–0xBFFF`:
  - if RAM enabled: write RAM (or RTC regs in MBC3)
  - else: ignore
- `0xFF50`:
  - on non-zero write: disable boot ROM mapping (deterministically; one-way)

Observability on failure (per `CONSTITUTION.md`):
- record a small “cartridge debug snapshot” on mismatch:
  - current ROM bank, RAM bank/RTC select, RAM enable, mode bits, bootrom enabled
  - last N MBC writes (addr,val,pc,cycle) in a ring buffer (dumped only on failure)

---

## F.3 — MBC1 Implementation (ROM + RAM Banking)

### MBC1 register decoding (writes in 0x0000–0x7FFF)

- `0x0000–0x1FFF`: RAM enable (`(val & 0x0F) == 0x0A`)
- `0x2000–0x3FFF`: ROM bank low 5 bits (bank 0 becomes bank 1 for the switchable region)
- `0x4000–0x5FFF`: upper 2 bits (ROM high bits or RAM bank depending on mode)
- `0x6000–0x7FFF`: mode select (0 = ROM banking, 1 = RAM banking)

### Mapping rules (reads)

- Fixed region `0x0000–0x3FFF`:
  - mode 0: bank 0
  - mode 1: bank `(upper_bits << 5)` (clamped/masked to available banks)
- Switchable region `0x4000–0x7FFF`:
  - bank `((upper_bits << 5) | low5)` with “bank 0 fixup” + clamping/masking
- Ext RAM `0xA000–0xBFFF`:
  - mode 0: RAM bank 0
  - mode 1: RAM bank = `upper_bits`

Edge cases to explicitly test:
- ROM sizes where computed bank exceeds available banks.
- RAM absent but RAM-enable toggled (must remain deterministic).
- Mode toggling effects on the `0x0000–0x3FFF` mapping (bank 0 vs shifted bank).

---

## F.4 — MBC3 Implementation (ROM + RAM + Deterministic RTC)

### MBC3 register decoding

- `0x0000–0x1FFF`: RAM enable
- `0x2000–0x3FFF`: ROM bank (7-bit; 0 becomes 1 for the switchable region)
- `0x4000–0x5FFF`: RAM bank (0–3) or RTC register select (0x08–0x0C)
- `0x6000–0x7FFF`: latch clock sequence (0 → 1 edge captures snapshot)

### Deterministic RTC strategy

Hard constraint: **no wall clock**.

Options (pick one and make it explicit):
1. Implement RTC counters driven from emulated cycles/frames (recommended).
2. Stub RTC registers to constant values but keep latch semantics deterministic (acceptable only if it unblocks ROM bring-up and is clearly marked).

If implementing RTC counters now:
- Track “RTC time” in a deterministic unit (e.g., seconds derived from cycle count / 4_194_304Hz).
- Implement latch to freeze a snapshot used for reads until next latch.

---

## F.5 — License-Safe MBC Micro-ROM Suite (Verifiable Rewards)

Extend the repo-owned ROM generator (preferred) to create deterministic tests that validate mapping without copyrighted content:

- `MBC1_SWITCH.gb`: write code/data in multiple banks, switch banks, write signatures to WRAM.
- `MBC1_RAM.gb`: enable RAM, write per-bank markers, switch, read back, write PASS signature.
- `MBC3_SWITCH.gb`: same for MBC3 ROM banking.
- `MBC3_RAM.gb`: RAM banking + enable gating.
- Optional `MBC3_RTC.gb`: latch + read RTC regs deterministically (only if RTC is implemented).

Add fast verify tests:
- `pyboy_single` vs `warp_vec_cpu` step-exact loop for each new ROM.
- CUDA parity tests where available, with minimal memory readback regions (avoid dumping huge buffers).

Keep the test pyramid inverted:
- unit tests for header parsing + bank-number computation helpers
- micro-ROM verify tests for end-to-end behavior
- “real ROM” tests remain opt-in (see F.6)

---

## F.6 — Real ROM Bring-Up Loop (Deterministic, Cheap-by-Default)

Add a harness mode intended for local iteration:

- Fixed action schedule + fixed seed.
- Run for N frames.
- Produce deterministic per-frame hashes (or per-VBlank hashes), and compare to PyBoy on the same inputs.
- On mismatch:
  - dump cartridge state snapshot + last N MBC writes
  - dump a small set of memory regions (WRAM, HRAM, key IO regs)
  - dump the first differing frame hash index (if hashing is enabled)

Constraints:
- Don’t ship copyrighted ROMs in-repo.
- Provide a documented local-only path for users to point to their own ROMs.
- Keep CI suite fast; gate heavy “real ROM” bring-up behind env var or `--runslow`.

---

## F.7 — Acceptance Checklist (Finish Line)

- MBC1 micro-ROMs pass `pyboy_single` vs `warp_vec_cpu` verification.
- MBC3 micro-ROMs pass `pyboy_single` vs `warp_vec_cpu` verification.
- (If CUDA available) CPU vs CUDA parity holds for the MBC micro-ROM suite.
- Boot ROM disable (`0xFF50`) behavior matches PyBoy for scenarios covered by tests.
- At least one user ROM that uses **MBC1** reaches a stable gameplay loop with deterministic hashes vs PyBoy for a fixed action trace.
- At least one user ROM that uses **MBC3** reaches a stable gameplay loop with deterministic hashes vs PyBoy for a fixed action trace.
- No nondeterminism remains (no host time; no uninitialized RAM; no “random open bus”).

---

## Suggested “File Touch Map” (for the implementation phase)

Host-side cartridge parsing + wiring:
- `src/gbxcule/backends/warp_vec.py` (load ROM once; initialize cart state; allocate cart RAM)
- `src/gbxcule/backends/pyboy_single.py` (optional: expose frame hashes / IO state hooks for bring-up harness)
- `src/gbxcule/core/abi.py` (if new canonical layouts are introduced)
- New: `src/gbxcule/core/cartridge.py` (or `core/cart.py`) for header parsing/types

Kernel-side MMIO + cartridge mapping:
- `src/gbxcule/kernels/cpu_step_builder.py` (inject `read8/write8`, route cartridge regions, bootrom mapping)
- `src/gbxcule/kernels/cpu_templates/*` (remove ad-hoc “ignore ROM writes” logic in templates once MMIO owns it)

Tests + ROM generators:
- `bench/roms/build_micro_rom.py` (add MBC ROM generators)
- `bench/roms/suite.yaml` (register new ROM ids)
- New/extend: `tests/test_warp_vec_ws*_*.py` for MBC verification vs PyBoy

