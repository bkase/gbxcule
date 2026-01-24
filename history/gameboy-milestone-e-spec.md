# Milestone E PPU Spec (Scanline-Latch Model)

Date: 2026-01-24

This spec defines the concrete behavioral model for Milestone E (window, sprites,
STAT interrupts, DMA) under the **scanline-latch** approach established in
Milestone D. If behavior or assumptions change, update this file and the tests
together.

---

## Global Model

- **Timing** is scanline-accurate, not dot-accurate.
- Per scanline, a single “latch point” captures PPU inputs for rendering.
- **Mid-scanline register effects are out of scope** for Milestone E; ROMs used
  for validation must not rely on mid-scanline changes.
- Env0 produces a framebuffer of **shade indices** (0..3) used for frame hashing.

---

## Window Rendering

Registers:
- `LCDC` bit 5: window enable
- `LCDC` bit 6: window tilemap select (`0x9800` or `0x9C00`)
- `WX` (`0xFF4B`), `WY` (`0xFF4A`)

Latch rules:
- `WX/WY` are **latched per scanline** (same cadence as `LCDC/SCX/SCY/BGP`).
- If LCD is disabled, latches are ignored until re-enabled.

Window activation:
- Window is considered active on scanline `LY` if:
  - `LCDC` window enable is set, and
  - `LY >= WY_latched`.
- Window X start is `WX_latched - 7`.
- If `WX_latched >= 167`, the window is effectively off-screen for that line,
  but `window_line` still increments for deterministic behavior.

Window line counter:
- `window_line` increments **once per scanline** when the window is active
  (`LY >= WY_latched` and window enabled), regardless of whether any pixels are
  actually visible (e.g., WX out of range).

Pixel selection:
- For each pixel, choose **BG or Window** source **before** sprite overlay.
- The BG/Window **color index** (0..3) is preserved for sprite priority checks.

---

## Sprite Rendering (OBJ)

Sprite evaluation (per scanline):
- Use OAM entries in index order.
- Select up to **10 sprites per scanline** (DMG rule).
- Coordinate offsets:
  - OAM Y is stored as `y + 16`
  - OAM X is stored as `x + 8`
- Sprite height:
  - 8x8 if `LCDC bit 2 == 0`
  - 8x16 if `LCDC bit 2 == 1`

Sprite pixel composition:
- Attributes:
  - bit 7: priority (1 = behind BG/Window)
  - bit 6: yflip
  - bit 5: xflip
  - bit 4: palette select (0 = OBP0, 1 = OBP1)
- OBJ color index `0` is transparent.
- Priority rule (DMG):
  - if OBJ priority is “behind”, the sprite pixel is **hidden** when
    BG/Window color index is **not** 0.
  - if BG/Window color index is 0, sprite pixel may show even when “behind”.

OBJ tie-break rule (deterministic):
- First by lower X (leftmost).
- If X is equal, lower OAM index wins.

---

## STAT Interrupts (edge-triggered)

STAT interrupt sources (DMG):
- Mode 0 (HBlank) enable: `STAT bit 3`
- Mode 1 (VBlank) enable: `STAT bit 4`
- Mode 2 (OAM) enable: `STAT bit 5`
- LYC=LY enable: `STAT bit 6`

Edge-trigger rule:
- Request IF bit 1 **only on transitions** into a true condition for any
  **enabled** source.
- Do **not** request repeatedly while a condition remains true.
- Enabling a STAT source while its condition is already true does **not**
  retroactively trigger an IRQ in this model.

LCD disabled behavior:
- When LCDC bit 7 is 0:
  - `LY` is forced to 0
  - `STAT` mode is 0 (HBlank)
  - No STAT interrupts are generated

---

## DMA (OAM DMA)

- Write to `0xFF46` triggers a copy of 160 bytes:
  - Source = `value << 8`
  - Dest = `0xFE00..0xFE9F`
- Milestone E uses an **immediate copy** model for correctness.
- Bus blocking / cycle-accurate DMA timing is explicitly out of scope here.

