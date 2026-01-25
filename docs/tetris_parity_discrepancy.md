# Tetris parity discrepancy (START hold)

## Context
- State: `states/tetris_start.state` captured at the first falling piece.
- Screenshots confirming the capture: `states/tetris_start_pre.png` (just after game start) and `states/tetris_start.png` (first fall).

## Repro (deterministic)
- Use `states/tetris_start.state` with `frames_per_step=24` and `release_after_frames=8`.
- Hold only START (action index 2 in the default action codec) for each step.

Observed on 2026-01-25:
- Divergence at step 4.
- PyBoy: PC `0x02ED`, cycles `47486584`.
- WarpVec: PC `0x02F0`, cycles `47486576` (Warp is 8 cycles behind).
- Both are in the busy-wait loop around `LDH A,(0x85)` / `AND A` / `JR Z` (ROM offsets `0x02E4-0x02F0`).
- LCDC changes to `0xDB` at this point in both runs.

## Notes
- The mismatch is tied to holding START for multiple frames; the same state with no input does not show this early divergence.
- JOYP register reads in PyBoy are cached and updated on writes to `0xFF00` (`Interaction.pull`), not on reads.
- WarpVec now mirrors this behavior, but a small timing drift remains.

## Next steps (if/when revisiting)
- Instrument JOYP writes/reads and IF (`0xFF0F`) timing around the `0x02ED` loop.
- Compare LCDC transitions and any interrupt service cycles during the step 4 window.
