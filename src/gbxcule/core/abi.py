"""ABI v4: Authoritative device buffer layouts.

This module defines the canonical layouts for state buffers used by Warp kernels
and downstream consumers. It is intentionally tiny and pure.

See ARCHITECTURE.md §6 for rationale and versioning policy.

ABI v2 migration note:
- Added serial output buffers (SERIAL_MAX, serial_buf/serial_len/serial_overflow).

ABI v3 migration note:
- Added interrupt + timer state buffers (IME/IME delay/HALT, DIV/TIMA edge tracking).

ABI v4 migration note:
- Added scanline-accurate PPU buffers (scanline cycle + LY), env0 BG latch arrays,
  and env0 BG shade framebuffer for Milestone D.

ABI v5 migration note:
- Added window latches (WX/WY + window line) and per-env window line counter
  for Milestone E.

ABI v6 migration note:
- Added OBJ palette latches (OBP0/OBP1) for Milestone E sprites.

ABI v7 migration note:
- Added STAT previous-condition buffer for edge-triggered STAT IRQs.
"""

from __future__ import annotations

from typing import Final

ABI_VERSION: Final[int] = 7

# Flat 64KB per environment (Game Boy address space).
MEM_SIZE: Final[int] = 65_536

# Default observation dimensionality for fused step kernels.
OBS_DIM_DEFAULT: Final[int] = 32

# Serial capture buffer length per environment (SB/SC debug output).
SERIAL_MAX: Final[int] = 1024

# Screen geometry (DMG).
SCREEN_W: Final[int] = 160
SCREEN_H: Final[int] = 144
DOWNSAMPLE_SCALE: Final[int] = 2
DOWNSAMPLE_W: Final[int] = SCREEN_W // DOWNSAMPLE_SCALE
DOWNSAMPLE_H: Final[int] = SCREEN_H // DOWNSAMPLE_SCALE

# Scanline timing (DMG).
CYCLES_PER_SCANLINE: Final[int] = 456


def mem_offset(env_idx: int, addr: int) -> int:
    """Compute the flat memory offset for (env_idx, addr).

    Args:
        env_idx: Environment index (0-based).
        addr: Address in [0, MEM_SIZE).

    Returns:
        Flat offset into a u8[num_envs * MEM_SIZE] buffer.
    """
    if env_idx < 0:
        raise ValueError(f"env_idx must be >= 0, got {env_idx}")
    if addr < 0 or addr >= MEM_SIZE:
        raise ValueError(f"addr must be in [0, {MEM_SIZE}), got {addr}")
    return env_idx * MEM_SIZE + addr


def mem_slice(env_idx: int, lo: int, hi: int) -> slice:
    """Compute a slice into a flat memory buffer for [lo, hi).

    Args:
        env_idx: Environment index (0-based).
        lo: Start address in [0, MEM_SIZE].
        hi: End address in [0, MEM_SIZE], must satisfy lo <= hi.

    Returns:
        A Python slice suitable for indexing a flat u8 buffer.
    """
    if env_idx < 0:
        raise ValueError(f"env_idx must be >= 0, got {env_idx}")
    if lo < 0 or hi < 0 or lo > MEM_SIZE or hi > MEM_SIZE:
        raise ValueError(f"lo/hi must be in [0, {MEM_SIZE}], got lo={lo} hi={hi}")
    if lo > hi:
        raise ValueError(f"lo must be <= hi, got lo={lo} hi={hi}")
    base = env_idx * MEM_SIZE
    return slice(base + lo, base + hi)


def obs_offset(env_idx: int, obs_idx: int, obs_dim: int = OBS_DIM_DEFAULT) -> int:
    """Compute the flat observation offset for (env_idx, obs_idx).

    Args:
        env_idx: Environment index (0-based).
        obs_idx: Observation index in [0, obs_dim).
        obs_dim: Observation dimension (default: OBS_DIM_DEFAULT).

    Returns:
        Flat offset into a f32[num_envs * obs_dim] buffer.
    """
    if env_idx < 0:
        raise ValueError(f"env_idx must be >= 0, got {env_idx}")
    if obs_dim <= 0:
        raise ValueError(f"obs_dim must be > 0, got {obs_dim}")
    if obs_idx < 0 or obs_idx >= obs_dim:
        raise ValueError(f"obs_idx must be in [0, {obs_dim}), got {obs_idx}")
    return env_idx * obs_dim + obs_idx


def obs_slice(env_idx: int, obs_dim: int = OBS_DIM_DEFAULT) -> slice:
    """Compute a slice into a flat observation buffer for one env.

    Args:
        env_idx: Environment index (0-based).
        obs_dim: Observation dimension (default: OBS_DIM_DEFAULT).

    Returns:
        A Python slice suitable for indexing a flat f32 buffer.
    """
    if env_idx < 0:
        raise ValueError(f"env_idx must be >= 0, got {env_idx}")
    if obs_dim <= 0:
        raise ValueError(f"obs_dim must be > 0, got {obs_dim}")
    base = env_idx * obs_dim
    return slice(base, base + obs_dim)
