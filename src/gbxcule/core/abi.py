"""ABI v0: Authoritative device buffer layouts.

This module defines the canonical layouts for state buffers used by Warp kernels
and downstream consumers. It is intentionally tiny and pure.

See ARCHITECTURE.md §6 for rationale and versioning policy.
"""

from __future__ import annotations

from typing import Final

ABI_VERSION: Final[int] = 0

# Flat 64KB per environment (Game Boy address space).
MEM_SIZE: Final[int] = 65_536


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
