"""Observation helpers for RAM-derived feature vectors."""

from __future__ import annotations

import numpy as np

from gbxcule.backends.common import NDArrayF32
from gbxcule.core.abi import OBS_DIM_DEFAULT


def build_obs_v3_from_state(
    *,
    pc: int,
    sp: int,
    a: int,
    f: int,
    b: int,
    c: int,
    d: int,
    e: int,
    h: int,
    l_reg: int,
    wram16: bytes | bytearray | memoryview | np.ndarray,
    obs_dim: int = OBS_DIM_DEFAULT,
) -> NDArrayF32:
    """Build the variant-3 observation vector from registers + WRAM window.

    Variant-3 layout (obs_dim=32):
    - [0:10): normalized PC, SP, A, F, B, C, D, E, H, L
    - [10:26): normalized bytes from WRAM 0xC000..0xC00F
    - [26:32): mixed features derived from WRAM + regs
    """
    if obs_dim <= 0:
        raise ValueError(f"obs_dim must be > 0, got {obs_dim}")
    if len(wram16) < 16:
        raise ValueError("wram16 must contain at least 16 bytes")

    obs = np.zeros((obs_dim,), dtype=np.float32)
    limit = min(obs_dim, OBS_DIM_DEFAULT)

    if limit >= 1:
        obs[0] = np.float32((pc & 0xFFFF) / 65535.0)
    if limit >= 2:
        obs[1] = np.float32((sp & 0xFFFF) / 65535.0)
    if limit >= 3:
        obs[2] = np.float32((a & 0xFF) / 255.0)
    if limit >= 4:
        obs[3] = np.float32((f & 0xF0) / 255.0)
    if limit >= 5:
        obs[4] = np.float32((b & 0xFF) / 255.0)
    if limit >= 6:
        obs[5] = np.float32((c & 0xFF) / 255.0)
    if limit >= 7:
        obs[6] = np.float32((d & 0xFF) / 255.0)
    if limit >= 8:
        obs[7] = np.float32((e & 0xFF) / 255.0)
    if limit >= 9:
        obs[8] = np.float32((h & 0xFF) / 255.0)
    if limit >= 10:
        obs[9] = np.float32((l_reg & 0xFF) / 255.0)

    if limit <= 10:
        return obs

    m = wram16
    mem_count = min(16, max(0, limit - 10))
    for idx in range(mem_count):
        obs[10 + idx] = np.float32((int(m[idx]) & 0xFF) / 255.0)

    if limit <= 26:
        return obs

    m0 = int(m[0]) & 0xFF
    m1 = int(m[1]) & 0xFF
    m2 = int(m[2]) & 0xFF
    m3 = int(m[3]) & 0xFF
    m4 = int(m[4]) & 0xFF
    m5 = int(m[5]) & 0xFF
    m6 = int(m[6]) & 0xFF
    m7 = int(m[7]) & 0xFF

    mix0 = (m0 ^ m1) & 0xFF
    mix1 = (m2 + (m3 * 5)) & 0xFF
    mix2 = (m4 ^ (a & 0xFF)) & 0xFF
    mix3 = (m5 + (b & 0xFF)) & 0xFF
    mix4 = (m6 ^ (c & 0xFF)) & 0xFF
    mix5 = (m7 + (d & 0xFF)) & 0xFF

    mix_vals = (mix0, mix1, mix2, mix3, mix4, mix5)
    mix_count = min(6, max(0, limit - 26))
    for idx in range(mix_count):
        obs[26 + idx] = np.float32(mix_vals[idx] / 255.0)

    return obs
