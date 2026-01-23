"""Post-step templates for fused kernel stages."""
# ruff: noqa: F841

from __future__ import annotations

import warp as wp

OBS_DIM = 32


def template_reward_v0(
    mem: int,
    base: int,
    a_i: int,
    reward_out: int,
    i: int,
) -> None:
    """Minimal deterministic reward template (v0)."""
    m0 = wp.int32(mem[base + 0xC000])
    m1 = wp.int32(mem[base + 0xC001])
    mix = (m0 + (m1 * 3)) & 0xFF
    val = (a_i ^ mix) & 0xFF
    reward_out[i] = wp.float32(val) / wp.float32(255.0)


def template_obs_v0(
    mem: int,
    base: int,
    pc_i: int,
    sp_i: int,
    a_i: int,
    b_i: int,
    c_i: int,
    d_i: int,
    e_i: int,
    h_i: int,
    l_i: int,
    f_i: int,
    obs_out: int,
    i: int,
) -> None:
    """Minimal deterministic obs template (v0)."""
    obs_base = i * OBS_DIM

    obs_out[obs_base + 0] = wp.float32(pc_i & 0xFFFF) / wp.float32(65535.0)
    obs_out[obs_base + 1] = wp.float32(sp_i & 0xFFFF) / wp.float32(65535.0)
    obs_out[obs_base + 2] = wp.float32(a_i & 0xFF) / wp.float32(255.0)
    obs_out[obs_base + 3] = wp.float32(f_i & 0xFF) / wp.float32(255.0)
    obs_out[obs_base + 4] = wp.float32(b_i & 0xFF) / wp.float32(255.0)
    obs_out[obs_base + 5] = wp.float32(c_i & 0xFF) / wp.float32(255.0)
    obs_out[obs_base + 6] = wp.float32(d_i & 0xFF) / wp.float32(255.0)
    obs_out[obs_base + 7] = wp.float32(e_i & 0xFF) / wp.float32(255.0)
    obs_out[obs_base + 8] = wp.float32(h_i & 0xFF) / wp.float32(255.0)
    obs_out[obs_base + 9] = wp.float32(l_i & 0xFF) / wp.float32(255.0)

    m0 = wp.int32(mem[base + 0xC000])
    m1 = wp.int32(mem[base + 0xC001])
    m2 = wp.int32(mem[base + 0xC002])
    m3 = wp.int32(mem[base + 0xC003])
    m4 = wp.int32(mem[base + 0xC004])
    m5 = wp.int32(mem[base + 0xC005])
    m6 = wp.int32(mem[base + 0xC006])
    m7 = wp.int32(mem[base + 0xC007])
    m8 = wp.int32(mem[base + 0xC008])
    m9 = wp.int32(mem[base + 0xC009])
    m10 = wp.int32(mem[base + 0xC00A])
    m11 = wp.int32(mem[base + 0xC00B])
    m12 = wp.int32(mem[base + 0xC00C])
    m13 = wp.int32(mem[base + 0xC00D])
    m14 = wp.int32(mem[base + 0xC00E])
    m15 = wp.int32(mem[base + 0xC00F])

    obs_out[obs_base + 10] = wp.float32(m0) / wp.float32(255.0)
    obs_out[obs_base + 11] = wp.float32(m1) / wp.float32(255.0)
    obs_out[obs_base + 12] = wp.float32(m2) / wp.float32(255.0)
    obs_out[obs_base + 13] = wp.float32(m3) / wp.float32(255.0)
    obs_out[obs_base + 14] = wp.float32(m4) / wp.float32(255.0)
    obs_out[obs_base + 15] = wp.float32(m5) / wp.float32(255.0)
    obs_out[obs_base + 16] = wp.float32(m6) / wp.float32(255.0)
    obs_out[obs_base + 17] = wp.float32(m7) / wp.float32(255.0)
    obs_out[obs_base + 18] = wp.float32(m8) / wp.float32(255.0)
    obs_out[obs_base + 19] = wp.float32(m9) / wp.float32(255.0)
    obs_out[obs_base + 20] = wp.float32(m10) / wp.float32(255.0)
    obs_out[obs_base + 21] = wp.float32(m11) / wp.float32(255.0)
    obs_out[obs_base + 22] = wp.float32(m12) / wp.float32(255.0)
    obs_out[obs_base + 23] = wp.float32(m13) / wp.float32(255.0)
    obs_out[obs_base + 24] = wp.float32(m14) / wp.float32(255.0)
    obs_out[obs_base + 25] = wp.float32(m15) / wp.float32(255.0)

    mix0 = (m0 ^ m1) & 0xFF
    mix1 = (m2 + (m3 * 5)) & 0xFF
    mix2 = (m4 ^ (a_i & 0xFF)) & 0xFF
    mix3 = (m5 + (b_i & 0xFF)) & 0xFF
    mix4 = (m6 ^ (c_i & 0xFF)) & 0xFF
    mix5 = (m7 + (d_i & 0xFF)) & 0xFF

    obs_out[obs_base + 26] = wp.float32(mix0) / wp.float32(255.0)
    obs_out[obs_base + 27] = wp.float32(mix1) / wp.float32(255.0)
    obs_out[obs_base + 28] = wp.float32(mix2) / wp.float32(255.0)
    obs_out[obs_base + 29] = wp.float32(mix3) / wp.float32(255.0)
    obs_out[obs_base + 30] = wp.float32(mix4) / wp.float32(255.0)
    obs_out[obs_base + 31] = wp.float32(mix5) / wp.float32(255.0)
