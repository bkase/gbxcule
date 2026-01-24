"""Rotate/shift/bit instruction templates for Warp CPU stepping."""
# ruff: noqa: F821, F841

from __future__ import annotations

import warp as wp


# Helper for type checking (injected as @wp.func in the kernel).
def make_flags(z: int, n: int, h: int, c: int) -> int: ...


def template_rlca(pc_i: int, a_i: int, f_i: int) -> None:
    """RLCA template (unprefixed)."""
    cflag = (a_i >> 7) & 0x1
    a_i = ((a_i << 1) & 0xFF) | cflag
    f_i = make_flags(0, 0, 0, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_rrca(pc_i: int, a_i: int, f_i: int) -> None:
    """RRCA template (unprefixed)."""
    cflag = a_i & 0x1
    a_i = ((a_i >> 1) & 0xFF) | (cflag << 7)
    f_i = make_flags(0, 0, 0, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_rla(pc_i: int, a_i: int, f_i: int) -> None:
    """RLA template (unprefixed)."""
    old_c = (f_i >> 4) & 0x1
    cflag = (a_i >> 7) & 0x1
    a_i = ((a_i << 1) & 0xFF) | old_c
    f_i = make_flags(0, 0, 0, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_rra(pc_i: int, a_i: int, f_i: int) -> None:
    """RRA template (unprefixed)."""
    old_c = (f_i >> 4) & 0x1
    cflag = a_i & 0x1
    a_i = ((a_i >> 1) & 0xFF) | (old_c << 7)
    f_i = make_flags(0, 0, 0, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_cb_rlc_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """CB RLC r8 template."""
    cflag = (REG_i >> 7) & 0x1
    REG_i = ((REG_i << 1) & 0xFF) | cflag
    z = wp.where(REG_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 8


def template_cb_rlc_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CB RLC (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    val = read8(
        i,
        base,
        hl,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    cflag = (val >> 7) & 0x1
    val = ((val << 1) & 0xFF) | cflag
    write8(
        i,
        base,
        hl,
        val,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
        div_counter,
        timer_prev_in,
        tima_reload_pending,
        tima_reload_delay,
    )
    z = wp.where(val == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 16


def template_cb_rrc_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """CB RRC r8 template."""
    cflag = REG_i & 0x1
    REG_i = ((REG_i >> 1) & 0xFF) | (cflag << 7)
    z = wp.where(REG_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 8


def template_cb_rrc_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CB RRC (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    val = read8(
        i,
        base,
        hl,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    cflag = val & 0x1
    val = ((val >> 1) & 0xFF) | (cflag << 7)
    write8(
        i,
        base,
        hl,
        val,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
        div_counter,
        timer_prev_in,
        tima_reload_pending,
        tima_reload_delay,
    )
    z = wp.where(val == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 16


def template_cb_rl_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """CB RL r8 template."""
    old_c = (f_i >> 4) & 0x1
    cflag = (REG_i >> 7) & 0x1
    REG_i = ((REG_i << 1) & 0xFF) | old_c
    z = wp.where(REG_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 8


def template_cb_rl_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CB RL (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    val = read8(
        i,
        base,
        hl,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    old_c = (f_i >> 4) & 0x1
    cflag = (val >> 7) & 0x1
    val = ((val << 1) & 0xFF) | old_c
    write8(
        i,
        base,
        hl,
        val,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
        div_counter,
        timer_prev_in,
        tima_reload_pending,
        tima_reload_delay,
    )
    z = wp.where(val == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 16


def template_cb_rr_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """CB RR r8 template."""
    old_c = (f_i >> 4) & 0x1
    cflag = REG_i & 0x1
    REG_i = ((REG_i >> 1) & 0xFF) | (old_c << 7)
    z = wp.where(REG_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 8


def template_cb_rr_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CB RR (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    val = read8(
        i,
        base,
        hl,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    old_c = (f_i >> 4) & 0x1
    cflag = val & 0x1
    val = ((val >> 1) & 0xFF) | (old_c << 7)
    write8(
        i,
        base,
        hl,
        val,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
        div_counter,
        timer_prev_in,
        tima_reload_pending,
        tima_reload_delay,
    )
    z = wp.where(val == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 16


def template_cb_sla_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """CB SLA r8 template."""
    cflag = (REG_i >> 7) & 0x1
    REG_i = (REG_i << 1) & 0xFF
    z = wp.where(REG_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 8


def template_cb_sla_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CB SLA (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    val = read8(
        i,
        base,
        hl,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    cflag = (val >> 7) & 0x1
    val = (val << 1) & 0xFF
    write8(
        i,
        base,
        hl,
        val,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
        div_counter,
        timer_prev_in,
        tima_reload_pending,
        tima_reload_delay,
    )
    z = wp.where(val == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 16


def template_cb_sra_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """CB SRA r8 template."""
    cflag = REG_i & 0x1
    REG_i = ((REG_i >> 1) & 0x7F) | (REG_i & 0x80)
    z = wp.where(REG_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 8


def template_cb_sra_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CB SRA (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    val = read8(
        i,
        base,
        hl,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    cflag = val & 0x1
    val = ((val >> 1) & 0x7F) | (val & 0x80)
    write8(
        i,
        base,
        hl,
        val,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
        div_counter,
        timer_prev_in,
        tima_reload_pending,
        tima_reload_delay,
    )
    z = wp.where(val == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 16


def template_cb_swap_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """CB SWAP r8 template."""
    REG_i = ((REG_i & 0x0F) << 4) | ((REG_i & 0xF0) >> 4)
    z = wp.where(REG_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, 0)
    cycles = 8


def template_cb_swap_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CB SWAP (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    val = read8(
        i,
        base,
        hl,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    val = ((val & 0x0F) << 4) | ((val & 0xF0) >> 4)
    write8(
        i,
        base,
        hl,
        val,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
        div_counter,
        timer_prev_in,
        tima_reload_pending,
        tima_reload_delay,
    )
    z = wp.where(val == 0, 1, 0)
    f_i = make_flags(z, 0, 0, 0)
    cycles = 16


def template_cb_srl_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """CB SRL r8 template."""
    cflag = REG_i & 0x1
    REG_i = (REG_i >> 1) & 0x7F
    z = wp.where(REG_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 8


def template_cb_srl_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CB SRL (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    val = read8(
        i,
        base,
        hl,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    cflag = val & 0x1
    val = (val >> 1) & 0x7F
    write8(
        i,
        base,
        hl,
        val,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
        div_counter,
        timer_prev_in,
        tima_reload_pending,
        tima_reload_delay,
    )
    z = wp.where(val == 0, 1, 0)
    f_i = make_flags(z, 0, 0, cflag)
    cycles = 16


def template_cb_bit_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """CB BIT b, r8 template."""
    bit_index = (cb_opcode >> 3) & 0x7
    bit = (REG_i >> bit_index) & 0x1
    z = wp.where(bit == 0, 1, 0)
    cflag = (f_i >> 4) & 0x1
    f_i = make_flags(z, 0, 1, cflag)
    cycles = 8


def template_cb_bit_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CB BIT b, (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    val = read8(
        i,
        base,
        hl,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    bit_index = (cb_opcode >> 3) & 0x7
    bit = (val >> bit_index) & 0x1
    z = wp.where(bit == 0, 1, 0)
    cflag = (f_i >> 4) & 0x1
    f_i = make_flags(z, 0, 1, cflag)
    cycles = 16


def template_cb_res_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """CB RES b, r8 template."""
    bit_index = (cb_opcode >> 3) & 0x7
    mask = wp.int32(1) << bit_index
    REG_i = REG_i & (0xFF ^ mask)
    cycles = 8


def template_cb_res_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CB RES b, (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    val = read8(
        i,
        base,
        hl,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    bit_index = (cb_opcode >> 3) & 0x7
    mask = wp.int32(1) << bit_index
    val = val & (0xFF ^ mask)
    write8(
        i,
        base,
        hl,
        val,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
        div_counter,
        timer_prev_in,
        tima_reload_pending,
        tima_reload_delay,
    )
    cycles = 16


def template_cb_set_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """CB SET b, r8 template."""
    bit_index = (cb_opcode >> 3) & 0x7
    mask = wp.int32(1) << bit_index
    REG_i = REG_i | mask
    cycles = 8


def template_cb_set_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CB SET b, (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    val = read8(
        i,
        base,
        hl,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    bit_index = (cb_opcode >> 3) & 0x7
    mask = wp.int32(1) << bit_index
    val = val | mask
    write8(
        i,
        base,
        hl,
        val,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
        div_counter,
        timer_prev_in,
        tima_reload_pending,
        tima_reload_delay,
    )
    cycles = 16
