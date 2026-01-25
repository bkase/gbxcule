"""ALU instruction templates for Warp CPU stepping."""
# ruff: noqa: F821, F841

from __future__ import annotations

import warp as wp


# Helper for type checking (injected as @wp.func in the kernel).
def make_flags(z: int, n: int, h: int, c: int) -> int: ...


def template_inc_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """8-bit INC template (REG_i placeholder)."""
    old = REG_i
    REG_i = (REG_i + 1) & 0xFF
    z = wp.where(REG_i == 0, 1, 0)
    hflag = wp.where((old & 0x0F) == 0x0F, 1, 0)
    cflag = (f_i >> 4) & 0x1
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_inc_r16(pc_i: int, HREG_i: int, LREG_i: int) -> None:
    """16-bit INC template (HREG_i/LREG_i placeholders)."""
    reg16 = ((HREG_i << 8) | LREG_i) & 0xFFFF
    reg16 = (reg16 + 1) & 0xFFFF
    HREG_i = (reg16 >> 8) & 0xFF
    LREG_i = reg16 & 0xFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_dec_r16(pc_i: int, HREG_i: int, LREG_i: int) -> None:
    """16-bit DEC template (HREG_i/LREG_i placeholders)."""
    reg16 = ((HREG_i << 8) | LREG_i) & 0xFFFF
    reg16 = (reg16 - 1) & 0xFFFF
    HREG_i = (reg16 >> 8) & 0xFF
    LREG_i = reg16 & 0xFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_inc_sp(pc_i: int, sp_i: int) -> None:
    """INC SP template."""
    sp_i = (sp_i + 1) & 0xFFFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_dec_sp(pc_i: int, sp_i: int) -> None:
    """DEC SP template."""
    sp_i = (sp_i - 1) & 0xFFFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_add_a_r8(pc_i: int, a_i: int, f_i: int, REG_i: int) -> None:
    """ADD A, r8 template (REG_i placeholder)."""
    sum_ab = a_i + REG_i
    res = sum_ab & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) + (REG_i & 0x0F)) > 0x0F, 1, 0)
    cflag = wp.where(sum_ab > 0xFF, 1, 0)
    a_i = res
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_add_hl_r16(
    pc_i: int,
    f_i: int,
    HREG_i: int,
    LREG_i: int,
    SRC_HREG_i: int,
    SRC_LREG_i: int,
) -> None:
    """ADD HL, rr template (SRC_HREG_i/SRC_LREG_i placeholders)."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    rr = ((SRC_HREG_i << 8) | SRC_LREG_i) & 0xFFFF
    sum_hl = hl + rr
    hflag = wp.where(((hl & 0x0FFF) + (rr & 0x0FFF)) > 0x0FFF, 1, 0)
    cflag = wp.where(sum_hl > 0xFFFF, 1, 0)
    z = (f_i >> 7) & 0x1
    HREG_i = (sum_hl >> 8) & 0xFF
    LREG_i = sum_hl & 0xFF
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_add_hl_sp(
    pc_i: int,
    f_i: int,
    HREG_i: int,
    LREG_i: int,
    sp_i: int,
) -> None:
    """ADD HL, SP template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    sum_hl = hl + sp_i
    hflag = wp.where(((hl & 0x0FFF) + (sp_i & 0x0FFF)) > 0x0FFF, 1, 0)
    cflag = wp.where(sum_hl > 0xFFFF, 1, 0)
    z = (f_i >> 7) & 0x1
    HREG_i = (sum_hl >> 8) & 0xFF
    LREG_i = sum_hl & 0xFF
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_add_sp_e8(
    pc_i: int, sp_i: int, f_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """ADD SP, e8 template."""
    off = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
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
    off_s = sign8(off)
    off_u = off & 0xFF
    hflag = wp.where(((sp_i & 0x0F) + (off_u & 0x0F)) > 0x0F, 1, 0)
    cflag = wp.where(((sp_i & 0xFF) + off_u) > 0xFF, 1, 0)
    sp_i = (sp_i + off_s) & 0xFFFF
    f_i = make_flags(0, 0, hflag, cflag)
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 16


def template_ld_hl_sp_e8(
    pc_i: int, sp_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD HL, SP+e8 template."""
    off = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
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
    off_s = sign8(off)
    off_u = off & 0xFF
    hflag = wp.where(((sp_i & 0x0F) + (off_u & 0x0F)) > 0x0F, 1, 0)
    cflag = wp.where(((sp_i & 0xFF) + off_u) > 0xFF, 1, 0)
    res = (sp_i + off_s) & 0xFFFF
    HREG_i = (res >> 8) & 0xFF
    LREG_i = res & 0xFF
    f_i = make_flags(0, 0, hflag, cflag)
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 12


def template_and_a_d8(pc_i: int, a_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """AND A, d8 template."""
    val = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
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
    res = a_i & val
    a_i = res & 0xFF
    z = wp.where(a_i == 0, 1, 0)
    f_i = make_flags(z, 0, 1, 0)
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 8


def template_dec_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """8-bit DEC template (REG_i placeholder)."""
    old = REG_i
    REG_i = (REG_i - 1) & 0xFF
    z = wp.where(REG_i == 0, 1, 0)
    hflag = wp.where((old & 0x0F) == 0x00, 1, 0)
    cflag = (f_i >> 4) & 0x1
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_add_a_hl(
    pc_i: int, a_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """ADD A, (HL) template."""
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
    sum_ab = a_i + val
    res = sum_ab & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) + (val & 0x0F)) > 0x0F, 1, 0)
    cflag = wp.where(sum_ab > 0xFF, 1, 0)
    a_i = res
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_add_a_d8(pc_i: int, a_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """ADD A, d8 template."""
    val = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
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
    sum_ab = a_i + val
    res = sum_ab & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) + (val & 0x0F)) > 0x0F, 1, 0)
    cflag = wp.where(sum_ab > 0xFF, 1, 0)
    a_i = res
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 8


def template_adc_a_r8(pc_i: int, a_i: int, f_i: int, REG_i: int) -> None:
    """ADC A, r8 template (REG_i placeholder)."""
    carry = (f_i >> 4) & 0x1
    sum_ab = a_i + REG_i + carry
    res = sum_ab & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) + (REG_i & 0x0F) + carry) > 0x0F, 1, 0)
    cflag = wp.where(sum_ab > 0xFF, 1, 0)
    a_i = res
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_adc_a_hl(
    pc_i: int, a_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """ADC A, (HL) template."""
    carry = (f_i >> 4) & 0x1
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
    sum_ab = a_i + val + carry
    res = sum_ab & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) + (val & 0x0F) + carry) > 0x0F, 1, 0)
    cflag = wp.where(sum_ab > 0xFF, 1, 0)
    a_i = res
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_adc_a_d8(pc_i: int, a_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """ADC A, d8 template."""
    carry = (f_i >> 4) & 0x1
    val = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
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
    sum_ab = a_i + val + carry
    res = sum_ab & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) + (val & 0x0F) + carry) > 0x0F, 1, 0)
    cflag = wp.where(sum_ab > 0xFF, 1, 0)
    a_i = res
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 8


def template_sub_a_r8(pc_i: int, a_i: int, f_i: int, REG_i: int) -> None:
    """SUB r8 template (REG_i placeholder)."""
    diff = a_i - REG_i
    res = diff & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) - (REG_i & 0x0F)) < 0, 1, 0)
    cflag = wp.where(a_i < REG_i, 1, 0)
    a_i = res
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_sub_a_hl(
    pc_i: int, a_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """SUB (HL) template."""
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
    diff = a_i - val
    res = diff & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) - (val & 0x0F)) < 0, 1, 0)
    cflag = wp.where(a_i < val, 1, 0)
    a_i = res
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_sub_a_d8(pc_i: int, a_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """SUB d8 template."""
    val = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
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
    diff = a_i - val
    res = diff & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) - (val & 0x0F)) < 0, 1, 0)
    cflag = wp.where(a_i < val, 1, 0)
    a_i = res
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 8


def template_sbc_a_r8(pc_i: int, a_i: int, f_i: int, REG_i: int) -> None:
    """SBC A, r8 template (REG_i placeholder)."""
    carry = (f_i >> 4) & 0x1
    diff = a_i - REG_i - carry
    res = diff & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) - ((REG_i & 0x0F) + carry)) < 0, 1, 0)
    cflag = wp.where(a_i < (REG_i + carry), 1, 0)
    a_i = res
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_sbc_a_hl(
    pc_i: int, a_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """SBC A, (HL) template."""
    carry = (f_i >> 4) & 0x1
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
    diff = a_i - val - carry
    res = diff & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) - ((val & 0x0F) + carry)) < 0, 1, 0)
    cflag = wp.where(a_i < (val + carry), 1, 0)
    a_i = res
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_sbc_a_d8(pc_i: int, a_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """SBC A, d8 template."""
    carry = (f_i >> 4) & 0x1
    val = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
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
    diff = a_i - val - carry
    res = diff & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) - ((val & 0x0F) + carry)) < 0, 1, 0)
    cflag = wp.where(a_i < (val + carry), 1, 0)
    a_i = res
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 8


def template_and_a_r8(pc_i: int, a_i: int, f_i: int, REG_i: int) -> None:
    """AND A, r8 template (REG_i placeholder)."""
    res = a_i & REG_i
    a_i = res & 0xFF
    z = wp.where(a_i == 0, 1, 0)
    f_i = make_flags(z, 0, 1, 0)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_and_a_hl(
    pc_i: int, a_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """AND A, (HL) template."""
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
    res = a_i & val
    a_i = res & 0xFF
    z = wp.where(a_i == 0, 1, 0)
    f_i = make_flags(z, 0, 1, 0)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_or_a_r8(pc_i: int, a_i: int, f_i: int, REG_i: int) -> None:
    """OR A, r8 template (REG_i placeholder)."""
    res = a_i | REG_i
    a_i = res & 0xFF
    z = wp.where(a_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, 0)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_or_a_hl(
    pc_i: int, a_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """OR A, (HL) template."""
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
    res = a_i | val
    a_i = res & 0xFF
    z = wp.where(a_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, 0)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_or_a_d8(pc_i: int, a_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """OR A, d8 template."""
    val = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
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
    res = a_i | val
    a_i = res & 0xFF
    z = wp.where(a_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, 0)
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 8


def template_xor_a_r8(pc_i: int, a_i: int, f_i: int, REG_i: int) -> None:
    """XOR A, r8 template (REG_i placeholder)."""
    res = a_i ^ REG_i
    a_i = res & 0xFF
    z = wp.where(a_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, 0)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_xor_a_hl(
    pc_i: int, a_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """XOR A, (HL) template."""
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
    res = a_i ^ val
    a_i = res & 0xFF
    z = wp.where(a_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, 0)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_xor_a_d8(pc_i: int, a_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """XOR A, d8 template."""
    val = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
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
    res = a_i ^ val
    a_i = res & 0xFF
    z = wp.where(a_i == 0, 1, 0)
    f_i = make_flags(z, 0, 0, 0)
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 8


def template_cp_a_r8(pc_i: int, a_i: int, f_i: int, REG_i: int) -> None:
    """CP r8 template (REG_i placeholder)."""
    diff = a_i - REG_i
    res = diff & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) - (REG_i & 0x0F)) < 0, 1, 0)
    cflag = wp.where(a_i < REG_i, 1, 0)
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_cp_a_hl(
    pc_i: int, a_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CP (HL) template."""
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
    diff = a_i - val
    res = diff & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) - (val & 0x0F)) < 0, 1, 0)
    cflag = wp.where(a_i < val, 1, 0)
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_cp_a_d8(pc_i: int, a_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """CP d8 template."""
    val = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
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
    diff = a_i - val
    res = diff & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) - (val & 0x0F)) < 0, 1, 0)
    cflag = wp.where(a_i < val, 1, 0)
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 8


def template_inc_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """INC (HL) template."""
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
    old = val
    val = (val + 1) & 0xFF
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
        actions,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    z = wp.where(val == 0, 1, 0)
    hflag = wp.where((old & 0x0F) == 0x0F, 1, 0)
    cflag = (f_i >> 4) & 0x1
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 12


def template_dec_hl(
    pc_i: int, f_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """DEC (HL) template."""
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
    old = val
    val = (val - 1) & 0xFF
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
        actions,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    z = wp.where(val == 0, 1, 0)
    hflag = wp.where((old & 0x0F) == 0x00, 1, 0)
    cflag = (f_i >> 4) & 0x1
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 12


def template_daa(pc_i: int, a_i: int, f_i: int) -> None:
    """DAA template."""
    a_val = a_i & 0xFF
    n_flag = (f_i >> 6) & 0x1
    h_flag = (f_i >> 5) & 0x1
    c_flag = (f_i >> 4) & 0x1
    adjust = 0
    carry = c_flag
    if n_flag == 0:
        if h_flag != 0 or (a_val & 0x0F) > 9:
            adjust |= 0x06
        if c_flag != 0 or a_val > 0x99:
            adjust |= 0x60
            carry = 1
        a_val = (a_val + adjust) & 0xFF
    else:
        if h_flag != 0:
            adjust |= 0x06
        if c_flag != 0:
            adjust |= 0x60
        a_val = (a_val - adjust) & 0xFF
    z = wp.where(a_val == 0, 1, 0)
    f_i = make_flags(z, n_flag, 0, carry)
    a_i = a_val
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_cpl(pc_i: int, a_i: int, f_i: int) -> None:
    """CPL template."""
    a_i = a_i ^ 0xFF
    z = (f_i >> 7) & 0x1
    cflag = (f_i >> 4) & 0x1
    f_i = make_flags(z, 1, 1, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_scf(pc_i: int, f_i: int) -> None:
    """SCF template."""
    z = (f_i >> 7) & 0x1
    f_i = make_flags(z, 0, 0, 1)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_ccf(pc_i: int, f_i: int) -> None:
    """CCF template."""
    z = (f_i >> 7) & 0x1
    cflag = (f_i >> 4) & 0x1
    cflag = wp.where(cflag != 0, 0, 1)
    f_i = make_flags(z, 0, 0, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4
