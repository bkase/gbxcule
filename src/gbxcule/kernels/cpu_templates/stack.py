"""Stack/control-flow helper templates for Warp CPU stepping."""
# ruff: noqa: F821, F841

from __future__ import annotations

import warp as wp


def template_push_r16(
    pc_i: int, sp_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """PUSH rr template (HREG_i/LREG_i placeholders)."""
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        HREG_i,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        LREG_i,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 16


def template_pop_r16(
    pc_i: int, sp_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """POP rr template (HREG_i/LREG_i placeholders)."""
    lo = read8(
        i,
        base,
        sp_i,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    sp_i = (sp_i + 1) & 0xFFFF
    hi = read8(
        i,
        base,
        sp_i,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    sp_i = (sp_i + 1) & 0xFFFF
    HREG_i = hi & 0xFF
    LREG_i = lo & 0xFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 12


def template_push_af(
    pc_i: int, sp_i: int, a_i: int, f_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """PUSH AF template."""
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        a_i,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        f_i & 0xF0,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 16


def template_pop_af(
    pc_i: int, sp_i: int, a_i: int, f_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """POP AF template."""
    lo = read8(
        i,
        base,
        sp_i,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    sp_i = (sp_i + 1) & 0xFFFF
    hi = read8(
        i,
        base,
        sp_i,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    sp_i = (sp_i + 1) & 0xFFFF
    a_i = hi & 0xFF
    f_i = lo & 0xF0
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 12


def template_call_a16(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """CALL a16 template."""
    lo = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    hi = read8(
        i,
        base,
        (pc_i + 2) & 0xFFFF,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    ret = (pc_i + 3) & 0xFFFF
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        (ret >> 8) & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        ret & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = ((hi << 8) | lo) & 0xFFFF
    cycles = 24


def template_call_nz_a16(
    pc_i: int, sp_i: int, f_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CALL NZ, a16 template."""
    z = (f_i >> 7) & 0x1
    take = wp.where(z == 0, 1, 0)
    if take != 0:
        lo = read8(
            i,
            base,
            (pc_i + 1) & 0xFFFF,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        hi = read8(
            i,
            base,
            (pc_i + 2) & 0xFFFF,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        ret = (pc_i + 3) & 0xFFFF
        sp_i = (sp_i - 1) & 0xFFFF
        write8(
            i,
            base,
            sp_i,
            (ret >> 8) & 0xFF,
            mem,
            joyp_select,
            serial_buf,
            serial_len,
            serial_overflow,
        )
        sp_i = (sp_i - 1) & 0xFFFF
        write8(
            i,
            base,
            sp_i,
            ret & 0xFF,
            mem,
            joyp_select,
            serial_buf,
            serial_len,
            serial_overflow,
        )
        pc_i = ((hi << 8) | lo) & 0xFFFF
        cycles = 24
    else:
        pc_i = (pc_i + 3) & 0xFFFF
        cycles = 12


def template_call_z_a16(
    pc_i: int, sp_i: int, f_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CALL Z, a16 template."""
    z = (f_i >> 7) & 0x1
    take = wp.where(z != 0, 1, 0)
    if take != 0:
        lo = read8(
            i,
            base,
            (pc_i + 1) & 0xFFFF,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        hi = read8(
            i,
            base,
            (pc_i + 2) & 0xFFFF,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        ret = (pc_i + 3) & 0xFFFF
        sp_i = (sp_i - 1) & 0xFFFF
        write8(
            i,
            base,
            sp_i,
            (ret >> 8) & 0xFF,
            mem,
            joyp_select,
            serial_buf,
            serial_len,
            serial_overflow,
        )
        sp_i = (sp_i - 1) & 0xFFFF
        write8(
            i,
            base,
            sp_i,
            ret & 0xFF,
            mem,
            joyp_select,
            serial_buf,
            serial_len,
            serial_overflow,
        )
        pc_i = ((hi << 8) | lo) & 0xFFFF
        cycles = 24
    else:
        pc_i = (pc_i + 3) & 0xFFFF
        cycles = 12


def template_call_nc_a16(
    pc_i: int, sp_i: int, f_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CALL NC, a16 template."""
    cflag = (f_i >> 4) & 0x1
    take = wp.where(cflag == 0, 1, 0)
    if take != 0:
        lo = read8(
            i,
            base,
            (pc_i + 1) & 0xFFFF,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        hi = read8(
            i,
            base,
            (pc_i + 2) & 0xFFFF,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        ret = (pc_i + 3) & 0xFFFF
        sp_i = (sp_i - 1) & 0xFFFF
        write8(
            i,
            base,
            sp_i,
            (ret >> 8) & 0xFF,
            mem,
            joyp_select,
            serial_buf,
            serial_len,
            serial_overflow,
        )
        sp_i = (sp_i - 1) & 0xFFFF
        write8(
            i,
            base,
            sp_i,
            ret & 0xFF,
            mem,
            joyp_select,
            serial_buf,
            serial_len,
            serial_overflow,
        )
        pc_i = ((hi << 8) | lo) & 0xFFFF
        cycles = 24
    else:
        pc_i = (pc_i + 3) & 0xFFFF
        cycles = 12


def template_call_c_a16(
    pc_i: int, sp_i: int, f_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """CALL C, a16 template."""
    cflag = (f_i >> 4) & 0x1
    take = wp.where(cflag != 0, 1, 0)
    if take != 0:
        lo = read8(
            i,
            base,
            (pc_i + 1) & 0xFFFF,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        hi = read8(
            i,
            base,
            (pc_i + 2) & 0xFFFF,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        ret = (pc_i + 3) & 0xFFFF
        sp_i = (sp_i - 1) & 0xFFFF
        write8(
            i,
            base,
            sp_i,
            (ret >> 8) & 0xFF,
            mem,
            joyp_select,
            serial_buf,
            serial_len,
            serial_overflow,
        )
        sp_i = (sp_i - 1) & 0xFFFF
        write8(
            i,
            base,
            sp_i,
            ret & 0xFF,
            mem,
            joyp_select,
            serial_buf,
            serial_len,
            serial_overflow,
        )
        pc_i = ((hi << 8) | lo) & 0xFFFF
        cycles = 24
    else:
        pc_i = (pc_i + 3) & 0xFFFF
        cycles = 12


def template_ret(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RET template."""
    lo = read8(
        i,
        base,
        sp_i,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    sp_i = (sp_i + 1) & 0xFFFF
    hi = read8(
        i,
        base,
        sp_i,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    sp_i = (sp_i + 1) & 0xFFFF
    pc_i = ((hi << 8) | lo) & 0xFFFF
    cycles = 16


def template_ret_nz(pc_i: int, sp_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RET NZ template."""
    z = (f_i >> 7) & 0x1
    take = wp.where(z == 0, 1, 0)
    if take != 0:
        lo = read8(
            i,
            base,
            sp_i,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        sp_i = (sp_i + 1) & 0xFFFF
        hi = read8(
            i,
            base,
            sp_i,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        sp_i = (sp_i + 1) & 0xFFFF
        pc_i = ((hi << 8) | lo) & 0xFFFF
        cycles = 20
    else:
        pc_i = (pc_i + 1) & 0xFFFF
        cycles = 8


def template_ret_z(pc_i: int, sp_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RET Z template."""
    z = (f_i >> 7) & 0x1
    take = wp.where(z != 0, 1, 0)
    if take != 0:
        lo = read8(
            i,
            base,
            sp_i,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        sp_i = (sp_i + 1) & 0xFFFF
        hi = read8(
            i,
            base,
            sp_i,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        sp_i = (sp_i + 1) & 0xFFFF
        pc_i = ((hi << 8) | lo) & 0xFFFF
        cycles = 20
    else:
        pc_i = (pc_i + 1) & 0xFFFF
        cycles = 8


def template_ret_nc(pc_i: int, sp_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RET NC template."""
    cflag = (f_i >> 4) & 0x1
    take = wp.where(cflag == 0, 1, 0)
    if take != 0:
        lo = read8(
            i,
            base,
            sp_i,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        sp_i = (sp_i + 1) & 0xFFFF
        hi = read8(
            i,
            base,
            sp_i,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        sp_i = (sp_i + 1) & 0xFFFF
        pc_i = ((hi << 8) | lo) & 0xFFFF
        cycles = 20
    else:
        pc_i = (pc_i + 1) & 0xFFFF
        cycles = 8


def template_ret_c(pc_i: int, sp_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RET C template."""
    cflag = (f_i >> 4) & 0x1
    take = wp.where(cflag != 0, 1, 0)
    if take != 0:
        lo = read8(
            i,
            base,
            sp_i,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        sp_i = (sp_i + 1) & 0xFFFF
        hi = read8(
            i,
            base,
            sp_i,
            mem,
            actions,
            joyp_select,
            frames_done,
            release_after_frames,
            action_codec_id,
        )
        sp_i = (sp_i + 1) & 0xFFFF
        pc_i = ((hi << 8) | lo) & 0xFFFF
        cycles = 20
    else:
        pc_i = (pc_i + 1) & 0xFFFF
        cycles = 8


def template_reti(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RETI template (IME not yet modeled)."""
    lo = read8(
        i,
        base,
        sp_i,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    sp_i = (sp_i + 1) & 0xFFFF
    hi = read8(
        i,
        base,
        sp_i,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    sp_i = (sp_i + 1) & 0xFFFF
    pc_i = ((hi << 8) | lo) & 0xFFFF
    cycles = 16


def template_rst(pc_i: int, sp_i: int, base: int, mem: wp.array, vector: int) -> None:  # type: ignore[name-defined]
    """RST n template (vector placeholder)."""
    ret = (pc_i + 1) & 0xFFFF
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        (ret >> 8) & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        ret & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = vector & 0xFFFF
    cycles = 16


def template_rst_00(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RST 00h template."""
    ret = (pc_i + 1) & 0xFFFF
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        (ret >> 8) & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        ret & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = 0x00
    cycles = 16


def template_rst_08(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RST 08h template."""
    ret = (pc_i + 1) & 0xFFFF
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        (ret >> 8) & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        ret & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = 0x08
    cycles = 16


def template_rst_10(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RST 10h template."""
    ret = (pc_i + 1) & 0xFFFF
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        (ret >> 8) & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        ret & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = 0x10
    cycles = 16


def template_rst_18(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RST 18h template."""
    ret = (pc_i + 1) & 0xFFFF
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        (ret >> 8) & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        ret & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = 0x18
    cycles = 16


def template_rst_20(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RST 20h template."""
    ret = (pc_i + 1) & 0xFFFF
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        (ret >> 8) & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        ret & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = 0x20
    cycles = 16


def template_rst_28(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RST 28h template."""
    ret = (pc_i + 1) & 0xFFFF
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        (ret >> 8) & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        ret & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = 0x28
    cycles = 16


def template_rst_30(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RST 30h template."""
    ret = (pc_i + 1) & 0xFFFF
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        (ret >> 8) & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        ret & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = 0x30
    cycles = 16


def template_rst_38(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """RST 38h template."""
    ret = (pc_i + 1) & 0xFFFF
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        (ret >> 8) & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    sp_i = (sp_i - 1) & 0xFFFF
    write8(
        i,
        base,
        sp_i,
        ret & 0xFF,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = 0x38
    cycles = 16
