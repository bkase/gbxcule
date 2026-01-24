"""LibCST-based builder for Warp cpu_step kernels."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import sys
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import libcst as cst


@dataclass(frozen=True)
class OpcodeTemplate:
    opcode: int
    template: Callable[..., Any]
    replacements: dict[str, str]


class TemplateSpecializer(cst.CSTTransformer):
    """Rename placeholder names in template bodies (e.g., REG_i -> a_i)."""

    def __init__(self, replacements: dict[str, str]):
        self._replacements = replacements

    def leave_Name(  # type: ignore[override]
        self, original: cst.Name, updated: cst.Name
    ) -> cst.Name:
        replacement = self._replacements.get(original.value)
        if replacement is None:
            return updated
        return updated.with_changes(value=replacement)


def _get_template_body(
    func_obj: Callable[..., Any], replacements: dict[str, str]
) -> list[cst.BaseStatement]:
    source = inspect.getsource(func_obj)
    tree = cst.parse_module(source)
    func_def = tree.body[0]
    if not isinstance(func_def, cst.FunctionDef):
        raise ValueError("Template source must be a function definition")
    transformer = TemplateSpecializer(replacements)
    modified_func = func_def.visit(transformer)
    assert isinstance(modified_func, cst.FunctionDef)
    return list(modified_func.body.body)  # type: ignore[misc]


def get_template_body(
    func_obj: Callable[..., Any], replacements: dict[str, str] | None = None
) -> list[cst.BaseStatement]:
    """Public wrapper for extracting a template body."""
    return _get_template_body(func_obj, replacements or {})


def _build_linear_dispatch(
    opcode_templates: Sequence[OpcodeTemplate],
    default_body: list[cst.BaseStatement],
    opcode_var: str,
) -> cst.If:
    current_node: cst.If = cst.If(
        test=cst.Name("True"),
        body=cst.IndentedBlock(body=default_body),
    )

    for spec in reversed(opcode_templates):
        body = _get_template_body(spec.template, spec.replacements)
        condition = cst.Comparison(
            left=cst.Name(opcode_var),
            comparisons=[
                cst.ComparisonTarget(
                    operator=cst.Equal(),
                    comparator=cst.Integer(f"0x{spec.opcode:02X}"),
                )
            ],
        )
        current_node = cst.If(
            test=condition,
            body=cst.IndentedBlock(body=body),
            orelse=cst.Else(body=cst.IndentedBlock(body=[current_node])),
        )

    return current_node


def _build_bucket_dispatch(
    opcode_templates: Sequence[OpcodeTemplate],
    default_body: list[cst.BaseStatement],
    opcode_var: str,
    opcode_hi_var: str,
) -> cst.If:
    buckets: dict[int, list[OpcodeTemplate]] = {}
    for spec in opcode_templates:
        buckets.setdefault(spec.opcode >> 4, []).append(spec)

    current_node: cst.If = cst.If(
        test=cst.Name("True"),
        body=cst.IndentedBlock(body=default_body),
    )

    for hi in reversed(sorted(buckets)):
        bucket = sorted(buckets[hi], key=lambda spec: spec.opcode)
        inner_dispatch = _build_linear_dispatch(bucket, default_body, opcode_var)
        condition = cst.Comparison(
            left=cst.Name(opcode_hi_var),
            comparisons=[
                cst.ComparisonTarget(
                    operator=cst.Equal(),
                    comparator=cst.Integer(str(hi)),
                )
            ],
        )
        current_node = cst.If(
            test=condition,
            body=cst.IndentedBlock(body=[inner_dispatch]),
            orelse=cst.Else(body=cst.IndentedBlock(body=[current_node])),
        )

    return current_node


_CPU_STEP_SKELETON = textwrap.dedent(
    """
    import warp as wp

    MEM_SIZE = {MEM_SIZE}
    CYCLES_PER_FRAME = {CYCLES_PER_FRAME}
    CYCLES_PER_SCANLINE = {CYCLES_PER_SCANLINE}
    LINES_PER_FRAME = {LINES_PER_FRAME}
    SCREEN_H = {SCREEN_H}
    ROM_LIMIT = {ROM_LIMIT}
    CART_RAM_START = {CART_RAM_START}
    CART_RAM_END = {CART_RAM_END}
    CART_ROM_BANK_SIZE = {CART_ROM_BANK_SIZE}
    CART_RAM_BANK_SIZE = {CART_RAM_BANK_SIZE}
    BOOTROM_SIZE = {BOOTROM_SIZE}
    CART_STATE_STRIDE = {CART_STATE_STRIDE}
    CART_STATE_MBC_KIND = {CART_STATE_MBC_KIND}
    CART_STATE_RAM_ENABLE = {CART_STATE_RAM_ENABLE}
    CART_STATE_ROM_BANK_LO = {CART_STATE_ROM_BANK_LO}
    CART_STATE_ROM_BANK_HI = {CART_STATE_ROM_BANK_HI}
    CART_STATE_RAM_BANK = {CART_STATE_RAM_BANK}
    CART_STATE_BANK_MODE = {CART_STATE_BANK_MODE}
    CART_STATE_BOOTROM_ENABLED = {CART_STATE_BOOTROM_ENABLED}
    CART_STATE_RTC_SELECT = {CART_STATE_RTC_SELECT}
    CART_STATE_RTC_LATCH = {CART_STATE_RTC_LATCH}
    CART_STATE_RTC_SECONDS = {CART_STATE_RTC_SECONDS}
    CART_STATE_RTC_MINUTES = {CART_STATE_RTC_MINUTES}
    CART_STATE_RTC_HOURS = {CART_STATE_RTC_HOURS}
    CART_STATE_RTC_DAYS_LOW = {CART_STATE_RTC_DAYS_LOW}
    CART_STATE_RTC_DAYS_HIGH = {CART_STATE_RTC_DAYS_HIGH}
    CART_STATE_RTC_LAST_CYCLE = {CART_STATE_RTC_LAST_CYCLE}
    MBC_KIND_ROM_ONLY = {MBC_KIND_ROM_ONLY}
    MBC_KIND_MBC1 = {MBC_KIND_MBC1}
    MBC_KIND_MBC3 = {MBC_KIND_MBC3}
    OBS_DIM = {OBS_DIM}
    SERIAL_MAX = {SERIAL_MAX}

    ACTION_CODEC_POKERED = 0

    DPAD_RIGHT = 1
    DPAD_LEFT = 2
    DPAD_UP = 4
    DPAD_DOWN = 8

    BUTTON_A = 1
    BUTTON_B = 2
    BUTTON_SELECT = 4
    BUTTON_START = 8

    @wp.func
    def sign8(x: wp.int32) -> wp.int32:
        return wp.where(x >= 128, x - 256, x)

    @wp.func
    def make_flags(z: wp.int32, n: wp.int32, h: wp.int32, c: wp.int32) -> wp.int32:
        return (z << 7) | (n << 6) | (h << 5) | (c << 4)

    @wp.func
    def action_dpad_mask(action: wp.int32, codec_id: wp.int32) -> wp.int32:
        dpad = wp.int32(0)
        if action == 3:
            dpad = DPAD_UP
        elif action == 4:
            dpad = DPAD_DOWN
        elif action == 5:
            dpad = DPAD_LEFT
        elif action == 6:
            dpad = DPAD_RIGHT
        return dpad

    @wp.func
    def action_button_mask(action: wp.int32, codec_id: wp.int32) -> wp.int32:
        btn = wp.int32(0)
        if action == 0:
            btn = BUTTON_A
        elif action == 1:
            btn = BUTTON_B
        elif action == 2:
            btn = BUTTON_START
        return btn

    @wp.func
    def joyp_read(
        action: wp.int32,
        frame_idx: wp.int32,
        release_after_frames: wp.int32,
        joyp_sel: wp.int32,
        codec_id: wp.int32,
    ) -> wp.int32:
        sel = joyp_sel & 0x30
        pressed = wp.int32(frame_idx < release_after_frames)
        dpad = wp.int32(0)
        btn = wp.int32(0)
        if pressed != 0:
            dpad = action_dpad_mask(action, codec_id)
            btn = action_button_mask(action, codec_id)
        low = wp.int32(0x0F)
        if (sel & 0x10) == 0:
            low = low & (0x0F ^ dpad)
        if (sel & 0x20) == 0:
            low = low & (0x0F ^ btn)
        low = low & 0x0F
        return 0xC0 | sel | low

    @wp.func
    def timer_div_bit(tac: wp.int32) -> wp.int32:
        sel = tac & 0x03
        if sel == 0:
            return 9
        if sel == 1:
            return 3
        if sel == 2:
            return 5
        return 7

    @wp.func
    def timer_in(div: wp.int32, tac: wp.int32) -> wp.int32:
        if (tac & 0x04) == 0:
            return 0
        bit = timer_div_bit(tac)
        return (div >> bit) & 0x1

    @wp.func
    def tima_inc(
        i: wp.int32,
        base: wp.int32,
        mem: wp.array(dtype=wp.uint8),
        tima_reload_pending: wp.array(dtype=wp.int32),
        tima_reload_delay: wp.array(dtype=wp.int32),
    ) -> None:
        tima = wp.int32(mem[base + 0xFF05]) & 0xFF
        tima2 = (tima + 1) & 0xFF
        mem[base + 0xFF05] = wp.uint8(tima2)
        if tima == 0xFF:
            tima_reload_pending[i] = 1
            tima_reload_delay[i] = 4

    @wp.func
    def tima_reload_now(
        i: wp.int32,
        base: wp.int32,
        mem: wp.array(dtype=wp.uint8),
        tima_reload_pending: wp.array(dtype=wp.int32),
        tima_reload_delay: wp.array(dtype=wp.int32),
    ) -> None:
        tma = wp.int32(mem[base + 0xFF06]) & 0xFF
        mem[base + 0xFF05] = wp.uint8(tma)
        if_addr = base + 0xFF0F
        mem[if_addr] = wp.uint8(mem[if_addr] | wp.uint8(0x04))
        tima_reload_pending[i] = 0
        tima_reload_delay[i] = 0

    @wp.func
    def timer_tick(
        i: wp.int32,
        base: wp.int32,
        cycles: wp.int32,
        mem: wp.array(dtype=wp.uint8),
        div_counter: wp.array(dtype=wp.int32),
        timer_prev_in: wp.array(dtype=wp.int32),
        tima_reload_pending: wp.array(dtype=wp.int32),
        tima_reload_delay: wp.array(dtype=wp.int32),
    ) -> None:
        if cycles <= 0:
            return

        remaining = cycles
        div = div_counter[i] & 0xFFFF

        while remaining > 0:
            if tima_reload_pending[i] != 0 and tima_reload_delay[i] <= 0:
                tima_reload_now(
                    i,
                    base,
                    mem,
                    tima_reload_pending,
                    tima_reload_delay,
                )

            tac = wp.int32(mem[base + 0xFF07]) & 0x07
            dt = remaining

            if tima_reload_pending[i] != 0:
                rd = tima_reload_delay[i]
                if rd < dt:
                    dt = rd

            do_edge = wp.int32(0)
            if (tac & 0x04) != 0:
                bit = timer_div_bit(tac)
                period = wp.int32(1) << (bit + 1)
                mask = period - 1
                mod = div & mask
                edge_dt = period - mod
                if edge_dt < dt:
                    dt = edge_dt
                if dt == edge_dt:
                    do_edge = 1

            div = (div + dt) & 0xFFFF

            if tima_reload_pending[i] != 0:
                tima_reload_delay[i] = tima_reload_delay[i] - dt

            remaining = remaining - dt

            if do_edge != 0:
                tima_inc(
                    i,
                    base,
                    mem,
                    tima_reload_pending,
                    tima_reload_delay,
                )

            if tima_reload_pending[i] != 0 and tima_reload_delay[i] <= 0:
                tima_reload_now(
                    i,
                    base,
                    mem,
                    tima_reload_pending,
                    tima_reload_delay,
                )

        div_counter[i] = div
        mem[base + 0xFF04] = wp.uint8((div >> 8) & 0xFF)
        tac = wp.int32(mem[base + 0xFF07]) & 0x07
        timer_prev_in[i] = timer_in(div, tac)

    @wp.func
    def ppu_update_stat(
        base: wp.int32,
        mem: wp.array(dtype=wp.uint8),
        ly_val: wp.int32,
        mode: wp.int32,
    ) -> None:
        lyc = wp.int32(mem[base + 0xFF45]) & 0xFF
        coincidence = wp.int32(0)
        if ly_val == lyc:
            coincidence = 1
        stat_keep = wp.int32(mem[base + 0xFF41]) & 0xF8
        mem[base + 0xFF41] = wp.uint8(stat_keep | (coincidence << 2) | mode)

    @wp.func
    def ppu_update_stat_irq(
        i: wp.int32,
        base: wp.int32,
        mem: wp.array(dtype=wp.uint8),
        ly_val: wp.int32,
        mode: wp.int32,
        stat_prev: wp.array(dtype=wp.uint8),
    ) -> None:
        lyc = wp.int32(mem[base + 0xFF45]) & 0xFF
        coincidence = wp.int32(0)
        if ly_val == lyc:
            coincidence = 1
        stat_keep = wp.int32(mem[base + 0xFF41]) & 0xF8
        mem[base + 0xFF41] = wp.uint8(stat_keep | (coincidence << 2) | mode)

        enable_mode0 = (stat_keep >> 3) & 1
        enable_mode1 = (stat_keep >> 4) & 1
        enable_mode2 = (stat_keep >> 5) & 1
        enable_lyc = (stat_keep >> 6) & 1

        curr_mode0 = wp.int32(0)
        curr_mode1 = wp.int32(0)
        curr_mode2 = wp.int32(0)
        if mode == 0:
            curr_mode0 = 1
        elif mode == 1:
            curr_mode1 = 1
        elif mode == 2:
            curr_mode2 = 1
        curr_lyc = coincidence

        prev = wp.int32(stat_prev[i]) & 0x0F
        prev_mode0 = prev & 1
        prev_mode1 = (prev >> 1) & 1
        prev_mode2 = (prev >> 2) & 1
        prev_lyc = (prev >> 3) & 1

        edge = wp.int32(0)
        if enable_mode0 != 0 and prev_mode0 == 0 and curr_mode0 != 0:
            edge = 1
        elif enable_mode1 != 0 and prev_mode1 == 0 and curr_mode1 != 0:
            edge = 1
        elif enable_mode2 != 0 and prev_mode2 == 0 and curr_mode2 != 0:
            edge = 1
        elif enable_lyc != 0 and prev_lyc == 0 and curr_lyc != 0:
            edge = 1

        if edge != 0:
            if_addr = base + 0xFF0F
            mem[if_addr] = wp.uint8(mem[if_addr] | wp.uint8(0x02))

        stat_prev[i] = wp.uint8(
            (curr_mode0 & 1)
            | ((curr_mode1 & 1) << 1)
            | ((curr_mode2 & 1) << 2)
            | ((curr_lyc & 1) << 3)
        )

    @wp.func
    def ppu_capture_latches_env0(
        i: wp.int32,
        ly_val: wp.int32,
        window_line: wp.int32,
        base: wp.int32,
        mem: wp.array(dtype=wp.uint8),
        bg_lcdc_latch_env0: wp.array(dtype=wp.uint8),
        bg_scx_latch_env0: wp.array(dtype=wp.uint8),
        bg_scy_latch_env0: wp.array(dtype=wp.uint8),
        bg_bgp_latch_env0: wp.array(dtype=wp.uint8),
        win_wx_latch_env0: wp.array(dtype=wp.uint8),
        win_wy_latch_env0: wp.array(dtype=wp.uint8),
        win_line_latch_env0: wp.array(dtype=wp.uint8),
        obj_obp0_latch_env0: wp.array(dtype=wp.uint8),
        obj_obp1_latch_env0: wp.array(dtype=wp.uint8),
    ) -> None:
        if i != 0:
            return
        if ly_val < 0 or ly_val >= SCREEN_H:
            return
        idx = wp.int32(ly_val)
        bg_lcdc_latch_env0[idx] = mem[base + 0xFF40]
        bg_scx_latch_env0[idx] = mem[base + 0xFF43]
        bg_scy_latch_env0[idx] = mem[base + 0xFF42]
        bg_bgp_latch_env0[idx] = mem[base + 0xFF47]
        win_wx_latch_env0[idx] = mem[base + 0xFF4B]
        win_wy_latch_env0[idx] = mem[base + 0xFF4A]
        win_line_latch_env0[idx] = wp.uint8(window_line & 0xFF)
        obj_obp0_latch_env0[idx] = mem[base + 0xFF48]
        obj_obp1_latch_env0[idx] = mem[base + 0xFF49]

    @wp.func
    def clamp_bank(
        bank: wp.int32,
        bank_count: wp.int32,
        bank_mask: wp.int32,
    ) -> wp.int32:
        if bank_count <= 0:
            return wp.int32(0)
        if bank_mask >= 0:
            return bank & bank_mask
        return bank % bank_count

    @wp.func
    def read8(
        i: wp.int32,
        base: wp.int32,
        addr: wp.int32,
        mem: wp.array(dtype=wp.uint8),
        rom: wp.array(dtype=wp.uint8),
        bootrom: wp.array(dtype=wp.uint8),
        cart_ram: wp.array(dtype=wp.uint8),
        cart_state: wp.array(dtype=wp.int32),
        rom_bank_count: wp.int32,
        rom_bank_mask: wp.int32,
        ram_bank_count: wp.int32,
        actions: wp.array(dtype=wp.int32),
        joyp_select: wp.array(dtype=wp.uint8),
        frames_done: wp.int32,
        release_after_frames: wp.int32,
        action_codec_id: wp.int32,
    ) -> wp.int32:
        addr16 = addr & 0xFFFF
        state_base = i * CART_STATE_STRIDE
        if addr16 < BOOTROM_SIZE and cart_state[
            state_base + CART_STATE_BOOTROM_ENABLED
        ] != 0:
            return wp.int32(bootrom[addr16])
        if addr16 < ROM_LIMIT:
            mbc_kind = cart_state[state_base + CART_STATE_MBC_KIND]
            rom_bank_lo_raw = cart_state[state_base + CART_STATE_ROM_BANK_LO]
            rom_bank_lo = rom_bank_lo_raw & 0x1F
            rom_bank_hi = cart_state[state_base + CART_STATE_ROM_BANK_HI] & 0x03
            bank_mode = cart_state[state_base + CART_STATE_BANK_MODE] & 0x01
            bank = wp.int32(0)
            if addr16 < 0x4000:
                if mbc_kind == MBC_KIND_MBC1 and bank_mode != 0:
                    bank = (rom_bank_hi & 0x03) << 5
            else:
                if mbc_kind == MBC_KIND_ROM_ONLY:
                    bank = wp.int32(1)
                elif mbc_kind == MBC_KIND_MBC1:
                    bank = (rom_bank_hi << 5) | rom_bank_lo
                    if bank == 0:
                        bank = wp.int32(1)
                else:
                    bank = rom_bank_lo_raw & 0x7F
                    if bank == 0:
                        bank = wp.int32(1)
            bank = clamp_bank(bank, rom_bank_count, rom_bank_mask)
            rom_index = wp.int32(0)
            if addr16 < 0x4000:
                rom_index = bank * CART_ROM_BANK_SIZE + addr16
            else:
                rom_index = bank * CART_ROM_BANK_SIZE + (addr16 - 0x4000)
            if rom_index < rom_bank_count * CART_ROM_BANK_SIZE:
                return wp.int32(rom[rom_index])
            return 0xFF
        if addr16 == 0xFF00:
            action_i = wp.int32(actions[i])
            joyp_sel = wp.int32(joyp_select[i])
            return joyp_read(
                action_i,
                frames_done,
                release_after_frames,
                joyp_sel,
                action_codec_id,
            )
        if addr16 >= CART_RAM_START and addr16 < CART_RAM_END:
            if ram_bank_count <= 0:
                return 0xFF
            if cart_state[state_base + CART_STATE_RAM_ENABLE] == 0:
                return 0xFF
            mbc_kind = cart_state[state_base + CART_STATE_MBC_KIND]
            if mbc_kind == MBC_KIND_MBC3:
                rtc_sel = cart_state[state_base + CART_STATE_RTC_SELECT] & 0x0F
                if rtc_sel == 0x08:
                    return cart_state[state_base + CART_STATE_RTC_SECONDS] & 0xFF
                if rtc_sel == 0x09:
                    return cart_state[state_base + CART_STATE_RTC_MINUTES] & 0xFF
                if rtc_sel == 0x0A:
                    return cart_state[state_base + CART_STATE_RTC_HOURS] & 0xFF
                if rtc_sel == 0x0B:
                    return cart_state[state_base + CART_STATE_RTC_DAYS_LOW] & 0xFF
                if rtc_sel == 0x0C:
                    return cart_state[state_base + CART_STATE_RTC_DAYS_HIGH] & 0xFF
            ram_bank = wp.int32(0)
            if mbc_kind == MBC_KIND_MBC1:
                bank_mode = cart_state[state_base + CART_STATE_BANK_MODE] & 0x01
                if bank_mode != 0:
                    ram_bank = cart_state[state_base + CART_STATE_RAM_BANK] & 0x03
            elif mbc_kind == MBC_KIND_MBC3:
                ram_bank = cart_state[state_base + CART_STATE_RAM_BANK] & 0x03
            ram_bank = clamp_bank(ram_bank, ram_bank_count, wp.int32(-1))
            ram_env_bytes = ram_bank_count * CART_RAM_BANK_SIZE
            if ram_env_bytes <= 0:
                return 0xFF
            offset = ram_bank * CART_RAM_BANK_SIZE + (addr16 - CART_RAM_START)
            if offset >= ram_env_bytes:
                return 0xFF
            ram_index = i * ram_env_bytes + offset
            return wp.int32(cart_ram[ram_index])
        return wp.int32(mem[base + addr16])

    @wp.func
    def write8(
        i: wp.int32,
        base: wp.int32,
        addr: wp.int32,
        val: wp.int32,
        mem: wp.array(dtype=wp.uint8),
        rom: wp.array(dtype=wp.uint8),
        bootrom: wp.array(dtype=wp.uint8),
        cart_ram: wp.array(dtype=wp.uint8),
        cart_state: wp.array(dtype=wp.int32),
        rom_bank_count: wp.int32,
        rom_bank_mask: wp.int32,
        ram_bank_count: wp.int32,
        joyp_select: wp.array(dtype=wp.uint8),
        serial_buf: wp.array(dtype=wp.uint8),
        serial_len: wp.array(dtype=wp.int32),
        serial_overflow: wp.array(dtype=wp.uint8),
        div_counter: wp.array(dtype=wp.int32),
        timer_prev_in: wp.array(dtype=wp.int32),
        tima_reload_pending: wp.array(dtype=wp.int32),
        tima_reload_delay: wp.array(dtype=wp.int32),
    ) -> None:
        addr16 = addr & 0xFFFF
        val8 = wp.uint8(val)
        state_base = i * CART_STATE_STRIDE
        if addr16 == 0xFF00:
            joyp_select[i] = val8 & wp.uint8(0x30)
            return
        if addr16 == 0xFF0F:
            mem[base + addr16] = val8 & wp.uint8(0x1F)
            return
        if addr16 == 0xFF01:
            mem[base + addr16] = val8
            return
        if addr16 == 0xFF02:
            mem[base + addr16] = val8
            if (val & 0x80) != 0 and (val & 0x01) != 0:
                sb = wp.int32(mem[base + 0xFF01]) & 0xFF
                idx = wp.int32(serial_len[i])
                if idx < SERIAL_MAX:
                    serial_buf[i * SERIAL_MAX + idx] = wp.uint8(sb)
                    serial_len[i] = idx + 1
                else:
                    serial_overflow[i] = wp.uint8(1)
                mem[base + addr16] = wp.uint8(val & 0x7F)
                if_addr = base + 0xFF0F
                mem[if_addr] = wp.uint8(mem[if_addr] | wp.uint8(0x08))
            return
        if addr16 == 0xFF50:
            if val != 0:
                cart_state[state_base + CART_STATE_BOOTROM_ENABLED] = 0
            return
        if addr16 == 0xFF04:
            tac = wp.int32(mem[base + 0xFF07]) & 0x07
            div = div_counter[i] & 0xFFFF
            pre_in = timer_in(div, tac)
            mem[base + addr16] = wp.uint8(0)
            div_counter[i] = 0
            post_in = timer_in(0, tac)
            timer_prev_in[i] = post_in
            if pre_in != 0 and post_in == 0:
                tima_inc(
                    i,
                    base,
                    mem,
                    tima_reload_pending,
                    tima_reload_delay,
                )
            return
        if addr16 == 0xFF05:
            mem[base + addr16] = val8
            return
        if addr16 == 0xFF06:
            mem[base + addr16] = val8
            return
        if addr16 == 0xFF07:
            div = div_counter[i] & 0xFFFF
            tac_old = wp.int32(mem[base + 0xFF07]) & 0x07
            pre_in = timer_in(div, tac_old)
            tac_new = wp.int32(val) & 0x07
            mem[base + addr16] = wp.uint8(tac_new)
            post_in = timer_in(div, tac_new)
            timer_prev_in[i] = post_in
            if pre_in != 0 and post_in == 0:
                tima_inc(
                    i,
                    base,
                    mem,
                    tima_reload_pending,
                    tima_reload_delay,
                )
            return
        if addr16 == 0xFF46:
            mem[base + addr16] = val8
            src = (wp.int32(val) & 0xFF) << 8
            for offset in range(160):
                mem[base + 0xFE00 + offset] = mem[base + ((src + offset) & 0xFFFF)]
            return
        if addr16 == 0xFFFF:
            mem[base + addr16] = val8 & wp.uint8(0x1F)
            return
        if addr16 >= CART_RAM_START and addr16 < CART_RAM_END:
            if ram_bank_count <= 0:
                return
            if cart_state[state_base + CART_STATE_RAM_ENABLE] == 0:
                return
            mbc_kind = cart_state[state_base + CART_STATE_MBC_KIND]
            if mbc_kind == MBC_KIND_MBC3:
                rtc_sel = cart_state[state_base + CART_STATE_RTC_SELECT] & 0x0F
                if rtc_sel == 0x08:
                    cart_state[state_base + CART_STATE_RTC_SECONDS] = val & 0xFF
                    return
                if rtc_sel == 0x09:
                    cart_state[state_base + CART_STATE_RTC_MINUTES] = val & 0xFF
                    return
                if rtc_sel == 0x0A:
                    cart_state[state_base + CART_STATE_RTC_HOURS] = val & 0xFF
                    return
                if rtc_sel == 0x0B:
                    cart_state[state_base + CART_STATE_RTC_DAYS_LOW] = val & 0xFF
                    return
                if rtc_sel == 0x0C:
                    cart_state[state_base + CART_STATE_RTC_DAYS_HIGH] = val & 0xFF
                    return
            ram_bank = wp.int32(0)
            if mbc_kind == MBC_KIND_MBC1:
                bank_mode = cart_state[state_base + CART_STATE_BANK_MODE] & 0x01
                if bank_mode != 0:
                    ram_bank = cart_state[state_base + CART_STATE_RAM_BANK] & 0x03
            elif mbc_kind == MBC_KIND_MBC3:
                ram_bank = cart_state[state_base + CART_STATE_RAM_BANK] & 0x03
            ram_bank = clamp_bank(ram_bank, ram_bank_count, wp.int32(-1))
            ram_env_bytes = ram_bank_count * CART_RAM_BANK_SIZE
            if ram_env_bytes <= 0:
                return
            offset = ram_bank * CART_RAM_BANK_SIZE + (addr16 - CART_RAM_START)
            if offset >= ram_env_bytes:
                return
            ram_index = i * ram_env_bytes + offset
            cart_ram[ram_index] = val8
            return
        if addr16 < ROM_LIMIT:
            mbc_kind = cart_state[state_base + CART_STATE_MBC_KIND]
            if mbc_kind == MBC_KIND_MBC1:
                if addr16 < 0x2000:
                    cart_state[state_base + CART_STATE_RAM_ENABLE] = (
                        1 if (val & 0x0F) == 0x0A else 0
                    )
                elif addr16 < 0x4000:
                    bank = val & 0x1F
                    if bank == 0:
                        bank = 1
                    cart_state[state_base + CART_STATE_ROM_BANK_LO] = bank
                elif addr16 < 0x6000:
                    bank = val & 0x03
                    cart_state[state_base + CART_STATE_ROM_BANK_HI] = bank
                    cart_state[state_base + CART_STATE_RAM_BANK] = bank
                else:
                    cart_state[state_base + CART_STATE_BANK_MODE] = val & 0x01
            return
        mem[base + addr16] = val8

    @wp.kernel
    def cpu_step(
        mem: wp.array(dtype=wp.uint8),
        rom: wp.array(dtype=wp.uint8),
        bootrom: wp.array(dtype=wp.uint8),
        cart_ram: wp.array(dtype=wp.uint8),
        cart_state: wp.array(dtype=wp.int32),
        rom_bank_count: wp.int32,
        rom_bank_mask: wp.int32,
        ram_bank_count: wp.int32,
        pc: wp.array(dtype=wp.int32),
        sp: wp.array(dtype=wp.int32),
        a: wp.array(dtype=wp.int32),
        b: wp.array(dtype=wp.int32),
        c: wp.array(dtype=wp.int32),
        d: wp.array(dtype=wp.int32),
        e: wp.array(dtype=wp.int32),
        h: wp.array(dtype=wp.int32),
        l_reg: wp.array(dtype=wp.int32),
        f: wp.array(dtype=wp.int32),
        instr_count: wp.array(dtype=wp.int64),
        cycle_count: wp.array(dtype=wp.int64),
        cycle_in_frame: wp.array(dtype=wp.int32),
        trap_flag: wp.array(dtype=wp.int32),
        trap_pc: wp.array(dtype=wp.int32),
        trap_opcode: wp.array(dtype=wp.int32),
        trap_kind: wp.array(dtype=wp.int32),
        actions: wp.array(dtype=wp.int32),
        joyp_select: wp.array(dtype=wp.uint8),
        serial_buf: wp.array(dtype=wp.uint8),
        serial_len: wp.array(dtype=wp.int32),
        serial_overflow: wp.array(dtype=wp.uint8),
        ime: wp.array(dtype=wp.int32),
        ime_delay: wp.array(dtype=wp.int32),
        halted: wp.array(dtype=wp.int32),
        div_counter: wp.array(dtype=wp.int32),
        timer_prev_in: wp.array(dtype=wp.int32),
        tima_reload_pending: wp.array(dtype=wp.int32),
        tima_reload_delay: wp.array(dtype=wp.int32),
        ppu_scanline_cycle: wp.array(dtype=wp.int32),
        ppu_ly: wp.array(dtype=wp.int32),
        ppu_window_line: wp.array(dtype=wp.int32),
        ppu_stat_prev: wp.array(dtype=wp.uint8),
        bg_lcdc_latch_env0: wp.array(dtype=wp.uint8),
        bg_scx_latch_env0: wp.array(dtype=wp.uint8),
        bg_scy_latch_env0: wp.array(dtype=wp.uint8),
        bg_bgp_latch_env0: wp.array(dtype=wp.uint8),
        win_wx_latch_env0: wp.array(dtype=wp.uint8),
        win_wy_latch_env0: wp.array(dtype=wp.uint8),
        win_line_latch_env0: wp.array(dtype=wp.uint8),
        obj_obp0_latch_env0: wp.array(dtype=wp.uint8),
        obj_obp1_latch_env0: wp.array(dtype=wp.uint8),
        action_codec_id: wp.int32,
        reward_out: wp.array(dtype=wp.float32),
        obs_out: wp.array(dtype=wp.float32),
        frames_to_run: wp.int32,
        release_after_frames: wp.int32,
    ):
        i = wp.tid()
        base = i * MEM_SIZE

        pc_i = pc[i] & 0xFFFF
        sp_i = sp[i] & 0xFFFF
        a_i = a[i] & 0xFF
        b_i = b[i] & 0xFF
        c_i = c[i] & 0xFF
        d_i = d[i] & 0xFF
        e_i = e[i] & 0xFF
        h_i = h[i] & 0xFF
        l_i = l_reg[i] & 0xFF
        f_i = f[i] & 0xF0

        instr_i = instr_count[i]
        cycles_i = cycle_count[i]
        cycle_frame = cycle_in_frame[i]
        trap_i = trap_flag[i]
        trap_pc_i = trap_pc[i]
        trap_opcode_i = trap_opcode[i]
        trap_kind_i = trap_kind[i]
        scanline_cycle_i = ppu_scanline_cycle[i]
        ppu_ly_i = ppu_ly[i]
        window_line_i = ppu_window_line[i]

        frames_done = wp.int32(0)

        while frames_done < frames_to_run:
            if trap_i != 0:
                break
            if halted[i] != 0:
                cycles = wp.int32(4)
                timer_tick(
                    i,
                    base,
                    cycles,
                    mem,
                    div_counter,
                    timer_prev_in,
                    tima_reload_pending,
                    tima_reload_delay,
                )
                lcdc = wp.int32(mem[base + 0xFF40]) & 0xFF
                if (lcdc & 0x80) == 0:
                    scanline_cycle_i = 0
                    ppu_ly_i = 0
                    window_line_i = 0
                    mem[base + 0xFF44] = wp.uint8(0)
                    ppu_update_stat(base, mem, wp.int32(0), wp.int32(0))
                    ppu_stat_prev[i] = wp.uint8(0)
                else:
                    total = scanline_cycle_i + cycles
                    lines = total // CYCLES_PER_SCANLINE
                    scanline_cycle_i = total - lines * CYCLES_PER_SCANLINE
                    if lines == 0 and scanline_cycle_i == 0:
                        ppu_capture_latches_env0(
                            i,
                            ppu_ly_i,
                            window_line_i,
                            base,
                            mem,
                            bg_lcdc_latch_env0,
                            bg_scx_latch_env0,
                            bg_scy_latch_env0,
                            bg_bgp_latch_env0,
                            win_wx_latch_env0,
                            win_wy_latch_env0,
                            win_line_latch_env0,
                            obj_obp0_latch_env0,
                            obj_obp1_latch_env0,
                        )
                    while lines > 0:
                        prev_ly = ppu_ly_i
                        if prev_ly < SCREEN_H:
                            if (lcdc & 0x20) != 0:
                                wy = wp.int32(mem[base + 0xFF4A]) & 0xFF
                                if prev_ly >= wy:
                                    window_line_i = window_line_i + 1
                        ppu_ly_i = (ppu_ly_i + 1) % LINES_PER_FRAME
                        if ppu_ly_i == 0:
                            window_line_i = 0
                        mem[base + 0xFF44] = wp.uint8(ppu_ly_i)
                        mode = wp.int32(0)
                        if ppu_ly_i >= SCREEN_H:
                            mode = 1
                        ppu_update_stat_irq(
                            i,
                            base,
                            mem,
                            ppu_ly_i,
                            mode,
                            ppu_stat_prev,
                        )
                        if prev_ly == 143 and ppu_ly_i == 144:
                            if_addr = base + 0xFF0F
                            mem[if_addr] = wp.uint8(mem[if_addr] | wp.uint8(0x01))
                        if ppu_ly_i < SCREEN_H:
                            ppu_capture_latches_env0(
                                i,
                                ppu_ly_i,
                                window_line_i,
                                base,
                                mem,
                                bg_lcdc_latch_env0,
                                bg_scx_latch_env0,
                                bg_scy_latch_env0,
                                bg_bgp_latch_env0,
                                win_wx_latch_env0,
                                win_wy_latch_env0,
                                win_line_latch_env0,
                                obj_obp0_latch_env0,
                                obj_obp1_latch_env0,
                            )
                        lines = lines - 1
                pending = (
                    wp.int32(mem[base + 0xFFFF])
                    & wp.int32(mem[base + 0xFF0F])
                    & 0x1F
                )
                service_cycles = wp.int32(0)
                if pending != 0:
                    halted[i] = 0
                    if ime[i] != 0:
                        ime[i] = 0
                        vector = wp.int32(0x40)
                        bit = wp.int32(0x01)
                        if (pending & 0x01) != 0:
                            vector = wp.int32(0x40)
                            bit = wp.int32(0x01)
                        elif (pending & 0x02) != 0:
                            vector = wp.int32(0x48)
                            bit = wp.int32(0x02)
                        elif (pending & 0x04) != 0:
                            vector = wp.int32(0x50)
                            bit = wp.int32(0x04)
                        elif (pending & 0x08) != 0:
                            vector = wp.int32(0x58)
                            bit = wp.int32(0x08)
                        else:
                            vector = wp.int32(0x60)
                            bit = wp.int32(0x10)
                        if_addr = base + 0xFF0F
                        mem[if_addr] = wp.uint8(
                            mem[if_addr] & wp.uint8(0xFF ^ bit)
                        )
                        ret = pc_i & 0xFFFF
                        sp_i = (sp_i - 1) & 0xFFFF
                        write8(
                            i,
                            base,
                            sp_i,
                            (ret >> 8) & 0xFF,
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
                        sp_i = (sp_i - 1) & 0xFFFF
                        write8(
                            i,
                            base,
                            sp_i,
                            ret & 0xFF,
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
                        pc_i = vector
                        service_cycles = 20
                        timer_tick(
                            i,
                            base,
                            service_cycles,
                            mem,
                            div_counter,
                            timer_prev_in,
                            tima_reload_pending,
                            tima_reload_delay,
                        )

                total_cycles = cycles + service_cycles
                cycles_i += wp.int64(total_cycles)
                cycle_frame += total_cycles

                while cycle_frame >= CYCLES_PER_FRAME:
                    cycle_frame -= CYCLES_PER_FRAME
                    frames_done += 1
                    if frames_done >= frames_to_run:
                        break
                continue
            opcode = read8(
                i,
                base,
                pc_i,
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
            opcode_hi = opcode >> 4
            cycles = wp.int32(0)

            if opcode == 0xCB:
                cb_opcode = read8(
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
                cb_opcode_hi = cb_opcode >> 4
                pc_i = (pc_i + 2) & 0xFFFF
                CB_DISPATCH
            else:
                INSTRUCTION_DISPATCH

            if trap_i != 0:
                break

            f_i = f_i & 0xF0
            instr_i += wp.int64(1)
            timer_tick(
                i,
                base,
                cycles,
                mem,
                div_counter,
                timer_prev_in,
                tima_reload_pending,
                tima_reload_delay,
            )
            if ime_delay[i] != 0:
                ime[i] = 1
                ime_delay[i] = 0
            lcdc = wp.int32(mem[base + 0xFF40]) & 0xFF
            if (lcdc & 0x80) == 0:
                scanline_cycle_i = 0
                ppu_ly_i = 0
                window_line_i = 0
                mem[base + 0xFF44] = wp.uint8(0)
                ppu_update_stat(base, mem, wp.int32(0), wp.int32(0))
                ppu_stat_prev[i] = wp.uint8(0)
            else:
                total = scanline_cycle_i + cycles
                lines = total // CYCLES_PER_SCANLINE
                scanline_cycle_i = total - lines * CYCLES_PER_SCANLINE
                if lines == 0 and scanline_cycle_i == 0:
                    ppu_capture_latches_env0(
                        i,
                        ppu_ly_i,
                        window_line_i,
                        base,
                        mem,
                        bg_lcdc_latch_env0,
                        bg_scx_latch_env0,
                        bg_scy_latch_env0,
                        bg_bgp_latch_env0,
                        win_wx_latch_env0,
                        win_wy_latch_env0,
                        win_line_latch_env0,
                        obj_obp0_latch_env0,
                        obj_obp1_latch_env0,
                    )
                while lines > 0:
                    prev_ly = ppu_ly_i
                    if prev_ly < SCREEN_H:
                        if (lcdc & 0x20) != 0:
                            wy = wp.int32(mem[base + 0xFF4A]) & 0xFF
                            if prev_ly >= wy:
                                window_line_i = window_line_i + 1
                    ppu_ly_i = (ppu_ly_i + 1) % LINES_PER_FRAME
                    if ppu_ly_i == 0:
                        window_line_i = 0
                    mem[base + 0xFF44] = wp.uint8(ppu_ly_i)
                    mode = wp.int32(0)
                    if ppu_ly_i >= SCREEN_H:
                        mode = 1
                    ppu_update_stat_irq(
                        i,
                        base,
                        mem,
                        ppu_ly_i,
                        mode,
                        ppu_stat_prev,
                    )
                    if prev_ly == 143 and ppu_ly_i == 144:
                        if_addr = base + 0xFF0F
                        mem[if_addr] = wp.uint8(mem[if_addr] | wp.uint8(0x01))
                    if ppu_ly_i < SCREEN_H:
                        ppu_capture_latches_env0(
                            i,
                            ppu_ly_i,
                            window_line_i,
                            base,
                            mem,
                            bg_lcdc_latch_env0,
                            bg_scx_latch_env0,
                            bg_scy_latch_env0,
                            bg_bgp_latch_env0,
                            win_wx_latch_env0,
                            win_wy_latch_env0,
                            win_line_latch_env0,
                            obj_obp0_latch_env0,
                            obj_obp1_latch_env0,
                        )
                    lines = lines - 1

            pending = (
                wp.int32(mem[base + 0xFFFF])
                & wp.int32(mem[base + 0xFF0F])
                & 0x1F
            )
            service_cycles = wp.int32(0)
            if pending != 0:
                if halted[i] != 0:
                    halted[i] = 0
                if ime[i] != 0:
                    ime[i] = 0
                    vector = wp.int32(0x40)
                    bit = wp.int32(0x01)
                    if (pending & 0x01) != 0:
                        vector = wp.int32(0x40)
                        bit = wp.int32(0x01)
                    elif (pending & 0x02) != 0:
                        vector = wp.int32(0x48)
                        bit = wp.int32(0x02)
                    elif (pending & 0x04) != 0:
                        vector = wp.int32(0x50)
                        bit = wp.int32(0x04)
                    elif (pending & 0x08) != 0:
                        vector = wp.int32(0x58)
                        bit = wp.int32(0x08)
                    else:
                        vector = wp.int32(0x60)
                        bit = wp.int32(0x10)
                    if_addr = base + 0xFF0F
                    mem[if_addr] = wp.uint8(
                        mem[if_addr] & wp.uint8(0xFF ^ bit)
                    )
                    ret = pc_i & 0xFFFF
                    sp_i = (sp_i - 1) & 0xFFFF
                    write8(
                        i,
                        base,
                        sp_i,
                        (ret >> 8) & 0xFF,
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
                    sp_i = (sp_i - 1) & 0xFFFF
                    write8(
                        i,
                        base,
                        sp_i,
                        ret & 0xFF,
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
                    pc_i = vector
                    service_cycles = 20
                    timer_tick(
                        i,
                        base,
                        service_cycles,
                        mem,
                        div_counter,
                        timer_prev_in,
                        tima_reload_pending,
                        tima_reload_delay,
                    )

            total_cycles = cycles + service_cycles
            cycles_i += wp.int64(total_cycles)
            cycle_frame += total_cycles

            while cycle_frame >= CYCLES_PER_FRAME:
                cycle_frame -= CYCLES_PER_FRAME
                frames_done += 1
                if frames_done >= frames_to_run:
                    break

        POST_STEP_DISPATCH

        pc[i] = pc_i & 0xFFFF
        sp[i] = sp_i & 0xFFFF
        a[i] = a_i & 0xFF
        b[i] = b_i & 0xFF
        c[i] = c_i & 0xFF
        d[i] = d_i & 0xFF
        e[i] = e_i & 0xFF
        h[i] = h_i & 0xFF
        l_reg[i] = l_i & 0xFF
        f[i] = f_i & 0xF0
        instr_count[i] = instr_i
        cycle_count[i] = cycles_i
        cycle_in_frame[i] = cycle_frame
        ppu_scanline_cycle[i] = scanline_cycle_i
        ppu_ly[i] = ppu_ly_i
        ppu_window_line[i] = window_line_i
        trap_flag[i] = trap_i
        trap_pc[i] = trap_pc_i
        trap_opcode[i] = trap_opcode_i
        trap_kind[i] = trap_kind_i
    """
)


def build_cpu_step_source(
    opcode_templates: Sequence[OpcodeTemplate],
    default_template: OpcodeTemplate,
    constants: dict[str, int],
    *,
    cb_opcode_templates: Sequence[OpcodeTemplate] | None = None,
    cb_default_template: OpcodeTemplate | None = None,
    post_step_body: Sequence[cst.BaseStatement] | None = None,
) -> str:
    default_body = _get_template_body(
        default_template.template, default_template.replacements
    )
    dispatch_tree = _build_bucket_dispatch(
        opcode_templates, default_body, "opcode", "opcode_hi"
    )
    cb_templates = cb_opcode_templates or ()
    cb_default = cb_default_template or default_template
    cb_default_body = _get_template_body(cb_default.template, cb_default.replacements)
    cb_dispatch_tree = _build_bucket_dispatch(
        cb_templates, cb_default_body, "cb_opcode", "cb_opcode_hi"
    )
    post_body = list(post_step_body) if post_step_body is not None else []

    skeleton_tree = cst.parse_module(_CPU_STEP_SKELETON.format(**constants))

    class Injector(cst.CSTTransformer):
        def __init__(self) -> None:
            super().__init__()
            self._in_cpu_step = False

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
            if node.name.value == "cpu_step":
                self._in_cpu_step = True
            return True

        def leave_FunctionDef(  # type: ignore[override]
            self, original: cst.FunctionDef, updated: cst.FunctionDef
        ) -> cst.FunctionDef:
            if original.name.value == "cpu_step":
                self._in_cpu_step = False
            return updated

        def leave_SimpleStatementLine(  # type: ignore[override]
            self,
            original: cst.SimpleStatementLine,
            updated: cst.SimpleStatementLine,
        ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
            if not self._in_cpu_step:
                return updated
            if (
                len(original.body) == 1
                and isinstance(original.body[0], cst.Expr)
                and isinstance(original.body[0].value, cst.Name)
                and original.body[0].value.value == "INSTRUCTION_DISPATCH"
            ):
                return cst.FlattenSentinel([dispatch_tree])
            if (
                len(original.body) == 1
                and isinstance(original.body[0], cst.Expr)
                and isinstance(original.body[0].value, cst.Name)
                and original.body[0].value.value == "CB_DISPATCH"
            ):
                return cst.FlattenSentinel([cb_dispatch_tree])
            if (
                len(original.body) == 1
                and isinstance(original.body[0], cst.Expr)
                and isinstance(original.body[0].value, cst.Name)
                and original.body[0].value.value == "POST_STEP_DISPATCH"
            ):
                if not post_body:
                    return cst.FlattenSentinel([])
                return cst.FlattenSentinel(post_body)
            return updated

    final_tree = skeleton_tree.visit(Injector())
    return final_tree.code


def build_cpu_step_kernel(
    opcode_templates: Sequence[OpcodeTemplate],
    default_template: OpcodeTemplate,
    constants: dict[str, int],
    *,
    cb_opcode_templates: Sequence[OpcodeTemplate] | None = None,
    cb_default_template: OpcodeTemplate | None = None,
    post_step_body: Sequence[cst.BaseStatement] | None = None,
) -> Callable[..., Any]:
    source = build_cpu_step_source(
        opcode_templates,
        default_template,
        constants,
        cb_opcode_templates=cb_opcode_templates,
        cb_default_template=cb_default_template,
        post_step_body=post_step_body,
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    module_name = f"gbxcule_warp_kernels.cpu_step_{digest}"
    cache_dir = Path(
        os.environ.get(
            "GBXCULE_WARP_CACHE_DIR",
            Path.home() / ".cache" / "gbxcule" / "warp_kernels",
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    module_path = cache_dir / f"cpu_step_{digest}.py"
    if not module_path.exists():
        module_path.write_text(source, encoding="utf-8")
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load kernel module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.cpu_step
