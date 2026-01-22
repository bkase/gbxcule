"""Warp kernels for CPU stepping."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

MEM_SIZE = 65_536
CYCLES_PER_FRAME = 70_224

OPCODE_NOP = 0x00
OPCODE_JP_A16 = 0xC3
OPCODE_JR_R8 = 0x18
OPCODE_LD_A_D8 = 0x3E
OPCODE_LD_B_D8 = 0x06
OPCODE_LD_HL_D16 = 0x21
OPCODE_INC_A = 0x3C
OPCODE_INC_B = 0x04
OPCODE_INC_HL = 0x23
OPCODE_ADD_A_B = 0x80
OPCODE_LD_HL_A = 0x77
OPCODE_LD_B_HL = 0x46
ROM_LIMIT = 0x8000
CART_RAM_START = 0xA000
CART_RAM_END = 0xC000

_wp: Any | None = None
_warp_initialized = False
_cpu_step_kernel: Callable[..., Any] | None = None
_warp_warmed_devices: set[str] = set()


def get_warp() -> Any:  # type: ignore[no-untyped-def]
    """Import Warp and initialize once."""
    global _wp, _warp_initialized
    if _wp is None:
        import warp as wp

        _wp = wp
        globals()["wp"] = wp
    if not _warp_initialized:
        _wp.init()
        _warp_initialized = True
    return _wp


def get_cpu_step_kernel():  # type: ignore[no-untyped-def]
    """Return the CPU step kernel (cached)."""
    global _cpu_step_kernel
    if _cpu_step_kernel is None:
        wp = get_warp()

        @wp.func
        def sign8(x: wp.int32) -> wp.int32:  # type: ignore[name-defined]
            return wp.where(x >= 128, x - 256, x)

        @wp.func
        def make_flags(z: wp.int32, n: wp.int32, h: wp.int32, c: wp.int32) -> wp.int32:  # type: ignore[name-defined]
            return (z << 7) | (n << 6) | (h << 5) | (c << 4)

        @wp.kernel
        def cpu_step(
            mem: wp.array(dtype=wp.uint8),  # type: ignore[name-defined]
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
            frames_to_run: wp.int32,
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

            frames_done = wp.int32(0)

            while frames_done < frames_to_run:
                opcode = wp.int32(mem[base + pc_i])
                cycles = wp.int32(0)

                if opcode == OPCODE_NOP:
                    pc_i = (pc_i + 1) & 0xFFFF
                    cycles = 4

                elif opcode == OPCODE_JP_A16:
                    lo = wp.int32(mem[base + ((pc_i + 1) & 0xFFFF)])
                    hi = wp.int32(mem[base + ((pc_i + 2) & 0xFFFF)])
                    pc_i = ((hi << 8) | lo) & 0xFFFF
                    cycles = 16

                elif opcode == OPCODE_JR_R8:
                    off = wp.int32(mem[base + ((pc_i + 1) & 0xFFFF)])
                    off = sign8(off)
                    pc_i = (pc_i + 2 + off) & 0xFFFF
                    cycles = 12

                elif opcode == OPCODE_LD_A_D8:
                    a_i = wp.int32(mem[base + ((pc_i + 1) & 0xFFFF)])
                    pc_i = (pc_i + 2) & 0xFFFF
                    cycles = 8

                elif opcode == OPCODE_LD_B_D8:
                    b_i = wp.int32(mem[base + ((pc_i + 1) & 0xFFFF)])
                    pc_i = (pc_i + 2) & 0xFFFF
                    cycles = 8

                elif opcode == OPCODE_LD_HL_D16:
                    lo = wp.int32(mem[base + ((pc_i + 1) & 0xFFFF)])
                    hi = wp.int32(mem[base + ((pc_i + 2) & 0xFFFF)])
                    hl = ((hi << 8) | lo) & 0xFFFF
                    h_i = (hl >> 8) & 0xFF
                    l_i = hl & 0xFF
                    pc_i = (pc_i + 3) & 0xFFFF
                    cycles = 12

                elif opcode == OPCODE_INC_A:
                    old = a_i
                    a_i = (a_i + 1) & 0xFF
                    z = wp.where(a_i == 0, 1, 0)
                    hflag = wp.where((old & 0x0F) == 0x0F, 1, 0)
                    cflag = (f_i >> 4) & 0x1
                    f_i = make_flags(z, 0, hflag, cflag)
                    pc_i = (pc_i + 1) & 0xFFFF
                    cycles = 4

                elif opcode == OPCODE_INC_B:
                    old = b_i
                    b_i = (b_i + 1) & 0xFF
                    z = wp.where(b_i == 0, 1, 0)
                    hflag = wp.where((old & 0x0F) == 0x0F, 1, 0)
                    cflag = (f_i >> 4) & 0x1
                    f_i = make_flags(z, 0, hflag, cflag)
                    pc_i = (pc_i + 1) & 0xFFFF
                    cycles = 4

                elif opcode == OPCODE_INC_HL:
                    hl = ((h_i << 8) | l_i) & 0xFFFF
                    hl = (hl + 1) & 0xFFFF
                    h_i = (hl >> 8) & 0xFF
                    l_i = hl & 0xFF
                    pc_i = (pc_i + 1) & 0xFFFF
                    cycles = 8

                elif opcode == OPCODE_ADD_A_B:
                    sum_ab = a_i + b_i
                    res = sum_ab & 0xFF
                    z = wp.where(res == 0, 1, 0)
                    hflag = wp.where(((a_i & 0x0F) + (b_i & 0x0F)) > 0x0F, 1, 0)
                    cflag = wp.where(sum_ab > 0xFF, 1, 0)
                    a_i = res
                    f_i = make_flags(z, 0, hflag, cflag)
                    pc_i = (pc_i + 1) & 0xFFFF
                    cycles = 4

                elif opcode == OPCODE_LD_HL_A:
                    hl = ((h_i << 8) | l_i) & 0xFFFF
                    # Cartridge ROM (0x0000-0x7FFF) is read-only on real hardware.
                    # MEM_RWB intentionally walks HL across the full address space,
                    # so we must ignore writes into ROM to avoid self-modifying code
                    # that diverges from PyBoy.
                    if hl >= ROM_LIMIT and not (
                        hl >= CART_RAM_START and hl < CART_RAM_END
                    ):
                        mem[base + hl] = wp.uint8(a_i)
                    pc_i = (pc_i + 1) & 0xFFFF
                    cycles = 8

                elif opcode == OPCODE_LD_B_HL:
                    hl = ((h_i << 8) | l_i) & 0xFFFF
                    # This repo's micro-ROMs are built with "no cart RAM".
                    # Reads in 0xA000-0xBFFF return open-bus (0xFF);
                    # writes are ignored.
                    if hl >= CART_RAM_START and hl < CART_RAM_END:
                        b_i = 0xFF
                    else:
                        b_i = wp.int32(mem[base + hl])
                    pc_i = (pc_i + 1) & 0xFFFF
                    cycles = 8

                else:
                    pc_i = (pc_i + 1) & 0xFFFF
                    cycles = 4

                f_i = f_i & 0xF0
                instr_i += wp.int64(1)
                cycles_i += wp.int64(cycles)
                cycle_frame += cycles

                while cycle_frame >= CYCLES_PER_FRAME:
                    cycle_frame -= CYCLES_PER_FRAME
                    frames_done += 1
                    if frames_done >= frames_to_run:
                        break

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

        _cpu_step_kernel = cpu_step
    return _cpu_step_kernel


def _warmup_warp_device(device_name: str) -> None:
    """Warm up Warp on a specific device by compiling the CPU kernel once."""
    if device_name in _warp_warmed_devices:
        return
    wp = get_warp()
    device = wp.get_device(device_name)
    mem = wp.zeros(1 * MEM_SIZE, dtype=wp.uint8, device=device)
    zeros_i32 = wp.zeros(1, dtype=wp.int32, device=device)
    zeros_i64 = wp.zeros(1, dtype=wp.int64, device=device)
    kernel = get_cpu_step_kernel()
    wp.launch(
        kernel,
        dim=1,
        inputs=[
            mem,
            zeros_i32,
            zeros_i32,
            zeros_i32,
            zeros_i32,
            zeros_i32,
            zeros_i32,
            zeros_i32,
            zeros_i32,
            zeros_i32,
            zeros_i32,
            zeros_i64,
            zeros_i64,
            zeros_i32,
            0,
        ],
        device=device,
    )
    wp.synchronize()
    _warp_warmed_devices.add(device_name)


def warmup_warp_cpu() -> None:
    """Warm up Warp on CPU by compiling the CPU kernel once."""
    _warmup_warp_device("cpu")


def warmup_warp_cuda(device: str = "cuda:0") -> None:
    """Warm up Warp on CUDA by compiling the CPU kernel once."""
    _warmup_warp_device(device)
