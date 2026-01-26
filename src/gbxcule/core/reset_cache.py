"""Cached reset states (CuLE-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W, MEM_SIZE, SERIAL_MAX
from gbxcule.core.cartridge import CART_STATE_STRIDE
from gbxcule.kernels.reset_mask import (
    ensure_reset_kernels_loaded,
    reset_copy_scalar_i32,
    reset_copy_scalar_i64,
    reset_copy_scalar_u8,
    reset_copy_strided_i32,
    reset_copy_strided_u8,
)


@dataclass
class ResetCache:
    """Snapshot of a single env state + masked restore helpers."""

    backend: Any
    num_envs: int
    mem: Any
    cart_state: Any
    cart_ram: Any | None
    serial_buf: Any
    pix: Any | None
    pc: Any
    sp: Any
    a: Any
    b: Any
    c: Any
    d: Any
    e: Any
    h: Any
    l_reg: Any
    f: Any
    instr_count: Any
    cycle_count: Any
    cycle_in_frame: Any
    trap_flag: Any
    trap_pc: Any
    trap_opcode: Any
    trap_kind: Any
    actions: Any
    joyp_select: Any
    serial_len: Any
    serial_overflow: Any
    ime: Any
    ime_delay: Any
    halted: Any
    div_counter: Any
    timer_prev_in: Any
    tima_reload_pending: Any
    tima_reload_delay: Any
    ppu_scanline_cycle: Any
    ppu_ly: Any
    ppu_window_line: Any
    ppu_stat_prev: Any
    cart_ram_stride: int

    @classmethod
    def from_backend(cls, backend: Any, *, env_idx: int = 0) -> ResetCache:
        """Capture a snapshot from env_idx into device-resident buffers."""
        if not backend._initialized:
            raise RuntimeError("Backend not initialized")
        wp = backend._wp
        device = backend._device
        num_envs = int(backend.num_envs)
        ensure_reset_kernels_loaded()

        def _snapshot_strided_u8(src, stride: int):
            if src is None:
                return None
            snap = wp.empty(stride, dtype=wp.uint8, device=device)
            wp.copy(
                snap,
                src,
                dest_offset=0,
                src_offset=int(env_idx) * stride,
                count=stride,
            )
            return snap

        def _snapshot_scalar(src, dtype):
            if src is None:
                return None
            snap = wp.empty(1, dtype=dtype, device=device)
            wp.copy(snap, src, dest_offset=0, src_offset=int(env_idx), count=1)
            return snap

        mem = _snapshot_strided_u8(backend._mem, MEM_SIZE)
        cart_state = wp.empty(CART_STATE_STRIDE, dtype=wp.int32, device=device)
        wp.copy(
            cart_state,
            backend._cart_state,
            dest_offset=0,
            src_offset=int(env_idx) * CART_STATE_STRIDE,
            count=CART_STATE_STRIDE,
        )
        cart_ram_stride = int(getattr(backend, "_ram_byte_length", 0))
        if cart_ram_stride <= 0:
            cart_ram_stride = 1
        cart_ram = _snapshot_strided_u8(backend._cart_ram, cart_ram_stride)
        serial_buf = _snapshot_strided_u8(backend._serial_buf, SERIAL_MAX)
        pix_stride = DOWNSAMPLE_H * DOWNSAMPLE_W
        pix = _snapshot_strided_u8(backend._pix, pix_stride) if backend._pix else None

        pc = _snapshot_scalar(backend._pc, wp.int32)
        sp = _snapshot_scalar(backend._sp, wp.int32)
        a = _snapshot_scalar(backend._a, wp.int32)
        b = _snapshot_scalar(backend._b, wp.int32)
        c = _snapshot_scalar(backend._c, wp.int32)
        d = _snapshot_scalar(backend._d, wp.int32)
        e = _snapshot_scalar(backend._e, wp.int32)
        h = _snapshot_scalar(backend._h, wp.int32)
        l_reg = _snapshot_scalar(backend._l, wp.int32)
        f = _snapshot_scalar(backend._f, wp.int32)
        instr_count = _snapshot_scalar(backend._instr_count, wp.int64)
        cycle_count = _snapshot_scalar(backend._cycle_count, wp.int64)
        cycle_in_frame = _snapshot_scalar(backend._cycle_in_frame, wp.int32)
        trap_flag = _snapshot_scalar(backend._trap_flag, wp.int32)
        trap_pc = _snapshot_scalar(backend._trap_pc, wp.int32)
        trap_opcode = _snapshot_scalar(backend._trap_opcode, wp.int32)
        trap_kind = _snapshot_scalar(backend._trap_kind, wp.int32)
        actions = _snapshot_scalar(backend._actions, wp.int32)
        joyp_select = _snapshot_scalar(backend._joyp_select, wp.uint8)
        serial_len = _snapshot_scalar(backend._serial_len, wp.int32)
        serial_overflow = _snapshot_scalar(backend._serial_overflow, wp.uint8)
        ime = _snapshot_scalar(backend._ime, wp.int32)
        ime_delay = _snapshot_scalar(backend._ime_delay, wp.int32)
        halted = _snapshot_scalar(backend._halted, wp.int32)
        div_counter = _snapshot_scalar(backend._div_counter, wp.int32)
        timer_prev_in = _snapshot_scalar(backend._timer_prev_in, wp.int32)
        tima_reload_pending = _snapshot_scalar(backend._tima_reload_pending, wp.int32)
        tima_reload_delay = _snapshot_scalar(backend._tima_reload_delay, wp.int32)
        ppu_scanline_cycle = _snapshot_scalar(backend._ppu_scanline_cycle, wp.int32)
        ppu_ly = _snapshot_scalar(backend._ppu_ly, wp.int32)
        ppu_window_line = _snapshot_scalar(backend._ppu_window_line, wp.int32)
        ppu_stat_prev = _snapshot_scalar(backend._ppu_stat_prev, wp.uint8)

        wp.synchronize()

        return cls(
            backend=backend,
            num_envs=num_envs,
            mem=mem,
            cart_state=cart_state,
            cart_ram=cart_ram,
            serial_buf=serial_buf,
            pix=pix,
            pc=pc,
            sp=sp,
            a=a,
            b=b,
            c=c,
            d=d,
            e=e,
            h=h,
            l_reg=l_reg,
            f=f,
            instr_count=instr_count,
            cycle_count=cycle_count,
            cycle_in_frame=cycle_in_frame,
            trap_flag=trap_flag,
            trap_pc=trap_pc,
            trap_opcode=trap_opcode,
            trap_kind=trap_kind,
            actions=actions,
            joyp_select=joyp_select,
            serial_len=serial_len,
            serial_overflow=serial_overflow,
            ime=ime,
            ime_delay=ime_delay,
            halted=halted,
            div_counter=div_counter,
            timer_prev_in=timer_prev_in,
            tima_reload_pending=tima_reload_pending,
            tima_reload_delay=tima_reload_delay,
            ppu_scanline_cycle=ppu_scanline_cycle,
            ppu_ly=ppu_ly,
            ppu_window_line=ppu_window_line,
            ppu_stat_prev=ppu_stat_prev,
            cart_ram_stride=cart_ram_stride,
        )

    def _launch_masked_copies(self, mask_wp: Any) -> None:
        """Launch Warp kernels to reset all masked envs (device-resident)."""
        backend = self.backend
        wp = backend._wp
        device = backend._device
        num_envs = self.num_envs

        if self.mem is not None and backend._mem is not None:
            wp.launch(
                reset_copy_strided_u8,
                dim=num_envs * MEM_SIZE,
                inputs=[mask_wp, self.mem, backend._mem, int(MEM_SIZE)],
                device=device,
            )
        if self.cart_state is not None and backend._cart_state is not None:
            wp.launch(
                reset_copy_strided_i32,
                dim=num_envs * CART_STATE_STRIDE,
                inputs=[
                    mask_wp,
                    self.cart_state,
                    backend._cart_state,
                    int(CART_STATE_STRIDE),
                ],
                device=device,
            )
        if self.cart_ram is not None and backend._cart_ram is not None:
            wp.launch(
                reset_copy_strided_u8,
                dim=num_envs * self.cart_ram_stride,
                inputs=[
                    mask_wp,
                    self.cart_ram,
                    backend._cart_ram,
                    int(self.cart_ram_stride),
                ],
                device=device,
            )
        if self.serial_buf is not None and backend._serial_buf is not None:
            wp.launch(
                reset_copy_strided_u8,
                dim=num_envs * SERIAL_MAX,
                inputs=[mask_wp, self.serial_buf, backend._serial_buf, int(SERIAL_MAX)],
                device=device,
            )
        if self.pix is not None and backend._pix is not None:
            stride = DOWNSAMPLE_H * DOWNSAMPLE_W
            wp.launch(
                reset_copy_strided_u8,
                dim=num_envs * stride,
                inputs=[mask_wp, self.pix, backend._pix, int(stride)],
                device=device,
            )

        def _copy_i32(src, dest):
            if src is None or dest is None:
                return
            wp.launch(
                reset_copy_scalar_i32,
                dim=num_envs,
                inputs=[mask_wp, src, dest],
                device=device,
            )

        def _copy_i64(src, dest):
            if src is None or dest is None:
                return
            wp.launch(
                reset_copy_scalar_i64,
                dim=num_envs,
                inputs=[mask_wp, src, dest],
                device=device,
            )

        def _copy_u8(src, dest):
            if src is None or dest is None:
                return
            wp.launch(
                reset_copy_scalar_u8,
                dim=num_envs,
                inputs=[mask_wp, src, dest],
                device=device,
            )

        _copy_i32(self.pc, backend._pc)
        _copy_i32(self.sp, backend._sp)
        _copy_i32(self.a, backend._a)
        _copy_i32(self.b, backend._b)
        _copy_i32(self.c, backend._c)
        _copy_i32(self.d, backend._d)
        _copy_i32(self.e, backend._e)
        _copy_i32(self.h, backend._h)
        _copy_i32(self.l_reg, backend._l)
        _copy_i32(self.f, backend._f)
        _copy_i64(self.instr_count, backend._instr_count)
        _copy_i64(self.cycle_count, backend._cycle_count)
        _copy_i32(self.cycle_in_frame, backend._cycle_in_frame)
        _copy_i32(self.trap_flag, backend._trap_flag)
        _copy_i32(self.trap_pc, backend._trap_pc)
        _copy_i32(self.trap_opcode, backend._trap_opcode)
        _copy_i32(self.trap_kind, backend._trap_kind)
        _copy_i32(self.actions, backend._actions)
        _copy_u8(self.joyp_select, backend._joyp_select)
        _copy_i32(self.serial_len, backend._serial_len)
        _copy_u8(self.serial_overflow, backend._serial_overflow)
        _copy_i32(self.ime, backend._ime)
        _copy_i32(self.ime_delay, backend._ime_delay)
        _copy_i32(self.halted, backend._halted)
        _copy_i32(self.div_counter, backend._div_counter)
        _copy_i32(self.timer_prev_in, backend._timer_prev_in)
        _copy_i32(self.tima_reload_pending, backend._tima_reload_pending)
        _copy_i32(self.tima_reload_delay, backend._tima_reload_delay)
        _copy_i32(self.ppu_scanline_cycle, backend._ppu_scanline_cycle)
        _copy_i32(self.ppu_ly, backend._ppu_ly)
        _copy_i32(self.ppu_window_line, backend._ppu_window_line)
        _copy_u8(self.ppu_stat_prev, backend._ppu_stat_prev)

    def apply_mask_np(self, mask: np.ndarray) -> None:
        """Apply reset mask for CPU backends using numpy writes."""
        backend = self.backend
        if backend.device != "cpu":
            raise RuntimeError("apply_mask_np is only supported on CPU backends.")
        if mask.shape != (self.num_envs,):
            raise ValueError("mask must have shape (num_envs,)")
        mask_bool = mask.astype(bool, copy=False)
        if not np.any(mask_bool):
            return

        mem_np = backend._mem.numpy().reshape(self.num_envs, MEM_SIZE)
        mem_np[mask_bool] = self.mem.numpy()

        cart_state_np = backend._cart_state.numpy().reshape(
            self.num_envs, CART_STATE_STRIDE
        )
        cart_state_np[mask_bool] = self.cart_state.numpy()

        if self.cart_ram is not None and backend._cart_ram is not None:
            ram_np = backend._cart_ram.numpy().reshape(
                self.num_envs, self.cart_ram_stride
            )
            ram_np[mask_bool] = self.cart_ram.numpy()

        serial_buf_np = backend._serial_buf.numpy().reshape(self.num_envs, SERIAL_MAX)
        serial_buf_np[mask_bool] = self.serial_buf.numpy()

        if self.pix is not None and backend._pix is not None:
            stride = DOWNSAMPLE_H * DOWNSAMPLE_W
            pix_np = backend._pix.numpy().reshape(self.num_envs, stride)
            pix_np[mask_bool] = self.pix.numpy()

        def _assign_scalar(dest, snap):
            if dest is None or snap is None:
                return
            dest_np = dest.numpy()
            dest_np[mask_bool] = snap.numpy()[0]

        _assign_scalar(backend._pc, self.pc)
        _assign_scalar(backend._sp, self.sp)
        _assign_scalar(backend._a, self.a)
        _assign_scalar(backend._b, self.b)
        _assign_scalar(backend._c, self.c)
        _assign_scalar(backend._d, self.d)
        _assign_scalar(backend._e, self.e)
        _assign_scalar(backend._h, self.h)
        _assign_scalar(backend._l, self.l_reg)
        _assign_scalar(backend._f, self.f)
        _assign_scalar(backend._instr_count, self.instr_count)
        _assign_scalar(backend._cycle_count, self.cycle_count)
        _assign_scalar(backend._cycle_in_frame, self.cycle_in_frame)
        _assign_scalar(backend._trap_flag, self.trap_flag)
        _assign_scalar(backend._trap_pc, self.trap_pc)
        _assign_scalar(backend._trap_opcode, self.trap_opcode)
        _assign_scalar(backend._trap_kind, self.trap_kind)
        _assign_scalar(backend._actions, self.actions)
        _assign_scalar(backend._joyp_select, self.joyp_select)
        _assign_scalar(backend._serial_len, self.serial_len)
        _assign_scalar(backend._serial_overflow, self.serial_overflow)
        _assign_scalar(backend._ime, self.ime)
        _assign_scalar(backend._ime_delay, self.ime_delay)
        _assign_scalar(backend._halted, self.halted)
        _assign_scalar(backend._div_counter, self.div_counter)
        _assign_scalar(backend._timer_prev_in, self.timer_prev_in)
        _assign_scalar(backend._tima_reload_pending, self.tima_reload_pending)
        _assign_scalar(backend._tima_reload_delay, self.tima_reload_delay)
        _assign_scalar(backend._ppu_scanline_cycle, self.ppu_scanline_cycle)
        _assign_scalar(backend._ppu_ly, self.ppu_ly)
        _assign_scalar(backend._ppu_window_line, self.ppu_window_line)
        _assign_scalar(backend._ppu_stat_prev, self.ppu_stat_prev)

    def apply_mask_torch(self, mask) -> None:  # type: ignore[no-untyped-def]
        """Apply reset mask for CUDA backends using torch mask + Warp kernels."""
        backend = self.backend
        if backend.device != "cuda":
            raise RuntimeError("apply_mask_torch is only supported on CUDA backends.")

        try:
            import importlib

            torch = importlib.import_module("torch")
        except Exception as exc:
            raise RuntimeError("Torch not available for apply_mask_torch().") from exc

        tensor_type = getattr(torch, "Tensor", None)
        if tensor_type is None or not isinstance(mask, tensor_type):
            raise TypeError("mask must be a torch.Tensor")
        if mask.device.type != "cuda":
            raise ValueError("mask must be a CUDA tensor")
        if mask.ndim != 1 or int(mask.shape[0]) != self.num_envs:
            raise ValueError(f"mask must have shape ({self.num_envs},)")
        if mask.dtype is torch.bool:
            mask_u8 = mask.to(torch.uint8)
        elif mask.dtype is torch.uint8:
            mask_u8 = mask
        else:
            raise ValueError("mask must be bool or uint8")

        mask_wp = backend._wp.from_torch(mask_u8)
        stream = backend._wp.stream_from_torch(torch.cuda.current_stream())
        with backend._wp.ScopedStream(stream):
            self._launch_masked_copies(mask_wp)
