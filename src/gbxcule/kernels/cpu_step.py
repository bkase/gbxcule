"""Warp kernels for CPU stepping."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gbxcule.backends.common import Stage
from gbxcule.core.abi import OBS_DIM_DEFAULT, SERIAL_MAX
from gbxcule.core.isa_sm83 import iter_cb, iter_unprefixed
from gbxcule.kernels.cpu_step_builder import (
    OpcodeTemplate,
    build_cpu_step_kernel,
    get_template_body,
)

MEM_SIZE = 65_536
CYCLES_PER_FRAME = 70_224

ROM_LIMIT = 0x8000
CART_RAM_START = 0xA000
CART_RAM_END = 0xC000

_wp: Any | None = None
_warp_initialized = False
_cpu_step_kernels: dict[tuple[str, int], Callable[..., Any]] = {}
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


def get_cpu_step_kernel(  # type: ignore[no-untyped-def]
    stage: Stage = "emulate_only",
    obs_dim: int = OBS_DIM_DEFAULT,
):
    """Return the CPU step kernel for a given stage (cached)."""
    global _cpu_step_kernels
    key = (stage, obs_dim)
    if key in _cpu_step_kernels:
        return _cpu_step_kernels[key]

    get_warp()

    from gbxcule.kernels.cpu_templates import alu, jumps, loads, misc, post_step

    template_map = {
        "nop": misc.template_nop,
        "jp_a16": jumps.template_jp_a16,
        "jr_r8": jumps.template_jr_r8,
        "jr_nz_r8": jumps.template_jr_nz_r8,
        "jr_z_r8": jumps.template_jr_z_r8,
        "ld_r8_d8": loads.template_ld_r8_d8,
        "ld_r16_d16": loads.template_ld_r16_d16,
        "ld_r8_r8": loads.template_ld_r8_r8,
        "ld_hl_d8": loads.template_ld_hl_d8,
        "ld_sp_d16": loads.template_ld_sp_d16,
        "ld_sp_hl": loads.template_ld_sp_hl,
        "ld_a16_a": loads.template_ld_a16_a,
        "ld_a_a16": loads.template_ld_a_a16,
        "ld_a16_sp": loads.template_ld_a16_sp,
        "ldh_a8_a": loads.template_ldh_a8_a,
        "ldh_a_a8": loads.template_ldh_a_a8,
        "ldh_c_a": loads.template_ldh_c_a,
        "ldh_a_c": loads.template_ldh_a_c,
        "ld_a_bc": loads.template_ld_a_bc,
        "ld_a_de": loads.template_ld_a_de,
        "ld_bc_a": loads.template_ld_bc_a,
        "ld_de_a": loads.template_ld_de_a,
        "ld_hl_inc_a": loads.template_ld_hl_inc_a,
        "ld_hl_dec_a": loads.template_ld_hl_dec_a,
        "ld_a_hl_inc": loads.template_ld_a_hl_inc,
        "ld_a_hl_dec": loads.template_ld_a_hl_dec,
        "inc_r8": alu.template_inc_r8,
        "dec_r8": alu.template_dec_r8,
        "inc_hl": alu.template_inc_hl,
        "dec_hl": alu.template_dec_hl,
        "inc_r16": alu.template_inc_r16,
        "add_a_r8": alu.template_add_a_r8,
        "add_a_hl": alu.template_add_a_hl,
        "add_a_d8": alu.template_add_a_d8,
        "adc_a_r8": alu.template_adc_a_r8,
        "adc_a_hl": alu.template_adc_a_hl,
        "adc_a_d8": alu.template_adc_a_d8,
        "sub_a_r8": alu.template_sub_a_r8,
        "sub_a_hl": alu.template_sub_a_hl,
        "sub_a_d8": alu.template_sub_a_d8,
        "sbc_a_r8": alu.template_sbc_a_r8,
        "sbc_a_hl": alu.template_sbc_a_hl,
        "sbc_a_d8": alu.template_sbc_a_d8,
        "and_a_r8": alu.template_and_a_r8,
        "and_a_hl": alu.template_and_a_hl,
        "and_a_d8": alu.template_and_a_d8,
        "or_a_r8": alu.template_or_a_r8,
        "or_a_hl": alu.template_or_a_hl,
        "or_a_d8": alu.template_or_a_d8,
        "xor_a_r8": alu.template_xor_a_r8,
        "xor_a_hl": alu.template_xor_a_hl,
        "xor_a_d8": alu.template_xor_a_d8,
        "cp_a_r8": alu.template_cp_a_r8,
        "cp_a_hl": alu.template_cp_a_hl,
        "cp_a_d8": alu.template_cp_a_d8,
        "daa": alu.template_daa,
        "cpl": alu.template_cpl,
        "scf": alu.template_scf,
        "ccf": alu.template_ccf,
        "ld_hl_r8": loads.template_ld_hl_r8,
        "ld_r8_hl": loads.template_ld_r8_hl,
    }

    opcode_templates: list[OpcodeTemplate] = []
    for spec in iter_unprefixed():
        if spec.template_key is None:
            continue
        template = template_map.get(spec.template_key)
        if template is None:
            raise KeyError(f"Missing template for key: {spec.template_key}")
        opcode_templates.append(
            OpcodeTemplate(spec.opcode, template, spec.replacements)
        )
    cb_opcode_templates: list[OpcodeTemplate] = []
    for spec in iter_cb():
        if spec.template_key is None:
            continue
        template = template_map.get(spec.template_key)
        if template is None:
            raise KeyError(f"Missing template for key: {spec.template_key}")
        cb_opcode_templates.append(
            OpcodeTemplate(spec.opcode, template, spec.replacements)
        )

    default_template = OpcodeTemplate(0x00, misc.template_trap_unprefixed, {})
    cb_default_template = OpcodeTemplate(0x00, misc.template_trap_cb, {})
    constants = {
        "MEM_SIZE": MEM_SIZE,
        "CYCLES_PER_FRAME": CYCLES_PER_FRAME,
        "ROM_LIMIT": ROM_LIMIT,
        "CART_RAM_START": CART_RAM_START,
        "CART_RAM_END": CART_RAM_END,
        "OBS_DIM": obs_dim,
        "SERIAL_MAX": SERIAL_MAX,
    }

    post_step_templates: list[Any] = []
    if stage == "reward_only":
        post_step_templates = [post_step.template_reward_v0]
    elif stage == "obs_only":
        post_step_templates = [post_step.template_obs_v0]
    elif stage == "full_step":
        post_step_templates = [
            post_step.template_obs_v0,
            post_step.template_reward_v0,
        ]

    post_step_body: list[Any] = []
    for template in post_step_templates:
        post_step_body.extend(get_template_body(template, {}))

    kernel = build_cpu_step_kernel(
        opcode_templates,
        default_template,
        constants,
        cb_opcode_templates=cb_opcode_templates,
        cb_default_template=cb_default_template,
        post_step_body=post_step_body,
    )
    _cpu_step_kernels[key] = kernel
    return kernel


def _warmup_warp_device(
    device_name: str,
    *,
    stage: Stage = "emulate_only",
    obs_dim: int = OBS_DIM_DEFAULT,
) -> None:
    """Warm up Warp on a specific device by compiling the CPU kernel once."""
    warmed_key = f"{device_name}:{stage}:{obs_dim}"
    if warmed_key in _warp_warmed_devices:
        return
    wp = get_warp()
    device = wp.get_device(device_name)
    mem = wp.zeros(1 * MEM_SIZE, dtype=wp.uint8, device=device)
    zeros_i32 = wp.zeros(1, dtype=wp.int32, device=device)
    zeros_i64 = wp.zeros(1, dtype=wp.int64, device=device)
    zeros_u8 = wp.zeros(1, dtype=wp.uint8, device=device)
    zeros_serial = wp.zeros(1 * SERIAL_MAX, dtype=wp.uint8, device=device)
    zeros_serial_len = wp.zeros(1, dtype=wp.int32, device=device)
    zeros_trap_flag = wp.zeros(1, dtype=wp.int32, device=device)
    zeros_trap_pc = wp.zeros(1, dtype=wp.int32, device=device)
    zeros_trap_opcode = wp.zeros(1, dtype=wp.int32, device=device)
    zeros_trap_kind = wp.zeros(1, dtype=wp.int32, device=device)
    zeros_f32 = wp.zeros(1, dtype=wp.float32, device=device)
    zeros_obs = wp.zeros(obs_dim, dtype=wp.float32, device=device)
    kernel = get_cpu_step_kernel(stage=stage, obs_dim=obs_dim)
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
            zeros_trap_flag,
            zeros_trap_pc,
            zeros_trap_opcode,
            zeros_trap_kind,
            zeros_i32,
            zeros_u8,
            zeros_serial,
            zeros_serial_len,
            zeros_u8,
            0,
            zeros_f32,
            zeros_obs,
            0,
            0,
        ],
        device=device,
    )
    wp.synchronize()
    _warp_warmed_devices.add(warmed_key)


def warmup_warp_cpu(
    *,
    stage: Stage = "emulate_only",
    obs_dim: int = OBS_DIM_DEFAULT,
) -> None:
    """Warm up Warp on CPU by compiling the CPU kernel once."""
    _warmup_warp_device("cpu", stage=stage, obs_dim=obs_dim)


def warmup_warp_cuda(
    device: str = "cuda:0",
    *,
    stage: Stage = "emulate_only",
    obs_dim: int = OBS_DIM_DEFAULT,
) -> None:
    """Warm up Warp on CUDA by compiling the CPU kernel once."""
    _warmup_warp_device(device, stage=stage, obs_dim=obs_dim)
