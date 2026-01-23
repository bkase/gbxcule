"""Warp kernels for CPU stepping."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gbxcule.backends.common import Stage
from gbxcule.core.abi import OBS_DIM_DEFAULT
from gbxcule.kernels.cpu_step_builder import (
    OpcodeTemplate,
    build_cpu_step_kernel,
    get_template_body,
)

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

    opcode_templates = [
        OpcodeTemplate(OPCODE_NOP, misc.template_nop, {}),
        OpcodeTemplate(OPCODE_JP_A16, jumps.template_jp_a16, {}),
        OpcodeTemplate(OPCODE_JR_R8, jumps.template_jr_r8, {}),
        OpcodeTemplate(OPCODE_LD_A_D8, loads.template_ld_r8_d8, {"REG_i": "a_i"}),
        OpcodeTemplate(OPCODE_LD_B_D8, loads.template_ld_r8_d8, {"REG_i": "b_i"}),
        OpcodeTemplate(
            OPCODE_LD_HL_D16,
            loads.template_ld_r16_d16,
            {"HREG_i": "h_i", "LREG_i": "l_i"},
        ),
        OpcodeTemplate(OPCODE_INC_A, alu.template_inc_r8, {"REG_i": "a_i"}),
        OpcodeTemplate(OPCODE_INC_B, alu.template_inc_r8, {"REG_i": "b_i"}),
        OpcodeTemplate(
            OPCODE_INC_HL,
            alu.template_inc_r16,
            {"HREG_i": "h_i", "LREG_i": "l_i"},
        ),
        OpcodeTemplate(
            OPCODE_ADD_A_B,
            alu.template_add_a_r8,
            {"REG_i": "b_i"},
        ),
        OpcodeTemplate(
            OPCODE_LD_HL_A,
            loads.template_ld_hl_r8,
            {"HREG_i": "h_i", "LREG_i": "l_i", "SRC_i": "a_i"},
        ),
        OpcodeTemplate(
            OPCODE_LD_B_HL,
            loads.template_ld_r8_hl,
            {"HREG_i": "h_i", "LREG_i": "l_i", "DST_i": "b_i"},
        ),
    ]
    default_template = OpcodeTemplate(0x00, misc.template_default, {})
    constants = {
        "MEM_SIZE": MEM_SIZE,
        "CYCLES_PER_FRAME": CYCLES_PER_FRAME,
        "ROM_LIMIT": ROM_LIMIT,
        "CART_RAM_START": CART_RAM_START,
        "CART_RAM_END": CART_RAM_END,
        "OBS_DIM": obs_dim,
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
            zeros_i32,
            zeros_u8,
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
