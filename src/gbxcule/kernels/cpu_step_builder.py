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

    def leave_Name(self, original: cst.Name, updated: cst.Name) -> cst.Name:
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
    return list(modified_func.body.body)


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
    ROM_LIMIT = {ROM_LIMIT}
    CART_RAM_START = {CART_RAM_START}
    CART_RAM_END = {CART_RAM_END}
    OBS_DIM = {OBS_DIM}
    SERIAL_MAX = {SERIAL_MAX}

    ACTION_CODEC_LEGACY = 0
    ACTION_CODEC_POKERED = 1

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
        if codec_id == ACTION_CODEC_LEGACY:
            if action == 1:
                dpad = DPAD_UP
            elif action == 2:
                dpad = DPAD_DOWN
            elif action == 3:
                dpad = DPAD_LEFT
            elif action == 4:
                dpad = DPAD_RIGHT
        else:
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
        if codec_id == ACTION_CODEC_LEGACY:
            if action == 5:
                btn = BUTTON_A
            elif action == 6:
                btn = BUTTON_B
            elif action == 7:
                btn = BUTTON_START
            elif action == 8:
                btn = BUTTON_SELECT
        else:
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
    def read8(
        i: wp.int32,
        base: wp.int32,
        addr: wp.int32,
        mem: wp.array(dtype=wp.uint8),
        actions: wp.array(dtype=wp.int32),
        joyp_select: wp.array(dtype=wp.uint8),
        frames_done: wp.int32,
        release_after_frames: wp.int32,
        action_codec_id: wp.int32,
    ) -> wp.int32:
        addr16 = addr & 0xFFFF
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
            return 0xFF
        return wp.int32(mem[base + addr16])

    @wp.func
    def write8(
        i: wp.int32,
        base: wp.int32,
        addr: wp.int32,
        val: wp.int32,
        mem: wp.array(dtype=wp.uint8),
        joyp_select: wp.array(dtype=wp.uint8),
        serial_buf: wp.array(dtype=wp.uint8),
        serial_len: wp.array(dtype=wp.int32),
        serial_overflow: wp.array(dtype=wp.uint8),
    ) -> None:
        addr16 = addr & 0xFFFF
        val8 = wp.uint8(val)
        if addr16 == 0xFF00:
            joyp_select[i] = val8 & wp.uint8(0x30)
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
        if addr16 >= ROM_LIMIT and not (
            addr16 >= CART_RAM_START and addr16 < CART_RAM_END
        ):
            mem[base + addr16] = val8

    @wp.kernel
    def cpu_step(
        mem: wp.array(dtype=wp.uint8),
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

        frames_done = wp.int32(0)

        while frames_done < frames_to_run:
            if trap_i != 0:
                break
            opcode = wp.int32(mem[base + pc_i])
            opcode_hi = opcode >> 4
            cycles = wp.int32(0)

            if opcode == 0xCB:
                cb_opcode = wp.int32(mem[base + ((pc_i + 1) & 0xFFFF)])
                cb_opcode_hi = cb_opcode >> 4
                pc_i = (pc_i + 2) & 0xFFFF
                CB_DISPATCH
            else:
                INSTRUCTION_DISPATCH

            if trap_i != 0:
                break

            f_i = f_i & 0xF0
            instr_i += wp.int64(1)
            cycles_i += wp.int64(cycles)
            cycle_frame += cycles

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
    cb_default_body = _get_template_body(
        cb_default.template, cb_default.replacements
    )
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

        def leave_FunctionDef(
            self, original: cst.FunctionDef, updated: cst.FunctionDef
        ) -> cst.FunctionDef:
            if original.name.value == "cpu_step":
                self._in_cpu_step = False
            return updated

        def leave_SimpleStatementLine(
            self,
            original: cst.SimpleStatementLine,
            updated: cst.SimpleStatementLine,
        ) -> cst.BaseStatement:
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
