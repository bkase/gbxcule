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


def _build_dispatch_tree(
    opcode_templates: Sequence[OpcodeTemplate],
    default_body: list[cst.BaseStatement],
) -> cst.If:
    current_node: cst.If = cst.If(
        test=cst.Name("True"),
        body=cst.IndentedBlock(body=default_body),
    )

    for spec in reversed(opcode_templates):
        body = _get_template_body(spec.template, spec.replacements)
        condition = cst.Comparison(
            left=cst.Name("opcode"),
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


_CPU_STEP_SKELETON = textwrap.dedent(
    """
    import warp as wp

    MEM_SIZE = {MEM_SIZE}
    CYCLES_PER_FRAME = {CYCLES_PER_FRAME}
    ROM_LIMIT = {ROM_LIMIT}
    CART_RAM_START = {CART_RAM_START}
    CART_RAM_END = {CART_RAM_END}

    @wp.func
    def sign8(x: wp.int32) -> wp.int32:
        return wp.where(x >= 128, x - 256, x)

    @wp.func
    def make_flags(z: wp.int32, n: wp.int32, h: wp.int32, c: wp.int32) -> wp.int32:
        return (z << 7) | (n << 6) | (h << 5) | (c << 4)

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

            INSTRUCTION_DISPATCH

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
    """
)


def build_cpu_step_source(
    opcode_templates: Sequence[OpcodeTemplate],
    default_template: OpcodeTemplate,
    constants: dict[str, int],
    post_step_body: Sequence[cst.BaseStatement] | None = None,
) -> str:
    default_body = _get_template_body(
        default_template.template, default_template.replacements
    )
    dispatch_tree = _build_dispatch_tree(opcode_templates, default_body)
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
    post_step_body: Sequence[cst.BaseStatement] | None = None,
) -> Callable[..., Any]:
    source = build_cpu_step_source(
        opcode_templates,
        default_template,
        constants,
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
