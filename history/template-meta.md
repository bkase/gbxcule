### The Strategy

1. **The Template:** Write a standalone Python function representing an instruction (e.g., `inc_r8`). Use a specific placeholder variable name (like `REG`).
2. **The Transpiler:** Use LibCST to parse that function, find every instance of `REG`, and swap it for the actual register name (e.g., `a`).
3. **The Injection:** Extract the _body_ of that modified function and insert it into the Warp kernel tree.

### 1. The Setup (Define Templates)

Create a file `cpu_templates.py`. This file is valid Python. You can run `ruff` and `mypy` on it.

```python
# cpu_templates.py
import warp as wp

# Helper for type checking (will be available in the generated kernel)
# We define stubs here so the IDE is happy.
def make_flags(z: int, n: int, h: int, c: int) -> int: ...

def template_inc_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """
    Template for 8-bit Increment.
    TARGET: We will replace 'REG_i' with 'a_i', 'b_i', etc.
    """
    # 1. Capture old value
    old = REG_i

    # 2. Perform Math
    REG_i = (REG_i + 1) & 0xFF

    # 3. Calculate Flags
    z = 1 if REG_i == 0 else 0 # Warp supports standard python ternary
    hflag = 1 if (old & 0x0F) == 0x0F else 0
    cflag = (f_i >> 4) & 0x1

    f_i = make_flags(z, 0, hflag, cflag)

    # 4. Advance PC and set cycles
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4

```

### 2. The Generator (Transform & Inject)

This script loads the template, customizes it for specific registers, and builds the final kernel.

```python
# builder.py
import inspect
import libcst as cst
import warp as wp
import cpu_templates # Import the templates file

class TemplateSpecializer(cst.CSTTransformer):
    """
    Swaps generic placeholder names (REG_i) for specific ones (a_i).
    """
    def __init__(self, replacements: dict[str, str]):
        self.replacements = replacements

    def leave_Name(self, original, updated):
        if original.value in self.replacements:
            return updated.with_changes(value=self.replacements[original.value])
        return updated

def get_template_logic(func_obj, replacements: dict[str, str]) -> list[cst.BaseStatement]:
    """
    1. Reads source of a function.
    2. Parses into CST.
    3. Renames variables based on 'replacements'.
    4. Returns just the body statements (stripping the 'def' wrapper).
    """
    # Get source code of the function
    source = inspect.getsource(func_obj)

    # Clean up indentation (inspect.getsource includes definitions' indent)
    # usually inspect returns dedented code if it's top level,
    # but we force parse it as a standalone module to be safe.
    tree = cst.parse_module(source)

    # The first statement in the module is the FunctionDef
    func_def = tree.body[0]
    if not isinstance(func_def, cst.FunctionDef):
        raise ValueError("Template source must be a function definition")

    # Apply variable renaming
    transformer = TemplateSpecializer(replacements)
    modified_func = func_def.visit(transformer)

    # Return the *body* of the function (the list of statements)
    return modified_func.body.body

def build_kernel_ast():
    # 1. Define our Opcode Map
    # We map Opcode -> (Template Function, Replacements)
    op_map = {
        0x3C: (cpu_templates.template_inc_r8, {"REG_i": "a_i"}), # INC A
        0x04: (cpu_templates.template_inc_r8, {"REG_i": "b_i"}), # INC B
        0x0C: (cpu_templates.template_inc_r8, {"REG_i": "c_i"}), # INC C
    }

    # 2. Build the If/Elif Tree Nodes
    # Start with default case
    current_node = cst.If(
        test=cst.parse_expression("cycles == 0"),
        body=cst.parse_module("pc_i += 1\ncycles = 4").body
    )

    # Build up the tree (bottom-up)
    for opcode, (tmpl_func, subs) in reversed(op_map.items()):

        # Get the specialized logic for this opcode
        logic_body = get_template_logic(tmpl_func, subs)

        # Create the 'if opcode == X' check
        condition = cst.Comparison(
            left=cst.Name("opcode"),
            comparisons=[
                cst.ComparisonTarget(
                    operator=cst.Equal(),
                    comparator=cst.Integer(f"0x{opcode:02X}")
                )
            ]
        )

        # Wrap in If/Else
        current_node = cst.If(
            test=condition,
            body=cst.IndentedBlock(body=logic_body),
            orelse=cst.If(
                test=cst.Name("True"),
                body=cst.IndentedBlock(body=[current_node])
            ) if isinstance(current_node, cst.If) else current_node
        )

    # 3. Inject into the Kernel Skeleton
    # (Same skeleton strategy as previous answer)
    skeleton = """
import warp as wp

# We can duplicate helper functions here or import them if Warp allows
@wp.func
def make_flags(z: int, n: int, h: int, c: int) -> int:
    return (z << 7) | (n << 6) | (h << 5) | (c << 4)

@wp.kernel
def cpu_step(mem: wp.array(dtype=wp.uint8), pc: wp.array(dtype=wp.int32), a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32), c: wp.array(dtype=wp.int32), f: wp.array(dtype=wp.int32)):
    tid = wp.tid()

    # Load registers to locals
    pc_i = pc[tid]
    a_i = a[tid]
    b_i = b[tid]
    c_i = c[tid]
    f_i = f[tid]

    opcode = wp.int32(mem[pc_i])
    cycles = 0

    # INJECTION_MARKER

    # Write back
    pc[tid] = pc_i
    a[tid] = a_i
    b[tid] = b_i
    c[tid] = c_i
    f[tid] = f_i
"""
    skeleton_tree = cst.parse_module(skeleton)

    # Transformer to find the marker and insert 'current_node' (the big tree)
    class Injector(cst.CSTTransformer):
        def leave_FunctionDef(self, original, updated):
            if original.name.value == "cpu_step":
                new_body = []
                for stmt in updated.body.body:
                    if isinstance(stmt, cst.Expr) and isinstance(stmt.value, cst.Name) and stmt.value.value == "INJECTION_MARKER":
                        # If marker was just a variable name "INJECTION_MARKER"
                        pass
                    # A better marker check:
                    elif "INJECTION_MARKER" in cst.Module(body=[stmt]).code:
                         new_body.append(current_node)
                    else:
                        new_body.append(stmt)
                return updated.with_changes(body=updated.body.with_changes(body=new_body))

    final_tree = skeleton_tree.visit(Injector())
    return final_tree.code

```

### Why this is safer

1. **Static Analysis:** `cpu_templates.py` is a normal file. If you make a typo in `template_inc_r8` (like `old & 0xZZ`), your IDE and Ruff will scream at you immediately.
2. **Scope Isolation:** You clearly define the _expected inputs_ (`pc_i`, `f_i`, `REG_i`) in the template function signature. This acts as a contract for what variables must be available in the Warp kernel scope.
3. **Refactoring:** If you rename `pc_i` to `program_counter` in your kernel, you just update the template signature and the generator replacements. You don't have to hunt through generic f-strings.
