"""ABI v0: Authoritative device buffer layouts.

This module defines the canonical layouts for state buffers used by
Warp kernels and downstream consumers.
"""

ABI_VERSION = 0

# ABI v0: flat 64KB memory + CPU regs + flags + instr counter
# See ARCHITECTURE.md Section 6 for rationale

# Register layout (per env):
# - pc: u16
# - sp: u16
# - a, f, b, c, d, e, h, l: u8
# - instr_count: u64
# - cycle_count: u64 (optional)

# Memory layout (per env):
# - mem: u8[65536] flat 64KB
