"""Hashing and trace comparison utilities.

This module provides deterministic hashing helpers for CPU state verification.
Uses stdlib only (no external dependencies beyond typing).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gbxcule.backends.common import CpuState


def hash_cpu_state(state: CpuState, *, include_counters: bool = True) -> str:
    """Compute a deterministic hash of CPU state for quick comparison.

    Uses blake2b over a canonicalized JSON representation with sorted keys
    and stable integer formatting.

    Args:
        state: CPU state dictionary with registers and flags.
        include_counters: If True, include instr_count and cycle_count in hash.
            If False, only hash core registers and flags.

    Returns:
        Hex-encoded blake2b hash (32 bytes / 64 hex chars).

    Example:
        >>> state = {"pc": 0x0150, "sp": 0xFFFE, "a": 0, "f": 0xB0, ...}
        >>> h1 = hash_cpu_state(state)
        >>> h2 = hash_cpu_state(state)
        >>> h1 == h2  # Deterministic
        True
    """
    # Build a canonicalized representation
    canonical: dict[str, int | dict[str, int] | None] = {}

    # Core registers (always include)
    for reg in ("pc", "sp", "a", "f", "b", "c", "d", "e", "h", "l"):
        if reg in state:
            canonical[reg] = state[reg]  # type: ignore[literal-required]

    # Flags (always include if present)
    if "flags" in state:
        canonical["flags"] = dict(state["flags"])

    # Counters (optional)
    if include_counters:
        if "instr_count" in state:
            canonical["instr_count"] = state["instr_count"]
        if "cycle_count" in state:
            canonical["cycle_count"] = state["cycle_count"]

    # Serialize with sorted keys for determinism
    json_bytes = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )

    # Hash using blake2b (fast, secure, stdlib)
    return hashlib.blake2b(json_bytes, digest_size=32).hexdigest()
