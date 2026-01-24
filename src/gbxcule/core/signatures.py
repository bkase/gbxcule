"""Hashing and trace comparison utilities.

This module provides deterministic hashing helpers for CPU state verification
and seed derivation.
Uses stdlib only (no external dependencies beyond typing).
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from typing import Any


def hash64(base_seed: int, env_idx: int, rom_sha: str) -> int:
    """Compute a deterministic 64-bit hash for per-env seed derivation.

    This provides a stable, reproducible seed for each environment that is:
    - Deterministic: same inputs always produce the same output
    - Well-distributed: different inputs produce uncorrelated outputs
    - Fast: uses blake2b which is optimized for speed

    Args:
        base_seed: The base random seed (from reset call or config).
        env_idx: Environment index (0-based).
        rom_sha: SHA-256 hex string of the ROM file.

    Returns:
        A 64-bit unsigned integer seed for this environment.

    Example:
        >>> seed1 = hash64(42, 0, "abc123...")
        >>> seed2 = hash64(42, 1, "abc123...")
        >>> seed1 != seed2  # Different env_idx gives different seed
        True
        >>> hash64(42, 0, "abc123...") == seed1  # Deterministic
        True
    """
    # Pack base_seed and env_idx as little-endian integers
    # base_seed as 64-bit signed (can be negative), env_idx as 32-bit unsigned
    data = struct.pack("<qI", base_seed, env_idx) + rom_sha.encode("utf-8")

    # Hash and take first 8 bytes as unsigned 64-bit int
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return struct.unpack("<Q", digest)[0]


def hash_cpu_state(state: Mapping[str, Any], *, include_counters: bool = True) -> str:
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
        # CpuFlags is TypedDict with int values, but dict() returns broader type
        canonical["flags"] = dict(state["flags"])  # type: ignore[assignment]

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
