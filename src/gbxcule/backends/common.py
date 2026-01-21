"""Common types, protocols, and validation helpers for backends.

This module defines the backend contract and shared types for all backend
implementations. Types are pure dataclasses/TypedDicts with no heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Core literals and type aliases
# ---------------------------------------------------------------------------

Device = Literal["cpu", "cuda"]
Stage = Literal["emulate_only", "full_step", "reward_only", "obs_only"]

# Numpy type aliases for clarity
NDArrayF32 = NDArray[np.float32]
NDArrayI32 = NDArray[np.int32]
NDArrayBool = NDArray[np.bool_]

# ---------------------------------------------------------------------------
# ArraySpec and BackendSpec dataclasses (pure + JSON-friendly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArraySpec:
    """Specification for an array (actions or observations).

    Attributes:
        shape: Expected shape of the array.
        dtype: String representation of the dtype (e.g., "float32").
        meaning: Short human-readable description.
    """

    shape: tuple[int, ...]
    dtype: str
    meaning: str

    def to_json_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "meaning": self.meaning,
        }


@dataclass(frozen=True)
class BackendSpec:
    """Specification describing a backend's configuration.

    Attributes:
        name: Backend identifier (e.g., "pyboy_single", "warp_vec").
        device: Device type ("cpu" or "cuda").
        num_envs: Number of parallel environments.
        action: Action array specification.
        obs: Observation array specification.
    """

    name: str
    device: Device
    num_envs: int
    action: ArraySpec
    obs: ArraySpec

    def to_json_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "device": self.device,
            "num_envs": self.num_envs,
            "action": self.action.to_json_dict(),
            "obs": self.obs.to_json_dict(),
        }


# ---------------------------------------------------------------------------
# StepOutput dataclass (internal convenience)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepOutput:
    """Output from a backend step call.

    Attributes:
        obs: Observations, shape (num_envs, obs_dim), dtype float32.
        reward: Rewards, shape (num_envs,), dtype float32.
        done: Done flags, shape (num_envs,), dtype bool.
        trunc: Truncation flags, shape (num_envs,), dtype bool.
        info: Additional info dictionary.
    """

    obs: NDArrayF32
    reward: NDArrayF32
    done: NDArrayBool
    trunc: NDArrayBool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# CPU state schema typing
# ---------------------------------------------------------------------------


class CpuFlags(TypedDict):
    """CPU flag register decomposition."""

    z: int  # Zero flag (bit 7 of F)
    n: int  # Subtract flag (bit 6 of F)
    h: int  # Half-carry flag (bit 5 of F)
    c: int  # Carry flag (bit 4 of F)


class CpuState(TypedDict, total=False):
    """CPU register state for verification.

    All register values are integers. Counters are optional and may be None
    if the backend cannot provide them.
    """

    pc: int  # Program counter
    sp: int  # Stack pointer
    a: int  # Accumulator
    f: int  # Flags register (raw byte)
    b: int
    c: int
    d: int
    e: int
    h: int
    l: int  # noqa: E741 - canonical register name
    flags: CpuFlags  # Decomposed flags
    instr_count: int | None  # Instruction count (optional)
    cycle_count: int | None  # Cycle count (optional)


# ---------------------------------------------------------------------------
# VecBackend Protocol
# ---------------------------------------------------------------------------


class VecBackend(Protocol):
    """Protocol for vectorized environment backends.

    All backends must implement this interface with consistent batch semantics:
    - actions: int32[num_envs]
    - obs: float32[num_envs, obs_dim]
    - reward: float32[num_envs]
    - done/trunc: bool[num_envs]
    """

    name: str
    device: Device
    num_envs: int
    action_spec: ArraySpec
    obs_spec: ArraySpec

    def reset(self, seed: int | None = None) -> tuple[NDArrayF32, dict[str, Any]]:
        """Reset all environments.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Tuple of (observations, info_dict).
        """
        ...

    def step(
        self, actions: NDArrayI32
    ) -> tuple[NDArrayF32, NDArrayF32, NDArrayBool, NDArrayBool, dict[str, Any]]:
        """Step all environments forward.

        Args:
            actions: Actions array, shape (num_envs,), dtype int32.

        Returns:
            Tuple of (obs, reward, done, trunc, info).
        """
        ...

    def get_cpu_state(self, env_idx: int) -> CpuState:
        """Get CPU register state for a specific environment.

        Args:
            env_idx: Environment index (0 for single-env backends).

        Returns:
            CpuState dictionary with register values and flags.
        """
        ...

    def close(self) -> None:
        """Clean up resources and release emulator instances."""
        ...


# ---------------------------------------------------------------------------
# Validation helpers (fast boundary checks)
# ---------------------------------------------------------------------------


def flags_from_f(f: int) -> CpuFlags:
    """Derive flag bits from the F register.

    The F register stores flags in the upper nibble:
    - Bit 7: Z (zero)
    - Bit 6: N (subtract)
    - Bit 5: H (half-carry)
    - Bit 4: C (carry)

    Args:
        f: Raw F register value (0-255).

    Returns:
        CpuFlags dict with z, n, h, c values (0 or 1).
    """
    return CpuFlags(
        z=(f >> 7) & 1,
        n=(f >> 6) & 1,
        h=(f >> 5) & 1,
        c=(f >> 4) & 1,
    )


def validate_actions(actions: NDArray[Any], num_envs: int) -> None:
    """Validate action array shape and dtype.

    Args:
        actions: Action array to validate.
        num_envs: Expected number of environments.

    Raises:
        ValueError: If validation fails with actionable message.
    """
    if actions.ndim != 1:
        raise ValueError(
            f"Actions must be 1D, got ndim={actions.ndim}, shape={actions.shape}"
        )
    if len(actions) != num_envs:
        raise ValueError(f"Actions length {len(actions)} != num_envs {num_envs}")
    if not np.issubdtype(actions.dtype, np.integer):
        raise ValueError(f"Actions dtype must be integer, got {actions.dtype}")


def as_i32_actions(actions: NDArray[Any], num_envs: int) -> NDArrayI32:
    """Validate and cast actions to int32.

    Args:
        actions: Action array to validate and cast.
        num_envs: Expected number of environments.

    Returns:
        Actions array as int32.

    Raises:
        ValueError: If validation fails.
    """
    validate_actions(actions, num_envs)
    if actions.dtype != np.int32:
        return actions.astype(np.int32)
    return actions


def validate_step_output(
    obs: NDArray[Any],
    reward: NDArray[Any],
    done: NDArray[Any],
    trunc: NDArray[Any],
    num_envs: int,
    obs_dim: int,
) -> None:
    """Validate step output shapes and dtypes.

    Args:
        obs: Observations array.
        reward: Rewards array.
        done: Done flags array.
        trunc: Truncation flags array.
        num_envs: Expected number of environments.
        obs_dim: Expected observation dimension.

    Raises:
        ValueError: If validation fails with actionable message.
    """
    # Validate obs
    expected_obs_shape = (num_envs, obs_dim)
    if obs.shape != expected_obs_shape:
        raise ValueError(f"obs shape {obs.shape} != expected {expected_obs_shape}")
    if obs.dtype != np.float32:
        raise ValueError(f"obs dtype {obs.dtype} != expected float32")

    # Validate reward
    expected_reward_shape = (num_envs,)
    if reward.shape != expected_reward_shape:
        raise ValueError(
            f"reward shape {reward.shape} != expected {expected_reward_shape}"
        )
    if reward.dtype != np.float32:
        raise ValueError(f"reward dtype {reward.dtype} != expected float32")

    # Validate done
    if done.shape != expected_reward_shape:
        raise ValueError(f"done shape {done.shape} != expected {expected_reward_shape}")
    if done.dtype != np.bool_:
        raise ValueError(f"done dtype {done.dtype} != expected bool")

    # Validate trunc
    if trunc.shape != expected_reward_shape:
        raise ValueError(
            f"trunc shape {trunc.shape} != expected {expected_reward_shape}"
        )
    if trunc.dtype != np.bool_:
        raise ValueError(f"trunc dtype {trunc.dtype} != expected bool")


def empty_obs(num_envs: int, obs_dim: int) -> NDArrayF32:
    """Create a zero-filled observation array.

    Args:
        num_envs: Number of environments.
        obs_dim: Observation dimension.

    Returns:
        Zero array of shape (num_envs, obs_dim), dtype float32.
    """
    return np.zeros((num_envs, obs_dim), dtype=np.float32)
