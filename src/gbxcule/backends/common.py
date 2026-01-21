"""Common types and protocols for backends."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ActionSpec:
    """Specification for action space."""

    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    num_actions: int


@dataclass(frozen=True)
class ObsSpec:
    """Specification for observation space."""

    shape: tuple[int, ...]
    dtype: np.dtype[Any]


@dataclass(frozen=True)
class StepOutput:
    """Output from a step call."""

    obs: NDArray[Any]
    reward: NDArray[np.float32]
    done: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    info: dict[str, Any]


class VecBackend(Protocol):
    """Protocol for vectorized environment backends."""

    @property
    def name(self) -> str:
        """Backend name."""
        ...

    @property
    def device(self) -> Literal["cpu", "cuda"]:
        """Device type."""
        ...

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        ...

    @property
    def action_spec(self) -> ActionSpec:
        """Action specification."""
        ...

    @property
    def obs_spec(self) -> ObsSpec:
        """Observation specification."""
        ...

    def reset(self, seed: int | None = None) -> tuple[NDArray[Any], dict[str, Any]]:
        """Reset all environments."""
        ...

    def step(self, actions: NDArray[Any]) -> StepOutput:
        """Step all environments."""
        ...

    def get_cpu_state(self, env_idx: int) -> dict[str, Any]:
        """Get CPU state for verification."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...
