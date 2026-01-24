"""Type stubs for gymnasium - all types as Any for now."""

from typing import Any

class Env:
    """Base Gym environment."""
    observation_space: Any
    action_space: Any

    def reset(self, *, seed: int | None = ..., options: dict[str, Any] | None = ...) -> tuple[Any, dict[str, Any]]: ...
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...
    def render(self) -> Any: ...
    def close(self) -> None: ...

def make(id: str, **kwargs: Any) -> Env: ...

# spaces module
class spaces:
    class Space:
        shape: tuple[int, ...]
        dtype: Any
        def sample(self) -> Any: ...
        def contains(self, x: Any) -> bool: ...

    class Box(Space):
        low: Any
        high: Any
        def __init__(
            self,
            low: float | Any = ...,
            high: float | Any = ...,
            shape: tuple[int, ...] | None = ...,
            dtype: Any = ...,
        ) -> None: ...

    class Discrete(Space):
        n: int
        def __init__(self, n: int, start: int = ...) -> None: ...

    class MultiDiscrete(Space):
        nvec: Any
        def __init__(self, nvec: Any, dtype: Any = ...) -> None: ...

    class MultiBinary(Space):
        def __init__(self, n: int | list[int]) -> None: ...

    class Dict(Space):
        spaces: dict[str, Space]
        def __init__(self, spaces: dict[str, Space] | None = ...) -> None: ...

    class Tuple(Space):
        def __init__(self, spaces: tuple[Space, ...]) -> None: ...
