"""Type stubs for pufferlib.vector - all types as Any for now."""

from typing import Any

def make(
    env_creator: Any,
    env_args: tuple[Any, ...] | None = ...,
    env_kwargs: dict[str, Any] | None = ...,
    num_envs: int = ...,
    envs_per_worker: int = ...,
    envs_per_batch: int | None = ...,
    env_pool: bool = ...,
    **kwargs: Any,
) -> Any: ...

def Serial(*args: Any, **kwargs: Any) -> Any: ...
def Multiprocessing(*args: Any, **kwargs: Any) -> Any: ...
