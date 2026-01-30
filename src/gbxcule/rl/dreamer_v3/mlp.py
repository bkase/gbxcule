"""Minimal MLP + LayerNorm utilities for Dreamer v3 RSSM (M3)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


torch = _require_torch()


class LayerNorm(torch.nn.LayerNorm):  # type: ignore[misc]
    """LayerNorm that preserves input dtype (matches sheeprl behavior)."""

    def forward(self, x):  # type: ignore[no-untyped-def]
        input_dtype = x.dtype
        out = super().forward(x)
        return out.to(input_dtype)


def _normalize_shape(value: int | Sequence[int]) -> list[int]:
    if isinstance(value, int):
        return [value]
    return list(value)


class MLP(torch.nn.Module):  # type: ignore[misc]
    """Simple MLP with optional LayerNorm after each hidden projection."""

    def __init__(
        self,
        input_dims: int | Sequence[int],
        output_dim: int | None = None,
        hidden_sizes: Sequence[int] = (),
        *,
        activation: Callable[..., Any] | type | None = None,
        layer_norm_cls: Callable[..., Any] | type | None = LayerNorm,
        layer_norm_kw: dict[str, Any] | None = None,
        bias: bool | None = None,
        flatten_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not hidden_sizes and output_dim is None:
            raise ValueError("MLP requires at least one hidden layer or an output_dim")
        if activation is None:
            activation = torch.nn.ReLU
        activation_cls = activation
        assert activation_cls is not None
        if layer_norm_kw is None:
            layer_norm_kw = {}

        input_dim = prod(_normalize_shape(input_dims))
        if bias is None:
            bias = layer_norm_cls in (None, torch.nn.Identity)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim, bias=bias))
            if layer_norm_cls not in (None, torch.nn.Identity):
                norm_kw = dict(layer_norm_kw)
                norm_shape = norm_kw.pop("normalized_shape", hidden_dim)
                layers.append(layer_norm_cls(norm_shape, **norm_kw))
            layers.append(activation_cls())
            prev_dim = hidden_dim
        if output_dim is not None:
            layers.append(torch.nn.Linear(prev_dim, output_dim))

        self._model = torch.nn.Sequential(*layers)
        self._output_dim = output_dim or prev_dim
        self._flatten_dim = flatten_dim

    @property
    def model(self):  # type: ignore[no-untyped-def]
        return self._model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x):  # type: ignore[no-untyped-def]
        if self._flatten_dim is not None:
            x = x.flatten(self._flatten_dim)
        return self._model(x)
