"""Heads for Dreamer v3 world model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


def _merge_time_batch(x):  # type: ignore[no-untyped-def]
    if x.ndim == 3:
        t, b = x.shape[:2]
        return x.reshape(t * b, x.shape[-1]), (t, b)
    return x, None


def _restore_time_batch(x, shape):  # type: ignore[no-untyped-def]
    if shape is None:
        return x
    t, b = shape
    return x.reshape(t, b, *x.shape[1:])


def _build_mlp(
    input_dim: int,
    output_dim: int,
    *,
    mlp_layers: int,
    dense_units: int,
    activation: Callable[[], Any] | None,
    layer_norm_cls: Callable[..., Any] | None,
    layer_norm_kw: dict[str, Any] | None,
):
    torch = _require_torch()
    activation_fn = activation or torch.nn.SiLU
    if layer_norm_kw is None:
        layer_norm_kw = {"eps": 1e-3}
    layers: list[Any] = []
    in_dim = input_dim
    for _ in range(mlp_layers):
        layers.append(torch.nn.Linear(in_dim, dense_units, bias=layer_norm_cls is None))
        if layer_norm_cls is not None:
            layers.append(layer_norm_cls(dense_units, **layer_norm_kw))
        layers.append(activation_fn())
        in_dim = dense_units
    layers.append(torch.nn.Linear(in_dim, output_dim))
    return torch.nn.Sequential(*layers)


class RewardHead(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        input_dim: int,
        bins: int,
        mlp_layers: int = 4,
        dense_units: int = 512,
        activation: Callable[[], Any] | None = None,
        layer_norm_cls: Callable[..., Any] | None = None,
        layer_norm_kw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if bins < 2:
            raise ValueError("bins must be >= 2")
        self.model = _build_mlp(
            input_dim,
            bins,
            mlp_layers=mlp_layers,
            dense_units=dense_units,
            activation=activation,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw,
        )

    def forward(self, latent_states):  # type: ignore[no-untyped-def]
        latent_states, shape = _merge_time_batch(latent_states)
        logits = self.model(latent_states)
        return _restore_time_batch(logits, shape)


class ContinueHead(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        input_dim: int,
        mlp_layers: int = 4,
        dense_units: int = 512,
        activation: Callable[[], Any] | None = None,
        layer_norm_cls: Callable[..., Any] | None = None,
        layer_norm_kw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = _build_mlp(
            input_dim,
            1,
            mlp_layers=mlp_layers,
            dense_units=dense_units,
            activation=activation,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw,
        )

    def forward(self, latent_states):  # type: ignore[no-untyped-def]
        latent_states, shape = _merge_time_batch(latent_states)
        logits = self.model(latent_states)
        return _restore_time_batch(logits, shape)
