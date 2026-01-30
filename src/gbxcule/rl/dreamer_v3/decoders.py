"""Decoders for Dreamer v3 world model."""

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


def _restore_time_batch_dict(outputs: dict[str, Any], shape):  # type: ignore[no-untyped-def]
    if shape is None:
        return outputs
    t, b = shape
    reshaped: dict[str, Any] = {}
    for key, value in outputs.items():
        reshaped[key] = value.reshape(t, b, *value.shape[1:])
    return reshaped


class LayerNormChannelLast(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(self, normalized_shape: int, eps: float = 1e-3) -> None:
        super().__init__()
        torch = _require_torch()
        self._ln = torch.nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):  # type: ignore[no-untyped-def]
        if x.ndim != 4:
            raise ValueError("expected input of shape [B, C, H, W]")
        x = x.permute(0, 2, 3, 1)
        x = self._ln(x)
        return x.permute(0, 3, 1, 2)


class CNNDecoder(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        keys: list[str],
        output_channels: list[int],
        channels_multiplier: int,
        latent_state_size: int,
        encoder_output_shape: tuple[int, int, int],
        activation: Callable[[], Any] | None = None,
        layer_norm_cls: Callable[..., Any] | None = LayerNormChannelLast,
        layer_norm_kw: dict[str, Any] | None = None,
        stages: int = 3,
    ) -> None:
        super().__init__()
        torch = _require_torch()
        activation_fn = activation or torch.nn.SiLU
        if layer_norm_kw is None:
            layer_norm_kw = {"eps": 1e-3}
        self.keys = list(keys)
        if not self.keys:
            raise ValueError("keys must be non-empty")
        self.output_channels = list(output_channels)
        if len(self.output_channels) != len(self.keys):
            raise ValueError("output_channels must match keys")
        self.encoder_output_shape = encoder_output_shape
        encoder_output_dim = int(
            encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2]
        )
        self.project = torch.nn.Sequential(
            torch.nn.Linear(latent_state_size, encoder_output_dim),
            torch.nn.Unflatten(1, encoder_output_shape),
        )
        layers: list[Any] = []
        in_ch = encoder_output_shape[0]
        for i in range(stages - 1):
            out_ch = channels_multiplier * (2 ** (stages - i - 2))
            layers.append(
                torch.nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=layer_norm_cls is None,
                )
            )
            if layer_norm_cls is not None:
                layers.append(layer_norm_cls(out_ch, **layer_norm_kw))
            layers.append(activation_fn())
            in_ch = out_ch
        layers.append(
            torch.nn.ConvTranspose2d(
                in_ch,
                sum(self.output_channels),
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        self.deconv = torch.nn.Sequential(*layers)

    def forward(self, latent_states):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        latent_states, shape = _merge_time_batch(latent_states)
        x = self.project(latent_states)
        x = self.deconv(x)
        outputs: dict[str, Any] = {}
        if len(self.output_channels) == 1:
            outputs[self.keys[0]] = x
        else:
            split = torch.split(x, self.output_channels, dim=-3)
            outputs = {k: v for k, v in zip(self.keys, split, strict=True)}
        return _restore_time_batch_dict(outputs, shape)


class MLPDecoder(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        keys: list[str],
        output_dims: list[int],
        latent_state_size: int,
        mlp_layers: int = 4,
        dense_units: int = 512,
        activation: Callable[[], Any] | None = None,
        layer_norm_cls: Callable[..., Any] | None = None,
        layer_norm_kw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        torch = _require_torch()
        activation_fn = activation or torch.nn.SiLU
        if layer_norm_kw is None:
            layer_norm_kw = {"eps": 1e-3}
        self.keys = list(keys)
        if not self.keys:
            raise ValueError("keys must be non-empty")
        if len(output_dims) != len(keys):
            raise ValueError("output_dims must match keys")
        layers: list[Any] = []
        in_dim = latent_state_size
        for _ in range(mlp_layers):
            layers.append(
                torch.nn.Linear(in_dim, dense_units, bias=layer_norm_cls is None)
            )
            if layer_norm_cls is not None:
                layers.append(layer_norm_cls(dense_units, **layer_norm_kw))
            layers.append(activation_fn())
            in_dim = dense_units
        self.model = torch.nn.Sequential(*layers)
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(dense_units, dim) for dim in output_dims]
        )
        self.output_dims = list(output_dims)

    def forward(self, latent_states):  # type: ignore[no-untyped-def]
        latent_states, shape = _merge_time_batch(latent_states)
        x = self.model(latent_states)
        outputs: dict[str, Any] = {}
        for key, head in zip(self.keys, self.heads, strict=True):
            outputs[key] = head(x)
        return _restore_time_batch_dict(outputs, shape)


class MultiDecoder(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        cnn_decoder: Any | None,
        mlp_decoder: Any | None,
    ) -> None:
        super().__init__()
        self.cnn_decoder = cnn_decoder
        self.mlp_decoder = mlp_decoder

    def forward(self, latent_states):  # type: ignore[no-untyped-def]
        outputs: dict[str, Any] = {}
        if self.cnn_decoder is not None:
            outputs.update(self.cnn_decoder(latent_states))
        if self.mlp_decoder is not None:
            outputs.update(self.mlp_decoder(latent_states))
        if not outputs:
            raise ValueError("no decoders configured")
        return outputs
