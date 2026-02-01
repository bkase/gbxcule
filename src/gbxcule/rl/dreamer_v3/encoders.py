"""Encoders for Dreamer v3 world model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gbxcule.rl.packed_pixels import unpack_2bpp_u8


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


def _merge_time_batch(x):  # type: ignore[no-untyped-def]
    if x.ndim == 5:
        t, b = x.shape[:2]
        return x.reshape(t * b, *x.shape[2:]), (t, b)
    return x, None


def _restore_time_batch(x, shape):  # type: ignore[no-untyped-def]
    if shape is None:
        return x
    t, b = shape
    return x.reshape(t, b, *x.shape[1:])


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


class CNNEncoder(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        keys: list[str],
        input_channels: list[int],
        image_size: tuple[int, int],
        channels_multiplier: int,
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
        self.input_dim = (sum(input_channels), *image_size)
        layers: list[Any] = []
        in_ch = self.input_dim[0]
        for i in range(stages):
            out_ch = channels_multiplier * (2**i)
            layers.append(
                torch.nn.Conv2d(
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
        self.conv = torch.nn.Sequential(*layers)
        self.model = torch.nn.Sequential(self.conv, torch.nn.Flatten(-3, -1))
        with torch.no_grad():
            dummy = torch.zeros((1, *self.input_dim))
            conv_out = self.conv(dummy)
            self.output_shape = conv_out.shape[-3:]
            self.output_dim = int(self.model(dummy).shape[-1])

    def forward(self, obs: dict[str, Any]):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        if not isinstance(obs, dict):
            raise TypeError("obs must be a dict")
        x = torch.cat([obs[k] for k in self.keys], dim=-3)
        x, shape = _merge_time_batch(x)
        out = self.model(x)
        return _restore_time_batch(out, shape)


class MLPEncoder(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        keys: list[str],
        input_dims: list[int],
        mlp_layers: int = 4,
        dense_units: int = 512,
        activation: Callable[[], Any] | None = None,
        layer_norm_cls: Callable[..., Any] | None = None,
        layer_norm_kw: dict[str, Any] | None = None,
        symlog_inputs: bool = True,
    ) -> None:
        super().__init__()
        torch = _require_torch()
        activation_fn = activation or torch.nn.SiLU
        if layer_norm_kw is None:
            layer_norm_kw = {"eps": 1e-3}
        self.keys = list(keys)
        if not self.keys:
            raise ValueError("keys must be non-empty")
        self.input_dim = int(sum(input_dims))
        self.symlog_inputs = symlog_inputs
        layers: list[Any] = []
        in_dim = self.input_dim
        for _ in range(mlp_layers):
            layers.append(
                torch.nn.Linear(in_dim, dense_units, bias=layer_norm_cls is None)
            )
            if layer_norm_cls is not None:
                layers.append(layer_norm_cls(dense_units, **layer_norm_kw))
            layers.append(activation_fn())
            in_dim = dense_units
        self.model = torch.nn.Sequential(*layers)
        self.output_dim = dense_units

    def forward(self, obs: dict[str, Any]):  # type: ignore[no-untyped-def]
        from gbxcule.rl.dreamer_v3.math import symlog

        torch = _require_torch()
        if not isinstance(obs, dict):
            raise TypeError("obs must be a dict")
        # Convert uint8 inputs to float32 (e.g., events tensor)
        parts = []
        for k in self.keys:
            val = obs[k]
            if val.dtype is torch.uint8:
                val = val.to(torch.float32)
            parts.append(val)
        x = torch.cat(parts, dim=-1)
        if self.symlog_inputs:
            x = symlog(x)
        x, shape = _merge_time_batch(x)
        out = self.model(x)
        return _restore_time_batch(out, shape)


class Packed2PixelEncoder(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        keys: list[str],
        image_size: tuple[int, int],
        channels_multiplier: int,
        activation: Callable[[], Any] | None = None,
        layer_norm_cls: Callable[..., Any] | None = LayerNormChannelLast,
        layer_norm_kw: dict[str, Any] | None = None,
        stages: int = 3,
    ) -> None:
        super().__init__()
        self.keys = list(keys)
        self._encoder = CNNEncoder(
            keys=keys,
            input_channels=[1 for _ in keys],
            image_size=image_size,
            channels_multiplier=channels_multiplier,
            activation=activation,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw,
            stages=stages,
        )

    @staticmethod
    def unpack_and_normalize(packed_obs):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        if packed_obs.dtype is not torch.uint8:
            raise ValueError("packed_obs must be uint8")
        unpacked = unpack_2bpp_u8(packed_obs)
        return unpacked.to(torch.float32) / 3.0 - 0.5

    @property
    def output_dim(self) -> int:
        return self._encoder.output_dim

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self._encoder.output_shape

    def forward(self, obs: dict[str, Any]):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        if not isinstance(obs, dict):
            raise TypeError("obs must be a dict")
        normalized: dict[str, Any] = {}
        for k in self.keys:
            val = obs[k]
            if val.dtype is not torch.uint8:
                raise ValueError("packed2 obs must be uint8")
            normalized[k] = self.unpack_and_normalize(val)
        return self._encoder(normalized)


class MultiEncoder(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        cnn_encoder: Any | None,
        mlp_encoder: Any | None,
    ) -> None:
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.mlp_encoder = mlp_encoder
        self.cnn_output_dim = 0 if cnn_encoder is None else int(cnn_encoder.output_dim)
        self.mlp_output_dim = 0 if mlp_encoder is None else int(mlp_encoder.output_dim)

    def forward(self, obs: dict[str, Any]):  # type: ignore[no-untyped-def]
        parts = []
        if self.cnn_encoder is not None:
            parts.append(self.cnn_encoder(obs))
        if self.mlp_encoder is not None:
            parts.append(self.mlp_encoder(obs))
        if not parts:
            raise ValueError("no encoders configured")
        torch = _require_torch()
        return torch.cat(parts, dim=-1)
