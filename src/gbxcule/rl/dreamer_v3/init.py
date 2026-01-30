"""Initialization utilities for Dreamer v3 (Hafner init)."""

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


def init_weights(module) -> None:  # type: ignore[no-untyped-def]
    """Hafner-style truncated normal init (matches sheeprl)."""

    torch = _require_torch()
    import numpy as np

    if isinstance(module, torch.nn.Linear):
        in_num = module.in_features
        out_num = module.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        torch.nn.init.trunc_normal_(
            module.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        space = module.kernel_size[0] * module.kernel_size[1]
        in_num = space * module.in_channels
        out_num = space * module.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        torch.nn.init.trunc_normal_(
            module.weight.data, mean=0.0, std=std, a=-2.0, b=2.0
        )
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, torch.nn.LayerNorm):
        module.weight.data.fill_(1.0)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def uniform_init_weights(given_scale: float) -> Callable[[Any], None]:
    """Uniform init for final layers (matches sheeprl)."""

    def _apply(module) -> None:  # type: ignore[no-untyped-def]
        torch = _require_torch()
        import numpy as np

        if isinstance(module, torch.nn.Linear):
            in_num = module.in_features
            out_num = module.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            torch.nn.init.uniform_(module.weight.data, a=-limit, b=limit)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    return _apply
