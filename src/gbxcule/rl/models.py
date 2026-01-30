"""Pixel-policy models (torch)."""

from __future__ import annotations

from typing import Any

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W
from gbxcule.rl.packed_pixels import get_unpack_lut


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl. Install with `uv sync --group rl`."
        ) from exc


class PixelActorCriticCNN:  # type: ignore[no-any-unimported]
    """Minimal Atari-ish CNN trunk with policy + value heads."""

    def __init__(
        self,
        *,
        num_actions: int = 7,
        in_frames: int = 4,
        height: int = DOWNSAMPLE_H,
        width: int = DOWNSAMPLE_W,
        input_format: str = "u8",
    ) -> None:
        torch = _require_torch()
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        if in_frames < 1:
            raise ValueError("in_frames must be >= 1")
        if height < 1 or width < 1:
            raise ValueError("height/width must be >= 1")

        if input_format not in ("u8", "packed2"):
            raise ValueError("input_format must be 'u8' or 'packed2'")

        self._torch = torch
        self.num_actions = int(num_actions)
        self.in_frames = int(in_frames)
        self.height = int(height)
        self.width = int(width)
        self.input_format = input_format
        if self.input_format == "packed2":
            if self.width % 4 != 0:
                raise ValueError("width must be divisible by 4 for packed2 inputs")
            self.input_width = self.width // 4
            self._unpack_lut = None
        else:
            self.input_width = self.width
            self._unpack_lut = None

        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_frames, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, self.in_frames, self.height, self.width))
            trunk_out = self.trunk(dummy)
            flat_dim = int(trunk_out.reshape(1, -1).shape[1])

        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(flat_dim, 512),
            torch.nn.ReLU(),
        )
        self.policy = torch.nn.Linear(512, self.num_actions)
        self.value = torch.nn.Linear(512, 1)

    def parameters(self):  # type: ignore[no-untyped-def]
        return (
            list(self.trunk.parameters())
            + list(self.fc.parameters())
            + list(self.policy.parameters())
            + list(self.value.parameters())
        )

    def state_dict(self):  # type: ignore[no-untyped-def]
        return {
            "trunk": self.trunk.state_dict(),
            "fc": self.fc.state_dict(),
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
        }

    def load_state_dict(self, state):  # type: ignore[no-untyped-def]
        self.trunk.load_state_dict(state["trunk"])
        self.fc.load_state_dict(state["fc"])
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        return self

    def train(self, mode: bool = True):  # type: ignore[no-untyped-def]
        self.trunk.train(mode)
        self.fc.train(mode)
        self.policy.train(mode)
        self.value.train(mode)
        return self

    def eval(self):  # type: ignore[no-untyped-def]
        return self.train(False)

    def to(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.trunk.to(*args, **kwargs)
        self.fc.to(*args, **kwargs)
        self.policy.to(*args, **kwargs)
        self.value.to(*args, **kwargs)
        return self

    def __call__(self, obs):  # type: ignore[no-untyped-def]
        torch = self._torch
        if obs.ndim != 4:
            raise ValueError("obs must be 4D [N, K, H, W]")
        if obs.dtype is not torch.uint8:
            raise ValueError("obs must be uint8 shade values")
        if obs.shape[1] != self.in_frames:
            raise ValueError("obs stack depth mismatch")
        if obs.shape[2] != self.height or obs.shape[3] != self.input_width:
            raise ValueError("obs spatial size mismatch")
        if self.input_format == "packed2":
            lut = self._unpack_lut
            if lut is None or lut.device != obs.device:
                lut = get_unpack_lut(device=obs.device, dtype=torch.uint8)
                self._unpack_lut = lut
            unpacked = lut[obs.to(torch.int64)].reshape(
                obs.shape[0],
                obs.shape[1],
                obs.shape[2],
                obs.shape[3] * 4,
            )
            x = unpacked.to(torch.float32) / 3.0
        else:
            if obs.shape[3] != self.width:
                raise ValueError("obs spatial size mismatch")
            x = obs.to(torch.float32) / 3.0
        features = self.trunk(x)
        features = self.fc(features)
        logits = self.policy(features)
        values = self.value(features).squeeze(-1)
        return logits, values
