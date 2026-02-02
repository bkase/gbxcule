"""Dual-Lobe Actor-Critic model for PPO with senses and events.

This model processes three input modalities:
1. Packed pixels (72x20 packed 2bpp) via CNN encoder
2. Senses (map_id, x, y, last_reward) via MLP encoder
3. Events (320 uint8 event flags) via MLP encoder

The features are combined and fed to policy + value heads.
This architecture allows the model to understand "quest state" explicitly,
enabling backtracking behavior (e.g., delivering the parcel back to Oak).
"""

from __future__ import annotations

from typing import Any

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W
from gbxcule.rl.packed_pixels import get_unpack_lut
from gbxcule.rl.pokered_packed_parcel_env import EVENTS_LENGTH, SENSES_DIM


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl. Install with `uv sync`."
        ) from exc


class DualLobeActorCritic:
    """Dual-lobe actor-critic for PPO with pixels, senses, and events.

    Architecture:
    - CNN Lobe: Processes packed 2bpp pixels (72x20) -> unpacked (72x80)
      Conv2d layers with final flattening
    - MLP Lobe: Processes senses (4) and events (320) via separate MLPs
    - Fusion: Concatenate CNN and MLP features -> shared MLP
    - Heads: Policy (discrete actions) + Value (scalar)

    The events encoding allows the model to know "quest state" (e.g., has_parcel)
    which is essential for backtracking behavior.
    """

    def __init__(
        self,
        *,
        num_actions: int = 18,
        senses_dim: int = SENSES_DIM,
        events_dim: int = EVENTS_LENGTH,
        height: int = DOWNSAMPLE_H,
        width: int = DOWNSAMPLE_W,
        cnn_channels: tuple[int, ...] = (32, 64, 64),
        mlp_hidden: int = 128,
        fusion_hidden: int = 512,
    ) -> None:
        torch = _require_torch()
        nn = torch.nn

        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        if senses_dim < 1:
            raise ValueError("senses_dim must be >= 1")
        if events_dim < 1:
            raise ValueError("events_dim must be >= 1")
        if height < 1 or width < 1:
            raise ValueError("height/width must be >= 1")

        self._torch = torch
        self.num_actions = int(num_actions)
        self.senses_dim = int(senses_dim)
        self.events_dim = int(events_dim)
        self.height = int(height)
        self.width = int(width)
        self.input_width = self.width // 4  # Packed 2bpp
        self._unpack_lut = None

        # CNN Lobe for pixels (Atari-style architecture)
        # Input: [N, 1, 72, 80] after unpacking
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output dimension
        with torch.no_grad():
            dummy = torch.zeros((1, 1, self.height, self.width))
            cnn_out = self.cnn(dummy)
            cnn_out_dim = int(cnn_out.shape[1])

        # MLP Lobe for senses (map_id, x, y, last_reward)
        self.senses_mlp = nn.Sequential(
            nn.Linear(senses_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        # MLP Lobe for events (320 event flags)
        # Use embedding-style processing: events -> compact representation
        self.events_mlp = nn.Sequential(
            nn.Linear(events_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        # Fusion layer: combine CNN + senses + events features
        fusion_input_dim = cnn_out_dim + mlp_hidden + mlp_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(),
        )

        # Policy head (discrete actions)
        self.policy = nn.Linear(fusion_hidden, self.num_actions)

        # Value head (scalar)
        self.value = nn.Linear(fusion_hidden, 1)

        self._all_modules = [
            self.cnn,
            self.senses_mlp,
            self.events_mlp,
            self.fusion,
            self.policy,
            self.value,
        ]

    def parameters(self):  # type: ignore[no-untyped-def]
        params = []
        for module in self._all_modules:
            params.extend(module.parameters())
        return params

    def state_dict(self):  # type: ignore[no-untyped-def]
        return {
            "cnn": self.cnn.state_dict(),
            "senses_mlp": self.senses_mlp.state_dict(),
            "events_mlp": self.events_mlp.state_dict(),
            "fusion": self.fusion.state_dict(),
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
        }

    def load_state_dict(self, state):  # type: ignore[no-untyped-def]
        self.cnn.load_state_dict(state["cnn"])
        self.senses_mlp.load_state_dict(state["senses_mlp"])
        self.events_mlp.load_state_dict(state["events_mlp"])
        self.fusion.load_state_dict(state["fusion"])
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        return self

    def train(self, mode: bool = True):  # type: ignore[no-untyped-def]
        for module in self._all_modules:
            module.train(mode)
        return self

    def eval(self):  # type: ignore[no-untyped-def]
        return self.train(False)

    def to(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        for module in self._all_modules:
            module.to(*args, **kwargs)
        return self

    def __call__(
        self,
        pixels: Any,
        senses: Any,
        events: Any,
    ) -> tuple[Any, Any]:
        """Forward pass.

        Args:
            pixels: uint8[N, 1, H, W_packed] - packed 2bpp pixels
            senses: float32[N, 4] - [map_id, x, y, last_reward]
            events: uint8[N, 320] - event flags

        Returns:
            logits: float32[N, num_actions] - action logits
            values: float32[N] - state values
        """
        torch = self._torch

        # Validate inputs
        if pixels.ndim != 4:
            raise ValueError("pixels must be 4D [N, 1, H, W]")
        if pixels.dtype is not torch.uint8:
            raise ValueError("pixels must be uint8")
        if pixels.shape[1] != 1:
            raise ValueError("pixels must have 1 channel")
        if pixels.shape[2] != self.height or pixels.shape[3] != self.input_width:
            exp = f"({self.height}, {self.input_width})"
            got = f"({pixels.shape[2]}, {pixels.shape[3]})"
            raise ValueError(f"pixels spatial size mismatch: expected {exp}, got {got}")

        if senses.ndim != 2:
            raise ValueError("senses must be 2D [N, senses_dim]")
        if senses.shape[1] != self.senses_dim:
            raise ValueError(
                f"senses dim mismatch: expected {self.senses_dim}, got {senses.shape[1]}"  # noqa: E501
            )

        if events.ndim != 2:
            raise ValueError("events must be 2D [N, events_dim]")
        if events.shape[1] != self.events_dim:
            raise ValueError(
                f"events dim mismatch: expected {self.events_dim}, got {events.shape[1]}"  # noqa: E501
            )

        batch_size = pixels.shape[0]
        if senses.shape[0] != batch_size or events.shape[0] != batch_size:
            raise ValueError("batch size mismatch between pixels, senses, events")

        # Unpack pixels: packed 2bpp -> unpacked
        lut = self._unpack_lut
        if lut is None or lut.device != pixels.device:
            lut = get_unpack_lut(device=pixels.device, dtype=torch.uint8)
            self._unpack_lut = lut

        unpacked = lut[pixels.to(torch.int64)].reshape(
            batch_size,
            1,
            self.height,
            self.width,
        )

        # Normalize to [0, 1]
        x_pixels = unpacked.to(torch.float32) / 3.0

        # CNN forward
        cnn_features = self.cnn(x_pixels)

        # Senses forward (already float32 from env)
        if senses.dtype is not torch.float32:
            senses = senses.to(torch.float32)
        senses_features = self.senses_mlp(senses)

        # Events forward (convert uint8 to float32)
        if events.dtype is not torch.float32:
            events = events.to(torch.float32) / 255.0  # Normalize to [0, 1]
        events_features = self.events_mlp(events)

        # Fusion
        combined = torch.cat([cnn_features, senses_features, events_features], dim=1)
        fusion_features = self.fusion(combined)

        # Heads
        logits = self.policy(fusion_features)
        values = self.value(fusion_features).squeeze(-1)

        return logits, values
