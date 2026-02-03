"""ImpalaResNet model with frame stacking for PPO.

This model provides:
- Deep residual architecture for better pattern recognition
- Frame stacking (4 frames) for short-term memory/motion detection
- Efficient throughput on modern GPUs (GB10/Blackwell)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    """Conv -> MaxPool -> ResBlock -> ResBlock."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res0 = ResidualBlock(out_channels)
        self.res1 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res0(x)
        x = self.res1(x)
        return x


class ImpalaResNet(nn.Module):
    """Impala ResNet for Pokemon Red with frame stacking.

    Architecture:
    - 3 ConvSequence blocks with residual connections
    - Frame stacking (in_frames=4) provides short-term memory
    - Dense layer -> policy head + value head

    Input: [batch, frames, height, width] where frames=4 for stacking
    For packed 2bpp pixels: height=72, width=80 (unpacked)
    """

    def __init__(
        self,
        num_actions: int = 8,
        in_frames: int = 4,
        base_channels: int = 64,
        height: int = 72,
        width: int = 80,
    ) -> None:
        super().__init__()
        self.in_frames = in_frames
        self.num_actions = num_actions

        # 3 Blocks of ResNets with pooling
        # Each ConvSequence halves spatial dims via stride-2 maxpool
        self.stem = nn.Sequential(
            ConvSequence(in_frames, base_channels),  # h/2, w/2
            ConvSequence(base_channels, base_channels * 2),  # h/4, w/4
            ConvSequence(base_channels * 2, base_channels * 2),  # h/8, w/8
        )

        # Calculate flat dimension: after 3x pooling, dims are h//8, w//8
        # For 72x80 input: 9x10 spatial, 128 channels = 11520
        flat_h = height // 8
        flat_w = width // 8
        flat_dim = base_channels * 2 * flat_h * flat_w

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: [batch, frames, height, width] tensor

        Returns:
            logits: [batch, num_actions] policy logits
            value: [batch] value estimate
        """
        x = self.stem(x)
        x = self.dense(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value
