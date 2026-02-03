"""Nature CNN model with frame stacking for PPO.

A simpler, faster model than ImpalaResNet that still provides
frame stacking for short-term memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """Nature DQN style CNN with frame stacking.

    Architecture:
    - 3 conv layers with ReLU
    - Flatten -> Dense -> policy/value heads

    Much faster than ImpalaResNet while still effective for Pokemon.
    """

    def __init__(
        self,
        num_actions: int = 8,
        in_frames: int = 4,
        height: int = 72,
        width: int = 80,
    ) -> None:
        super().__init__()
        self.in_frames = in_frames
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(in_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate flat dimension
        with torch.no_grad():
            dummy = torch.zeros(1, in_frames, height, width)
            flat_dim = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
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
        x = self.conv(x)
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value
