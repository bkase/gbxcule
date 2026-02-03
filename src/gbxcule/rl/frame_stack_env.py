"""Frame-stacking wrapper for Pokemon Red environment.

Provides short-term memory by stacking the last N frames, allowing
the agent to perceive motion and avoid getting stuck in loops.
"""

from __future__ import annotations

from typing import Any

import torch

from gbxcule.rl.packed_pixels import get_unpack_lut
from gbxcule.rl.pokered_packed_parcel_env import PokeredPackedParcelEnv


class FrameStackEnv:
    """Wrapper that stacks unpacked frames for ImpalaResNet.

    Takes a PokeredPackedParcelEnv and provides:
    - Unpacked pixels (72x80 grayscale)
    - Frame stacking (default 4 frames)
    - Observations as [batch, frames, height, width] tensor

    This gives the agent short-term memory to detect motion and
    avoid infinite loops when facing walls or obstacles.
    """

    def __init__(
        self,
        env: PokeredPackedParcelEnv,
        num_frames: int = 4,
        device: str = "cuda",
    ) -> None:
        self.env = env
        self.num_frames = num_frames
        self.device = device
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions

        # Unpack LUT for 2bpp -> grayscale
        self._unpack_lut = get_unpack_lut(device=device, dtype=torch.uint8)

        # Frame buffer: [num_envs, num_frames, height, width]
        self._frame_buffer: torch.Tensor | None = None
        self._frame_idx = 0

    def _unpack_pixels(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack 2bpp packed pixels to grayscale [batch, height, width]."""
        # packed: [batch, 1, 72, 20] -> unpacked: [batch, 72, 80]
        unpacked = self._unpack_lut[packed.to(torch.int64)]
        return unpacked.reshape(self.num_envs, 72, 80)

    def _update_frame_buffer(self, frame: torch.Tensor) -> None:
        """Add new frame to circular buffer."""
        if self._frame_buffer is None:
            # Initialize buffer with copies of first frame
            self._frame_buffer = frame.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        else:
            # Roll buffer and insert new frame
            self._frame_buffer = torch.roll(self._frame_buffer, -1, dims=1)
            self._frame_buffer[:, -1] = frame

    def _get_stacked_obs(self) -> torch.Tensor:
        """Get stacked frames as observation (uint8 for memory efficiency)."""
        assert self._frame_buffer is not None
        # Return raw uint8 (0-3), normalize in training loop to save memory
        return self._frame_buffer

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset environment and initialize frame buffer."""
        obs = self.env.reset_torch(seed=seed)
        packed_pixels = obs["pixels"]

        # Unpack and initialize frame buffer
        frame = self._unpack_pixels(packed_pixels)
        self._frame_buffer = frame.unsqueeze(1).repeat(1, self.num_frames, 1, 1)

        return {
            "pixels": self._get_stacked_obs(),
            "senses": obs["senses"],
            "events": obs["events"],
        }

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step environment and update frame buffer."""
        next_obs, reward, terminated, truncated, info = self.env.step_torch(actions)
        packed_pixels = next_obs["pixels"]

        # Unpack and update frame buffer
        frame = self._unpack_pixels(packed_pixels)
        self._update_frame_buffer(frame)

        return (
            {
                "pixels": self._get_stacked_obs(),
                "senses": next_obs["senses"],
                "events": next_obs["events"],
            },
            reward,
            terminated,
            truncated,
            info,
        )

    def reset_mask(self, mask: torch.Tensor) -> None:
        """Reset specific environments and their frame buffers."""
        self.env.reset_mask(mask)

        # Reset frame buffer for masked envs
        if mask.any() and self._frame_buffer is not None:
            # Get current packed pixels for reset envs
            packed = self.env._pixels
            frame = self._unpack_pixels(packed)

            # Reset buffer for masked envs (fill with current frame)
            for i in range(self.num_frames):
                self._frame_buffer[mask, i] = frame[mask]

    def close(self) -> None:
        """Close underlying environment."""
        self.env.close()
