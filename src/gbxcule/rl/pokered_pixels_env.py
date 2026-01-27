"""Torch-only pixel env wrapper for Pokémon Red (GPU).

This module provides a minimal, GPU-native stepping loop:
pixels (uint8) -> torch policy -> actions (int32) -> WarpVecCudaBackend -> pixels.

Torch is an optional dependency of the overall project; this module imports it
only dynamically to keep base workflows (make test/typecheck) torch-free.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from gbxcule.backends.warp_vec import WarpVecCudaBackend
from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl. Install with `uv sync --group rl`."
        ) from exc


class PokeredPixelsEnv:
    """Torch-facing pixels-only environment wrapper (GPU).

    Observations returned are uint8 shade values in [0,3], stacked over K frames:
      obs_u8: uint8[num_envs, K, 72, 80] on CUDA
    """

    def __init__(
        self,
        rom_path: str,
        *,
        state_path: str | None = None,
        num_envs: int = 1,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        stack_k: int = 4,
        action_codec: str | None = None,
        force_lcdc_on_render: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if stack_k < 1:
            raise ValueError(f"stack_k must be >= 1, got {stack_k}")

        torch = _require_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA torch is required for PokeredPixelsEnv.")

        self._torch = torch
        self.num_envs = num_envs
        self.stack_k = stack_k

        self._state_path = str(Path(state_path)) if state_path is not None else None

        backend_kwargs: dict[str, Any] = {}
        if action_codec is not None:
            backend_kwargs["action_codec"] = action_codec

        self.backend = WarpVecCudaBackend(
            rom_path,
            num_envs=num_envs,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            obs_dim=32,
            render_pixels=True,
            force_lcdc_on_render=force_lcdc_on_render,
            **backend_kwargs,
        )

        self._pix = None
        self._stack = None
        self._episode_step = None

    @property
    def pixels(self):  # type: ignore[no-untyped-def]
        """Current downsampled pixel frame view: uint8[N,72,80] on CUDA."""
        if self._pix is None:
            self._pix = self.backend.pixels_torch()
        return self._pix

    @property
    def obs(self):  # type: ignore[no-untyped-def]
        """Current stacked observation: uint8[N,K,72,80] on CUDA."""
        if self._stack is None:
            raise RuntimeError("Call reset() before accessing obs.")
        return self._stack

    @property
    def episode_step(self):  # type: ignore[no-untyped-def]
        """Per-env episode step counter: int32[N] on CUDA."""
        if self._episode_step is None:
            raise RuntimeError("Call reset() before accessing episode_step.")
        return self._episode_step

    def reset(self, seed: int | None = None):  # type: ignore[no-untyped-def]
        """Reset all envs and initialize the frame stack (no stepping)."""
        self._pix = None
        self.backend.reset(seed=seed)
        if self._state_path is not None:
            for env_idx in range(self.num_envs):
                self.backend.load_state_file(self._state_path, env_idx=env_idx)

        self.backend.render_pixels_snapshot_torch()
        pix = self.backend.pixels_torch()
        self._pix = pix

        if self._stack is None:
            self._stack = self._torch.empty(
                (self.num_envs, self.stack_k, DOWNSAMPLE_H, DOWNSAMPLE_W),
                dtype=self._torch.uint8,
                device=pix.device,
            )
        if self.stack_k == 1:
            self._stack[:, 0].copy_(pix)
        else:
            for k in range(self.stack_k):
                self._stack[:, k].copy_(pix)

        if self._episode_step is None:
            self._episode_step = self._torch.zeros(
                (self.num_envs,), dtype=self._torch.int32, device=pix.device
            )
        else:
            self._episode_step.zero_()

        return self._stack

    def step(self, actions):  # type: ignore[no-untyped-def]
        """Step all envs using CUDA torch actions and update the frame stack."""
        if self._stack is None or self._episode_step is None:
            raise RuntimeError("Call reset() before step().")

        torch = self._torch
        tensor_type = getattr(torch, "Tensor", None)
        if tensor_type is None or not isinstance(actions, tensor_type):
            raise TypeError("actions must be a torch.Tensor")
        if actions.device.type != "cuda":
            raise ValueError("actions must be a CUDA tensor")
        if actions.ndim != 1 or int(actions.shape[0]) != self.num_envs:
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        if actions.dtype is not torch.int32:
            raise ValueError("actions must have dtype torch.int32 (no implicit cast)")
        if not actions.is_contiguous():
            raise ValueError("actions must be contiguous")

        self.backend.step_torch(actions)
        self.backend.render_pixels_snapshot_torch()

        pix = self.pixels
        if self.stack_k == 1:
            self._stack[:, 0].copy_(pix)
        else:
            self._stack[:, :-1].copy_(self._stack[:, 1:].clone())
            self._stack[:, -1].copy_(pix)

        self._episode_step.add_(1)
        return self._stack

    def close(self) -> None:
        self.backend.close()
        self._pix = None
        self._stack = None
        self._episode_step = None
