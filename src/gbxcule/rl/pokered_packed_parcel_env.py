"""Packed2 parcel environment for Pokemon Red (CUDA, Dreamer-ready).

Observations are dicts:
  pixels: uint8[num_envs, 1, 72, 20] on CUDA (packed 2bpp)
  senses: float32[num_envs, 4] on CUDA - [map_id, x, y, last_reward]
  events: uint8[num_envs, 320] on CUDA - raw event flag bytes
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Final

from gbxcule.backends.warp_vec import WarpVecCudaBackend
from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES
from gbxcule.core.reset_cache import ResetCache
from gbxcule.rl.pokered_parcel_detectors import (
    EVENTS_LENGTH,
    EVENTS_START,
    delivered_parcel,
    get_bag_item_count,
    has_parcel,
    is_in_dialogue,
)

# Re-export for tools/rl_train_gpu.py
__all__ = ["PokeredPackedParcelEnv", "EVENTS_LENGTH", "EVENTS_START", "SENSES_DIM"]

# RAM addresses for location
MAP_ID_ADDR: Final[int] = 0xD35E
PLAYER_Y_ADDR: Final[int] = 0xD361
PLAYER_X_ADDR: Final[int] = 0xD362

# Snow hash primes
HASH_PRIME_MAP: Final[int] = 73856093
HASH_PRIME_X: Final[int] = 19349663
HASH_PRIME_Y: Final[int] = 83492791
HASH_PRIME_STAGE: Final[int] = 536870909

# Senses layout: [map_id, x, y, last_reward] (events are separate uint8 tensor)
SENSES_DIM: Final[int] = 4


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.pokered_packed_parcel_env. "
            "Install with `uv sync`."
        ) from exc


class PokeredPackedParcelEnv:
    """Packed2 RL env with dict obs, epoch-trick snow, and parcel rewards.

    Supports "curiosity reset" - when key events occur (like getting the parcel),
    the exploration hash table is cleared so the agent will re-explore familiar
    areas with fresh curiosity, enabling backtracking behavior.
    """

    def __init__(
        self,
        rom_path: str,
        *,
        state_path: str | None = None,
        num_envs: int = 1,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        action_codec: str | None = None,
        max_steps: int = 2048,
        snow_bonus: float = 0.01,
        get_parcel_bonus: float = 5.0,
        deliver_bonus: float = 10.0,
        snow_size: int = 65536,
        force_lcdc_on_render: bool = True,
        skip_reset_if_empty: bool = False,
        info_mode: str = "full",
        curiosity_reset_on_parcel: bool = True,
        # Interaction rewards
        dialogue_bonus: float = 0.01,
        item_pickup_bonus: float = 0.05,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")

        torch = _require_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA torch is required for PokeredPackedParcelEnv.")

        backend_kwargs: dict[str, Any] = {}
        if action_codec is not None:
            backend_kwargs["action_codec"] = action_codec

        self.backend = WarpVecCudaBackend(
            rom_path,
            num_envs=num_envs,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            obs_dim=32,
            render_pixels=False,
            render_pixels_packed=True,
            render_on_step=False,
            force_lcdc_on_render=force_lcdc_on_render,
            **backend_kwargs,
        )

        self._torch = torch
        self.num_envs = int(num_envs)
        self.num_actions = int(self.backend.num_actions)
        self.max_steps = int(max_steps)
        self._state_path = str(Path(state_path)) if state_path is not None else None

        # Reward config
        self._snow_bonus = float(snow_bonus)
        self._get_parcel_bonus = float(get_parcel_bonus)
        self._deliver_bonus = float(deliver_bonus)

        # Snow table config (epoch-trick)
        self._snow_size = int(snow_size)

        self._skip_reset_if_empty = bool(skip_reset_if_empty)
        if info_mode not in {"full", "stats", "none"}:
            raise ValueError(f"info_mode must be full, stats, or none, got {info_mode}")
        self._info_mode = info_mode
        self._curiosity_reset_on_parcel = bool(curiosity_reset_on_parcel)

        # Interaction reward config
        self._dialogue_bonus = float(dialogue_bonus)
        self._item_pickup_bonus = float(item_pickup_bonus)

        # Buffers (allocated on first reset)
        self._obs: dict[str, Any] | None = None
        self._pixels: Any | None = None
        self._senses: Any | None = None
        self._events: Any | None = None
        self._start_pixels: Any | None = None
        self._episode_step: Any | None = None
        self._last_reward: Any | None = None
        self._stage_u8: Any | None = None  # 0=no parcel, 1=has parcel
        self._reset_cache: ResetCache | None = None

        # Interaction tracking
        self._prev_in_dialogue: Any | None = None  # Previous dialogue state
        self._prev_item_count: Any | None = None  # Previous bag item count

        # Epoch-trick snow table
        self._snow_table: Any | None = None  # [num_envs, snow_size] int16
        self._episode_id: Any | None = None  # [num_envs] int16

    @property
    def obs(self) -> dict[str, Any]:
        if self._obs is None:
            raise RuntimeError("Call reset() before accessing obs.")
        return self._obs

    def reset_torch(self, seed: int | None = None) -> dict[str, Any]:
        return self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        torch = self._torch
        self.backend.reset(seed=seed)
        if self._state_path is not None:
            self.backend.load_state_file(self._state_path, env_idx=0)

        # Allocate pixels buffer
        if self._pixels is None:
            self._pixels = torch.empty(
                (self.num_envs, 1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES),
                dtype=torch.uint8,
                device="cuda",
            )
        self.backend.render_pixels_snapshot_packed_to_torch(self._pixels, 0)

        # Allocate senses buffer (small: map_id, x, y, last_reward)
        if self._senses is None:
            self._senses = torch.zeros(
                (self.num_envs, SENSES_DIM),
                dtype=torch.float32,
                device="cuda",
            )

        # Allocate events buffer (uint8, memory-efficient)
        if self._events is None:
            self._events = torch.zeros(
                (self.num_envs, EVENTS_LENGTH),
                dtype=torch.uint8,
                device="cuda",
            )

        # Setup reset cache
        if self._reset_cache is None:
            self._reset_cache = ResetCache.from_backend(self.backend, env_idx=0)

        mask_all = torch.ones((self.num_envs,), device="cuda", dtype=torch.uint8)
        self._reset_cache.apply_mask_torch(mask_all)

        # Cache start state pixels
        assert self._pixels is not None
        self._start_pixels = self._pixels[0].clone()
        self._pixels[:] = self._start_pixels

        # Allocate episode counters
        if self._episode_step is None:
            self._episode_step = torch.zeros(
                (self.num_envs,), dtype=torch.int32, device="cuda"
            )
        else:
            self._episode_step.zero_()

        if self._last_reward is None:
            self._last_reward = torch.zeros(
                (self.num_envs,), dtype=torch.float32, device="cuda"
            )
        else:
            self._last_reward.zero_()

        if self._stage_u8 is None:
            self._stage_u8 = torch.zeros(
                (self.num_envs,), dtype=torch.uint8, device="cuda"
            )
        else:
            self._stage_u8.zero_()

        # Initialize interaction tracking
        mem = self.backend.memory_torch()
        if self._prev_in_dialogue is None:
            self._prev_in_dialogue = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device="cuda"
            )
        else:
            # Set to current dialogue state
            self._prev_in_dialogue = is_in_dialogue(mem)

        if self._prev_item_count is None:
            self._prev_item_count = torch.zeros(
                (self.num_envs,), dtype=torch.uint8, device="cuda"
            )
        else:
            # Set to current item count
            self._prev_item_count = get_bag_item_count(mem)

        # Allocate epoch-trick snow table
        if self._snow_table is None:
            self._snow_table = torch.zeros(
                (self.num_envs, self._snow_size), dtype=torch.int16, device="cuda"
            )
        if self._episode_id is None:
            self._episode_id = torch.ones(
                (self.num_envs,), dtype=torch.int16, device="cuda"
            )
        else:
            # Reset episode IDs
            assert self._episode_id is not None
            assert self._snow_table is not None
            self._episode_id.fill_(1)
            self._snow_table.zero_()

        # Build initial senses and events
        self._update_senses_and_events()

        self._obs = {
            "pixels": self._pixels,
            "senses": self._senses,
            "events": self._events,
        }
        return self._obs

    def _update_senses_and_events(self) -> None:
        """Update senses and events buffers from current RAM state."""
        mem = self.backend.memory_torch()

        # Read location into senses
        map_id = mem[:, MAP_ID_ADDR].float()
        x = mem[:, PLAYER_X_ADDR].float()
        y = mem[:, PLAYER_Y_ADDR].float()

        # Fill senses: [map_id, x, y, last_reward]
        assert self._senses is not None
        assert self._last_reward is not None
        self._senses[:, 0] = map_id
        self._senses[:, 1] = x
        self._senses[:, 2] = y
        self._senses[:, 3] = self._last_reward

        # Copy events directly as uint8 (no float conversion)
        assert self._events is not None
        self._events.copy_(mem[:, EVENTS_START : EVENTS_START + EVENTS_LENGTH])

    def _compute_snow_reward(
        self, map_id: Any, x: Any, y: Any, stage: Any
    ) -> tuple[Any, Any]:
        """Compute fresh-snow exploration reward using epoch trick."""
        assert self._snow_table is not None
        assert self._episode_id is not None

        # Hash: XOR/prime mix of (map_id, x, y, stage)
        hash_idx = (
            (map_id.int() * HASH_PRIME_MAP)
            ^ (x.int() * HASH_PRIME_X)
            ^ (y.int() * HASH_PRIME_Y)
            ^ (stage.int() * HASH_PRIME_STAGE)
        ) % self._snow_size

        # Check if novel
        stored_id = self._snow_table.gather(1, hash_idx.unsqueeze(1)).squeeze(1)
        is_novel = stored_id != self._episode_id

        # Mark visited
        self._snow_table.scatter_(
            1, hash_idx.unsqueeze(1), self._episode_id.unsqueeze(1)
        )

        snow_reward = is_novel.float() * self._snow_bonus
        return snow_reward, is_novel

    def curiosity_reset(self, mask: Any) -> None:
        """Reset curiosity (fresh snow) for masked environments.

        This implements the "Event-Triggered Curiosity Reset" pattern:
        when a key event occurs (e.g., picking up the parcel), the exploration
        hash table is cleared for those envs. This makes the whole world "fresh"
        again, so the agent will happily re-explore areas it has already visited.

        This is critical for backtracking tasks like delivering the parcel,
        where the agent needs to return to a previously-visited location.
        """
        torch = self._torch
        assert self._episode_id is not None
        assert self._snow_table is not None

        # Convert mask to bool
        if mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)

        if not mask.any():
            return

        # Increment episode_id for masked envs (epoch trick: clears snow without memset)
        new_episode_id = self._episode_id + 1
        self._episode_id = torch.where(
            mask, new_episode_id.to(torch.int16), self._episode_id
        )

        # Handle overflow (wrap to 0 means we need to clear those env rows)
        assert self._episode_id is not None
        assert self._snow_table is not None
        episode_id = self._episode_id
        snow_table = self._snow_table
        overflowed = mask & (episode_id == 0)
        if overflowed.any():
            snow_table[overflowed] = 0
            episode_id[overflowed] = 1

    def step_torch(self, actions: Any) -> tuple[dict[str, Any], Any, Any, Any, Any]:
        return self.step(actions)

    def step(self, actions: Any) -> tuple[dict[str, Any], Any, Any, Any, Any]:
        if self._pixels is None or self._episode_step is None:
            raise RuntimeError("Call reset() before step().")
        if self._obs is None:
            raise RuntimeError("Obs dict not initialized. Call reset() first.")
        if self._reset_cache is None:
            raise RuntimeError("Reset cache not initialized. Call reset() first.")

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

        # Step the backend
        self.backend.step_torch(actions)
        self.backend.render_pixels_snapshot_packed_to_torch(self._pixels, 0)

        # Read memory for detectors and location
        mem = self.backend.memory_torch()
        events = mem[:, EVENTS_START : EVENTS_START + EVENTS_LENGTH]
        map_id = mem[:, MAP_ID_ADDR]
        x = mem[:, PLAYER_X_ADDR]
        y = mem[:, PLAYER_Y_ADDR]

        # Compute detectors
        has_parcel_now = has_parcel(mem, events)
        delivered_now = delivered_parcel(mem, events)

        # Stage transitions
        assert self._stage_u8 is not None
        stage_prev = self._stage_u8.clone()
        self._stage_u8 = has_parcel_now.to(torch.uint8)

        # Compute rewards
        got_parcel = (stage_prev == 0) & (self._stage_u8 == 1)

        # Curiosity reset: when getting parcel, reset snow table so world is fresh
        # This enables backtracking behavior - agent will re-explore to deliver
        if self._curiosity_reset_on_parcel and got_parcel.any():
            self.curiosity_reset(got_parcel)

        snow_reward, is_novel = self._compute_snow_reward(map_id, x, y, self._stage_u8)

        # Compute interaction rewards
        assert self._prev_in_dialogue is not None
        assert self._prev_item_count is not None

        # Dialogue reward: bonus for entering dialogue/menu (False->True)
        curr_in_dialogue = is_in_dialogue(mem)
        entered_dialogue = (~self._prev_in_dialogue) & curr_in_dialogue
        dialogue_reward = entered_dialogue.float() * self._dialogue_bonus

        # Item pickup reward: bonus for increasing bag item count
        curr_item_count = get_bag_item_count(mem)
        item_delta = (curr_item_count.int() - self._prev_item_count.int()).clamp(min=0)
        item_reward = item_delta.float() * self._item_pickup_bonus

        # Update tracking state
        self._prev_in_dialogue = curr_in_dialogue
        self._prev_item_count = curr_item_count

        reward = (
            snow_reward
            + got_parcel.float() * self._get_parcel_bonus
            + delivered_now.float() * self._deliver_bonus
            + dialogue_reward
            + item_reward
        )

        # Done/truncation
        terminated = delivered_now
        self._episode_step.add_(1)
        truncated = self._episode_step >= self.max_steps

        # Update senses/events with *previous* reward (for proprioception)
        self._update_senses_and_events()

        # Now update last_reward for next step's senses
        assert self._last_reward is not None
        self._last_reward = reward.clone()

        # Build info
        if self._info_mode == "full":
            info = {
                "map_id": map_id,
                "x": x,
                "y": y,
                "stage": self._stage_u8,
                "is_novel": is_novel,
                "got_parcel": got_parcel,
                "delivered": delivered_now,
                "reset_mask": terminated | truncated,
            }
        elif self._info_mode == "stats":
            info = {
                "stage": self._stage_u8,
                "got_parcel": got_parcel,
                "delivered": delivered_now,
            }
        else:
            info = {}

        return self._obs, reward, terminated, truncated, info

    def reset_mask(self, mask: Any) -> None:
        if self._reset_cache is None:
            raise RuntimeError("Reset cache not initialized. Call reset() first.")
        if self._pixels is None:
            raise RuntimeError("Obs buffer not initialized. Call reset() first.")

        torch = self._torch
        tensor_type = getattr(torch, "Tensor", None)
        if tensor_type is None or not isinstance(mask, tensor_type):
            raise TypeError("mask must be a torch.Tensor")
        if mask.device.type != "cuda":
            raise ValueError("mask must be a CUDA tensor")
        if mask.ndim != 1 or int(mask.shape[0]) != self.num_envs:
            raise ValueError(f"mask must have shape ({self.num_envs},)")

        reset_mask = mask.to(torch.bool)
        if self._skip_reset_if_empty and not torch.any(reset_mask):
            return

        self._reset_cache.apply_mask_torch(reset_mask.to(torch.uint8))

        # Reset counters for masked envs
        assert self._episode_step is not None
        assert self._last_reward is not None
        assert self._stage_u8 is not None
        zeros_i32 = torch.zeros_like(self._episode_step)
        zeros_f32 = torch.zeros_like(self._last_reward)
        zeros_u8 = torch.zeros_like(self._stage_u8)

        self._episode_step = torch.where(reset_mask, zeros_i32, self._episode_step)
        self._last_reward = torch.where(reset_mask, zeros_f32, self._last_reward)
        self._stage_u8 = torch.where(reset_mask, zeros_u8, self._stage_u8)

        # Reset interaction tracking for masked envs
        assert self._prev_in_dialogue is not None
        assert self._prev_item_count is not None
        zeros_bool = torch.zeros_like(self._prev_in_dialogue)
        self._prev_in_dialogue = torch.where(
            reset_mask, zeros_bool, self._prev_in_dialogue
        )
        self._prev_item_count = torch.where(reset_mask, zeros_u8, self._prev_item_count)

        # Epoch trick: increment episode_id for masked envs (no full clear)
        assert self._episode_id is not None
        assert self._snow_table is not None
        new_episode_id = self._episode_id + 1
        self._episode_id = torch.where(
            reset_mask, new_episode_id.to(torch.int16), self._episode_id
        )

        # Handle overflow (wrap to 0 means we need to clear those env rows)
        assert self._snow_table is not None
        assert self._episode_id is not None
        snow_table = self._snow_table
        episode_id = self._episode_id
        overflowed = reset_mask & (episode_id == 0)
        if overflowed.any():
            snow_table[overflowed] = 0
            episode_id[overflowed] = 1

        # Restore start pixels
        if self._start_pixels is not None:
            self._pixels[reset_mask] = self._start_pixels

        # Update senses/events for reset envs
        self._update_senses_and_events()

    def close(self) -> None:
        self.backend.close()
        self._obs = None
        self._pixels = None
        self._senses = None
        self._events = None
        self._start_pixels = None
        self._episode_step = None
        self._last_reward = None
        self._stage_u8 = None
        self._reset_cache = None
        self._snow_table = None
        self._episode_id = None
        self._prev_in_dialogue = None
        self._prev_item_count = None
