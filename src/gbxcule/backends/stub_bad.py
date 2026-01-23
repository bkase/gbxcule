"""Intentionally incorrect backend (for mismatch bundle smoke tests)."""

from __future__ import annotations

from typing import Any

import numpy as np

from gbxcule.backends.common import (
    DEFAULT_ACTION_CODEC_ID,
    ArraySpec,
    CpuState,
    Device,
    NDArrayBool,
    NDArrayF32,
    NDArrayI32,
    action_codec_spec,
    as_i32_actions,
    empty_obs,
    flags_from_f,
    resolve_action_codec,
)


class StubBadBackend:
    """A deterministic backend that always disagrees with the oracle.

    This is used to exercise verify-mode mismatch bundle creation in CI/hooks
    without relying on a real backend being broken.
    """

    name: str = "stub_bad"
    device: Device = "cpu"
    num_envs: int = 1

    def __init__(self, rom_path: str, *, obs_dim: int = 32) -> None:
        self._rom_path = rom_path
        self._obs_dim = obs_dim
        self._step_count = 0
        self._action_codec = resolve_action_codec(DEFAULT_ACTION_CODEC_ID)
        self.action_codec = action_codec_spec(DEFAULT_ACTION_CODEC_ID)
        self.num_actions = self._action_codec.num_actions

        self.action_spec = ArraySpec(
            shape=(self.num_envs,),
            dtype="int32",
            meaning="Action index per environment",
        )
        self.obs_spec = ArraySpec(
            shape=(self.num_envs, obs_dim),
            dtype="float32",
            meaning="Observation vector per environment",
        )

    def reset(self, seed: int | None = None) -> tuple[NDArrayF32, dict[str, Any]]:
        self._step_count = 0
        return empty_obs(self.num_envs, self._obs_dim), {"seed": seed}

    def step(
        self, actions: NDArrayI32
    ) -> tuple[NDArrayF32, NDArrayF32, NDArrayBool, NDArrayBool, dict[str, Any]]:
        actions = as_i32_actions(actions, self.num_envs)
        invalid = (actions < 0) | (actions >= self.num_actions)
        if np.any(invalid):
            bad = int(actions[invalid][0])
            raise ValueError(f"Action {bad} out of range [0, {self.num_actions})")

        self._step_count += 1
        obs = empty_obs(self.num_envs, self._obs_dim)
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.zeros((self.num_envs,), dtype=np.bool_)
        trunc = np.zeros((self.num_envs,), dtype=np.bool_)
        return obs, reward, done, trunc, {}

    def get_cpu_state(self, env_idx: int) -> CpuState:
        if env_idx != 0:
            raise ValueError(f"env_idx must be 0 for {self.name}, got {env_idx}")
        # Deliberately incorrect state (pc=0 etc). Keep schema stable.
        f = 0
        return CpuState(
            pc=0,
            sp=0,
            a=0,
            f=f,
            b=0,
            c=0,
            d=0,
            e=0,
            h=0,
            l=0,
            flags=flags_from_f(f),
            instr_count=self._step_count,
            cycle_count=self._step_count,
        )

    def read_memory(self, env_idx: int, lo: int, hi: int) -> bytes:
        if env_idx != 0:
            raise ValueError(f"env_idx must be 0 for {self.name}, got {env_idx}")
        if lo < 0 or hi > 0x10000 or lo >= hi:
            raise ValueError(f"Invalid memory range: {lo:#06x}:{hi:#06x}")
        return bytes([0] * (hi - lo))

    def close(self) -> None:
        return
