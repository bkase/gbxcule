"""JOY_DIVERGE_PERSIST verification and divergence tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bench.harness import diff_states, hash_memory, normalize_cpu_state
from bench.roms.build_micro_rom import build_joy_diverge_persist
from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCpuBackend
from gbxcule.core.action_codec import POKERED_PUFFER_V1_ID, get_action_codec


def _write_rom(tmp_path: Path) -> str:
    rom_path = tmp_path / "JOY_DIVERGE_PERSIST.gb"
    rom_path.write_bytes(build_joy_diverge_persist())
    return str(rom_path)


def test_verify_joy_diverge_persist_memory_and_state(
    tmp_path: Path,
) -> None:
    rom_path = _write_rom(tmp_path)
    codec = get_action_codec(POKERED_PUFFER_V1_ID)
    action_names = list(codec.action_names)
    actions_cycle = [
        action_names.index("UP"),
        action_names.index("DOWN"),
        action_names.index("LEFT"),
        action_names.index("RIGHT"),
    ]
    ref = PyBoySingleBackend(
        rom_path,
        frames_per_step=24,
        release_after_frames=8,
        obs_dim=32,
    )
    dut = WarpVecCpuBackend(
        rom_path,
        frames_per_step=24,
        release_after_frames=8,
        obs_dim=32,
    )
    try:
        ref.reset()
        dut.reset()
        for step_idx in range(16):
            action = actions_cycle[step_idx % len(actions_cycle)]
            actions = np.array([action], dtype=np.int32)
            ref.step(actions)
            dut.step(actions)
            ref_state = normalize_cpu_state(ref.get_cpu_state(0))
            dut_state = normalize_cpu_state(dut.get_cpu_state(0))
            diff = diff_states(ref_state, dut_state)
            assert diff is None, f"Mismatch at step {step_idx}: {diff}"
            ref_mem = ref.read_memory(0, 0xC000, 0xC010)
            dut_mem = dut.read_memory(0, 0xC000, 0xC010)
            assert hash_memory(ref_mem) == hash_memory(dut_mem)
    finally:
        ref.close()
        dut.close()


def test_warp_vec_cpu_diverges_across_envs(tmp_path: Path) -> None:
    rom_path = _write_rom(tmp_path)
    codec = get_action_codec(POKERED_PUFFER_V1_ID)
    action_names = list(codec.action_names)
    actions = np.array(
        [
            action_names.index("UP"),
            action_names.index("DOWN"),
            action_names.index("LEFT"),
            action_names.index("RIGHT"),
        ],
        dtype=np.int32,
    )
    backend = WarpVecCpuBackend(
        rom_path,
        num_envs=4,
        frames_per_step=24,
        release_after_frames=8,
        obs_dim=32,
    )
    try:
        backend.reset()
        for _ in range(8):
            backend.step(actions)
        hashes = []
        for env_idx in range(4):
            mem = backend.read_memory(env_idx, 0xC000, 0xC010)
            hashes.append(hash_memory(mem))
        assert len(set(hashes)) > 1, "Expected divergence across envs"
    finally:
        backend.close()
