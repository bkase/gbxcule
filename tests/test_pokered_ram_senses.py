"""RAM sense observation addresses (Pokemon Red) match PyBoy state."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gbxcule.backends.warp_vec import WarpVecCpuBackend, WarpVecCudaBackend

ROM_PATH = Path(__file__).parent.parent / "red.gb"
STATE_PATH = Path(__file__).parent.parent / "states" / "gary-fought.state"

# Addresses from pokemonred_puffer (wCurMap, wYCoord, wXCoord).
MAP_ID_ADDR = 0xD35E
PLAYER_Y_ADDR = 0xD361
PLAYER_X_ADDR = 0xD362

# Event flags in RAM (pokemonred_puffer/data/events.py).
EVENTS_START = 0xD747
EVENTS_LENGTH = 320


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    wp = pytest.importorskip("warp")
    wp.init()
    return wp.is_cuda_available()


def _expected_from_pyboy() -> tuple[int, int, int, bytes]:
    import importlib.util

    if importlib.util.find_spec("pyboy") is None:
        pytest.skip("PyBoy not installed")

    from pyboy import PyBoy

    pyboy = PyBoy(str(ROM_PATH), window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)
    try:
        with open(STATE_PATH, "rb") as f:
            pyboy.load_state(f)
        mem = pyboy.memory
        map_id = mem[MAP_ID_ADDR]
        y_pos = mem[PLAYER_Y_ADDR]
        x_pos = mem[PLAYER_X_ADDR]
        events = bytes(mem[EVENTS_START : EVENTS_START + EVENTS_LENGTH])
    finally:
        pyboy.stop(save=False)
    return map_id, y_pos, x_pos, events


def _assert_backend_matches_state(backend_cls) -> None:  # type: ignore[no-untyped-def]
    assert ROM_PATH.exists(), f"ROM not found: {ROM_PATH}"
    assert STATE_PATH.exists(), f"State not found: {STATE_PATH}"

    expected_map_id, expected_y, expected_x, expected_events = _expected_from_pyboy()

    backend = backend_cls(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        stage="emulate_only",
    )
    try:
        backend.reset()
        backend.load_state_file(str(STATE_PATH), env_idx=0)

        mem = backend.memory_torch()
        map_id = int(mem[0, MAP_ID_ADDR].item())
        y_pos = int(mem[0, PLAYER_Y_ADDR].item())
        x_pos = int(mem[0, PLAYER_X_ADDR].item())
        events_t = mem[0, EVENTS_START : EVENTS_START + EVENTS_LENGTH]
        events = events_t.cpu().numpy().tobytes()

        assert map_id == expected_map_id
        assert y_pos == expected_y
        assert x_pos == expected_x
        assert events == expected_events
    finally:
        backend.close()


def test_ram_senses_match_state_cpu() -> None:
    _assert_backend_matches_state(WarpVecCpuBackend)


def test_ram_senses_match_state_cuda() -> None:
    if not _cuda_available():
        pytest.skip("CUDA disabled for dev runs")
    _assert_backend_matches_state(WarpVecCudaBackend)
