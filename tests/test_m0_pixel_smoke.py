"""Deterministic smoke tests for env0 BG shade frames (m0)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from gbxcule.backends.warp_vec import WarpVecCpuBackend, WarpVecCudaBackend

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "m0_pokered_smoke.json"
GOLDEN_PATH = Path(__file__).parent / "data" / "m0_pokered_frame_hashes.json"

ACTION_MIN = 0
ACTION_MAX = 6


def _blake2b_hex(data: bytes) -> str:
    import hashlib

    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _load_config() -> tuple[dict, Path, Path]:
    assert CONFIG_PATH.exists(), f"Missing config: {CONFIG_PATH}"
    cfg = json.loads(CONFIG_PATH.read_text())
    rom = Path(cfg.get("rom", ""))
    state = Path(cfg.get("state", ""))
    assert rom.exists(), f"Missing ROM: {rom}"
    assert state.exists(), f"Missing state: {state}"
    return cfg, rom, state


def _validate_actions(actions: Iterable[int]) -> list[int]:
    out: list[int] = []
    for action in actions:
        action = int(action)
        if action < ACTION_MIN or action > ACTION_MAX:
            raise ValueError(
                f"Action {action} out of range [{ACTION_MIN}, {ACTION_MAX}]"
            )
        out.append(action)
    if not out:
        raise ValueError("No actions provided")
    return out


def _run_trace(backend, actions: list[int]) -> tuple[list[str], bytes]:
    hashes: list[str] = []
    last_frame = b""
    for action in actions:
        backend.step(np.array([action], dtype=np.int32))
        frame_bytes = backend.read_frame_bg_shade_env0()
        hashes.append(_blake2b_hex(frame_bytes))
        last_frame = frame_bytes
    return hashes, last_frame


def _assert_frame_invariants(frame_bytes: bytes) -> None:
    frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    assert frame.size == 160 * 144
    assert frame.min() >= 0
    assert frame.max() <= 3
    assert np.unique(frame).size >= 2


def test_m0_pixel_smoke_cpu_deterministic() -> None:
    cfg, rom, state = _load_config()
    actions = _validate_actions(cfg.get("actions", []))
    steps = int(cfg.get("steps", len(actions)))
    actions = actions[:steps]

    frames_per_step = int(cfg.get("frames_per_step", 24))
    assert frames_per_step == 24

    backend = WarpVecCpuBackend(
        rom_path=str(rom),
        num_envs=1,
        frames_per_step=frames_per_step,
        release_after_frames=int(cfg.get("release_after_frames", 8)),
        render_bg=True,
        action_codec=str(cfg.get("action_codec", "pokemonred_puffer_v0")),
    )
    try:
        backend.reset(seed=0)
        backend.load_state_file(str(state), env_idx=0)
        hashes_a, last_frame = _run_trace(backend, actions)
        _assert_frame_invariants(last_frame)

        backend.reset(seed=0)
        backend.load_state_file(str(state), env_idx=0)
        hashes_b, _ = _run_trace(backend, actions)
        assert hashes_a == hashes_b

        if GOLDEN_PATH.exists():
            expected = json.loads(GOLDEN_PATH.read_text())
            assert hashes_a == expected
    finally:
        backend.close()


def test_m0_pixel_smoke_cuda_deterministic() -> None:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        pytest.skip("CUDA disabled")
    if os.environ.get("GBXCULE_M0_CUDA_SMOKE") != "1":
        pytest.skip("Set GBXCULE_M0_CUDA_SMOKE=1 to enable CUDA smoke")
    try:
        import warp as wp
    except Exception:
        pytest.skip("Warp not available")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA not available")

    cfg, rom, state = _load_config()
    actions = _validate_actions(cfg.get("actions", []))
    steps = int(cfg.get("steps", len(actions)))
    actions = actions[:steps]

    frames_per_step = int(cfg.get("frames_per_step", 24))
    assert frames_per_step == 24

    def run_once() -> tuple[list[str], bytes]:
        backend = WarpVecCudaBackend(
            rom_path=str(rom),
            num_envs=1,
            frames_per_step=frames_per_step,
            release_after_frames=int(cfg.get("release_after_frames", 8)),
            render_bg=True,
            action_codec=str(cfg.get("action_codec", "pokemonred_puffer_v0")),
        )
        try:
            backend.reset(seed=0)
            backend.load_state_file(str(state), env_idx=0)
            return _run_trace(backend, actions)
        finally:
            backend.close()

    hashes_a, frame_a = run_once()
    _assert_frame_invariants(frame_a)
    hashes_b, _ = run_once()
    assert hashes_a == hashes_b
