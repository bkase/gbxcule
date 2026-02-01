"""Parcel detector tests (Pokemon Red)."""

from __future__ import annotations

from pathlib import Path

from gbxcule.backends.warp_vec import WarpVecCpuBackend
from gbxcule.rl.pokered_parcel_detectors import (
    EVENTS_LENGTH,
    EVENTS_START,
    delivered_parcel,
    has_parcel,
)

ROM_PATH = Path(__file__).parent.parent / "red.gb"
STATE_DIR = Path(__file__).parent.parent / "states" / "rl_oak_parcel"
STATE_START = STATE_DIR / "start.state"
STATE_AFTER_MART = STATE_DIR / "after_mart.state"
STATE_AFTER_DELIVER = STATE_DIR / "after_deliver.state"


def _detect(state_path: Path) -> tuple[bool, bool]:
    assert ROM_PATH.exists(), f"ROM not found: {ROM_PATH}"
    assert state_path.exists(), f"State not found: {state_path}"

    backend = WarpVecCpuBackend(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        stage="emulate_only",
    )
    try:
        backend.reset()
        backend.load_state_file(str(state_path), env_idx=0)
        mem = backend.memory_torch()
        events = mem[:, EVENTS_START : EVENTS_START + EVENTS_LENGTH]
        has = has_parcel(mem, events)
        delivered = delivered_parcel(mem, events)
        return bool(has[0].item()), bool(delivered[0].item())
    finally:
        backend.close()


def test_start_state_has_no_parcel() -> None:
    has, delivered = _detect(STATE_START)
    assert has is False
    assert delivered is False


def test_after_mart_state_has_parcel() -> None:
    has, delivered = _detect(STATE_AFTER_MART)
    assert has is True
    assert delivered is False


def test_after_deliver_state_delivered() -> None:
    has, delivered = _detect(STATE_AFTER_DELIVER)
    assert has is False
    assert delivered is True
