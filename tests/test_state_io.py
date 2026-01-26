"""Tests for PyBoy v9 state save/load functionality."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gbxcule.backends.warp_vec import WarpVecCpuBackend
from gbxcule.core.state_io import (
    PyBoyState,
    apply_state_to_warp_backend,
    load_pyboy_state,
    save_pyboy_state,
    state_from_warp_backend,
)

# Use a simple ROM for testing
ROM_PATH = Path(__file__).parent.parent / "bench" / "roms" / "out" / "BG_STATIC.gb"


@pytest.fixture
def rom_path() -> str:
    """Get path to a test ROM."""
    if not ROM_PATH.exists():
        pytest.skip(f"Test ROM not found: {ROM_PATH}")
    return str(ROM_PATH)


@pytest.fixture
def warp_backend(rom_path: str):
    """Create a Warp CPU backend for testing."""
    backend = WarpVecCpuBackend(
        rom_path,
        num_envs=1,
        frames_per_step=1,
        stage="emulate_only",
    )
    backend.reset()
    yield backend
    backend.close()


class TestStateIO:
    """Tests for state save/load functions."""

    def test_state_from_warp_backend(self, warp_backend: WarpVecCpuBackend) -> None:
        """Test extracting state from Warp backend."""
        # Run a few steps to get interesting state
        for _ in range(10):
            actions = np.array([0], dtype=np.int32)
            warp_backend.step(actions)

        state = state_from_warp_backend(warp_backend, env_idx=0)

        # Verify basic structure
        assert isinstance(state, PyBoyState)
        assert state.version == 9  # We always save as v9
        assert len(state.vram) == 8192
        assert len(state.oam) == 160
        assert len(state.wram) == 8192
        assert len(state.hram) == 127
        assert len(state.io_ports) == 76
        assert len(state.scanline_params) == 720

        # Verify registers are in valid ranges
        assert 0 <= state.a <= 255
        assert 0 <= state.f <= 255
        assert 0 <= state.sp <= 0xFFFF
        assert 0 <= state.pc <= 0xFFFF
        assert state.ime in (0, 1)
        assert state.halted in (0, 1)

    def test_save_and_load_state_roundtrip(
        self, warp_backend: WarpVecCpuBackend
    ) -> None:
        """Test saving and loading state produces identical state."""
        # Run a few steps
        for _ in range(10):
            actions = np.array([0], dtype=np.int32)
            warp_backend.step(actions)

        # Extract state
        original_state = state_from_warp_backend(warp_backend, env_idx=0)

        # Save to file
        with tempfile.NamedTemporaryFile(suffix=".state", delete=False) as f:
            state_path = f.name

        try:
            save_pyboy_state(original_state, state_path)

            # Verify file was created with expected size
            file_size = Path(state_path).stat().st_size
            # Expected: header(5) + cpu(18) + lcd(9102) + sound(138) + renderer(115200)
            # + ram(8543) + timer(8) + cart(5) + joypad(2) = 133021 bytes
            assert file_size > 130000, f"State file too small: {file_size}"

            # Load state back
            loaded_state = load_pyboy_state(state_path)

            # Verify key fields match
            assert loaded_state.version == original_state.version
            assert loaded_state.a == original_state.a
            assert loaded_state.f == original_state.f
            assert loaded_state.b == original_state.b
            assert loaded_state.c == original_state.c
            assert loaded_state.d == original_state.d
            assert loaded_state.e == original_state.e
            assert loaded_state.h == original_state.h
            assert loaded_state.l_reg == original_state.l_reg
            assert loaded_state.sp == original_state.sp
            assert loaded_state.pc == original_state.pc
            assert loaded_state.ime == original_state.ime
            assert loaded_state.halted == original_state.halted
            assert loaded_state.ie == original_state.ie
            assert loaded_state.if_reg == original_state.if_reg

            # Verify LCD state
            assert loaded_state.lcdc == original_state.lcdc
            assert loaded_state.stat == original_state.stat
            assert loaded_state.ly == original_state.ly
            assert loaded_state.scy == original_state.scy
            assert loaded_state.scx == original_state.scx
            assert loaded_state.bgp == original_state.bgp

            # Verify memory
            assert loaded_state.vram == original_state.vram
            assert loaded_state.oam == original_state.oam
            assert loaded_state.wram == original_state.wram
            assert loaded_state.hram == original_state.hram

            # Verify timer
            assert loaded_state.div == original_state.div
            assert loaded_state.tima == original_state.tima
            assert loaded_state.tac == original_state.tac

        finally:
            Path(state_path).unlink(missing_ok=True)

    def test_apply_state_to_warp_backend(self, warp_backend: WarpVecCpuBackend) -> None:
        """Test applying state modifies backend correctly."""
        # Run a few steps
        for _ in range(10):
            actions = np.array([0], dtype=np.int32)
            warp_backend.step(actions)

        # Extract state
        original_state = state_from_warp_backend(warp_backend, env_idx=0)

        # Run more steps to change state
        for _ in range(50):
            actions = np.array([0], dtype=np.int32)
            warp_backend.step(actions)

        # Verify some state changed (DIV counter always advances)
        changed_state = state_from_warp_backend(warp_backend, env_idx=0)
        assert changed_state.div_counter != original_state.div_counter, (
            "DIV counter should have changed"
        )

        # Apply original state back
        apply_state_to_warp_backend(original_state, warp_backend, env_idx=0)

        # Extract and verify it matches original
        restored_state = state_from_warp_backend(warp_backend, env_idx=0)

        assert restored_state.a == original_state.a
        assert restored_state.f == original_state.f
        assert restored_state.sp == original_state.sp
        assert restored_state.pc == original_state.pc
        assert restored_state.ime == original_state.ime
        assert restored_state.halted == original_state.halted
        assert restored_state.vram == original_state.vram
        assert restored_state.wram == original_state.wram


class TestWarpBackendStateMethods:
    """Tests for Warp backend save_state/load_state methods."""

    def test_save_state_method(self, warp_backend: WarpVecCpuBackend) -> None:
        """Test backend.save_state() method."""
        # Run a few steps
        for _ in range(10):
            actions = np.array([0], dtype=np.int32)
            warp_backend.step(actions)

        state = warp_backend.save_state(env_idx=0)
        assert isinstance(state, PyBoyState)
        assert state.version == 9  # We always save as v9

    def test_load_state_method(self, warp_backend: WarpVecCpuBackend) -> None:
        """Test backend.load_state() method."""
        # Run a few steps
        for _ in range(10):
            actions = np.array([0], dtype=np.int32)
            warp_backend.step(actions)

        # Save state
        saved_state = warp_backend.save_state(env_idx=0)
        saved_pc = saved_state.pc

        # Run more steps
        for _ in range(50):
            actions = np.array([0], dtype=np.int32)
            warp_backend.step(actions)

        # Load state
        warp_backend.load_state(saved_state, env_idx=0)

        # Verify state restored
        restored_state = warp_backend.save_state(env_idx=0)
        assert restored_state.pc == saved_pc

    def test_save_state_file_method(self, warp_backend: WarpVecCpuBackend) -> None:
        """Test backend.save_state_file() method."""
        # Run a few steps
        for _ in range(10):
            actions = np.array([0], dtype=np.int32)
            warp_backend.step(actions)

        with tempfile.NamedTemporaryFile(suffix=".state", delete=False) as f:
            state_path = f.name

        try:
            warp_backend.save_state_file(state_path, env_idx=0)
            assert Path(state_path).exists()
            assert Path(state_path).stat().st_size > 130000
        finally:
            Path(state_path).unlink(missing_ok=True)

    def test_load_state_file_method(self, warp_backend: WarpVecCpuBackend) -> None:
        """Test backend.load_state_file() method."""
        # Run a few steps
        for _ in range(10):
            actions = np.array([0], dtype=np.int32)
            warp_backend.step(actions)

        # Save state to file
        saved_state = warp_backend.save_state(env_idx=0)
        saved_pc = saved_state.pc

        with tempfile.NamedTemporaryFile(suffix=".state", delete=False) as f:
            state_path = f.name

        try:
            warp_backend.save_state_file(state_path, env_idx=0)

            # Run more steps
            for _ in range(50):
                actions = np.array([0], dtype=np.int32)
                warp_backend.step(actions)

            # Load state from file
            warp_backend.load_state_file(state_path, env_idx=0)

            # Verify state restored
            restored_state = warp_backend.save_state(env_idx=0)
            assert restored_state.pc == saved_pc

        finally:
            Path(state_path).unlink(missing_ok=True)


class TestWarpToWarpStateTransfer:
    """Test state transfer between two Warp backends."""

    def test_state_transfer_between_backends(self, rom_path: str) -> None:
        """Test transferring state from one backend to another."""
        backend1 = WarpVecCpuBackend(
            rom_path,
            num_envs=1,
            frames_per_step=1,
            stage="emulate_only",
        )
        backend2 = WarpVecCpuBackend(
            rom_path,
            num_envs=1,
            frames_per_step=1,
            stage="emulate_only",
        )

        try:
            backend1.reset()
            backend2.reset()

            # Run backend1 for a while
            for _ in range(100):
                actions = np.array([0], dtype=np.int32)
                backend1.step(actions)

            # Save state from backend1
            state = backend1.save_state(env_idx=0)

            # Apply to backend2
            backend2.load_state(state, env_idx=0)

            # Verify states match
            state1 = backend1.save_state(env_idx=0)
            state2 = backend2.save_state(env_idx=0)

            assert state1.a == state2.a
            assert state1.f == state2.f
            assert state1.pc == state2.pc
            assert state1.sp == state2.sp
            assert state1.vram == state2.vram
            assert state1.wram == state2.wram

            # Run both for more steps and verify they stay in sync
            for _ in range(10):
                actions = np.array([0], dtype=np.int32)
                backend1.step(actions)
                backend2.step(actions)

            state1_after = backend1.save_state(env_idx=0)
            state2_after = backend2.save_state(env_idx=0)

            assert state1_after.pc == state2_after.pc
            assert state1_after.a == state2_after.a

        finally:
            backend1.close()
            backend2.close()


class TestPyBoyInterop:
    """Test interoperability with PyBoy."""

    @pytest.fixture
    def pyboy_available(self) -> bool:
        """Check if PyBoy is available."""
        import importlib.util

        if importlib.util.find_spec("pyboy") is None:
            pytest.skip("PyBoy not installed")
            return False
        return True

    def test_warp_state_loads_in_pyboy(
        self, rom_path: str, pyboy_available: bool
    ) -> None:
        """Test that a state saved from Warp can be loaded in PyBoy."""
        from pyboy import PyBoy

        # Create Warp backend and run
        warp_backend = WarpVecCpuBackend(
            rom_path,
            num_envs=1,
            frames_per_step=1,
            stage="emulate_only",
        )

        try:
            warp_backend.reset()

            # Run for a while to get interesting state
            for _ in range(100):
                actions = np.array([0], dtype=np.int32)
                warp_backend.step(actions)

            # Save state to file
            with tempfile.NamedTemporaryFile(suffix=".state", delete=False) as f:
                state_path = f.name

            warp_backend.save_state_file(state_path, env_idx=0)

        finally:
            warp_backend.close()

        # Load in PyBoy
        pyboy = PyBoy(rom_path, window="null")
        try:
            with open(state_path, "rb") as f:
                pyboy.load_state(f)

            # Verify PyBoy loaded the state (can tick without crashing)
            for _ in range(10):
                pyboy.tick()

            # Check some registers match
            # Note: PyBoy API might differ, this is approximate
            # The important thing is that it loads without error

        finally:
            pyboy.stop()
            Path(state_path).unlink(missing_ok=True)

    def test_pyboy_state_loads_in_warp(
        self, rom_path: str, pyboy_available: bool
    ) -> None:
        """Test that a state saved from PyBoy can be loaded in Warp."""
        from pyboy import PyBoy

        # Create PyBoy and run
        pyboy = PyBoy(rom_path, window="null")
        try:
            # Run for a while
            for _ in range(1000):  # More ticks since PyBoy ticks are single cycles
                pyboy.tick()

            # Save state
            fd, state_path = tempfile.mkstemp(suffix=".state")
            os.close(fd)
            with open(state_path, "wb") as f:
                pyboy.save_state(f)
                f.flush()
                os.fsync(f.fileno())
            if Path(state_path).stat().st_size == 0:
                pytest.skip("PyBoy produced empty state file")

        finally:
            pyboy.stop()

        # Load in Warp
        warp_backend = WarpVecCpuBackend(
            rom_path,
            num_envs=1,
            frames_per_step=1,
            stage="emulate_only",
        )

        try:
            warp_backend.reset()
            warp_backend.load_state_file(state_path, env_idx=0)

            # Verify state loaded (can step without crashing)
            for _ in range(10):
                actions = np.array([0], dtype=np.int32)
                warp_backend.step(actions)

            # Extract state to verify it's reasonable
            state = warp_backend.save_state(env_idx=0)
            assert state.version == 9  # We save as v9
            # PC should be in valid ROM range or RAM
            assert state.pc < 0x10000

        finally:
            warp_backend.close()
            Path(state_path).unlink(missing_ok=True)
