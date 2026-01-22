"""Multi-environment PyBoy backend using multiprocessing.

This backend provides a CPU-parallel baseline for benchmarking by running
multiple PyBoy instances across worker processes.
"""

from __future__ import annotations

import contextlib
import hashlib
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np

from gbxcule.backends.common import (
    ArraySpec,
    CpuState,
    Device,
    NDArrayBool,
    NDArrayF32,
    NDArrayI32,
    action_to_button,
    as_i32_actions,
    empty_obs,
    flags_from_f,
    get_pyboy_class,
)
from gbxcule.core.signatures import hash64

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

BOOTROM_PATH = Path("bench/roms/bootrom_fast_dmg.bin")


@dataclass(frozen=True)
class PyBoyMpConfig:
    """Configuration for the multiprocessing PyBoy backend.

    Attributes:
        num_envs: Total number of environments.
        num_workers: Number of worker processes.
        frames_per_step: Frames to advance per step call.
        release_after_frames: Frames after which to release button.
        rom_path: Path to the ROM file.
        base_seed: Optional base seed for deterministic seeding.
        headless: Whether to run headless (always True for now).
        obs_dim: Observation vector dimension.
    """

    num_envs: int
    num_workers: int
    frames_per_step: int
    release_after_frames: int
    rom_path: str
    base_seed: int | None = None
    headless: bool = True
    obs_dim: int = 32

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")
        if self.num_workers > self.num_envs:
            raise ValueError(
                f"num_workers ({self.num_workers}) cannot exceed "
                f"num_envs ({self.num_envs})"
            )
        if self.release_after_frames < 0:
            raise ValueError(
                f"release_after_frames must be >= 0, got {self.release_after_frames}"
            )
        if self.release_after_frames > self.frames_per_step:
            raise ValueError(
                f"release_after_frames ({self.release_after_frames}) cannot exceed "
                f"frames_per_step ({self.frames_per_step})"
            )
        if not Path(self.rom_path).exists():
            raise ValueError(f"ROM file not found: {self.rom_path}")


def _compute_rom_sha256(rom_path: str) -> str:
    """Compute SHA-256 hash of a ROM file.

    Args:
        rom_path: Path to the ROM file.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    sha256 = hashlib.sha256()
    with open(rom_path, "rb") as f:
        # Read in chunks for memory efficiency (though ROMs are small)
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# IPC Protocol
# ---------------------------------------------------------------------------


class Command(Enum):
    """Commands sent from master to workers."""

    RESET = auto()
    STEP = auto()
    GET_CPU_STATE = auto()
    READ_MEMORY = auto()
    CLOSE = auto()


class ResponseType(Enum):
    """Response types from workers."""

    OK = auto()
    STATE = auto()
    MEMORY = auto()
    ERR = auto()


@dataclass
class WorkerMessage:
    """Message sent from master to worker."""

    cmd: Command
    data: Any = None


@dataclass
class WorkerResponse:
    """Response from worker to master."""

    type: ResponseType
    data: Any = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Worker process function (top-level for spawn pickling)
# ---------------------------------------------------------------------------


def _worker_main(
    conn: Connection,
    worker_id: int,
    env_indices: list[int],
    rom_path: str,
    frames_per_step: int,
    release_after_frames: int,
) -> None:
    """Worker process main loop.

    Args:
        conn: Connection to master process.
        worker_id: Worker identifier.
        env_indices: List of environment indices this worker owns.
        rom_path: Path to the ROM file.
        frames_per_step: Frames per step.
        release_after_frames: Frames before button release.
    """
    PyBoy = get_pyboy_class()
    if not BOOTROM_PATH.exists():
        raise FileNotFoundError(
            f"Boot ROM not found: {BOOTROM_PATH}. Expected repo-local fast boot ROM."
        )

    pyboys: list[Any] = []

    try:
        while True:
            msg: WorkerMessage = conn.recv()

            if msg.cmd == Command.RESET:
                # Close existing instances
                for pb in pyboys:
                    pb.stop(save=False)
                pyboys.clear()

                # Create fresh PyBoy instances
                for _ in env_indices:
                    pb = PyBoy(
                        rom_path,
                        window="null",
                        sound_emulated=False,
                        bootrom=str(BOOTROM_PATH),
                    )
                    pb.set_emulation_speed(0)
                    pyboys.append(pb)

                conn.send(WorkerResponse(ResponseType.OK))

            elif msg.cmd == Command.STEP:
                actions: list[int] = msg.data

                if len(actions) != len(env_indices):
                    conn.send(
                        WorkerResponse(
                            ResponseType.ERR,
                            error=f"Expected {len(env_indices)} actions, "
                            f"got {len(actions)}",
                        )
                    )
                    continue

                # Step each environment
                for pb, action in zip(pyboys, actions, strict=True):
                    button = action_to_button(action)

                    if button is not None:
                        pb.button_press(button)
                        for _ in range(release_after_frames):
                            pb.tick(render=False)
                        pb.button_release(button)
                        remaining = frames_per_step - release_after_frames
                        for _ in range(remaining):
                            pb.tick(render=False)
                    else:
                        for _ in range(frames_per_step):
                            pb.tick(render=False)

                conn.send(WorkerResponse(ResponseType.OK))

            elif msg.cmd == Command.GET_CPU_STATE:
                local_idx: int = msg.data

                if local_idx < 0 or local_idx >= len(pyboys):
                    conn.send(
                        WorkerResponse(
                            ResponseType.ERR,
                            error=f"Invalid local_idx {local_idx}, "
                            f"worker has {len(pyboys)} envs",
                        )
                    )
                    continue

                pb = pyboys[local_idx]
                reg = pb.register_file

                pc = int(reg.PC)
                sp = int(reg.SP)
                a = int(reg.A)
                f = int(reg.F)
                b = int(reg.B)
                c = int(reg.C)
                d = int(reg.D)
                e = int(reg.E)
                hl = int(reg.HL)
                h = (hl >> 8) & 0xFF
                l_val = hl & 0xFF

                state = CpuState(
                    pc=pc,
                    sp=sp,
                    a=a,
                    f=f,
                    b=b,
                    c=c,
                    d=d,
                    e=e,
                    h=h,
                    l=l_val,
                    flags=flags_from_f(f),
                    instr_count=None,
                    cycle_count=None,
                )
                conn.send(WorkerResponse(ResponseType.STATE, data=state))

            elif msg.cmd == Command.READ_MEMORY:
                local_idx, lo, hi = msg.data

                if local_idx < 0 or local_idx >= len(pyboys):
                    conn.send(
                        WorkerResponse(
                            ResponseType.ERR,
                            error=f"Invalid local_idx {local_idx}, "
                            f"worker has {len(pyboys)} envs",
                        )
                    )
                    continue
                if lo < 0 or hi > 0x10000 or lo >= hi:
                    conn.send(
                        WorkerResponse(
                            ResponseType.ERR,
                            error=f"Invalid memory range {lo:#06x}:{hi:#06x}",
                        )
                    )
                    continue

                pb = pyboys[local_idx]
                mem_slice = bytes(pb.memory[lo:hi])
                conn.send(WorkerResponse(ResponseType.MEMORY, data=mem_slice))

            elif msg.cmd == Command.CLOSE:
                for pb in pyboys:
                    pb.stop(save=False)
                pyboys.clear()
                conn.send(WorkerResponse(ResponseType.OK))
                break

    except Exception as exc:
        # Send error back to master
        with contextlib.suppress(Exception):
            conn.send(
                WorkerResponse(ResponseType.ERR, error=f"{type(exc).__name__}: {exc}")
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main backend class
# ---------------------------------------------------------------------------


class PyBoyVecMpBackend:
    """Multi-environment PyBoy backend using multiprocessing.

    This backend runs multiple PyBoy instances across worker processes,
    providing a CPU-parallel baseline for benchmarking.

    Attributes:
        name: Backend identifier ("pyboy_vec_mp").
        device: Device type ("cpu").
        num_envs: Number of parallel environments.
        action_spec: Action array specification.
        obs_spec: Observation array specification.
    """

    name: str = "pyboy_vec_mp"
    device: Device = "cpu"

    def __init__(
        self,
        rom_path: str,
        *,
        num_envs: int = 360,
        num_workers: int | None = 20,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        obs_dim: int = 32,
        base_seed: int | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            rom_path: Path to the ROM file.
            num_envs: Number of environments (default: 360).
            num_workers: Number of worker processes (default: 20).
            frames_per_step: Frames to advance per step call.
            release_after_frames: Frames after which to release button.
            obs_dim: Observation vector dimension.
            base_seed: Optional base seed for reproducibility.
        """
        if num_workers is None:
            num_workers = num_envs

        self._config = PyBoyMpConfig(
            num_envs=num_envs,
            num_workers=num_workers,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            rom_path=rom_path,
            base_seed=base_seed,
            obs_dim=obs_dim,
        )

        self.num_envs = num_envs
        self._obs_dim = obs_dim

        # Compute ROM SHA for deterministic seeding
        self._rom_sha = _compute_rom_sha256(rom_path)

        # Specs
        self.action_spec = ArraySpec(
            shape=(self.num_envs,),
            dtype="int32",
            meaning="action index [0, 9)",
        )
        self.obs_spec = ArraySpec(
            shape=(self.num_envs, obs_dim),
            dtype="float32",
            meaning="normalized register features",
        )

        # Worker state
        self._workers: list[mp.Process] = []
        self._conns: list[Connection] = []
        self._env_partition: list[list[int]] = []
        self._initialized = False

        # Derived seeds (populated on reset)
        self._derived_seeds: list[int] = []

    def __del__(self) -> None:
        """Ensure workers are cleaned up on garbage collection."""
        self.close()

    def _partition_envs(self) -> list[list[int]]:
        """Partition environment indices across workers."""
        num_envs = self._config.num_envs
        num_workers = self._config.num_workers

        partition: list[list[int]] = [[] for _ in range(num_workers)]
        for env_idx in range(num_envs):
            worker_idx = env_idx % num_workers
            partition[worker_idx].append(env_idx)

        return partition

    def _start_workers(self) -> None:
        """Start worker processes."""
        self._env_partition = self._partition_envs()

        for worker_id, env_indices in enumerate(self._env_partition):
            parent_conn, child_conn = mp.Pipe()

            proc = mp.Process(
                target=_worker_main,
                args=(
                    child_conn,
                    worker_id,
                    env_indices,
                    self._config.rom_path,
                    self._config.frames_per_step,
                    self._config.release_after_frames,
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # Close child end in parent

            self._workers.append(proc)
            self._conns.append(parent_conn)

    def _send_all(self, msg: WorkerMessage) -> None:
        """Send message to all workers."""
        for conn in self._conns:
            conn.send(msg)

    def _recv_all(self) -> list[WorkerResponse]:
        """Receive responses from all workers.

        Raises:
            RuntimeError: If a worker has died (broken pipe/EOF).
        """
        responses = []
        for i, conn in enumerate(self._conns):
            try:
                if not conn.poll(timeout=30.0):  # 30 second timeout
                    self.close()
                    raise RuntimeError(f"Worker {i} timed out waiting for response")
                resp = conn.recv()
                responses.append(resp)
            except EOFError:
                self.close()
                raise RuntimeError(
                    f"Worker {i} died unexpectedly (pipe closed)"
                ) from None
            except BrokenPipeError:
                self.close()
                raise RuntimeError(f"Worker {i} connection broken") from None
        return responses

    def _check_responses(self, responses: list[WorkerResponse]) -> None:
        """Check for errors in worker responses.

        Raises:
            RuntimeError: If any worker reported an error.
        """
        for i, resp in enumerate(responses):
            if resp.type == ResponseType.ERR:
                self.close()
                raise RuntimeError(f"Worker {i} error: {resp.error}")

    def reset(self, seed: int | None = None) -> tuple[NDArrayF32, dict[str, Any]]:
        """Reset all environments.

        Args:
            seed: Optional random seed (recorded for reproducibility).

        Returns:
            Tuple of (observations, info_dict).
        """
        # Start workers if not started
        if not self._initialized:
            self._start_workers()
            self._initialized = True

        # Send reset to all workers
        self._send_all(WorkerMessage(Command.RESET, data=seed))
        responses = self._recv_all()
        self._check_responses(responses)

        # Compute derived seeds for each environment
        # Use the reset seed if provided, otherwise fall back to config base_seed
        effective_seed = seed if seed is not None else self._config.base_seed
        if effective_seed is not None:
            self._derived_seeds = [
                hash64(effective_seed, env_idx, self._rom_sha)
                for env_idx in range(self.num_envs)
            ]
        else:
            # No seed provided - clear derived seeds
            self._derived_seeds = []

        # Build observation (zeros for M0)
        obs = empty_obs(self.num_envs, self._obs_dim)

        info: dict[str, Any] = {
            "seed": seed,
            "base_seed": self._config.base_seed,
            "rom_sha256": self._rom_sha,
            "derived_seeds": self._derived_seeds if self._derived_seeds else None,
        }
        return obs, info

    def step(
        self, actions: NDArrayI32
    ) -> tuple[NDArrayF32, NDArrayF32, NDArrayBool, NDArrayBool, dict[str, Any]]:
        """Step all environments forward.

        Args:
            actions: Action array, shape (num_envs,), dtype int32.

        Returns:
            Tuple of (obs, reward, done, trunc, info).
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        # Validate and cast actions
        actions = as_i32_actions(actions, self.num_envs)

        # Partition actions across workers
        for worker_idx, env_indices in enumerate(self._env_partition):
            worker_actions = [int(actions[env_idx]) for env_idx in env_indices]
            self._conns[worker_idx].send(
                WorkerMessage(Command.STEP, data=worker_actions)
            )

        # Receive responses
        responses = self._recv_all()
        self._check_responses(responses)

        # Build outputs (zeros for M0)
        obs = empty_obs(self.num_envs, self._obs_dim)
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.zeros((self.num_envs,), dtype=np.bool_)
        trunc = np.zeros((self.num_envs,), dtype=np.bool_)
        info: dict[str, Any] = {}

        return obs, reward, done, trunc, info

    def get_cpu_state(self, env_idx: int) -> CpuState:
        """Get CPU register state for a specific environment.

        Args:
            env_idx: Environment index.

        Returns:
            CpuState dictionary with register values and flags.

        Raises:
            ValueError: If env_idx is out of range.
            RuntimeError: If backend not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")

        # Find owning worker and local index
        for worker_idx, env_indices in enumerate(self._env_partition):
            if env_idx in env_indices:
                local_idx = env_indices.index(env_idx)
                self._conns[worker_idx].send(
                    WorkerMessage(Command.GET_CPU_STATE, data=local_idx)
                )
                resp = self._conns[worker_idx].recv()

                if resp.type == ResponseType.ERR:
                    raise RuntimeError(f"Worker error: {resp.error}")

                return resp.data

        # Should not reach here
        raise RuntimeError(f"Could not find worker for env_idx {env_idx}")

    def read_memory(self, env_idx: int, lo: int, hi: int) -> bytes:
        """Read a slice of memory for a specific environment.

        Args:
            env_idx: Environment index.
            lo: Lower address (inclusive).
            hi: Upper address (exclusive).

        Returns:
            Bytes for the requested memory slice.
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if lo < 0 or hi > 0x10000 or lo >= hi:
            raise ValueError(f"Invalid memory range: {lo:#06x}:{hi:#06x}")

        # Find owning worker and local index
        for worker_idx, env_indices in enumerate(self._env_partition):
            if env_idx in env_indices:
                local_idx = env_indices.index(env_idx)
                self._conns[worker_idx].send(
                    WorkerMessage(Command.READ_MEMORY, data=(local_idx, lo, hi))
                )
                resp = self._conns[worker_idx].recv()

                if resp.type == ResponseType.ERR:
                    raise RuntimeError(f"Worker error: {resp.error}")

                if resp.type != ResponseType.MEMORY:
                    raise RuntimeError(
                        f"Unexpected response type: {resp.type} (expected MEMORY)"
                    )

                return bytes(resp.data)

        # Should not reach here
        raise RuntimeError(f"Could not find worker for env_idx {env_idx}")

    def close(self) -> None:
        """Clean up resources and terminate workers."""
        if not self._initialized:
            return

        # Send close to all workers
        for conn in self._conns:
            with contextlib.suppress(Exception):
                conn.send(WorkerMessage(Command.CLOSE))

        # Wait for workers with timeout
        for proc in self._workers:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)

        # Close connections
        for conn in self._conns:
            with contextlib.suppress(Exception):
                conn.close()

        self._workers.clear()
        self._conns.clear()
        self._initialized = False
