#!/usr/bin/env python3
"""GBxCuLE Benchmark Harness CLI.

This module provides a CLI for running benchmarks and verification against
the reference and DUT backends.

Usage:
    uv run python bench/harness.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from gbxcule.backends.common import Stage, VecBackend

from gbxcule.backends.common import DEFAULT_ACTION_CODEC_ID
from gbxcule.core.action_codec import get_action_codec, list_action_codecs

# ---------------------------------------------------------------------------
# Backend Registry
# ---------------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, type] = {}


def register_backends() -> None:
    """Lazily register available backends."""
    global BACKEND_REGISTRY

    if BACKEND_REGISTRY:
        return  # Already registered

    from gbxcule.backends.pyboy_puffer_single import PyBoyPufferSingleBackend
    from gbxcule.backends.pyboy_puffer_vec import PyBoyPufferVecBackend
    from gbxcule.backends.pyboy_single import PyBoySingleBackend
    from gbxcule.backends.pyboy_vec_mp import PyBoyVecMpBackend
    from gbxcule.backends.stub_bad import StubBadBackend
    from gbxcule.backends.warp_vec import (
        WarpVecBackend,
        WarpVecCpuBackend,
        WarpVecCudaBackend,
    )

    BACKEND_REGISTRY["pyboy_single"] = PyBoySingleBackend
    BACKEND_REGISTRY["pyboy_puffer_single"] = PyBoyPufferSingleBackend
    BACKEND_REGISTRY["pyboy_puffer_vec"] = PyBoyPufferVecBackend
    BACKEND_REGISTRY["pyboy_vec_mp"] = PyBoyVecMpBackend
    BACKEND_REGISTRY["stub_bad"] = StubBadBackend
    BACKEND_REGISTRY["warp_vec_cpu"] = WarpVecCpuBackend
    BACKEND_REGISTRY["warp_vec_cuda"] = WarpVecCudaBackend
    BACKEND_REGISTRY["warp_vec"] = WarpVecBackend


def create_backend(
    name: str,
    rom_path: str,
    *,
    num_envs: int = 1,
    num_workers: int | None = None,
    frames_per_step: int = 24,
    release_after_frames: int = 8,
    stage: Stage = "emulate_only",
    obs_dim: int = 32,
    base_seed: int | None = None,
    action_codec: str = DEFAULT_ACTION_CODEC_ID,
    puffer_vec_backend: str = "puffer_mp_sync",
) -> VecBackend:
    """Create a backend instance by name.

    Args:
        name: Backend name (pyboy_single, pyboy_vec_mp).
        rom_path: Path to ROM file.
        num_envs: Number of environments.
        num_workers: Number of workers (MP backend only).
        frames_per_step: Frames per step.
        release_after_frames: Frames before button release.
        stage: Execution stage (warp backends only).
        obs_dim: Observation dimension.
        base_seed: Optional base seed.
        action_codec: Action codec id.

    Returns:
        Configured backend instance.

    Raises:
        ValueError: If backend name is unknown.
    """
    register_backends()

    if name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(BACKEND_REGISTRY.keys())}"
        )

    backend_cls = BACKEND_REGISTRY[name]

    if name == "pyboy_single" or name == "pyboy_puffer_single":
        return backend_cls(
            rom_path,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            obs_dim=obs_dim,
            action_codec=action_codec,
        )
    elif name == "pyboy_puffer_vec":
        return backend_cls(
            rom_path,
            num_envs=num_envs,
            num_workers=num_workers,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            obs_dim=obs_dim,
            action_codec=action_codec,
            vec_backend=puffer_vec_backend,
        )
    elif name == "pyboy_vec_mp":
        return backend_cls(
            rom_path,
            num_envs=num_envs,
            num_workers=num_workers,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            obs_dim=obs_dim,
            base_seed=base_seed,
            action_codec=action_codec,
        )
    elif name == "stub_bad":
        return backend_cls(rom_path, obs_dim=obs_dim, action_codec=action_codec)
    elif name in {"warp_vec", "warp_vec_cpu", "warp_vec_cuda"}:
        return backend_cls(
            rom_path,
            num_envs=num_envs,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            stage=stage,
            obs_dim=obs_dim,
            base_seed=base_seed,
            action_codec=action_codec,
        )
    else:
        # Generic fallback
        return backend_cls(rom_path)


# ---------------------------------------------------------------------------
# Action Generation (Pure)
# ---------------------------------------------------------------------------

# Generator version for reproducibility tracking
ACTION_GEN_VERSION = "1.0"


def _striped_pattern(action_names: list[str] | None, num_actions: int) -> np.ndarray:
    if num_actions < 4:
        raise ValueError("striped generator requires at least 4 actions")

    normalized_names = (
        [str(name).upper() for name in action_names] if action_names else None
    )

    if normalized_names:
        target = ["UP", "DOWN", "LEFT", "RIGHT"]
        indices: list[int] = []
        for name in target:
            if name in normalized_names:
                indices.append(normalized_names.index(name))
        if len(indices) == len(target):
            return np.array(indices, dtype=np.int32)

    candidates: list[int]
    if normalized_names:
        candidates = [
            idx
            for idx, name in enumerate(normalized_names)
            if name not in {"NOOP", "NO-OP", "NONE"}
        ]
    else:
        candidates = list(range(num_actions))

    if len(candidates) < 4:
        candidates = list(range(num_actions))
    if len(candidates) < 4:
        raise ValueError("striped generator requires at least 4 actions")

    return np.array(candidates[:4], dtype=np.int32)


def generate_actions(
    step_idx: int,
    num_envs: int,
    seed: int | None,
    gen_name: str,
    num_actions: int,
    action_names: list[str] | None = None,
) -> np.ndarray:
    """Generate actions for a single step (pure function).

    This function is deterministic given the same inputs. It has no side effects.

    Args:
        step_idx: The current step index (0-based).
        num_envs: Number of environments to generate actions for.
        seed: Base seed for random generation. None means noop.
        gen_name: Generator name ("noop", "seeded_random", or "striped").
        num_actions: Number of available actions.

    Returns:
        int32 array of shape (num_envs,) with action indices.

    Raises:
        ValueError: If gen_name is unknown.
    """
    if num_actions < 1:
        raise ValueError("num_actions must be >= 1")

    if gen_name == "noop":
        return np.zeros((num_envs,), dtype=np.int32)

    if gen_name == "seeded_random":
        if seed is None:
            raise ValueError("seeded_random generator requires a seed")
        # Create a deterministic seed for this step
        # Using step_idx in the seed ensures different actions per step
        step_seed = seed + step_idx
        rng = np.random.default_rng(step_seed)
        return rng.integers(0, num_actions, size=(num_envs,), dtype=np.int32)

    if gen_name == "striped":
        pattern = _striped_pattern(action_names, num_actions)
        return pattern[np.arange(num_envs) % len(pattern)].astype(np.int32)

    raise ValueError(f"Unknown action generator: {gen_name}")


def get_action_gen_metadata(
    gen_name: str,
    seed: int | None,
    *,
    action_names: list[str] | None = None,
    num_actions: int | None = None,
) -> dict[str, Any]:
    """Get metadata for action generator (for artifact recording).

    Args:
        gen_name: Generator name.
        seed: Base seed (or None).

    Returns:
        Dict with name, version, and seed.
    """
    meta: dict[str, Any] = {
        "name": gen_name,
        "version": ACTION_GEN_VERSION,
        "seed": seed,
    }
    if gen_name == "striped":
        if num_actions is None:
            raise ValueError("striped metadata requires num_actions")
        pattern = _striped_pattern(action_names, num_actions).tolist()
        meta["pattern"] = pattern
        if action_names:
            meta["pattern_action_names"] = [action_names[i] for i in pattern]
    return meta


def get_action_codec_metadata(codec_id: str) -> dict[str, Any]:
    """Get metadata for action codec (for artifact recording)."""
    codec = get_action_codec(codec_id)
    return {
        "id": codec_id,
        "name": codec.name,
        "version": codec.version,
        "num_actions": codec.num_actions,
        "action_names": list(codec.action_names),
    }


def get_action_schedule_metadata(
    frames_per_step: int, release_after_frames: int
) -> dict[str, Any]:
    """Get metadata for action press/release schedule."""
    tick_split = [max(frames_per_step - 1, 0), 1 if frames_per_step > 0 else 0]
    return {
        "frames_per_step": frames_per_step,
        "release_after_frames": release_after_frames,
        "tick_split": tick_split,
    }


# ---------------------------------------------------------------------------
# State Comparison (Pure)
# ---------------------------------------------------------------------------

# Canonical register keys for normalization
CANONICAL_REGISTER_KEYS = ["pc", "sp", "a", "f", "b", "c", "d", "e", "h", "l"]
CANONICAL_FLAG_KEYS = ["z", "n", "h", "c"]

# Memory hash configuration
MEM_HASH_VERSION = "blake2b-128-v1"
MAX_MEM_DUMP_BYTES = 1024


def _parse_hex_u16(text: str) -> int:
    """Parse a hex string into an integer within [0, 0x10000]."""
    cleaned = text.strip().lower()
    if cleaned.startswith("0x"):
        cleaned = cleaned[2:]
    if not cleaned:
        raise ValueError("empty hex value")
    value = int(cleaned, 16)
    if value < 0 or value > 0x10000:
        raise ValueError(f"hex value out of range: {text}")
    return value


def parse_mem_region(region: str) -> tuple[int, int]:
    """Parse a memory region string of the form LO:HI (hex).

    HI is exclusive, matching Python slice semantics.
    """
    if ":" not in region:
        raise ValueError("mem region must be in LO:HI form")
    lo_str, hi_str = region.split(":", 1)
    lo = _parse_hex_u16(lo_str)
    hi = _parse_hex_u16(hi_str)
    if lo < 0 or hi > 0x10000 or lo >= hi:
        raise ValueError(f"invalid memory range: {region}")
    return lo, hi


def parse_mem_regions(regions: list[str] | None) -> list[tuple[int, int]]:
    """Parse a list of memory region strings."""
    if not regions:
        return []
    return [parse_mem_region(region) for region in regions]


def hash_memory(data: bytes) -> str:
    """Hash memory bytes deterministically."""
    import hashlib

    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _coerce_memory_slice(data: Any, expected_len: int) -> bytes:
    """Coerce backend memory read into bytes and validate length."""
    mem_bytes = data if isinstance(data, bytes) else bytes(data)
    if len(mem_bytes) != expected_len:
        raise ValueError(
            f"memory slice length {len(mem_bytes)} != expected {expected_len}"
        )
    return mem_bytes


def normalize_cpu_state(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize CPU state for comparison (pure function).

    Ensures consistent key ordering and types for deterministic comparison.

    Args:
        state: Raw CPU state dictionary from backend.

    Returns:
        Normalized dictionary with canonical key order and Python int types.
    """
    normalized: dict[str, Any] = {}

    # Normalize registers in canonical order
    for key in CANONICAL_REGISTER_KEYS:
        if key in state:
            normalized[key] = int(state[key])
        else:
            normalized[key] = None

    # Normalize flags
    if "flags" in state and state["flags"]:
        normalized["flags"] = {
            key: int(state["flags"].get(key, 0)) for key in CANONICAL_FLAG_KEYS
        }
    else:
        normalized["flags"] = {key: 0 for key in CANONICAL_FLAG_KEYS}

    # Optional counters
    normalized["instr_count"] = (
        int(state["instr_count"]) if state.get("instr_count") is not None else None
    )
    normalized["cycle_count"] = (
        int(state["cycle_count"]) if state.get("cycle_count") is not None else None
    )

    return normalized


def diff_states(
    ref_state: dict[str, Any], dut_state: dict[str, Any]
) -> dict[str, Any] | None:
    """Compute differences between normalized CPU states (pure function).

    Args:
        ref_state: Normalized reference state.
        dut_state: Normalized DUT state.

    Returns:
        None if states match, otherwise a dict with field-level differences.
    """
    diffs: dict[str, Any] = {}

    # Compare registers
    for key in CANONICAL_REGISTER_KEYS:
        ref_val = ref_state.get(key)
        dut_val = dut_state.get(key)
        if ref_val != dut_val:
            diffs[key] = {"ref": ref_val, "dut": dut_val}

    # Compare flags
    ref_flags = ref_state.get("flags", {})
    dut_flags = dut_state.get("flags", {})
    flag_diffs: dict[str, Any] = {}
    for key in CANONICAL_FLAG_KEYS:
        ref_val = ref_flags.get(key)
        dut_val = dut_flags.get(key)
        if ref_val != dut_val:
            flag_diffs[key] = {"ref": ref_val, "dut": dut_val}
    if flag_diffs:
        diffs["flags"] = flag_diffs

    # Compare counters (optional, skip if both None)
    for key in ["instr_count", "cycle_count"]:
        ref_val = ref_state.get(key)
        dut_val = dut_state.get(key)
        if ref_val is not None and dut_val is not None and ref_val != dut_val:
            diffs[key] = {"ref": ref_val, "dut": dut_val}

    return diffs if diffs else None


# ---------------------------------------------------------------------------
# System Info
# ---------------------------------------------------------------------------


def get_system_info() -> dict[str, Any]:
    """Collect system information for reproducibility."""
    import csv
    import os

    import numpy as np

    info: dict[str, Any] = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": platform.processor() or None,
        "gpu": None,
        "gpu_names": [],
        "driver_version": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pufferlib": None,
        "pufferlib_dist_name": None,
        "pufferlib_dist_version": None,
        "pufferlib_direct_url": None,
        "warp_dist_name": None,
        "warp_dist_version": None,
        "warp_direct_url": None,
        "warp_wheel_source": "unknown",
    }

    # Try to get PyBoy version
    try:
        import pyboy

        info["pyboy"] = pyboy.__version__
    except (ImportError, AttributeError):
        info["pyboy"] = None

    # Try to get Warp version
    try:
        import warp

        info["warp"] = warp.__version__
    except (ImportError, AttributeError):
        info["warp"] = None

    # Try to get PufferLib version
    try:
        import pufferlib

        info["pufferlib"] = pufferlib.__version__
    except (ImportError, AttributeError):
        info["pufferlib"] = None

    # Try to get Warp distribution + provenance (PEP 610 direct_url.json)
    try:
        from importlib import metadata as importlib_metadata

        def _try_get_dist(name: str) -> importlib_metadata.Distribution | None:
            try:
                return importlib_metadata.distribution(name)
            except importlib_metadata.PackageNotFoundError:
                return None

        warp_dist = (
            _try_get_dist("warp-lang")
            or _try_get_dist("warp_lang")
            or _try_get_dist("warp")
        )

        if warp_dist is not None:
            info["warp_dist_name"] = warp_dist.metadata.get("Name") or None
            info["warp_dist_version"] = warp_dist.version

            direct_url_path = None
            for file in warp_dist.files or []:
                if str(file).endswith("direct_url.json"):
                    direct_url_path = warp_dist.locate_file(file)
                    break

            if direct_url_path is not None:
                try:
                    direct_url_json = json.loads(Path(direct_url_path).read_text())
                    direct_url = direct_url_json.get("url")
                except Exception:
                    direct_url = None
                if isinstance(direct_url, str) and direct_url.strip():
                    info["warp_direct_url"] = direct_url
                    info["warp_wheel_source"] = "pep610"
    except Exception:
        # Best-effort only; don't fail artifact writing.
        pass

    # Try to get PufferLib distribution + provenance (PEP 610 direct_url.json)
    try:
        from importlib import metadata as importlib_metadata

        puffer_dist = None
        try:
            puffer_dist = importlib_metadata.distribution("pufferlib")
        except importlib_metadata.PackageNotFoundError:
            puffer_dist = None

        if puffer_dist is not None:
            info["pufferlib_dist_name"] = puffer_dist.metadata.get("Name") or None
            info["pufferlib_dist_version"] = puffer_dist.version

            direct_url_path = None
            for file in puffer_dist.files or []:
                if str(file).endswith("direct_url.json"):
                    direct_url_path = puffer_dist.locate_file(file)
                    break

            if direct_url_path is not None:
                try:
                    direct_url_json = json.loads(Path(direct_url_path).read_text())
                    direct_url = direct_url_json.get("url")
                except Exception:
                    direct_url = None
                if isinstance(direct_url, str) and direct_url.strip():
                    info["pufferlib_direct_url"] = direct_url
    except Exception:
        # Best-effort only; don't fail artifact writing.
        pass

    # Try to get GPU + driver info
    try:
        import io
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            gpu_names: list[str] = []
            driver_version: str | None = None
            for row in csv.reader(io.StringIO(result.stdout)):
                if len(row) < 2:
                    continue
                name = row[0].strip()
                version = row[1].strip()
                if name:
                    gpu_names.append(name)
                if version and driver_version is None:
                    driver_version = version
            info["gpu_names"] = gpu_names
            info["gpu"] = gpu_names[0] if gpu_names else None
            info["driver_version"] = driver_version
    except Exception:
        # Best-effort only; don't fail artifact writing.
        pass

    # Try to get git commit SHA
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["git_commit"] = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        info["git_commit"] = None

    return info


# ---------------------------------------------------------------------------
# ROM SHA256
# ---------------------------------------------------------------------------


def compute_rom_sha256(rom_path: str) -> str:
    """Compute SHA-256 hash of a ROM file."""
    import hashlib

    sha256 = hashlib.sha256()
    with open(rom_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


def run_benchmark(
    backend: VecBackend,
    *,
    steps: int,
    warmup_steps: int,
    action_gen: str,
    actions_seed: int | None,
    frames_per_step: int,
    sync_every: int | None,
    num_actions: int,
    action_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run a benchmark and return results.

    Args:
        backend: The backend to benchmark.
        steps: Number of steps to measure.
        warmup_steps: Number of warmup steps (not timed).
        action_gen: Action generator name ("noop", "seeded_random", or "striped").
        actions_seed: Seed for action generation (required for seeded_random).
        frames_per_step: Frames per step (for FPS calculation).
        sync_every: Sync interval for CUDA benchmarks (0 = end only).
        num_actions: Number of available actions.

    Returns:
        Dictionary with benchmark results.
    """
    num_envs = backend.num_envs

    # Reset
    backend.reset(seed=actions_seed)

    # Warmup (step indices are negative to distinguish from measured steps)
    for i in range(warmup_steps):
        actions = generate_actions(
            step_idx=-(warmup_steps - i),
            num_envs=num_envs,
            seed=actions_seed,
            gen_name=action_gen,
            num_actions=num_actions,
            action_names=action_names,
        )
        backend.step(actions)

    synchronize_backend(backend)

    # Measured run
    start_time = time.perf_counter()

    for i in range(steps):
        actions = generate_actions(
            step_idx=i,
            num_envs=num_envs,
            seed=actions_seed,
            gen_name=action_gen,
            num_actions=num_actions,
            action_names=action_names,
        )
        backend.step(actions)
        if sync_every is not None and sync_every > 0 and (i + 1) % sync_every == 0:
            synchronize_backend(backend)

    synchronize_backend(backend)
    elapsed = time.perf_counter() - start_time

    # Compute metrics
    total_env_steps = steps * num_envs
    total_sps = total_env_steps / elapsed if elapsed > 0 else 0.0
    per_env_sps = steps / elapsed if elapsed > 0 else 0.0
    frames_per_sec = total_sps * frames_per_step

    return {
        "warmup_steps": warmup_steps,
        "measured_steps": steps,
        "num_envs": num_envs,
        "seconds": elapsed,
        "total_env_steps": total_env_steps,
        "total_sps": total_sps,
        "per_env_sps": per_env_sps,
        "frames_per_step": frames_per_step,
        "frames_per_sec": frames_per_sec,
    }


def get_divergence_receipt(backend: VecBackend) -> dict[str, Any] | None:
    """Return a cheap divergence receipt if the backend can provide PCs."""
    snapshot_fn = getattr(backend, "get_pc_snapshot", None)
    if not callable(snapshot_fn):
        return None

    pcs = snapshot_fn()
    pcs_np = np.array(pcs, dtype=np.int32).reshape(-1)
    if pcs_np.size == 0:
        return {
            "metric": "unique_pc_count",
            "unique_pc_count": 0,
            "num_envs": 0,
            "unique_pc_ratio": None,
        }
    unique_pc_count = int(np.unique(pcs_np).size)
    return {
        "metric": "unique_pc_count",
        "unique_pc_count": unique_pc_count,
        "num_envs": int(pcs_np.size),
        "unique_pc_ratio": unique_pc_count / float(pcs_np.size),
    }


def synchronize_backend(backend: VecBackend) -> None:
    """Synchronize device work for accurate timing (CUDA only)."""
    if backend.device != "cuda":
        return
    sync_fn = getattr(backend, "synchronize", None)
    if callable(sync_fn):
        sync_fn()
        return
    try:
        import warp as wp
    except Exception as exc:  # pragma: no cover - CUDA-only path
        raise RuntimeError(
            "CUDA backend requires synchronize support for accurate timing."
        ) from exc
    wp.synchronize()


# ---------------------------------------------------------------------------
# Artifact Writer
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON data to path atomically using temp file + rename.

    This ensures partial writes never exist at the target path ("Crash-Only" pattern).

    Args:
        path: Destination file path.
        data: Dictionary to serialize as JSON.
    """
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (ensures same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.rename(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def write_artifact(
    output_dir: Path,
    *,
    run_id: str,
    system_info: dict[str, Any],
    config: dict[str, Any],
    results: dict[str, Any],
) -> Path:
    """Write a benchmark artifact to JSON atomically.

    Args:
        output_dir: Directory for output files.
        run_id: Unique run identifier.
        system_info: System information dict.
        config: Run configuration dict.
        results: Benchmark results dict.

    Returns:
        Path to the written artifact file.
    """
    from gbxcule.backends.common import RESULT_SCHEMA_VERSION

    artifact = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "system": system_info,
        "config": config,
        "results": results,
    }

    artifact_path = output_dir / f"{run_id}.json"
    _write_json_atomic(artifact_path, artifact)

    return artifact_path


def write_scaling_artifact(
    output_dir: Path,
    *,
    run_id: str,
    system_info: dict[str, Any],
    sweep_config: dict[str, Any],
    results_list: list[dict[str, Any]],
) -> Path:
    """Write a scaling benchmark artifact to JSON atomically.

    Args:
        output_dir: Directory for output files.
        run_id: Unique run identifier.
        system_info: System information dict.
        sweep_config: Sweep configuration (shared across env counts).
        results_list: List of results for each env count.

    Returns:
        Path to the written artifact file.
    """
    from gbxcule.backends.common import RESULT_SCHEMA_VERSION

    artifact = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "system": system_info,
        "sweep_config": sweep_config,
        "results": results_list,
    }

    artifact_path = output_dir / f"{run_id}__scaling.json"
    _write_json_atomic(artifact_path, artifact)

    return artifact_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GBxCuLE Benchmark Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Backend selection (required for benchmark mode, ignored in verify mode)
    parser.add_argument(
        "--backend",
        choices=[
            "pyboy_single",
            "pyboy_puffer_single",
            "pyboy_puffer_vec",
            "pyboy_vec_mp",
            "warp_vec",
            "warp_vec_cpu",
            "warp_vec_cuda",
        ],
        default=None,
        help="Backend to benchmark (required for benchmark mode)",
    )

    # ROM selection (mutually exclusive)
    rom_group = parser.add_mutually_exclusive_group(required=True)
    rom_group.add_argument("--rom", type=str, help="Path to ROM file")
    rom_group.add_argument("--suite", type=str, help="Path to suite YAML file")

    # Run configuration
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps to measure",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=None,
        help="Sync interval for CUDA benchmarks (default: 64 on CUDA, None on CPU). "
        "Use 0 to sync only at end of window.",
    )
    parser.add_argument(
        "--stage",
        choices=["emulate_only", "full_step", "reward_only", "obs_only"],
        default="emulate_only",
        help="Execution stage",
    )

    # Backend configuration
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of environments"
    )
    parser.add_argument(
        "--env-counts",
        type=str,
        default=None,
        help="Comma-separated env counts for scaling sweep (e.g. '1,2,4,8')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Number of workers for MP backends (defaults to num_envs or "
            "a divisor for puffer vec)"
        ),
    )
    parser.add_argument(
        "--puffer-vec-backend",
        choices=["puffer_mp_sync", "puffer_serial"],
        default="puffer_mp_sync",
        help="PufferLib vectorization backend",
    )
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=24,
        help="Frames per step",
    )
    parser.add_argument(
        "--release-after-frames",
        type=int,
        default=8,
        help="Frames before button release",
    )

    # Action configuration
    parser.add_argument(
        "--actions-seed",
        type=int,
        default=None,
        help="Seed for action generation (None = noop)",
    )
    parser.add_argument(
        "--action-gen",
        choices=["noop", "seeded_random", "striped"],
        default="noop",
        help="Action generator type",
    )
    parser.add_argument(
        "--action-codec",
        choices=list_action_codecs(),
        default=DEFAULT_ACTION_CODEC_ID,
        help="Action codec id",
    )

    # Verify mode configuration
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable verification mode (ref vs DUT comparison)",
    )
    parser.add_argument(
        "--ref-backend",
        choices=["pyboy_single", "pyboy_puffer_single", "pyboy_vec_mp"],
        default="pyboy_single",
        help="Reference backend for verification",
    )
    parser.add_argument(
        "--dut-backend",
        choices=[
            "pyboy_single",
            "pyboy_puffer_single",
            "pyboy_vec_mp",
            "stub_bad",
            "warp_vec",
            "warp_vec_cpu",
            "warp_vec_cuda",
        ],
        default="warp_vec",
        help="Device-under-test backend for verification",
    )
    parser.add_argument(
        "--verify-steps",
        type=int,
        default=100,
        help="Number of verification steps",
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=1,
        help="Compare states every N steps",
    )
    parser.add_argument(
        "--mem-region",
        action="append",
        default=None,
        help="Memory region to hash/compare (hex LO:HI, HI exclusive). "
        "Can be specified multiple times.",
    )
    parser.add_argument(
        "--actions-file",
        type=str,
        default=None,
        help="Path to actions.jsonl file for replay (overrides generator)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bench/runs",
        help="Output directory for artifacts",
    )

    return parser.parse_args(argv)


def resolve_sync_every(sync_every: int | None, device: str) -> int | None:
    """Resolve sync policy for a given device.

    Defaults to 64 for CUDA when not explicitly provided.
    """
    if sync_every is not None and sync_every < 0:
        raise ValueError("--sync-every must be >= 0")
    if sync_every is None and device == "cuda":
        return 64
    return sync_every


def load_suite(suite_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a ROM suite YAML file and return defaults + ROM entries."""
    import yaml

    if not suite_path.exists():
        raise FileNotFoundError(f"Suite file not found: {suite_path}")

    with suite_path.open() as f:
        suite = yaml.safe_load(f)

    roms = suite.get("roms") if isinstance(suite, dict) else None
    if not roms:
        raise ValueError("Suite file has no ROMs defined")

    defaults = suite.get("defaults", {}) if isinstance(suite, dict) else {}

    validated: list[dict[str, Any]] = []
    for entry in roms:
        if not isinstance(entry, dict):
            raise ValueError("Suite ROM entry must be a mapping")
        if "path" not in entry:
            raise ValueError("Suite ROM entry missing 'path'")
        validated.append(entry)

    return defaults, validated


def resolve_suite_value(
    entry: dict[str, Any],
    defaults: dict[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    """Resolve a value for a suite ROM entry with defaults and fallback."""
    if key in entry:
        return entry[key]
    if key in defaults:
        return defaults[key]
    return fallback


def resolve_suite_rom_path(suite_path: Path, path_str: str) -> Path:
    """Resolve a suite ROM path relative to CWD or suite directory."""
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return suite_path.parent / candidate


def run_suite_scaling_sweep(
    args: argparse.Namespace,
    suite_path: Path,
    env_counts: list[int],
    *,
    defaults: dict[str, Any],
    roms: list[dict[str, Any]],
) -> int:
    """Run scaling sweeps across all ROMs in a suite."""
    output_root = Path(args.output_dir)
    overall_rc = 0

    print(f"Running suite scaling sweep: {suite_path}")
    print(f"ROMs: {len(roms)}")
    print()

    for entry in roms:
        rom_id = str(entry.get("id") or Path(entry["path"]).stem)
        rom_path = resolve_suite_rom_path(suite_path, str(entry["path"]))

        if not rom_path.exists():
            print(f"Error: ROM file not found: {rom_path}", file=sys.stderr)
            return 1

        rom_args = argparse.Namespace(**vars(args))
        rom_args.frames_per_step = int(
            resolve_suite_value(
                entry, defaults, "frames_per_step", args.frames_per_step
            )
        )
        rom_args.release_after_frames = int(
            resolve_suite_value(
                entry, defaults, "release_after_frames", args.release_after_frames
            )
        )
        rom_args.action_gen = resolve_suite_value(
            entry, defaults, "action_gen", args.action_gen
        )
        rom_args.actions_seed = resolve_suite_value(
            entry, defaults, "actions_seed", args.actions_seed
        )
        rom_args.output_dir = str(output_root / rom_id)

        print(f"Suite ROM: {rom_id}")
        print(
            "  schedule: "
            f"{rom_args.frames_per_step} frames/step, "
            f"release_after_frames={rom_args.release_after_frames}"
        )
        if rom_args.action_gen != args.action_gen:
            print(f"  action_gen override: {rom_args.action_gen}")
        print(f"  output_dir: {rom_args.output_dir}")
        print()

        rc = run_scaling_sweep(rom_args, rom_path, env_counts)
        if rc != 0:
            overall_rc = rc

    return overall_rc


def run_scaling_sweep(
    args: argparse.Namespace,
    rom_path: Path,
    env_counts: list[int],
) -> int:
    """Run a scaling sweep across multiple env counts.

    Args:
        args: Parsed arguments.
        rom_path: Path to ROM file.
        env_counts: List of env counts to sweep.

    Returns:
        Exit code (0 for success).
    """
    # Collect system info once
    system_info = get_system_info()
    rom_sha256 = compute_rom_sha256(str(rom_path))

    # Use seed for seeded_random generator, None for noop
    effective_seed = args.actions_seed if args.action_gen == "seeded_random" else None
    action_codec_meta = get_action_codec_metadata(args.action_codec)
    action_schedule_meta = get_action_schedule_metadata(
        args.frames_per_step, args.release_after_frames
    )

    # Build sweep config (shared across all runs)
    sweep_config = {
        "backend": args.backend,
        "rom_path": str(rom_path),
        "rom_sha256": rom_sha256,
        "stage": args.stage,
        "env_counts": env_counts,
        "num_workers_cap": (
            args.num_workers
            if args.backend in {"pyboy_vec_mp", "pyboy_puffer_vec"}
            else None
        ),
        "vec_backend": (
            args.puffer_vec_backend if args.backend == "pyboy_puffer_vec" else None
        ),
        "frames_per_step": args.frames_per_step,
        "release_after_frames": args.release_after_frames,
        "action_schedule": action_schedule_meta,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "action_generator": get_action_gen_metadata(
            args.action_gen,
            effective_seed,
            action_names=action_codec_meta["action_names"],
            num_actions=action_codec_meta["num_actions"],
        ),
        "action_codec": action_codec_meta,
        "sync_every": None,
    }

    results_list: list[dict[str, Any]] = []

    print(f"Running scaling sweep: env_counts={env_counts}")
    print(f"Backend: {args.backend}")
    print(f"ROM: {rom_path.name}")
    print()

    for num_envs in env_counts:
        # Single-env backends only support 1 env
        if args.backend in {"pyboy_single", "pyboy_puffer_single"} and num_envs != 1:
            print(f"  Skipping num_envs={num_envs} ({args.backend} only supports 1)")
            continue

        try:
            effective_num_workers: int | None = args.num_workers
            if args.backend == "pyboy_vec_mp":
                if effective_num_workers is None:
                    effective_num_workers = os.cpu_count() or 1
                effective_num_workers = min(num_envs, effective_num_workers)

            backend = create_backend(
                args.backend,
                str(rom_path),
                num_envs=num_envs,
                num_workers=effective_num_workers,
                frames_per_step=args.frames_per_step,
                release_after_frames=args.release_after_frames,
                stage=args.stage,
                base_seed=args.actions_seed,
                action_codec=args.action_codec,
                puffer_vec_backend=args.puffer_vec_backend,
            )
        except Exception as e:
            print(f"  Error creating backend for num_envs={num_envs}: {e}")
            continue

        try:
            args.sync_every = resolve_sync_every(args.sync_every, backend.device)
            if sweep_config["sync_every"] is None:
                sweep_config["sync_every"] = args.sync_every
            results = run_benchmark(
                backend,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                action_gen=args.action_gen,
                actions_seed=effective_seed,
                frames_per_step=args.frames_per_step,
                sync_every=args.sync_every,
                num_actions=action_codec_meta["num_actions"],
                action_names=action_codec_meta["action_names"],
            )

            # Add entry with env_count + results
            entry = {
                "num_envs": num_envs,
                "device": backend.device,
                "num_workers": effective_num_workers
                if args.backend == "pyboy_vec_mp"
                else None,
                **results,
            }
            receipt = get_divergence_receipt(backend)
            if receipt is not None:
                entry["divergence_receipt"] = receipt
            vec_backend = getattr(backend, "vec_backend", None)
            if vec_backend is not None:
                entry["vec_backend"] = vec_backend
            num_workers_actual = getattr(backend, "num_workers", None)
            if num_workers_actual is not None:
                entry["num_workers"] = num_workers_actual
            results_list.append(entry)

            print(f"  num_envs={num_envs}: {results['total_sps']:.1f} SPS")

        finally:
            backend.close()

    # Generate run ID and write artifact
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.backend}_{rom_path.stem}"

    output_dir = Path(args.output_dir)
    artifact_path = write_scaling_artifact(
        output_dir,
        run_id=run_id,
        system_info=system_info,
        sweep_config=sweep_config,
        results_list=results_list,
    )

    print()
    print(f"Scaling artifact: {artifact_path}")

    return 0


# ---------------------------------------------------------------------------
# Mismatch Bundle Schema Version
# ---------------------------------------------------------------------------

MISMATCH_SCHEMA_VERSION = 1


def load_actions_trace(actions_file: str) -> list[np.ndarray]:
    """Load actions trace from JSONL file.

    Args:
        actions_file: Path to actions.jsonl file.

    Returns:
        List of action arrays (one per step).

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid.
    """
    actions_list: list[np.ndarray] = []
    with open(actions_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                actions_data = json.loads(line)
                actions_list.append(np.array(actions_data, dtype=np.int32))
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"Invalid action trace at line {line_num}: {e}") from e
    return actions_list


def write_actions_trace(actions_trace: list[list[int]], output_path: Path) -> None:
    """Write actions trace to JSONL file.

    Args:
        actions_trace: List of action lists (one per step).
        output_path: Path to write the JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for actions in actions_trace:
            f.write(json.dumps(actions) + "\n")


def write_mismatch_bundle(
    output_dir: Path,
    *,
    timestamp: str,
    rom_path: Path,
    rom_sha256: str,
    ref_backend: str,
    dut_backend: str,
    mismatch_step: int,
    env_idx: int,
    ref_state: dict[str, Any],
    dut_state: dict[str, Any],
    diff: dict[str, Any],
    action_codec: dict[str, Any],
    mem_regions: list[tuple[int, int]] | None = None,
    mem_hash_version: str | None = None,
    mem_dumps: list[tuple[int, int, bytes, bytes]] | None = None,
    mem_region_flags: list[str] | None = None,
    actions_trace: list[list[int]],
    system_info: dict[str, Any],
    action_gen_name: str,
    action_gen_seed: int | None,
    frames_per_step: int,
    release_after_frames: int,
    compare_every: int,
    verify_steps: int,
) -> Path:
    """Write a mismatch repro bundle atomically.

    Creates a directory with all necessary files to reproduce the mismatch.

    Args:
        output_dir: Base output directory (e.g., bench/runs).
        timestamp: Timestamp string for the bundle name.
        rom_path: Path to the ROM file.
        rom_sha256: SHA-256 hash of the ROM.
        ref_backend: Reference backend name.
        dut_backend: DUT backend name.
        mismatch_step: Step index where mismatch occurred.
        env_idx: Environment index where mismatch occurred.
        ref_state: Normalized reference CPU state.
        dut_state: Normalized DUT CPU state.
        diff: State difference dictionary.
        action_codec: Action codec metadata dict.
        mem_regions: Optional list of memory regions (lo, hi).
        mem_hash_version: Hash version label for memory hashing.
        mem_dumps: Optional list of memory dumps (lo, hi, ref_bytes, dut_bytes).
        mem_region_flags: Optional list of mem-region strings for repro.
        actions_trace: List of actions up to mismatch.
        system_info: System information dictionary.
        action_gen_name: Action generator name.
        action_gen_seed: Action generator seed.
        frames_per_step: Frames per step setting.
        release_after_frames: Release after frames setting.
        compare_every: Compare every N steps setting.
        verify_steps: Total verification steps setting.

    Returns:
        Path to the final bundle directory.
    """
    import os
    import shutil
    import uuid

    # Create bundle name
    bundle_name = f"{timestamp}_{rom_path.stem}_{ref_backend}_vs_{dut_backend}"
    final_dir = output_dir / "mismatch" / bundle_name
    temp_dir = output_dir / "mismatch" / f".tmp_{uuid.uuid4()}"
    rom_filename = "rom.gb"

    try:
        # Create temp directory
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Embed ROM in bundle for hermetic repros
        rom_dest = temp_dir / rom_filename
        shutil.copyfile(rom_path, rom_dest)
        rom_size = rom_dest.stat().st_size

        # Write metadata.json
        metadata = {
            "schema_version": MISMATCH_SCHEMA_VERSION,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "rom_path": str(rom_path),
            "rom_filename": rom_filename,
            "rom_size": rom_size,
            "rom_sha256": rom_sha256,
            "ref_backend": ref_backend,
            "dut_backend": dut_backend,
            "mismatch_step": mismatch_step,
            "env_idx": env_idx,
            "verify_steps": verify_steps,
            "compare_every": compare_every,
            "frames_per_step": frames_per_step,
            "release_after_frames": release_after_frames,
            "action_generator": {
                "name": action_gen_name,
                "version": ACTION_GEN_VERSION,
                "seed": action_gen_seed,
            },
            "action_codec": action_codec,
            "system": system_info,
        }
        if mem_regions:
            metadata["mem_regions"] = [{"lo": lo, "hi": hi} for lo, hi in mem_regions]
            metadata["mem_hash_version"] = mem_hash_version
        with open(temp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Write ref_state.json
        with open(temp_dir / "ref_state.json", "w") as f:
            json.dump(ref_state, f, indent=2)

        # Write dut_state.json
        with open(temp_dir / "dut_state.json", "w") as f:
            json.dump(dut_state, f, indent=2)

        # Write diff.json
        with open(temp_dir / "diff.json", "w") as f:
            json.dump(diff, f, indent=2)

        # Write memory dumps (optional, for small regions)
        if mem_dumps:
            for lo, hi, ref_bytes, dut_bytes in mem_dumps:
                ref_name = f"mem_ref_{lo:04X}_{hi:04X}.bin"
                dut_name = f"mem_dut_{lo:04X}_{hi:04X}.bin"
                (temp_dir / ref_name).write_bytes(ref_bytes)
                (temp_dir / dut_name).write_bytes(dut_bytes)

        # Write actions.jsonl
        with open(temp_dir / "actions.jsonl", "w") as f:
            for actions in actions_trace:
                f.write(json.dumps(actions) + "\n")

        mem_region_args = mem_region_flags or (
            [f"{lo:04X}:{hi:04X}" for lo, hi in mem_regions] if mem_regions else []
        )
        mem_region_lines = "".join(
            [f"    --mem-region {flag} \\\n" for flag in mem_region_args]
        )

        # Write repro.sh
        repro_script = f"""#!/bin/bash
# Reproduction script for mismatch at step {mismatch_step}
# Generated: {datetime.now(UTC).isoformat()}

set -e

# Navigate to project root (adjust if needed)
cd "$(dirname "$0")/../../../.."

    # Run verification with action replay
uv run python bench/harness.py \\
    --verify \\
    --ref-backend {ref_backend} \\
    --dut-backend {dut_backend} \\
    --rom "$(dirname "$0")/{rom_filename}" \\
    --verify-steps {verify_steps} \\
    --compare-every {compare_every} \\
    --frames-per-step {frames_per_step} \\
    --release-after-frames {release_after_frames} \\
    --action-codec {action_codec["id"]} \\
{mem_region_lines}    --actions-file "$(dirname "$0")/actions.jsonl"
"""
        repro_path = temp_dir / "repro.sh"
        with open(repro_path, "w") as f:
            f.write(repro_script)
        os.chmod(repro_path, 0o755)

        # Atomically rename temp dir to final
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.move(str(temp_dir), str(final_dir))

        return final_dir

    except Exception:
        # Clean up temp dir on failure
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def run_verify(
    args: argparse.Namespace,
    rom_path: Path,
) -> int:
    """Run verification mode comparing ref vs DUT.

    Args:
        args: Parsed arguments.
        rom_path: Path to ROM file.

    Returns:
        Exit code (0 for match, 1 for mismatch or error).
    """
    # Load actions from file if provided (overrides generator)
    replay_actions: list[np.ndarray] | None = None
    if args.actions_file:
        try:
            replay_actions = load_actions_trace(args.actions_file)
            print(f"Replaying actions from: {args.actions_file}")
            # Override verify_steps to match action trace length
            if len(replay_actions) < args.verify_steps:
                print(
                    f"Warning: action trace has {len(replay_actions)} steps, "
                    f"using that instead of {args.verify_steps}",
                    file=sys.stderr,
                )
                args.verify_steps = len(replay_actions)
        except FileNotFoundError:
            print(
                f"Error: actions file not found: {args.actions_file}", file=sys.stderr
            )
            return 1
        except ValueError as e:
            print(f"Error loading actions file: {e}", file=sys.stderr)
            return 1

    # Parse memory regions (if any)
    try:
        mem_regions = parse_mem_regions(args.mem_region)
    except ValueError as e:
        print(f"Error parsing --mem-region: {e}", file=sys.stderr)
        return 1

    mem_region_flags = [f"{lo:04X}:{hi:04X}" for lo, hi in mem_regions]

    print(f"Verification mode: ref={args.ref_backend} vs dut={args.dut_backend}")
    print(f"ROM: {rom_path.name}")
    print(f"Steps: {args.verify_steps}, compare every {args.compare_every}")
    if mem_regions:
        print(f"Memory regions: {', '.join(mem_region_flags)}")
    print()

    # Use seed for seeded_random generator, None for noop
    effective_seed = args.actions_seed if args.action_gen == "seeded_random" else None
    action_codec_meta = get_action_codec_metadata(args.action_codec)

    # Create ref backend
    try:
        ref_backend = create_backend(
            args.ref_backend,
            str(rom_path),
            num_envs=1,  # Verify uses single env for now
            frames_per_step=args.frames_per_step,
            release_after_frames=args.release_after_frames,
            stage=args.stage,
            base_seed=effective_seed,
            action_codec=args.action_codec,
        )
    except Exception as e:
        print(f"Error creating ref backend: {e}", file=sys.stderr)
        return 1

    # Create DUT backend
    try:
        dut_backend = create_backend(
            args.dut_backend,
            str(rom_path),
            num_envs=1,
            frames_per_step=args.frames_per_step,
            release_after_frames=args.release_after_frames,
            stage=args.stage,
            base_seed=effective_seed,
            action_codec=args.action_codec,
        )
    except Exception as e:
        ref_backend.close()
        print(f"Error creating DUT backend: {e}", file=sys.stderr)
        return 1

    try:
        # Reset both backends with same seed
        ref_backend.reset(seed=effective_seed)
        dut_backend.reset(seed=effective_seed)

        # Validate memory support if requested
        if mem_regions:
            for backend, backend_name in [
                (ref_backend, args.ref_backend),
                (dut_backend, args.dut_backend),
            ]:
                reader = getattr(backend, "read_memory", None)
                if not callable(reader):
                    print(
                        f"Error: backend '{backend_name}' does not support memory "
                        "reads required by --mem-region",
                        file=sys.stderr,
                    )
                    return 1

        # Action trace for recording
        actions_trace: list[list[int]] = []

        # Verify loop
        for step_idx in range(args.verify_steps):
            # Get actions (from replay or generate)
            if replay_actions is not None:
                actions = replay_actions[step_idx]
            else:
                actions = generate_actions(
                    step_idx=step_idx,
                    num_envs=1,
                    seed=effective_seed,
                    gen_name=args.action_gen,
                    num_actions=action_codec_meta["num_actions"],
                    action_names=action_codec_meta["action_names"],
                )

            # Record actions
            actions_trace.append(actions.tolist())

            # Step both backends
            ref_backend.step(actions)
            dut_backend.step(actions)

            # Compare states every compare_every steps
            if step_idx % args.compare_every == 0:
                ref_state = normalize_cpu_state(ref_backend.get_cpu_state(0))
                dut_state = normalize_cpu_state(dut_backend.get_cpu_state(0))

                diff = diff_states(ref_state, dut_state)
                mem_dumps: list[tuple[int, int, bytes, bytes]] = []

                if mem_regions:
                    mem_diffs: list[dict[str, Any]] = []
                    try:
                        for lo, hi in mem_regions:
                            expected_len = hi - lo
                            ref_bytes = _coerce_memory_slice(
                                ref_backend.read_memory(0, lo, hi), expected_len
                            )
                            dut_bytes = _coerce_memory_slice(
                                dut_backend.read_memory(0, lo, hi), expected_len
                            )
                            ref_hash = hash_memory(ref_bytes)
                            dut_hash = hash_memory(dut_bytes)
                            if ref_hash != dut_hash:
                                mem_diffs.append(
                                    {
                                        "lo": lo,
                                        "hi": hi,
                                        "ref_hash": ref_hash,
                                        "dut_hash": dut_hash,
                                    }
                                )
                                if expected_len <= MAX_MEM_DUMP_BYTES:
                                    mem_dumps.append((lo, hi, ref_bytes, dut_bytes))
                    except Exception as e:
                        print(f"Error reading memory: {e}", file=sys.stderr)
                        return 1

                    if mem_diffs:
                        if diff is None:
                            diff = {}
                        diff["memory"] = mem_diffs

                if diff is not None:
                    # Mismatch found
                    print(f"MISMATCH at step {step_idx}")
                    print(f"First differing fields: {list(diff.keys())[:5]}")

                    # Write full mismatch bundle
                    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                    rom_sha256 = compute_rom_sha256(str(rom_path))
                    system_info = get_system_info()

                    bundle_path = write_mismatch_bundle(
                        output_dir=Path(args.output_dir),
                        timestamp=timestamp,
                        rom_path=rom_path,
                        rom_sha256=rom_sha256,
                        ref_backend=args.ref_backend,
                        dut_backend=args.dut_backend,
                        mismatch_step=step_idx,
                        env_idx=0,
                        ref_state=ref_state,
                        dut_state=dut_state,
                        diff=diff,
                        action_codec=action_codec_meta,
                        mem_regions=mem_regions if mem_regions else None,
                        mem_hash_version=MEM_HASH_VERSION if mem_regions else None,
                        mem_dumps=mem_dumps if mem_dumps else None,
                        mem_region_flags=mem_region_flags if mem_regions else None,
                        actions_trace=actions_trace,
                        system_info=system_info,
                        action_gen_name=args.action_gen,
                        action_gen_seed=effective_seed,
                        frames_per_step=args.frames_per_step,
                        release_after_frames=args.release_after_frames,
                        compare_every=args.compare_every,
                        verify_steps=args.verify_steps,
                    )

                    print(f"Bundle: {bundle_path}")
                    print(f"Repro: {bundle_path / 'repro.sh'}")
                    return 1

        print(f"PASS: No mismatches in {args.verify_steps} steps")
        return 0

    finally:
        ref_backend.close()
        dut_backend.close()


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    suite_defaults: dict[str, Any] | None = None
    suite_roms: list[dict[str, Any]] | None = None
    suite_path: Path | None = None

    # Validate ROM path
    if args.rom:
        rom_path = Path(args.rom)
        if not rom_path.exists():
            print(f"Error: ROM file not found: {rom_path}", file=sys.stderr)
            return 1
    elif args.suite:
        suite_path = Path(args.suite)
        try:
            suite_defaults, suite_roms = load_suite(suite_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        # Use first ROM in suite for non-sweep modes
        first_rom = suite_roms[0]
        rom_path = resolve_suite_rom_path(suite_path, str(first_rom["path"]))
        if not rom_path.exists():
            print(f"Error: ROM file not found: {rom_path}", file=sys.stderr)
            return 1
    else:
        print("Error: Must specify --rom or --suite", file=sys.stderr)
        return 1

    # Check for scaling sweep mode
    if args.env_counts:
        if args.backend is None:
            print(
                "Error: --backend is required for scaling sweep mode", file=sys.stderr
            )
            return 1
        try:
            env_counts = [int(x.strip()) for x in args.env_counts.split(",")]
            if not env_counts:
                print("Error: --env-counts must not be empty", file=sys.stderr)
                return 1
            if any(x < 1 for x in env_counts):
                print("Error: All env counts must be >= 1", file=sys.stderr)
                return 1
        except ValueError as e:
            print(f"Error parsing --env-counts: {e}", file=sys.stderr)
            return 1

        if (
            suite_path is not None
            and suite_defaults is not None
            and suite_roms is not None
        ):
            return run_suite_scaling_sweep(
                args,
                suite_path,
                env_counts,
                defaults=suite_defaults,
                roms=suite_roms,
            )

        return run_scaling_sweep(args, rom_path, env_counts)

    # Check for verify mode
    if args.verify:
        return run_verify(args, rom_path)

    # Single run mode (benchmark)
    # Require --backend for benchmark mode
    if args.backend is None:
        print("Error: --backend is required for benchmark mode", file=sys.stderr)
        return 1

    # Validate backend constraints
    if args.backend == "pyboy_single" and args.num_envs != 1:
        print(
            "Warning: pyboy_single only supports num_envs=1, ignoring --num-envs",
            file=sys.stderr,
        )
        args.num_envs = 1

    # Create backend
    try:
        backend = create_backend(
            args.backend,
            str(rom_path),
            num_envs=args.num_envs,
            num_workers=args.num_workers,
            frames_per_step=args.frames_per_step,
            release_after_frames=args.release_after_frames,
            stage=args.stage,
            base_seed=args.actions_seed,
            action_codec=args.action_codec,
            puffer_vec_backend=args.puffer_vec_backend,
        )
    except Exception as e:
        print(f"Error creating backend: {e}", file=sys.stderr)
        return 1

    try:
        args.sync_every = resolve_sync_every(args.sync_every, backend.device)
        # Collect system info
        system_info = get_system_info()
        action_codec_meta = get_action_codec_metadata(args.action_codec)
        action_schedule_meta = get_action_schedule_metadata(
            args.frames_per_step, args.release_after_frames
        )

        # Build config
        rom_sha256 = compute_rom_sha256(str(rom_path))
        # Use seed for seeded_random generator, None for noop
        effective_seed = (
            args.actions_seed if args.action_gen == "seeded_random" else None
        )
        config = {
            "backend": args.backend,
            "device": backend.device,
            "rom_path": str(rom_path),
            "rom_sha256": rom_sha256,
            "stage": args.stage,
            "num_envs": backend.num_envs,
            "num_workers": getattr(backend, "num_workers", args.num_workers),
            "vec_backend": getattr(backend, "vec_backend", None),
            "frames_per_step": args.frames_per_step,
            "release_after_frames": args.release_after_frames,
            "action_schedule": action_schedule_meta,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "action_generator": get_action_gen_metadata(
                args.action_gen,
                effective_seed,
                action_names=action_codec_meta["action_names"],
                num_actions=action_codec_meta["num_actions"],
            ),
            "action_codec": action_codec_meta,
            "sync_every": args.sync_every,
        }

        # Run benchmark
        results = run_benchmark(
            backend,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            action_gen=args.action_gen,
            actions_seed=effective_seed,
            frames_per_step=args.frames_per_step,
            sync_every=args.sync_every,
            num_actions=action_codec_meta["num_actions"],
            action_names=action_codec_meta["action_names"],
        )
        receipt = get_divergence_receipt(backend)
        if receipt is not None:
            results["divergence_receipt"] = receipt

        # Generate run ID
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{args.backend}_{rom_path.stem}"

        # Write artifact
        output_dir = Path(args.output_dir)
        artifact_path = write_artifact(
            output_dir,
            run_id=run_id,
            system_info=system_info,
            config=config,
            results=results,
        )

        # Print summary
        print(f"Backend: {args.backend}")
        print(f"ROM: {rom_path.name}")
        print(f"Envs: {backend.num_envs}")
        print(f"Steps: {args.steps} (warmup: {args.warmup_steps})")
        print(f"Time: {results['seconds']:.3f}s")
        print(f"Total SPS: {results['total_sps']:.1f}")
        print(f"Per-env SPS: {results['per_env_sps']:.1f}")
        print(f"Frames/sec: {results['frames_per_sec']:.1f}")
        print(f"Artifact: {artifact_path}")

        return 0

    finally:
        backend.close()


if __name__ == "__main__":
    sys.exit(main())
