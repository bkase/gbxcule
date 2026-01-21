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
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from gbxcule.backends.common import VecBackend

# ---------------------------------------------------------------------------
# Backend Registry
# ---------------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, type] = {}


def register_backends() -> None:
    """Lazily register available backends."""
    global BACKEND_REGISTRY

    if BACKEND_REGISTRY:
        return  # Already registered

    from gbxcule.backends.pyboy_single import PyBoySingleBackend
    from gbxcule.backends.pyboy_vec_mp import PyBoyVecMpBackend
    from gbxcule.backends.warp_vec import WarpVecBackend

    BACKEND_REGISTRY["pyboy_single"] = PyBoySingleBackend
    BACKEND_REGISTRY["pyboy_vec_mp"] = PyBoyVecMpBackend
    BACKEND_REGISTRY["warp_vec"] = WarpVecBackend


def create_backend(
    name: str,
    rom_path: str,
    *,
    num_envs: int = 1,
    num_workers: int | None = None,
    frames_per_step: int = 24,
    release_after_frames: int = 8,
    obs_dim: int = 32,
    base_seed: int | None = None,
) -> VecBackend:
    """Create a backend instance by name.

    Args:
        name: Backend name (pyboy_single, pyboy_vec_mp).
        rom_path: Path to ROM file.
        num_envs: Number of environments.
        num_workers: Number of workers (MP backend only).
        frames_per_step: Frames per step.
        release_after_frames: Frames before button release.
        obs_dim: Observation dimension.
        base_seed: Optional base seed.

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

    if name == "pyboy_single":
        return backend_cls(
            rom_path,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            obs_dim=obs_dim,
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
        )
    elif name == "warp_vec":
        return backend_cls(
            rom_path,
            num_envs=num_envs,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            obs_dim=obs_dim,
            base_seed=base_seed,
        )
    else:
        # Generic fallback
        return backend_cls(rom_path)


# ---------------------------------------------------------------------------
# Action Generation (Pure)
# ---------------------------------------------------------------------------

# Generator version for reproducibility tracking
ACTION_GEN_VERSION = "1.0"


def generate_actions(
    step_idx: int,
    num_envs: int,
    seed: int | None,
    gen_name: str,
) -> np.ndarray:
    """Generate actions for a single step (pure function).

    This function is deterministic given the same inputs. It has no side effects.

    Args:
        step_idx: The current step index (0-based).
        num_envs: Number of environments to generate actions for.
        seed: Base seed for random generation. None means noop.
        gen_name: Generator name ("noop" or "seeded_random").

    Returns:
        int32 array of shape (num_envs,) with action indices.

    Raises:
        ValueError: If gen_name is unknown.
    """
    if gen_name == "noop":
        return np.zeros((num_envs,), dtype=np.int32)

    if gen_name == "seeded_random":
        if seed is None:
            raise ValueError("seeded_random generator requires a seed")
        # Create a deterministic seed for this step
        # Using step_idx in the seed ensures different actions per step
        step_seed = seed + step_idx
        rng = np.random.default_rng(step_seed)
        return rng.integers(0, 9, size=(num_envs,), dtype=np.int32)

    raise ValueError(f"Unknown action generator: {gen_name}")


def get_action_gen_metadata(gen_name: str, seed: int | None) -> dict[str, Any]:
    """Get metadata for action generator (for artifact recording).

    Args:
        gen_name: Generator name.
        seed: Base seed (or None).

    Returns:
        Dict with name, version, and seed.
    """
    return {
        "name": gen_name,
        "version": ACTION_GEN_VERSION,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# State Comparison (Pure)
# ---------------------------------------------------------------------------

# Canonical register keys for normalization
CANONICAL_REGISTER_KEYS = ["pc", "sp", "a", "f", "b", "c", "d", "e", "h", "l"]
CANONICAL_FLAG_KEYS = ["z", "n", "h", "c"]


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
    import numpy as np

    info: dict[str, Any] = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": platform.processor() or None,
        "gpu": None,  # Placeholder for CUDA info
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
) -> dict[str, Any]:
    """Run a benchmark and return results.

    Args:
        backend: The backend to benchmark.
        steps: Number of steps to measure.
        warmup_steps: Number of warmup steps (not timed).
        action_gen: Action generator name ("noop" or "seeded_random").
        actions_seed: Seed for action generation (required for seeded_random).
        frames_per_step: Frames per step (for FPS calculation).

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
        )
        backend.step(actions)

    # Measured run
    start_time = time.perf_counter()

    for i in range(steps):
        actions = generate_actions(
            step_idx=i,
            num_envs=num_envs,
            seed=actions_seed,
            gen_name=action_gen,
        )
        backend.step(actions)

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


# ---------------------------------------------------------------------------
# Artifact Writer
# ---------------------------------------------------------------------------


def write_artifact(
    output_dir: Path,
    *,
    run_id: str,
    system_info: dict[str, Any],
    config: dict[str, Any],
    results: dict[str, Any],
) -> Path:
    """Write a benchmark artifact to JSON.

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

    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "system": system_info,
        "config": config,
        "results": results,
    }

    artifact_path = output_dir / f"{run_id}.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)

    return artifact_path


def write_scaling_artifact(
    output_dir: Path,
    *,
    run_id: str,
    system_info: dict[str, Any],
    sweep_config: dict[str, Any],
    results_list: list[dict[str, Any]],
) -> Path:
    """Write a scaling benchmark artifact to JSON.

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

    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "system": system_info,
        "sweep_config": sweep_config,
        "results": results_list,
    }

    artifact_path = output_dir / f"{run_id}__scaling.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)

    return artifact_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GBxCuLE Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Backend selection (required for benchmark mode, ignored in verify mode)
    parser.add_argument(
        "--backend",
        choices=["pyboy_single", "pyboy_vec_mp", "warp_vec"],
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
        help="Number of steps to measure (default: 100)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps (default: 10)",
    )
    parser.add_argument(
        "--stage",
        choices=["emulate_only", "full_step", "reward_only", "obs_only"],
        default="emulate_only",
        help="Execution stage (default: emulate_only)",
    )

    # Backend configuration
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of environments (default: 1)"
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
        help="Number of workers for MP backend (default: num_envs)",
    )
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=24,
        help="Frames per step (default: 24)",
    )
    parser.add_argument(
        "--release-after-frames",
        type=int,
        default=8,
        help="Frames before button release (default: 8)",
    )

    # Action configuration
    parser.add_argument(
        "--actions-seed",
        type=int,
        default=None,
        help="Seed for action generation (default: None = noop)",
    )
    parser.add_argument(
        "--action-gen",
        choices=["noop", "seeded_random"],
        default="noop",
        help="Action generator type (default: noop)",
    )

    # Verify mode configuration
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable verification mode (ref vs DUT comparison)",
    )
    parser.add_argument(
        "--ref-backend",
        choices=["pyboy_single", "pyboy_vec_mp"],
        default="pyboy_single",
        help="Reference backend for verification (default: pyboy_single)",
    )
    parser.add_argument(
        "--dut-backend",
        choices=["pyboy_single", "pyboy_vec_mp", "warp_vec"],
        default="warp_vec",
        help="Device-under-test backend for verification (default: warp_vec)",
    )
    parser.add_argument(
        "--verify-steps",
        type=int,
        default=100,
        help="Number of verification steps (default: 100)",
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=1,
        help="Compare states every N steps (default: 1)",
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
        help="Output directory for artifacts (default: bench/runs)",
    )

    return parser.parse_args(argv)


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

    # Build sweep config (shared across all runs)
    sweep_config = {
        "backend": args.backend,
        "rom_path": str(rom_path),
        "rom_sha256": rom_sha256,
        "stage": args.stage,
        "env_counts": env_counts,
        "frames_per_step": args.frames_per_step,
        "release_after_frames": args.release_after_frames,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "action_generator": get_action_gen_metadata(args.action_gen, effective_seed),
        "sync_every": None,
    }

    results_list: list[dict[str, Any]] = []

    print(f"Running scaling sweep: env_counts={env_counts}")
    print(f"Backend: {args.backend}")
    print(f"ROM: {rom_path.name}")
    print()

    for num_envs in env_counts:
        # pyboy_single only supports 1 env
        if args.backend == "pyboy_single" and num_envs != 1:
            print(f"  Skipping num_envs={num_envs} (pyboy_single only supports 1)")
            continue

        try:
            backend = create_backend(
                args.backend,
                str(rom_path),
                num_envs=num_envs,
                num_workers=args.num_workers,
                frames_per_step=args.frames_per_step,
                release_after_frames=args.release_after_frames,
                base_seed=args.actions_seed,
            )
        except Exception as e:
            print(f"  Error creating backend for num_envs={num_envs}: {e}")
            continue

        try:
            results = run_benchmark(
                backend,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                action_gen=args.action_gen,
                actions_seed=effective_seed,
                frames_per_step=args.frames_per_step,
            )

            # Add entry with env_count + results
            entry = {
                "num_envs": num_envs,
                "device": backend.device,
                **results,
            }
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

    print(f"Verification mode: ref={args.ref_backend} vs dut={args.dut_backend}")
    print(f"ROM: {rom_path.name}")
    print(f"Steps: {args.verify_steps}, compare every {args.compare_every}")
    print()

    # Use seed for seeded_random generator, None for noop
    effective_seed = args.actions_seed if args.action_gen == "seeded_random" else None

    # Create ref backend
    try:
        ref_backend = create_backend(
            args.ref_backend,
            str(rom_path),
            num_envs=1,  # Verify uses single env for now
            frames_per_step=args.frames_per_step,
            release_after_frames=args.release_after_frames,
            base_seed=effective_seed,
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
            base_seed=effective_seed,
        )
    except Exception as e:
        ref_backend.close()
        print(f"Error creating DUT backend: {e}", file=sys.stderr)
        return 1

    try:
        # Reset both backends with same seed
        ref_backend.reset(seed=effective_seed)
        dut_backend.reset(seed=effective_seed)

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
                if diff is not None:
                    # Mismatch found
                    print(f"MISMATCH at step {step_idx}")
                    print(f"First differing fields: {list(diff.keys())[:5]}")
                    # Return mismatch info (bundle writing will be added later)
                    # For now, write action trace to output dir
                    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                    trace_path = (
                        Path(args.output_dir)
                        / "mismatch"
                        / f"{timestamp}_{rom_path.stem}"
                        / "actions.jsonl"
                    )
                    write_actions_trace(actions_trace, trace_path)
                    print(f"Action trace: {trace_path}")
                    return 1

        print(f"PASS: No mismatches in {args.verify_steps} steps")
        return 0

    finally:
        ref_backend.close()
        dut_backend.close()


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Validate ROM path
    if args.rom:
        rom_path = Path(args.rom)
        if not rom_path.exists():
            print(f"Error: ROM file not found: {rom_path}", file=sys.stderr)
            return 1
    elif args.suite:
        # Suite support - for now just use the first ROM in the suite
        import yaml

        suite_path = Path(args.suite)
        if not suite_path.exists():
            print(f"Error: Suite file not found: {suite_path}", file=sys.stderr)
            return 1

        with open(suite_path) as f:
            suite = yaml.safe_load(f)

        if not suite.get("roms"):
            print("Error: Suite file has no ROMs defined", file=sys.stderr)
            return 1

        # Use first ROM in suite
        first_rom = suite["roms"][0]
        rom_path = suite_path.parent / first_rom["path"]
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
            base_seed=args.actions_seed,
        )
    except Exception as e:
        print(f"Error creating backend: {e}", file=sys.stderr)
        return 1

    try:
        # Collect system info
        system_info = get_system_info()

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
            "num_workers": args.num_workers,
            "frames_per_step": args.frames_per_step,
            "release_after_frames": args.release_after_frames,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "action_generator": get_action_gen_metadata(
                args.action_gen, effective_seed
            ),
            "sync_every": None,  # For future GPU backends
        }

        # Run benchmark
        results = run_benchmark(
            backend,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            action_gen=args.action_gen,
            actions_seed=effective_seed,
            frames_per_step=args.frames_per_step,
        )

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
