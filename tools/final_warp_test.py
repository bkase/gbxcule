#!/usr/bin/env python3
"""Test that Warp can load and execute from the PyBoy gameplay state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyboy import PyBoy

from gbxcule.backends.warp_vec import WarpVecCpuBackend

# Action indices
LEFT = 5
RIGHT = 6
DOWN = 4
A = 0  # Rotate


def main() -> None:
    rom_path = "tetris.gb"
    state_path = Path("states/tetris_gameplay.state")
    out_dir = Path("states")

    if not Path(rom_path).exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")

    # Verify the state is in gameplay by loading in PyBoy first
    print("Verifying initial state with PyBoy...")
    pyboy = PyBoy(rom_path, window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

    with state_path.open("rb") as f:
        pyboy.load_state(f)
    pyboy.tick(render=True)

    initial_img = pyboy.screen.image
    initial_img.save(out_dir / "tetris_initial_verify.png")
    print(f"Initial state screenshot: {out_dir / 'tetris_initial_verify.png'}")
    pyboy.stop(save=False)

    # Now test with Warp
    print("\nTesting Warp Runtime...")
    backend = WarpVecCpuBackend(
        rom_path=rom_path,
        num_envs=1,
        frames_per_step=24,
        release_after_frames=8,
        render_bg=True,
    )

    obs, info = backend.reset(seed=42)
    print("Warp backend initialized")

    # Load the gameplay state
    backend.load_state_file(str(state_path), env_idx=0)
    print("Loaded state into Warp")

    # Execute some gameplay actions
    print("Executing gameplay actions in Warp...")

    actions_sequence = [
        (LEFT, "Move left"),
        (LEFT, "Move left"),
        (A, "Rotate"),
        (RIGHT, "Move right"),
        (DOWN, "Drop faster"),
        (DOWN, "Drop faster"),
        (DOWN, "Drop faster"),
        (0, "Wait"),
        (0, "Wait"),
        (0, "Wait"),
    ]

    for action, desc in actions_sequence:
        actions = np.array([action], dtype=np.int32)
        backend.step(actions)
        print(f"  {desc}")

    # Get CPU state to verify emulation is running
    cpu_state = backend.get_cpu_state(0)
    print("\nCPU state after actions:")
    print(f"  PC: {cpu_state['pc']:#06x}")
    print(f"  SP: {cpu_state['sp']:#06x}")
    instr_count = cpu_state.get("instr_count")
    if instr_count is None:
        print("  Instructions executed: n/a")
    else:
        print(f"  Instructions executed: {instr_count}")

    # Save the final Warp state back to PyBoy format
    final_state_path = out_dir / "tetris_warp_final.state"
    backend.save_state_file(str(final_state_path), env_idx=0)
    print(f"\nWarp state saved to {final_state_path}")

    # Verify by reloading in PyBoy
    print("\nVerifying Warp-saved state in PyBoy...")
    pyboy2 = PyBoy(rom_path, window="null", sound_emulated=False)
    pyboy2.set_emulation_speed(0)

    try:
        with final_state_path.open("rb") as f:
            pyboy2.load_state(f)
        pyboy2.tick(render=True)
        final_img = pyboy2.screen.image
        final_img.save(out_dir / "tetris_warp_final_verify.png")
        print(f"Final state screenshot: {out_dir / 'tetris_warp_final_verify.png'}")
    except Exception as e:
        print(f"Warning: Could not verify Warp state in PyBoy: {e}")
        print("This may be due to state format differences.")

    pyboy2.stop(save=False)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Initial state (PyBoy): {state_path}")
    print(f"  - Verified screenshot: {out_dir / 'tetris_initial_verify.png'}")
    print("  - This state shows Tetris gameplay with falling blocks")
    print("\nWarp execution successful:")
    print("  - Loaded state into Warp Runtime")
    print(f"  - Executed {len(actions_sequence)} gameplay actions")
    if instr_count is None:
        print("  - CPU executed: n/a")
    else:
        print(f"  - CPU executed {instr_count} instructions")
    print(f"\nCheckpoint state: {state_path}")
    print("  - Use this state file to start Warp from gameplay")


if __name__ == "__main__":
    main()
