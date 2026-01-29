#!/usr/bin/env python3
"""Random exploration using CPU backend to find successful trajectories.

When a goal is reached, saves the successful end state which can be used
as a starting point for the next stage or for curriculum learning.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gbxcule.backends.warp_vec import WarpVecCpuBackend
from gbxcule.rl.goal_template import load_goal_template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", required=True, help="Path to start .state file")
    parser.add_argument("--goal-dir", required=True, help="Goal template directory")
    parser.add_argument(
        "--num-envs", type=int, default=16, help="Parallel environments"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Max steps per episode"
    )
    parser.add_argument("--episodes", type=int, default=100, help="Episodes to run")
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument(
        "--tau", type=float, default=0.05, help="Goal distance threshold"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Directory to save successful states"
    )
    parser.add_argument(
        "--save-first-n", type=int, default=3, help="Save first N successful states"
    )
    args = parser.parse_args()

    # Create backend
    print(f"Creating CPU backend with {args.num_envs} envs...")
    backend = WarpVecCpuBackend(
        args.rom,
        num_envs=args.num_envs,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        render_pixels=True,
    )

    # Load goal template
    print(f"Loading goal template from {args.goal_dir}...")
    template, meta = load_goal_template(
        Path(args.goal_dir),
        action_codec_id=backend.action_codec.id,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        stack_k=1,
        dist_metric=None,
        pipeline_version=None,
    )
    goal_np = np.array(template, dtype=np.float32)  # [1, 72, 80]

    # Load initial state for all envs
    print(f"Loading initial state from {args.state}...")
    from gbxcule.core.state_io import apply_state_to_warp_backend, load_pyboy_state

    # MBC3 ROMs like Pokemon Red have 32KB cart RAM
    initial_state = load_pyboy_state(args.state, expected_cart_ram_size=32768)

    num_actions = backend.num_actions
    print(f"Action space: {num_actions} actions")
    print(f"Max steps: {args.max_steps}")
    print(f"Goal threshold (tau): {args.tau}")

    # Setup save directory
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Will save first {args.save_first_n} successful states to: {save_dir}")
    print()

    # Stats
    total_episodes = 0
    total_goals = 0
    min_dist_ever = 1.0
    best_steps_to_goal = None
    saved_states = 0

    start_time = time.time()

    try:
        for _ep_batch in range(args.episodes // args.num_envs + 1):
            if total_episodes >= args.episodes:
                break

            # Reset all envs to initial state
            backend.reset()
            for env_idx in range(args.num_envs):
                apply_state_to_warp_backend(initial_state, backend, env_idx=env_idx)

            episode_done = np.zeros(args.num_envs, dtype=bool)
            steps_to_goal = np.full(args.num_envs, -1, dtype=np.int32)
            min_dist_episode = np.ones(args.num_envs, dtype=np.float32)
            action_traces = [
                [] for _ in range(args.num_envs)
            ]  # Track actions for replay

            for step in range(args.max_steps):
                # Random actions
                actions = np.random.randint(
                    0, num_actions, size=args.num_envs, dtype=np.int32
                )

                # Record actions for trace
                for i in range(args.num_envs):
                    if not episode_done[i]:
                        action_traces[i].append(int(actions[i]))

                # Step
                backend.step(actions)

                # Get observations (render pixels)
                # Note: pixels_wp returns flat [num_envs * 72 * 80] array
                backend.render_pixels_snapshot()
                pixels_flat = backend.pixels_wp().numpy()  # [num_envs * 72 * 80]
                obs = pixels_flat.reshape(args.num_envs, 72, 80)  # [num_envs, 72, 80]

                # Compute distance to goal
                obs_float = obs.astype(np.float32)
                for i in range(args.num_envs):
                    if not episode_done[i]:
                        dist = np.abs(obs_float[i] - goal_np).mean() / 3.0
                        min_dist_episode[i] = min(min_dist_episode[i], dist)

                        if dist < args.tau:
                            episode_done[i] = True
                            steps_to_goal[i] = step + 1
                            total_goals += 1
                            if (
                                best_steps_to_goal is None
                                or (step + 1) < best_steps_to_goal
                            ):
                                best_steps_to_goal = step + 1
                            ep_num = total_episodes + i
                            print(f"GOAL! Ep {ep_num}: step {step + 1}, d={dist:.4f}")

                            # Save successful state and action trace
                            if save_dir and saved_states < args.save_first_n:
                                s = step + 1
                                prefix = f"success_{saved_states:02d}_ep{ep_num}_s{s}"

                                # Save final state
                                state_path = save_dir / f"{prefix}.state"
                                backend.save_state_file(str(state_path), env_idx=i)

                                # Save action trace as JSON
                                trace_path = save_dir / f"{prefix}_trace.json"
                                import json

                                trace_data = {
                                    "start_state": args.state,
                                    "goal_dir": args.goal_dir,
                                    "steps": step + 1,
                                    "final_dist": float(dist),
                                    "actions": action_traces[i],
                                }
                                with open(trace_path, "w") as tf:
                                    json.dump(trace_data, tf)

                                print(f"  Saved state to: {state_path}")
                                print(f"  Saved trace to: {trace_path}")
                                saved_states += 1

                # Early exit if all done
                if episode_done.all():
                    break

            # Update stats
            for i in range(args.num_envs):
                if total_episodes + i < args.episodes:
                    min_dist_ever = min(min_dist_ever, min_dist_episode[i])

            batch_size = min(args.num_envs, args.episodes - total_episodes)
            total_episodes += batch_size

            # Progress
            elapsed = time.time() - start_time
            eps_per_sec = total_episodes / elapsed
            print(
                f"Episodes: {total_episodes}/{args.episodes}, "
                f"Goals: {total_goals} ({100 * total_goals / total_episodes:.1f}%), "
                f"Min dist: {min_dist_ever:.4f}, "
                f"Speed: {eps_per_sec:.1f} ep/s"
            )

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        backend.close()

    # Final stats
    elapsed = time.time() - start_time
    print()
    print("=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Total episodes: {total_episodes}")
    print(f"Goals reached: {total_goals} ({100 * total_goals / total_episodes:.1f}%)")
    print(f"Minimum distance ever: {min_dist_ever:.4f}")
    if best_steps_to_goal:
        print(f"Best steps to goal: {best_steps_to_goal}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Episodes/sec: {total_episodes / elapsed:.2f}")


if __name__ == "__main__":
    main()
