#!/usr/bin/env python3
"""Unified RL evaluation CLI.

Usage (Dreamer v3):
  uv run python tools/rl_eval.py --algo dreamer_v3 --checkpoint <ckpt> \
    --rom red.gb --state states/rl_stage1_exit_oak/start.state \
    --goal-dir states/rl_stage1_exit_oak --episodes 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _require_torch():  # type: ignore[no-untyped-def]
    import importlib

    return importlib.import_module("torch")


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", default="dreamer_v3")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--rom", default=None, help="Path to ROM (optional)")
    parser.add_argument("--state", default=None, help="Path to .state file (optional)")
    parser.add_argument("--goal-dir", default=None, help="Goal template dir (optional)")
    parser.add_argument("--frames-per-step", type=int, default=None)
    parser.add_argument("--release-after-frames", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", help="Use greedy actions")
    parser.add_argument(
        "--trajectory",
        default=None,
        help="Optional JSONL trajectory path (num-envs must be 1)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional summary JSON path (prints to stdout regardless)",
    )
    return parser.parse_args()


def _lower_median_int(values: list[int]) -> int | None:
    data = sorted(values)
    if not data:
        return None
    idx = (len(data) - 1) // 2
    return int(data[idx])


def _lower_median_float(values: list[float]) -> float | None:
    data = sorted(values)
    if not data:
        return None
    idx = (len(data) - 1) // 2
    return float(data[idx])


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _resolve_path(arg: str | None, cfg_val: Any, label: str) -> str:
    if arg:
        return str(Path(arg))
    if cfg_val:
        return str(Path(cfg_val))
    raise ValueError(f"{label} must be provided (missing from args and checkpoint)")


def _resolve_int(arg: int | None, cfg_val: Any) -> int:
    if arg is not None:
        return int(arg)
    if cfg_val is not None:
        return int(cfg_val)
    raise ValueError("missing required integer config")


def _dreamer_eval(args: argparse.Namespace) -> dict[str, Any]:
    if not _cuda_available():
        return {"skipped": "CUDA not available"}
    torch = _require_torch()
    if not torch.cuda.is_available():
        return {"skipped": "torch CUDA not available"}

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cuda")
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError("Checkpoint missing config dict")

    rom = _resolve_path(args.rom, cfg.get("rom"), "rom")
    state = _resolve_path(args.state, cfg.get("state"), "state")
    goal_dir = _resolve_path(args.goal_dir, cfg.get("goal_dir"), "goal_dir")
    frames_per_step = _resolve_int(args.frames_per_step, cfg.get("frames_per_step"))
    release_after_frames = _resolve_int(
        args.release_after_frames, cfg.get("release_after_frames")
    )

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W
    from gbxcule.rl.dreamer_v3.behavior import Actor
    from gbxcule.rl.dreamer_v3.decoders import CNNDecoder, MultiDecoder
    from gbxcule.rl.dreamer_v3.encoders import MultiEncoder, Packed2PixelEncoder
    from gbxcule.rl.dreamer_v3.heads import ContinueHead, RewardHead
    from gbxcule.rl.dreamer_v3.player import DreamerActorCore
    from gbxcule.rl.dreamer_v3.rssm import build_rssm
    from gbxcule.rl.dreamer_v3.world_model import WorldModel
    from gbxcule.rl.pokered_packed_goal_env import PokeredPackedGoalEnv

    rom_path = Path(rom)
    state_path = Path(state)
    goal_path = Path(goal_dir)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")
    if not goal_path.exists():
        raise FileNotFoundError(f"Goal dir not found: {goal_path}")

    env = PokeredPackedGoalEnv(
        rom,
        goal_dir=goal_dir,
        state_path=state,
        num_envs=int(args.num_envs),
        frames_per_step=frames_per_step,
        release_after_frames=release_after_frames,
        action_codec=cfg.get("action_codec"),
        max_steps=int(cfg.get("max_steps", 128)),
        step_cost=float(cfg.get("step_cost", -0.01)),
        alpha=float(cfg.get("alpha", 1.0)),
        goal_bonus=float(cfg.get("goal_bonus", 10.0)),
        tau=cfg.get("tau"),
        k_consecutive=cfg.get("k_consecutive"),
        info_mode="full",
    )

    action_dim = int(env.num_actions)
    image_size = (DOWNSAMPLE_H, DOWNSAMPLE_W)

    cnn_encoder = Packed2PixelEncoder(
        keys=["pixels"],
        image_size=image_size,
        channels_multiplier=int(cfg.get("cnn_channels_multiplier", 4)),
        stages=int(cfg.get("cnn_stages", 3)),
    )
    encoder = MultiEncoder(cnn_encoder, None)
    rssm = build_rssm(
        action_dim=action_dim,
        embed_dim=encoder.cnn_output_dim,
        stochastic_size=int(cfg.get("stochastic_size", 32)),
        discrete_size=int(cfg.get("discrete_size", 32)),
        recurrent_state_size=int(cfg.get("recurrent_state_size", 512)),
        dense_units=int(cfg.get("dense_units", 512)),
        hidden_size=int(cfg.get("hidden_size", 512)),
        unimix=0.01,
        rnn_dtype=torch.float32,
    )
    latent_state_size = int(cfg.get("stochastic_size", 32)) * int(
        cfg.get("discrete_size", 32)
    ) + int(cfg.get("recurrent_state_size", 512))
    decoder = CNNDecoder(
        keys=["pixels"],
        output_channels=[1],
        channels_multiplier=int(cfg.get("cnn_channels_multiplier", 4)),
        latent_state_size=latent_state_size,
        encoder_output_shape=cnn_encoder.output_shape,
        stages=int(cfg.get("cnn_stages", 3)),
    )
    observation_model = MultiDecoder(decoder, None)
    reward_model = RewardHead(
        input_dim=latent_state_size,
        bins=int(cfg.get("reward_bins", 255)),
        mlp_layers=int(cfg.get("head_mlp_layers", 2)),
        dense_units=int(cfg.get("dense_units", 512)),
    )
    continue_model = ContinueHead(
        input_dim=latent_state_size,
        mlp_layers=int(cfg.get("head_mlp_layers", 2)),
        dense_units=int(cfg.get("dense_units", 512)),
    )
    world_model = WorldModel(
        encoder=encoder,
        rssm=rssm,
        observation_model=observation_model,
        reward_model=reward_model,
        continue_model=continue_model,
        cnn_keys=["pixels"],
        mlp_keys=[],
        reward_low=float(cfg.get("reward_low", -20.0)),
        reward_high=float(cfg.get("reward_high", 20.0)),
        continue_scale_factor=float(cfg.get("continue_scale_factor", 1.0)),
    ).to("cuda")

    actor = Actor(
        latent_state_size=latent_state_size,
        actions_dim=[action_dim],
        is_continuous=False,
        distribution_cfg={"type": "discrete"},
        dense_units=int(cfg.get("dense_units", 512)),
        mlp_layers=int(cfg.get("head_mlp_layers", 2)),
        unimix=0.01,
        action_clip=1.0,
        init_std=0.0,
        min_std=1.0,
        max_std=1.0,
    ).to("cuda")

    world_model.load_state_dict(ckpt.get("world_model", {}))
    actor.load_state_dict(ckpt.get("actor", {}))
    world_model.eval()
    actor.eval()

    actor_core = DreamerActorCore(
        encoder=world_model.encoder,
        rssm=world_model.rssm,
        actor=actor,
        action_dim=action_dim,
        greedy=bool(args.greedy),
    )
    actor_core.sync_player()

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    action_gen = torch.Generator(device="cuda")
    action_gen.manual_seed(int(args.seed) + 1)

    if int(args.episodes) < 1:
        raise ValueError("episodes must be >= 1")

    if args.trajectory is not None and int(args.num_envs) != 1:
        raise ValueError("trajectory is only supported for num_envs == 1")

    obs = env.reset(seed=int(args.seed))
    device = obs.device
    state = actor_core.init_state(int(args.num_envs), device)
    is_first = torch.ones((int(args.num_envs),), dtype=torch.bool, device=device)

    ep_returns = torch.zeros((int(args.num_envs),), device=device)
    ep_steps = torch.zeros((int(args.num_envs),), device=device, dtype=torch.int32)

    episodes_collected = 0
    successes = 0
    returns: list[float] = []
    steps_all: list[int] = []
    steps_success: list[int] = []
    success_returns: list[float] = []
    fail_returns: list[float] = []
    dist_end: list[float] = []

    trajectory: list[dict[str, Any]] = []
    episode_idx = 0
    start_time = time.time()

    with torch.inference_mode():
        while episodes_collected < int(args.episodes):
            actions, state = actor_core.act(obs, is_first, state, generator=action_gen)
            next_obs, reward, done, trunc, info = env.step(actions)

            ep_returns.add_(reward)
            ep_steps.add_(1)

            terminated = done | trunc

            if int(args.num_envs) == 1 and args.trajectory is not None:
                dist_val = None
                if isinstance(info, dict) and "dist" in info:
                    dist_val = float(info["dist"][0].item())
                trajectory.append(
                    {
                        "episode_idx": episode_idx,
                        "t": int(ep_steps[0].item()),
                        "action": int(actions[0].item()),
                        "reward": float(reward[0].item()),
                        "done": bool(done[0].item()),
                        "trunc": bool(trunc[0].item()),
                        "dist": dist_val,
                    }
                )

            if torch.any(terminated):
                idxs = torch.nonzero(terminated, as_tuple=False).flatten()
                for idx in idxs.tolist():
                    episodes_collected += 1
                    steps_all.append(int(ep_steps[idx].item()))
                    if bool(done[idx].item()):
                        successes += 1
                        steps_success.append(int(ep_steps[idx].item()))
                        success_returns.append(float(ep_returns[idx].item()))
                    else:
                        fail_returns.append(float(ep_returns[idx].item()))
                    if isinstance(info, dict) and "dist" in info:
                        dist_end.append(float(info["dist"][idx].item()))
                    returns.append(float(ep_returns[idx].item()))
                    ep_returns[idx] = 0.0
                    ep_steps[idx] = 0
                    if int(args.num_envs) == 1 and args.trajectory is not None:
                        episode_idx += 1
                    if episodes_collected >= int(args.episodes):
                        break

            reset_mask = None
            if isinstance(info, dict):
                reset_mask = info.get("reset_mask")
            if reset_mask is None:
                reset_mask = terminated

            if hasattr(env, "reset_mask"):
                env.reset_mask(reset_mask)

            is_first = reset_mask.to(torch.bool)
            obs = env.obs if hasattr(env, "obs") else next_obs

            if (
                args.max_steps is not None
                and episodes_collected < int(args.episodes)
                and int(ep_steps.max().item()) >= int(args.max_steps)
            ):
                raise RuntimeError(
                    f"max_steps {args.max_steps} reached without episode termination"
                )

    if args.trajectory is not None:
        _write_jsonl(Path(args.trajectory), trajectory)

    mean_return = float(sum(returns) / max(1, len(returns)))
    success_rate = float(successes / int(args.episodes))
    median_steps = _lower_median_int(steps_success)
    steps_p50_success = _lower_median_int(steps_success)
    return_mean_success = (
        float(sum(success_returns) / len(success_returns)) if success_returns else None
    )
    return_mean_fail = (
        float(sum(fail_returns) / len(fail_returns)) if fail_returns else None
    )
    dist_at_end_p50 = _lower_median_float(dist_end)
    mean_ep_len = float(sum(steps_all) / max(1, len(steps_all)))

    summary = {
        "episodes": int(args.episodes),
        "successes": int(successes),
        "success_rate": success_rate,
        "median_steps_to_goal": median_steps,
        "mean_return": mean_return,
        "steps_p50_success": steps_p50_success,
        "return_mean_success": return_mean_success,
        "return_mean_fail": return_mean_fail,
        "dist_at_end_p50": dist_at_end_p50,
        "Rewards/rew_avg": mean_return,
        "Game/ep_len_avg": mean_ep_len,
        "wall_time_s": float(time.time() - start_time),
    }

    env.close()
    return summary


def main() -> int:
    args = _parse_args()
    if args.algo != "dreamer_v3":
        raise ValueError("Only --algo dreamer_v3 is supported")

    summary = _dreamer_eval(args)
    print(json.dumps(summary))

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
