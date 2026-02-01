#!/usr/bin/env python3
"""Visualize Dreamer v3 rollouts and imagined frames (GPU).

Usage:
  uv run python tools/rl_dreamer_visualize.py --checkpoint <checkpoint.pt> \
    --rom red.gb --state states/rl_stage1_exit_oak/start.state \
    --goal-dir states/rl_stage1_exit_oak --steps 256 --imagine-horizon 64
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _require_torch():
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
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--rom", default=None, help="Path to ROM (optional)")
    parser.add_argument("--state", default=None, help="Path to .state file (optional)")
    parser.add_argument("--goal-dir", default=None, help="Goal template dir (optional)")
    parser.add_argument("--task", default=None, help="Task name (goal or parcel)")
    parser.add_argument("--events-start", type=int, default=None)
    parser.add_argument("--events-length", type=int, default=None)
    parser.add_argument("--snow-bonus", type=float, default=None)
    parser.add_argument("--snow-size", type=int, default=None)
    parser.add_argument("--get-parcel-bonus", type=float, default=None)
    parser.add_argument("--deliver-bonus", type=float, default=None)
    parser.add_argument("--frames-per-step", type=int, default=None)
    parser.add_argument("--release-after-frames", type=int, default=None)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--imagine-horizon", type=int, default=64)
    parser.add_argument(
        "--imagine-from-step",
        type=int,
        default=-1,
        help="Step index to seed imagination (default: last step)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--stop-on-done", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output-tag", default="dreamer_visuals")
    parser.add_argument("--run-root", default="bench/runs/rl")
    parser.add_argument("--skip-recon", action="store_true")
    parser.add_argument("--skip-imagined", action="store_true")
    return parser.parse_args()


def _resolve_path(arg: str | None, cfg_val: Any, label: str) -> str:
    if arg:
        return str(Path(arg))
    if cfg_val:
        return str(Path(cfg_val))
    raise ValueError(f"{label} must be provided (missing from args and checkpoint)")


def _resolve_int(arg: int | None, cfg_val: Any, default: int | None = None) -> int:
    if arg is not None:
        return int(arg)
    if cfg_val is not None:
        return int(cfg_val)
    if default is not None:
        return int(default)
    raise ValueError("missing required integer config")


def _resolve_float(
    arg: float | None, cfg_val: Any, default: float | None = None
) -> float:
    if arg is not None:
        return float(arg)
    if cfg_val is not None:
        return float(cfg_val)
    if default is not None:
        return float(default)
    raise ValueError("missing required float config")


def _resolve_task(arg: str | None, cfg: dict[str, Any]) -> str:
    if arg:
        return str(arg)
    cfg_task = cfg.get("task")
    if cfg_task:
        return str(cfg_task)
    if cfg.get("events_start") is not None or cfg.get("events_length") is not None:
        return "parcel"
    if cfg.get("snow_bonus") is not None or cfg.get("get_parcel_bonus") is not None:
        return "parcel"
    if cfg.get("goal_dir") is not None:
        return "goal"
    return "parcel"


def _write_frame_png(frame: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("PIL is required to write frames") from exc
    palette = np.array([255, 170, 85, 0], dtype=np.uint8)
    img = Image.fromarray(palette[frame], mode="L")
    img.save(path)


def _encode_mp4(frames_dir: Path, fps: int, output: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode MP4")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def _write_video(frames: list[np.ndarray], *, fps: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir)
        for idx, frame in enumerate(frames):
            _write_frame_png(frame, frames_dir / f"frame_{idx:06d}.png")
        _encode_mp4(frames_dir, fps, output)


def _packed_to_shades(packed) -> np.ndarray:  # type: ignore[no-untyped-def]
    from gbxcule.rl.packed_pixels import unpack_2bpp_u8

    packed_cpu = packed.detach().to("cpu")
    unpacked = unpack_2bpp_u8(packed_cpu)
    return unpacked.squeeze().numpy()


def _pred_to_shades(pred) -> np.ndarray:  # type: ignore[no-untyped-def]
    torch = _require_torch()
    pred_cpu = pred.detach().to("cpu")
    shades = torch.clamp((pred_cpu + 0.5) * 3.0, 0.0, 3.0)
    shades = shades.round().to(torch.uint8)
    return shades.squeeze().numpy()


def _extract_info_scalar(info: Any, key: str) -> Any | None:
    if not isinstance(info, dict) or key not in info:
        return None
    value = info[key]
    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.detach().to("cpu").reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[0]
    return value


def _cast_int(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _cast_float(value: Any | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _cast_bool(value: Any | None) -> bool | None:
    if value is None:
        return None
    try:
        return bool(int(value))
    except Exception:
        return None


def _senses_xyz(obs: Any) -> tuple[int | None, int | None, int | None]:
    if not isinstance(obs, dict) or "senses" not in obs:
        return None, None, None
    senses = obs["senses"]
    if hasattr(senses, "numel") and senses.numel() < 3:
        return None, None, None
    try:
        if senses.ndim == 1:
            map_id = senses[0].item()
            x = senses[1].item()
            y = senses[2].item()
        else:
            map_id = senses[0, 0].item()
            x = senses[0, 1].item()
            y = senses[0, 2].item()
        return int(map_id), int(x), int(y)
    except Exception:
        return None, None, None


def _collect_step_info(info: Any, obs: Any) -> dict[str, Any]:
    map_id = _extract_info_scalar(info, "map_id")
    x_val = _extract_info_scalar(info, "x")
    y_val = _extract_info_scalar(info, "y")
    novelty = _extract_info_scalar(info, "novelty")
    got_parcel = _extract_info_scalar(info, "got_parcel")
    delivered = _extract_info_scalar(info, "delivered")
    dist = _extract_info_scalar(info, "dist")

    if map_id is None or x_val is None or y_val is None:
        fallback_map, fallback_x, fallback_y = _senses_xyz(obs)
        if map_id is None:
            map_id = fallback_map
        if x_val is None:
            x_val = fallback_x
        if y_val is None:
            y_val = fallback_y

    return {
        "map_id": _cast_int(map_id),
        "x": _cast_int(x_val),
        "y": _cast_int(y_val),
        "novelty": _cast_bool(novelty),
        "got_parcel": _cast_bool(got_parcel),
        "delivered": _cast_bool(delivered),
        "dist": _cast_float(dist),
    }


@dataclass(frozen=True)
class VisualConfig:
    steps: int
    imagine_horizon: int
    imagine_from_step: int
    seed: int
    greedy: bool
    stop_on_done: bool
    fps: int
    output_tag: str
    run_root: str
    skip_recon: bool
    skip_imagined: bool


DEFAULT_EVENTS_START = 0xD747
DEFAULT_EVENTS_LENGTH = 320
DEFAULT_SNOW_BONUS = 0.01
DEFAULT_SNOW_SIZE = 65536
DEFAULT_GET_PARCEL_BONUS = 5.0
DEFAULT_DELIVER_BONUS = 10.0


def main() -> int:
    args = _parse_args()
    if not _cuda_available():
        print(json.dumps({"skipped": "CUDA not available"}))
        return 0
    torch = _require_torch()
    if not torch.cuda.is_available():
        print(json.dumps({"skipped": "torch CUDA not available"}))
        return 0

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cuda")
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError("Checkpoint missing config dict")

    task = _resolve_task(args.task, cfg).lower()
    if task in ("goal", "goal_template", "goal-template", "template"):
        task = "goal"
    elif task in ("parcel", "oak_parcel", "oaks_parcel", "parcel_task"):
        task = "parcel"
    else:
        raise ValueError(f"unknown task: {task}")

    rom = _resolve_path(args.rom, cfg.get("rom"), "rom")
    state = _resolve_path(args.state, cfg.get("state"), "state")
    goal_dir = args.goal_dir if args.goal_dir is not None else cfg.get("goal_dir")
    frames_per_step = _resolve_int(args.frames_per_step, cfg.get("frames_per_step"))
    release_after_frames = _resolve_int(
        args.release_after_frames, cfg.get("release_after_frames")
    )

    steps = int(args.steps)
    if steps < 1:
        raise ValueError("steps must be >= 1")

    imagine_horizon = int(args.imagine_horizon)
    if imagine_horizon < 1:
        raise ValueError("imagine-horizon must be >= 1")

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W
    from gbxcule.rl.dreamer_v3.behavior import Actor, Critic
    from gbxcule.rl.dreamer_v3.decoders import CNNDecoder, MLPDecoder, MultiDecoder
    from gbxcule.rl.dreamer_v3.encoders import (
        MLPEncoder,
        MultiEncoder,
        Packed2PixelEncoder,
    )
    from gbxcule.rl.dreamer_v3.heads import ContinueHead, RewardHead
    from gbxcule.rl.dreamer_v3.imagination import imagine_rollout
    from gbxcule.rl.dreamer_v3.player import DreamerActorCore
    from gbxcule.rl.dreamer_v3.rssm import build_rssm
    from gbxcule.rl.dreamer_v3.world_model import WorldModel
    from gbxcule.rl.experiment import Experiment

    rom_path = Path(rom)
    state_path = Path(state)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")

    goal_path = None
    senses_dim = None
    events_start = None
    events_length = None
    snow_bonus = None
    snow_size = None
    get_parcel_bonus = None
    deliver_bonus = None

    if task == "parcel":
        parcel_module = importlib.import_module("gbxcule.rl.pokered_packed_parcel_env")
        PokeredPackedParcelEnv = parcel_module.PokeredPackedParcelEnv

        events_start = _resolve_int(
            args.events_start, cfg.get("events_start"), DEFAULT_EVENTS_START
        )
        events_length = _resolve_int(
            args.events_length, cfg.get("events_length"), DEFAULT_EVENTS_LENGTH
        )
        snow_bonus = _resolve_float(
            args.snow_bonus, cfg.get("snow_bonus"), DEFAULT_SNOW_BONUS
        )
        snow_size = _resolve_int(
            args.snow_size, cfg.get("snow_size"), DEFAULT_SNOW_SIZE
        )
        get_parcel_bonus = _resolve_float(
            args.get_parcel_bonus,
            cfg.get("get_parcel_bonus"),
            DEFAULT_GET_PARCEL_BONUS,
        )
        deliver_bonus = _resolve_float(
            args.deliver_bonus, cfg.get("deliver_bonus"), DEFAULT_DELIVER_BONUS
        )
        senses_dim = 4 + int(events_length)
        env = PokeredPackedParcelEnv(
            rom,
            state_path=state,
            num_envs=1,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            action_codec=cfg.get("action_codec"),
            max_steps=int(cfg.get("max_steps", 2048)),
            events_start=int(events_start),
            events_length=int(events_length),
            snow_bonus=float(snow_bonus),
            snow_size=int(snow_size),
            get_parcel_bonus=float(get_parcel_bonus),
            deliver_bonus=float(deliver_bonus),
            info_mode="full",
        )
    else:
        from gbxcule.rl.pokered_packed_goal_env import PokeredPackedGoalEnv

        goal_dir = _resolve_path(args.goal_dir, cfg.get("goal_dir"), "goal_dir")
        goal_path = Path(goal_dir)
        if not goal_path.exists():
            raise FileNotFoundError(f"Goal dir not found: {goal_path}")
        env = PokeredPackedGoalEnv(
            rom,
            goal_dir=goal_dir,
            state_path=state,
            num_envs=1,
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
    mlp_encoder = None
    if task == "parcel":
        mlp_encoder = MLPEncoder(
            keys=["senses"],
            input_dims=[int(senses_dim or 0)],
            mlp_layers=int(cfg.get("head_mlp_layers", 2)),
            dense_units=int(cfg.get("dense_units", 512)),
        )
    encoder = MultiEncoder(cnn_encoder, mlp_encoder)
    rssm = build_rssm(
        action_dim=action_dim,
        embed_dim=encoder.cnn_output_dim + encoder.mlp_output_dim,
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
    mlp_decoder = None
    if task == "parcel":
        mlp_decoder = MLPDecoder(
            keys=["senses"],
            output_dims=[int(senses_dim or 0)],
            latent_state_size=latent_state_size,
            mlp_layers=int(cfg.get("head_mlp_layers", 2)),
            dense_units=int(cfg.get("dense_units", 512)),
        )
    observation_model = MultiDecoder(decoder, mlp_decoder)
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
        mlp_keys=["senses"] if task == "parcel" else [],
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
    critic = Critic(
        latent_state_size=latent_state_size,
        bins=int(cfg.get("value_bins", 255)),
        dense_units=int(cfg.get("dense_units", 512)),
        mlp_layers=int(cfg.get("head_mlp_layers", 2)),
    ).to("cuda")

    world_model.load_state_dict(ckpt.get("world_model", {}))
    actor.load_state_dict(ckpt.get("actor", {}))
    critic.load_state_dict(ckpt.get("critic", {}))

    world_model.eval()
    actor.eval()
    critic.eval()

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

    meta: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "rom": str(rom_path),
        "state": str(state_path),
        "task": task,
        "frames_per_step": int(frames_per_step),
        "release_after_frames": int(release_after_frames),
        "num_envs": 1,
    }
    if goal_path is not None:
        meta["goal_dir"] = str(goal_path)
    if task == "parcel":
        meta.update(
            {
                "events_start": int(events_start or 0),
                "events_length": int(events_length or 0),
                "senses_dim": int(senses_dim or 0),
                "snow_bonus": float(snow_bonus or 0.0),
                "snow_size": int(snow_size or 0),
                "get_parcel_bonus": float(get_parcel_bonus or 0.0),
                "deliver_bonus": float(deliver_bonus or 0.0),
            }
        )
    visual_cfg = VisualConfig(
        steps=steps,
        imagine_horizon=imagine_horizon,
        imagine_from_step=int(args.imagine_from_step),
        seed=int(args.seed),
        greedy=bool(args.greedy),
        stop_on_done=bool(args.stop_on_done),
        fps=int(args.fps),
        output_tag=str(args.output_tag),
        run_root=str(args.run_root),
        skip_recon=bool(args.skip_recon),
        skip_imagined=bool(args.skip_imagined),
    )

    experiment = Experiment(
        algo="dreamer_v3",
        rom_id=rom_path.stem,
        tag=str(args.output_tag),
        run_root=str(args.run_root),
        meta=meta,
        config=asdict(visual_cfg),
    )

    obs = env.reset(seed=int(args.seed))
    obs_is_dict = isinstance(obs, dict)
    obs_keys: list[str] = []
    obs_list: dict[str, list[Any]] = {}
    obs_tensor_list: list[Any] = []
    if obs_is_dict:
        device = next(iter(obs.values())).device
        obs_keys = list(obs.keys())
        obs_list = {k: [] for k in obs_keys}
    else:
        device = obs.device
    state_obj = actor_core.init_state(1, device)
    is_first = torch.ones((1,), dtype=torch.bool, device=device)
    actions_list = []
    is_first_list = []
    terminated_list = []
    truncated_list = []
    rewards_list = []
    info_list = []

    with torch.no_grad():
        for _ in range(steps):
            if obs_is_dict:
                for key in obs_keys:
                    obs_list[key].append(obs[key].clone())
            else:
                obs_tensor_list.append(obs.clone())
            is_first_list.append(is_first.clone())

            actions, state_obj = actor_core.act(
                obs, is_first, state_obj, generator=action_gen
            )
            actions_list.append(actions.clone())

            next_obs, reward, done, trunc, info = env.step(actions)
            terminated_list.append(done.clone())
            truncated_list.append(trunc.clone())
            rewards_list.append(reward.clone())
            info_list.append(_collect_step_info(info, obs))

            reset_mask = done | trunc
            if hasattr(env, "reset_mask"):
                env.reset_mask(reset_mask)
            is_first = reset_mask.to(torch.bool)
            obs = env.obs if hasattr(env, "obs") else next_obs

            if args.stop_on_done and bool(reset_mask[0].item()):
                break

    env.close()

    if obs_is_dict:
        obs_seq = {k: torch.stack(v, dim=0) for k, v in obs_list.items()}
    else:
        obs_seq = torch.stack(obs_tensor_list, dim=0)
    actions_seq = torch.stack(actions_list, dim=0)
    is_first_seq = torch.stack(is_first_list, dim=0)
    terminated_seq = torch.stack(terminated_list, dim=0)

    continues_seq = (~terminated_seq).to(torch.float32)

    rollout_path = experiment.run_dir / "rollout.jsonl"
    with rollout_path.open("w", encoding="utf-8") as handle:
        total_steps = actions_seq.shape[0]
        for idx in range(total_steps):
            step_info = info_list[idx] if idx < len(info_list) else {}
            handle.write(
                json.dumps(
                    {
                        "t": idx,
                        "action": int(actions_seq[idx, 0].item()),
                        "reward": float(rewards_list[idx][0].item()),
                        "done": bool(terminated_seq[idx, 0].item()),
                        "trunc": bool(truncated_list[idx][0].item()),
                        "dist": step_info.get("dist"),
                        "map_id": step_info.get("map_id"),
                        "x": step_info.get("x"),
                        "y": step_info.get("y"),
                        "novelty": step_info.get("novelty"),
                        "got_parcel": step_info.get("got_parcel"),
                        "delivered": step_info.get("delivered"),
                    }
                )
                + "\n"
            )

    trace_path = experiment.run_dir / "trace.json"
    trace_payload: dict[str, Any] = {
        "task": task,
        "start_state": str(state_path),
        "steps": int(actions_seq.shape[0]),
        "actions": [int(a.item()) for a in actions_seq[:, 0]],
        "checkpoint": str(ckpt_path),
        "frames_per_step": int(frames_per_step),
        "release_after_frames": int(release_after_frames),
    }
    if goal_path is not None:
        trace_payload["goal_dir"] = str(goal_path)
    if task == "parcel":
        trace_payload.update(
            {
                "events_start": int(events_start or 0),
                "events_length": int(events_length or 0),
                "senses_dim": int(senses_dim or 0),
            }
        )
    trace_path.write_text(json.dumps(trace_payload, indent=2) + "\n", encoding="utf-8")

    with torch.no_grad():
        if obs_is_dict:
            enc_obs, _ = world_model.prepare_obs(obs_seq, obs_format="packed2")
        else:
            enc_obs, _ = world_model.prepare_obs(
                {"pixels": obs_seq}, obs_format="packed2"
            )
        outputs = world_model(enc_obs, actions_seq, is_first_seq, action_dim=action_dim)

    pixels_seq: Any = obs_seq["pixels"] if obs_is_dict else obs_seq
    real_frames = [
        _packed_to_shades(pixels_seq[idx, 0, 0]) for idx in range(pixels_seq.shape[0])
    ]
    recon_frames = []
    if not args.skip_recon:
        recon = outputs.reconstructions["pixels"]
        recon_frames = [
            _pred_to_shades(recon[idx, 0, 0]) for idx in range(recon.shape[0])
        ]

    real_mp4 = experiment.run_dir / "rollout_real.mp4"
    _write_video(real_frames, fps=int(args.fps), output=real_mp4)

    recon_mp4 = None
    if recon_frames:
        recon_mp4 = experiment.run_dir / "rollout_recon.mp4"
        _write_video(recon_frames, fps=int(args.fps), output=recon_mp4)

    imagined_mp4 = None
    if not args.skip_imagined:
        start_idx = int(args.imagine_from_step)
        if start_idx < 0:
            start_idx = actions_seq.shape[0] + start_idx
        if start_idx < 0 or start_idx >= actions_seq.shape[0]:
            raise ValueError("imagine-from-step is out of range")

        true_continue = continues_seq[start_idx : start_idx + 1]
        with torch.no_grad():
            imagination = imagine_rollout(
                rssm=world_model.rssm,
                actor=actor,
                critic=critic,
                reward_model=world_model.reward_model,
                continue_model=world_model.continue_model,
                posteriors=outputs.posteriors[start_idx : start_idx + 1],
                recurrent_states=outputs.recurrent_states[start_idx : start_idx + 1],
                true_continue=true_continue,
                horizon=imagine_horizon,
                gamma=float(cfg.get("gamma", 0.99)),
                reward_low=float(cfg.get("reward_low", -20.0)),
                reward_high=float(cfg.get("reward_high", 20.0)),
                value_low=float(cfg.get("reward_low", -20.0)),
                value_high=float(cfg.get("reward_high", 20.0)),
                greedy=bool(args.greedy),
                generator=action_gen,
                sample_state=True,
            )

            imagined = world_model.observation_model(imagination.latent_states)
        imag_frames = [
            _pred_to_shades(imagined["pixels"][idx, 0, 0])
            for idx in range(imagined["pixels"].shape[0])
        ]

        imagined_mp4 = experiment.run_dir / "rollout_imagined.mp4"
        _write_video(imag_frames, fps=int(args.fps), output=imagined_mp4)

        imagine_meta = {
            "start_idx": int(start_idx),
            "horizon": int(imagine_horizon),
            "greedy": bool(args.greedy),
            "seed": int(args.seed),
        }
        (experiment.run_dir / "imagined_meta.json").write_text(
            json.dumps(imagine_meta, indent=2) + "\n", encoding="utf-8"
        )

    print(
        json.dumps(
            {
                "run_dir": str(experiment.run_dir),
                "real_mp4": str(real_mp4),
                "recon_mp4": str(recon_mp4) if recon_mp4 else None,
                "imagined_mp4": str(imagined_mp4) if imagined_mp4 else None,
                "trace": str(trace_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
