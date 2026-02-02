#!/usr/bin/env python3
"""Dreamer v3 GPU training (packed2, Pokemon Red).

Usage:
  uv run python tools/rl_train_gpu.py --algo dreamer_v3 --rom red.gb \
    --state states/rl_stage1_exit_oak/start.state --goal-dir states/rl_stage1_exit_oak
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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


def _git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", True


def _system_meta() -> dict[str, Any]:
    torch = _require_torch()
    warp_version = None
    try:
        import warp as wp

        warp_version = getattr(wp, "__version__", None)
    except Exception:
        warp_version = None
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch_version": torch.__version__,
        "warp_version": warp_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
    }


@dataclass(frozen=True)
class TrainConfig:
    algo: str
    task: str
    mode: str
    rom: str
    state: str
    goal_dir: str
    frames_per_step: int
    release_after_frames: int
    num_envs: int
    action_codec: str
    max_steps: int
    step_cost: float
    alpha: float
    goal_bonus: float
    snow_bonus: float
    get_parcel_bonus: float
    deliver_bonus: float
    tau: float | None
    k_consecutive: int | None
    seed: int
    iterations: int | None
    total_env_steps: int
    steps_per_rollout: int
    replay_capacity: int
    seq_len: int
    batch_size: int
    commit_stride: int
    replay_ratio: float
    learning_starts: int
    pretrain_steps: int
    min_ready_steps: int
    safety_margin: int
    max_learner_steps_per_tick: int | None
    debug: bool
    checkpoint_every: int
    output_tag: str
    run_root: str
    standing_still_action: int
    # model + loss params
    stochastic_size: int
    discrete_size: int
    recurrent_state_size: int
    dense_units: int
    hidden_size: int
    cnn_channels_multiplier: int
    cnn_stages: int
    head_mlp_layers: int
    reward_bins: int
    reward_low: float
    reward_high: float
    continue_scale_factor: float
    value_bins: int
    gamma: float
    lmbda: float
    horizon: int
    ent_coef: float
    actor_lr: float
    critic_lr: float
    world_model_lr: float
    actor_clip_grad: float | None
    critic_clip_grad: float | None
    world_model_clip_grad: float | None
    critic_tau: float
    target_update_freq: int
    moments_decay: float
    moments_low: float
    moments_high: float
    moments_max: float

    def validate(self) -> None:
        if self.algo != "dreamer_v3":
            raise ValueError("algo must be dreamer_v3")
        if self.task not in ("goal_template", "oak_parcel"):
            raise ValueError("task must be goal_template or oak_parcel")
        if self.mode not in ("full", "standing_still"):
            raise ValueError("mode must be full or standing_still")
        if self.task == "goal_template" and not self.goal_dir:
            raise ValueError("goal_dir is required for goal_template task")
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if self.steps_per_rollout < 1:
            raise ValueError("steps_per_rollout must be >= 1")
        if self.replay_capacity < self.seq_len + 1:
            raise ValueError("replay_capacity must be >= seq_len + 1")
        if self.commit_stride < 1:
            raise ValueError("commit_stride must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be >= 0")
        if self.pretrain_steps < 0:
            raise ValueError("pretrain_steps must be >= 0")
        if self.min_ready_steps < self.seq_len:
            raise ValueError("min_ready_steps must be >= seq_len")
        if self.safety_margin < self.seq_len:
            raise ValueError("safety_margin must be >= seq_len")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        if not (0.0 <= self.lmbda <= 1.0):
            raise ValueError("lmbda must be in [0, 1]")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", default="dreamer_v3")
    parser.add_argument(
        "--task", default="goal_template", choices=("goal_template", "oak_parcel")
    )
    parser.add_argument("--mode", default="full", choices=("full", "standing_still"))
    parser.add_argument("--rom", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--goal-dir", default="")
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--action-codec", default="pokemonred_puffer_v1")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--step-cost", type=float, default=-0.01)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--goal-bonus", type=float, default=10.0)
    parser.add_argument("--snow-bonus", type=float, default=0.01)
    parser.add_argument("--get-parcel-bonus", type=float, default=5.0)
    parser.add_argument("--deliver-bonus", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--k-consecutive", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--total-env-steps", type=int, default=200_000)
    parser.add_argument("--steps-per-rollout", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=49152)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--commit-stride", type=int, default=4)
    parser.add_argument("--replay-ratio", type=float, default=0.25)
    parser.add_argument("--learning-starts", type=int, default=0)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--min-ready-steps", type=int, default=16)
    parser.add_argument("--safety-margin", type=int, default=16)
    parser.add_argument("--max-learner-steps-per-tick", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--resume", default=None, help="Path to checkpoint.pt to resume from"
    )
    parser.add_argument("--output-tag", default="dreamer_v3")
    parser.add_argument("--run-root", default="bench/runs/rl")
    parser.add_argument("--standing-still-action", type=int, default=0)

    parser.add_argument("--stochastic-size", type=int, default=32)
    parser.add_argument("--discrete-size", type=int, default=32)
    parser.add_argument("--recurrent-state-size", type=int, default=512)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--cnn-channels-multiplier", type=int, default=16)
    parser.add_argument("--cnn-stages", type=int, default=3)
    parser.add_argument("--head-mlp-layers", type=int, default=2)
    parser.add_argument("--reward-bins", type=int, default=255)
    parser.add_argument("--reward-low", type=float, default=-20.0)
    parser.add_argument("--reward-high", type=float, default=20.0)
    parser.add_argument("--continue-scale-factor", type=float, default=1.0)
    parser.add_argument("--value-bins", type=int, default=255)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.95)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--ent-coef", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=8e-5)
    parser.add_argument("--critic-lr", type=float, default=8e-5)
    parser.add_argument("--world-model-lr", type=float, default=1e-4)
    parser.add_argument("--actor-clip-grad", type=float, default=100.0)
    parser.add_argument("--critic-clip-grad", type=float, default=100.0)
    parser.add_argument("--world-model-clip-grad", type=float, default=1000.0)
    parser.add_argument("--critic-tau", type=float, default=0.02)
    parser.add_argument("--target-update-freq", type=int, default=1)
    parser.add_argument("--moments-decay", type=float, default=0.99)
    parser.add_argument("--moments-low", type=float, default=0.05)
    parser.add_argument("--moments-high", type=float, default=0.95)
    parser.add_argument("--moments-max", type=float, default=1.0)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (bfloat16 is enabled by default)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous phased training (collect then train) instead of async",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=32.0,
        help="Training updates per env step (for sync mode, matches DreamerV3 paper)",
    )
    return parser.parse_args()


def _resolve_iterations(cfg: TrainConfig) -> int:
    if cfg.iterations is not None:
        return int(cfg.iterations)
    steps_per_iter = int(cfg.num_envs * cfg.steps_per_rollout)
    return max(1, int((cfg.total_env_steps + steps_per_iter - 1) // steps_per_iter))


def _get_state_dict(model):  # type: ignore[no-untyped-def]
    """Get state dict from model, handling torch.compile wrapped models."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def _save_checkpoint(path, payload) -> None:  # type: ignore[no-untyped-def]
    torch = _require_torch()
    tmp = Path(path).with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> int:
    args = _parse_args()
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = 2048 if args.task == "oak_parcel" else 128
    cfg = TrainConfig(
        algo=str(args.algo),
        task=str(args.task),
        mode=str(args.mode),
        rom=str(args.rom),
        state=str(args.state),
        goal_dir=str(args.goal_dir),
        frames_per_step=int(args.frames_per_step),
        release_after_frames=int(args.release_after_frames),
        num_envs=int(args.num_envs),
        action_codec=str(args.action_codec),
        max_steps=int(max_steps),
        step_cost=float(args.step_cost),
        alpha=float(args.alpha),
        goal_bonus=float(args.goal_bonus),
        snow_bonus=float(args.snow_bonus),
        get_parcel_bonus=float(args.get_parcel_bonus),
        deliver_bonus=float(args.deliver_bonus),
        tau=args.tau,
        k_consecutive=args.k_consecutive,
        seed=int(args.seed),
        iterations=args.iterations,
        total_env_steps=int(args.total_env_steps),
        steps_per_rollout=int(args.steps_per_rollout),
        replay_capacity=int(args.replay_capacity),
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size),
        commit_stride=int(args.commit_stride),
        replay_ratio=float(args.replay_ratio),
        learning_starts=int(args.learning_starts),
        pretrain_steps=int(args.pretrain_steps),
        min_ready_steps=int(args.min_ready_steps),
        safety_margin=int(args.safety_margin),
        max_learner_steps_per_tick=(
            None
            if args.max_learner_steps_per_tick in (None, 0)
            else int(args.max_learner_steps_per_tick)
        ),
        debug=bool(args.debug),
        checkpoint_every=int(args.checkpoint_every),
        output_tag=str(args.output_tag),
        run_root=str(args.run_root),
        standing_still_action=int(args.standing_still_action),
        stochastic_size=int(args.stochastic_size),
        discrete_size=int(args.discrete_size),
        recurrent_state_size=int(args.recurrent_state_size),
        dense_units=int(args.dense_units),
        hidden_size=int(args.hidden_size),
        cnn_channels_multiplier=int(args.cnn_channels_multiplier),
        cnn_stages=int(args.cnn_stages),
        head_mlp_layers=int(args.head_mlp_layers),
        reward_bins=int(args.reward_bins),
        reward_low=float(args.reward_low),
        reward_high=float(args.reward_high),
        continue_scale_factor=float(args.continue_scale_factor),
        value_bins=int(args.value_bins),
        gamma=float(args.gamma),
        lmbda=float(args.lmbda),
        horizon=int(args.horizon),
        ent_coef=float(args.ent_coef),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        world_model_lr=float(args.world_model_lr),
        actor_clip_grad=float(args.actor_clip_grad)
        if args.actor_clip_grad is not None
        else None,
        critic_clip_grad=float(args.critic_clip_grad)
        if args.critic_clip_grad is not None
        else None,
        world_model_clip_grad=float(args.world_model_clip_grad)
        if args.world_model_clip_grad is not None
        else None,
        critic_tau=float(args.critic_tau),
        target_update_freq=int(args.target_update_freq),
        moments_decay=float(args.moments_decay),
        moments_low=float(args.moments_low),
        moments_high=float(args.moments_high),
        moments_max=float(args.moments_max),
    )
    cfg.validate()

    torch = _require_torch()
    if not _cuda_available() or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Dreamer v3 training.")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W, DOWNSAMPLE_W_BYTES
    from gbxcule.rl.dreamer_v3.async_dreamer_v3_engine import AsyncDreamerV3Engine
    from gbxcule.rl.dreamer_v3.behavior import Actor, Critic, behavior_step
    from gbxcule.rl.dreamer_v3.config import DreamerEngineConfig
    from gbxcule.rl.dreamer_v3.decoders import CNNDecoder, MLPDecoder, MultiDecoder
    from gbxcule.rl.dreamer_v3.encoders import (
        MLPEncoder,
        MultiEncoder,
        Packed2PixelEncoder,
    )
    from gbxcule.rl.dreamer_v3.heads import ContinueHead, RewardHead
    from gbxcule.rl.dreamer_v3.player import ConstantActorCore, DreamerActorCore
    from gbxcule.rl.dreamer_v3.return_ema import ReturnEMA
    from gbxcule.rl.dreamer_v3.rssm import build_rssm
    from gbxcule.rl.dreamer_v3.world_model import WorldModel
    from gbxcule.rl.experiment import Experiment
    from gbxcule.rl.goal_template import compute_sha256

    if cfg.task == "goal_template":
        from gbxcule.rl.pokered_packed_goal_env import PokeredPackedGoalEnv

        env = PokeredPackedGoalEnv(
            cfg.rom,
            goal_dir=cfg.goal_dir,
            state_path=cfg.state,
            num_envs=cfg.num_envs,
            frames_per_step=cfg.frames_per_step,
            release_after_frames=cfg.release_after_frames,
            action_codec=cfg.action_codec,
            max_steps=cfg.max_steps,
            step_cost=cfg.step_cost,
            alpha=cfg.alpha,
            goal_bonus=cfg.goal_bonus,
            tau=cfg.tau,
            k_consecutive=cfg.k_consecutive,
            info_mode="stats",
        )
        senses_dim = None
        senses_dim_val = 0
        events_length = 0
    else:
        import importlib

        parcel_mod = importlib.import_module("gbxcule.rl.pokered_packed_parcel_env")
        PokeredPackedParcelEnv = parcel_mod.PokeredPackedParcelEnv
        events_length = int(parcel_mod.EVENTS_LENGTH)
        senses_dim_val = int(parcel_mod.SENSES_DIM)

        env = PokeredPackedParcelEnv(
            cfg.rom,
            state_path=cfg.state,
            num_envs=cfg.num_envs,
            frames_per_step=cfg.frames_per_step,
            release_after_frames=cfg.release_after_frames,
            action_codec=cfg.action_codec,
            max_steps=cfg.max_steps,
            snow_bonus=cfg.snow_bonus,
            get_parcel_bonus=cfg.get_parcel_bonus,
            deliver_bonus=cfg.deliver_bonus,
            info_mode="stats",
        )
        # senses (float32) + events (uint8 -> float32)
        senses_dim = senses_dim_val + events_length

    action_dim = int(env.num_actions)
    obs_shape = (1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
    image_size = (DOWNSAMPLE_H, DOWNSAMPLE_W)

    cnn_encoder = Packed2PixelEncoder(
        keys=["pixels"],
        image_size=image_size,
        channels_multiplier=cfg.cnn_channels_multiplier,
        stages=cfg.cnn_stages,
    )
    mlp_encoder = None
    mlp_keys: list[str] = []
    if senses_dim is not None:
        if cfg.task == "oak_parcel":
            # oak_parcel: senses (float32) and events (uint8 -> float32)
            mlp_keys = ["senses", "events"]
            mlp_encoder = MLPEncoder(
                keys=mlp_keys,
                input_dims=[senses_dim_val, events_length],
                mlp_layers=cfg.head_mlp_layers,
                dense_units=cfg.dense_units,
            )
        else:
            mlp_keys = ["senses"]
            mlp_encoder = MLPEncoder(
                keys=mlp_keys,
                input_dims=[senses_dim],
                mlp_layers=cfg.head_mlp_layers,
                dense_units=cfg.dense_units,
            )
    encoder = MultiEncoder(cnn_encoder, mlp_encoder)
    rssm = build_rssm(
        action_dim=action_dim,
        embed_dim=encoder.cnn_output_dim + encoder.mlp_output_dim,
        stochastic_size=cfg.stochastic_size,
        discrete_size=cfg.discrete_size,
        recurrent_state_size=cfg.recurrent_state_size,
        dense_units=cfg.dense_units,
        hidden_size=cfg.hidden_size,
        unimix=0.01,
        rnn_dtype=torch.float32,
    )
    latent_state_size = (
        cfg.stochastic_size * cfg.discrete_size + cfg.recurrent_state_size
    )
    decoder = CNNDecoder(
        keys=["pixels"],
        output_channels=[1],
        channels_multiplier=cfg.cnn_channels_multiplier,
        latent_state_size=latent_state_size,
        encoder_output_shape=cnn_encoder.output_shape,
        stages=cfg.cnn_stages,
    )
    mlp_decoder = None
    # Decoder only reconstructs senses, not events (events are input-only features)
    if senses_dim is not None:
        if cfg.task == "oak_parcel":
            # Only reconstruct senses (4 dims), not events
            mlp_decoder = MLPDecoder(
                keys=["senses"],
                output_dims=[senses_dim_val],
                latent_state_size=latent_state_size,
                mlp_layers=cfg.head_mlp_layers,
                dense_units=cfg.dense_units,
            )
        else:
            mlp_decoder = MLPDecoder(
                keys=["senses"],
                output_dims=[senses_dim],
                latent_state_size=latent_state_size,
                mlp_layers=cfg.head_mlp_layers,
                dense_units=cfg.dense_units,
            )
    observation_model = MultiDecoder(decoder, mlp_decoder)
    reward_model = RewardHead(
        input_dim=latent_state_size,
        bins=cfg.reward_bins,
        mlp_layers=cfg.head_mlp_layers,
        dense_units=cfg.dense_units,
    )
    continue_model = ContinueHead(
        input_dim=latent_state_size,
        mlp_layers=cfg.head_mlp_layers,
        dense_units=cfg.dense_units,
    )
    # mlp_keys for world model: decoder only reconstructs senses
    world_model_mlp_keys = ["senses"] if senses_dim is not None else []
    world_model = WorldModel(
        encoder=encoder,
        rssm=rssm,
        observation_model=observation_model,
        reward_model=reward_model,
        continue_model=continue_model,
        cnn_keys=["pixels"],
        mlp_keys=world_model_mlp_keys,
        reward_low=cfg.reward_low,
        reward_high=cfg.reward_high,
        continue_scale_factor=cfg.continue_scale_factor,
    ).to("cuda")

    actor = Actor(
        latent_state_size=latent_state_size,
        actions_dim=[action_dim],
        is_continuous=False,
        distribution_cfg={"type": "discrete"},
        dense_units=cfg.dense_units,
        mlp_layers=cfg.head_mlp_layers,
        unimix=0.01,
        action_clip=1.0,
        init_std=0.0,
        min_std=1.0,
        max_std=1.0,
    ).to("cuda")
    critic = Critic(
        latent_state_size=latent_state_size,
        bins=cfg.value_bins,
        dense_units=cfg.dense_units,
        mlp_layers=cfg.head_mlp_layers,
    ).to("cuda")
    target_critic = Critic(
        latent_state_size=latent_state_size,
        bins=cfg.value_bins,
        dense_units=cfg.dense_units,
        mlp_layers=cfg.head_mlp_layers,
    ).to("cuda")
    target_critic.load_state_dict(critic.state_dict())

    # Resume from checkpoint if provided (must be before torch.compile)
    resume_env_steps = 0
    resume_train_steps = 0
    ckpt = None
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cuda")
        world_model.load_state_dict(ckpt["world_model"])
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        target_critic.load_state_dict(ckpt["target_critic"])
        resume_env_steps = int(ckpt.get("env_steps", 0) or 0)
        resume_train_steps = int(ckpt.get("train_steps", 0) or 0)
        print(f"  Resumed: env={resume_env_steps}, train={resume_train_steps}")

    # NOTE: torch.compile disabled due to convolution_backward shape mismatch bug
    # TODO: Re-enable once PyTorch fixes the issue or try mode="reduce-overhead"
    # world_model = torch.compile(world_model)
    # actor = torch.compile(actor)
    # critic = torch.compile(critic)
    # target_critic = torch.compile(target_critic)

    world_opt = torch.optim.Adam(world_model.parameters(), lr=cfg.world_model_lr)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    # Load optimizer states if resuming
    if ckpt is not None:
        if "world_opt" in ckpt:
            world_opt.load_state_dict(ckpt["world_opt"])
        if "actor_opt" in ckpt:
            actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            critic_opt.load_state_dict(ckpt["critic_opt"])

    moments = ReturnEMA(
        decay=cfg.moments_decay,
        percentiles=(cfg.moments_low, cfg.moments_high),
        max_value=cfg.moments_max,
    )
    behavior_gen = torch.Generator(device="cuda")
    behavior_gen.manual_seed(cfg.seed + 2)

    if cfg.mode == "standing_still":
        actor_core = ConstantActorCore(action=cfg.standing_still_action)
    else:
        actor_core = DreamerActorCore(
            encoder=world_model.encoder,
            rssm=world_model.rssm,
            actor=actor,
            action_dim=action_dim,
            greedy=False,
        )
        actor_core.sync_player()

    rom_path = Path(cfg.rom)
    state_path = Path(cfg.state)
    rom_sha = compute_sha256(rom_path)
    state_sha = compute_sha256(state_path)
    git_commit, git_dirty = _git_info()
    meta = {
        "rom": {"rom_path": str(rom_path), "rom_sha256": rom_sha},
        "state": {"state_path": str(state_path), "state_sha256": state_sha},
        "env": {
            "num_envs": cfg.num_envs,
            "frames_per_step": cfg.frames_per_step,
            "release_after_frames": cfg.release_after_frames,
            "stack_k": 1,
        },
        "pipeline": {"obs_format": "packed2", "action_codec_id": cfg.action_codec},
        "algo": {"algo_name": cfg.algo, "algo_version": 1},
        "code": {"git_commit": git_commit, "git_dirty": git_dirty},
        "system": _system_meta(),
        "task": cfg.task,
        "mode": cfg.mode,
    }
    if cfg.task == "goal_template":
        goal_path = Path(cfg.goal_dir) / "goal_template.npy"
        goal_sha = compute_sha256(goal_path) if goal_path.exists() else "unknown"
        meta["goal"] = {"goal_dir": cfg.goal_dir, "goal_sha256": goal_sha}
    else:
        meta["env"].update(
            {
                "snow_bonus": cfg.snow_bonus,
                "get_parcel_bonus": cfg.get_parcel_bonus,
                "deliver_bonus": cfg.deliver_bonus,
                "max_steps": cfg.max_steps,
            }
        )

    experiment = Experiment(
        algo=cfg.algo,
        rom_id=rom_path.stem,
        tag=cfg.output_tag,
        run_root=cfg.run_root,
        meta=meta,
        config=asdict(cfg),
    )

    engine_cfg = DreamerEngineConfig(
        num_envs=cfg.num_envs,
        obs_shape=obs_shape,
        replay_capacity=cfg.replay_capacity,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        steps_per_rollout=cfg.steps_per_rollout,
        commit_stride=cfg.commit_stride,
        replay_ratio=cfg.replay_ratio,
        learning_starts=cfg.learning_starts,
        pretrain_steps=cfg.pretrain_steps,
        min_ready_steps=cfg.min_ready_steps,
        safety_margin=cfg.safety_margin,
        max_learner_steps_per_tick=cfg.max_learner_steps_per_tick,
        device="cuda",
        seed=cfg.seed,
        debug=cfg.debug,
    )

    # AMP context manager (bfloat16 for safe dynamic range, enabled by default)
    amp_enabled = not bool(args.no_amp)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if amp_enabled
        else torch.inference_mode(mode=False)  # no-op context
    )
    if amp_enabled:
        print("AMP enabled (bfloat16)")

    def world_model_update(batch):  # type: ignore[no-untyped-def]
        obs = batch.obs
        if not isinstance(obs, dict):
            obs = {"pixels": obs}
        rewards = batch.reward
        continues = batch.continues
        if rewards.ndim == 2:
            rewards = rewards.unsqueeze(-1)
        if continues.ndim == 2:
            continues = continues.unsqueeze(-1)
        enc_obs, loss_obs = world_model.prepare_obs(obs, obs_format="packed2")
        with amp_ctx:
            outputs = world_model(
                enc_obs, batch.action, batch.is_first, action_dim=action_dim
            )
            metrics = world_model.loss(
                outputs,
                loss_obs,
                rewards,
                continues,
                kl_dynamic=0.5,
                kl_representation=0.1,
                kl_free_nats=1.0,
                kl_regularizer=1.0,
            )
        world_opt.zero_grad(set_to_none=True)
        metrics.loss.backward()
        grad_norm = None
        if cfg.world_model_clip_grad is not None and cfg.world_model_clip_grad > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                world_model.parameters(), cfg.world_model_clip_grad
            )
        world_opt.step()

        from torch.distributions import Independent, OneHotCategorical

        post_entropy = Independent(
            OneHotCategorical(logits=outputs.posterior_logits.detach()), 1
        ).entropy()
        prior_entropy = Independent(
            OneHotCategorical(logits=outputs.prior_logits.detach()), 1
        ).entropy()

        out = {
            "Loss/world_model_loss": metrics.loss.detach(),
            "Loss/observation_loss": metrics.observation_loss.detach(),
            "Loss/reward_loss": metrics.reward_loss.detach(),
            "Loss/state_loss": metrics.kl_loss.detach(),
            "Loss/continue_loss": metrics.continue_loss.detach(),
            "State/kl": metrics.kl.detach(),
            "State/post_entropy": post_entropy.mean().detach(),
            "State/prior_entropy": prior_entropy.mean().detach(),
        }
        if grad_norm is not None:
            out["Grads/world_model"] = grad_norm.detach()
        return out

    def behavior_update(batch):  # type: ignore[no-untyped-def]
        if cfg.mode != "full":
            return {}
        with torch.no_grad(), amp_ctx:
            obs = batch.obs
            if not isinstance(obs, dict):
                obs = {"pixels": obs}
            enc_obs, _ = world_model.prepare_obs(obs, obs_format="packed2")
            outputs = world_model(
                enc_obs, batch.action, batch.is_first, action_dim=action_dim
            )
        with amp_ctx:
            losses = behavior_step(
                rssm=world_model.rssm,
                actor=actor,
                critic=critic,
                target_critic=target_critic,
                reward_model=world_model.reward_model,
                continue_model=world_model.continue_model,
                posteriors=outputs.posteriors,
                recurrent_states=outputs.recurrent_states,
                continues=batch.continues,
                horizon=cfg.horizon,
                gamma=cfg.gamma,
                lmbda=cfg.lmbda,
                moments=moments,
                actor_optimizer=actor_opt,
                critic_optimizer=critic_opt,
                actor_clip_grad=cfg.actor_clip_grad,
                critic_clip_grad=cfg.critic_clip_grad,
                target_update_freq=cfg.target_update_freq,
                tau=cfg.critic_tau,
                ent_coef=cfg.ent_coef,
                generator=behavior_gen,
                sample_state=True,
            )

        def _grad_norm(params):  # type: ignore[no-untyped-def]
            grads = [p.grad for p in params if p.grad is not None]
            if not grads:
                return None
            flat = torch.cat([g.detach().flatten() for g in grads])
            return torch.linalg.norm(flat)

        ret_low = moments.low
        ret_high = moments.high
        ret_scale = None
        if ret_low is not None and ret_high is not None:
            ret_scale = torch.max(
                torch.tensor(1.0 / cfg.moments_max, device=ret_low.device),
                ret_high - ret_low,
            )

        with torch.no_grad():
            _, policies = actor(losses.imagination.latent_states.detach(), greedy=True)
            if actor.is_continuous:
                action_entropy = policies[0].entropy().mean()
            else:
                action_entropy = torch.stack(
                    [dist.entropy() for dist in policies], dim=-1
                ).sum(dim=-1)
                action_entropy = action_entropy.mean()

        out = {
            "Loss/policy_loss": losses.policy_loss.detach(),
            "Loss/value_loss": losses.value_loss.detach(),
            "action_entropy": action_entropy.detach(),
            "adv_mean": losses.advantage.mean().detach(),
            "adv_std": losses.advantage.std().detach(),
            "value_mean": losses.imagination.values.mean().detach(),
            "value_std": losses.imagination.values.std().detach(),
        }
        actor_grad = _grad_norm(actor.parameters())
        critic_grad = _grad_norm(critic.parameters())
        if actor_grad is not None:
            out["Grads/actor"] = actor_grad
        if critic_grad is not None:
            out["Grads/critic"] = critic_grad
        if ret_low is not None:
            out["ret_p05"] = ret_low.detach()
        if ret_high is not None:
            out["ret_p95"] = ret_high.detach()
        if ret_scale is not None:
            out["ret_scale"] = ret_scale.detach()
        return out

    iterations = _resolve_iterations(cfg)
    last_metrics: dict[str, Any] = {}
    start = time.time()
    repro = " ".join([json.dumps(arg) if " " in arg else arg for arg in sys.argv])

    # Choose sync or async training
    sync_mode = bool(args.sync)
    train_ratio = float(args.train_ratio)

    if sync_mode:
        # ========== SYNCHRONOUS PHASED TRAINING ==========
        print(f"Sync mode: collect {cfg.steps_per_rollout} steps, train {train_ratio}x")
        from gbxcule.rl.dreamer_v3.async_dreamer_v3_engine import DreamerBatch
        from gbxcule.rl.dreamer_v3.replay_cuda import ReplayRingCUDA

        # Build replay buffer
        obs = env.reset_torch(seed=cfg.seed)
        obs_spec: dict[str, tuple[tuple[int, ...], Any]] = {}
        for key, value in obs.items():
            obs_spec[key] = (tuple(int(x) for x in value.shape[1:]), value.dtype)

        replay = ReplayRingCUDA(
            capacity=cfg.replay_capacity,
            num_envs=cfg.num_envs,
            device="cuda",
            obs_spec=obs_spec,
        )

        # Actor state
        is_first = torch.ones((cfg.num_envs,), dtype=torch.bool, device="cuda")
        episode_id = torch.zeros((cfg.num_envs,), dtype=torch.int32, device="cuda")
        actor_state = actor_core.init_state(cfg.num_envs, "cuda")
        action_gen = torch.Generator(device="cuda")
        action_gen.manual_seed(cfg.seed + 1)
        sample_gen = torch.Generator(device="cuda")
        sample_gen.manual_seed(cfg.seed)

        env_steps = 0
        train_steps = 0

        try:
            for step_idx in range(iterations):
                iter_start = time.time()

                # === PHASE 1: COLLECT ===
                collect_steps = cfg.steps_per_rollout
                with torch.no_grad():
                    for _ in range(collect_steps):
                        actions, actor_state = actor_core.act(
                            obs, is_first, actor_state, generator=action_gen
                        )
                        next_obs, reward, terminated, truncated, _ = env.step_torch(
                            actions
                        )
                        continues = (~terminated).to(dtype=torch.float32)

                        replay.push_step(
                            obs=obs,
                            action=actions,
                            reward=reward,
                            is_first=is_first,
                            continue_=continues,
                            episode_id=episode_id,
                            terminated=terminated,
                            truncated=truncated,
                        )

                        reset_mask = terminated | truncated
                        episode_id = episode_id + reset_mask.to(torch.int32)
                        is_first = reset_mask
                        if hasattr(env, "reset_mask"):
                            env.reset_mask(reset_mask)
                        obs = next_obs

                env_steps += cfg.num_envs * collect_steps
                torch.cuda.synchronize()

                # === PHASE 2: TRAIN ===
                # Train ratio: updates per rollout step (not per env step)
                # With train_ratio=32, steps_per_rollout=32: 1024 updates per iteration
                num_updates = max(1, int(train_ratio * collect_steps))

                # Wait for enough samples in replay
                if replay.size >= cfg.min_ready_steps:
                    wm_losses: dict[str, float] = {}
                    beh_losses: dict[str, float] = {}

                    for _ in range(num_updates):
                        sample = replay.sample_sequences(
                            batch=cfg.batch_size,
                            seq_len=cfg.seq_len,
                            gen=sample_gen,
                            committed_t=replay.total_steps - 1,
                            safety_margin=cfg.safety_margin,
                            exclude_head=True,
                            return_indices=True,
                        )
                        batch = DreamerBatch(
                            obs=sample["obs"],
                            action=sample["action"],
                            reward=sample["reward"],
                            is_first=sample["is_first"],
                            continues=sample["continue"],
                            episode_id=sample["episode_id"],
                            start_times=sample.get("meta", {}).get("start_offset"),
                            env_indices=sample.get("meta", {}).get("env_idx"),
                        )
                        wm_metrics = world_model_update(batch)
                        beh_metrics = behavior_update(batch)

                        for key, value in wm_metrics.items():
                            if isinstance(value, torch.Tensor):
                                wm_losses[key] = float(value.detach().mean().item())
                        for key, value in beh_metrics.items():
                            if isinstance(value, torch.Tensor):
                                beh_losses[key] = float(value.detach().mean().item())

                        train_steps += 1

                    # Sync actor weights after training
                    actor_core.sync_player()
                    torch.cuda.synchronize()

                iter_time = time.time() - iter_start
                wall_time = time.time() - start
                sps = (cfg.num_envs * collect_steps) / iter_time

                last_metrics = {
                    "env_steps": env_steps,
                    "train_steps": train_steps,
                    "wall_time_s": wall_time,
                    "sps": sps,
                    "train_updates_per_iter": num_updates,
                    "replay_size": replay.size,
                    **wm_losses,
                    **beh_losses,
                }
                experiment.log_metrics(last_metrics)

                if (
                    cfg.checkpoint_every > 0
                    and (step_idx + 1) % cfg.checkpoint_every == 0
                ):
                    payload = {
                        "world_model": _get_state_dict(world_model),
                        "actor": _get_state_dict(actor),
                        "critic": _get_state_dict(critic),
                        "target_critic": _get_state_dict(target_critic),
                        "world_opt": world_opt.state_dict(),
                        "actor_opt": actor_opt.state_dict(),
                        "critic_opt": critic_opt.state_dict(),
                        "config": asdict(cfg),
                        "env_steps": env_steps,
                        "train_steps": train_steps,
                    }
                    ckpt_name = f"ckpt_{step_idx + 1}.pt"
                    experiment.save_checkpoint(ckpt_name, payload)

        except Exception as exc:
            experiment.write_failure_bundle(
                kind="dreamer_v3_sync_train",
                error=exc,
                extra={"last_metrics": last_metrics},
                repro=repro,
            )
            raise
        finally:
            env.close()

    else:
        # ========== ASYNC TRAINING (original) ==========
        engine = AsyncDreamerV3Engine(
            engine_cfg,
            env=env,
            actor_core=actor_core,
            world_model_update=world_model_update,
            behavior_update=behavior_update,
            experiment=experiment,
        )

        try:
            for step_idx in range(iterations):
                last_metrics = engine.run(num_iterations=1)
                if (
                    cfg.checkpoint_every > 0
                    and (step_idx + 1) % cfg.checkpoint_every == 0
                ):
                    payload = {
                        "world_model": _get_state_dict(world_model),
                        "actor": _get_state_dict(actor),
                        "critic": _get_state_dict(critic),
                        "target_critic": _get_state_dict(target_critic),
                        "world_opt": world_opt.state_dict(),
                        "actor_opt": actor_opt.state_dict(),
                        "critic_opt": critic_opt.state_dict(),
                        "config": asdict(cfg),
                        "env_steps": last_metrics.get("env_steps"),
                        "train_steps": last_metrics.get("train_steps"),
                    }
                    ckpt_name = f"ckpt_{step_idx + 1}.pt"
                    experiment.save_checkpoint(ckpt_name, payload)
        except Exception as exc:
            experiment.write_failure_bundle(
                kind="dreamer_v3_train",
                error=exc,
                extra={"last_metrics": last_metrics},
                repro=repro,
            )
            raise
        finally:
            engine.close()

    payload = {
        "world_model": _get_state_dict(world_model),
        "actor": _get_state_dict(actor),
        "critic": _get_state_dict(critic),
        "target_critic": _get_state_dict(target_critic),
        "world_opt": world_opt.state_dict(),
        "actor_opt": actor_opt.state_dict(),
        "critic_opt": critic_opt.state_dict(),
        "config": asdict(cfg),
        "env_steps": last_metrics.get("env_steps"),
        "train_steps": last_metrics.get("train_steps"),
    }
    experiment.save_checkpoint("checkpoint.pt", payload)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s. Run dir: {experiment.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
